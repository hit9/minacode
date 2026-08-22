"""Explicit ViewImage routing through an optional [vision] provider."""

import json

import pytest
from PIL import Image

from minacode.base import ConfigError, ModelError, ToolError
from minacode.config import Config, ProviderConfig
from minacode.context import ContextManager
from minacode.image import IMAGE_REFS_KEY, ImageInputs
from minacode.model import ModelClient
from minacode.prompts import VISION_OBSERVE_DEFAULT_QUESTION, VISION_OBSERVE_PROMPT
from minacode.runner import ToolRunner
from minacode.session import Session
from minacode.tools import Tool, ViewImageTool


OBSERVATION = "The screenshot shows a terminal error."


def image_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), (12, 34, 56)).save(path, format="PNG")
    return path


def session(tmp_path, *, vision=True, api="chat"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = Config(data_dir=str(tmp_path / "data"))
    config.providers = {"default": ProviderConfig(url="http://main.test", key="key", model="main-model")}
    if vision:
        config.providers["v"] = ProviderConfig(url="http://vision.test", key="vkey", model="vision-model", api=api)
        config.vision_provider = "v"
    return Session(cwd=str(tmp_path), config=config)


def call_view_image(s, args):
    tool = ViewImageTool(s, args)
    output = ToolRunner(s, ContextManager(s), output_fn=lambda _text: None).call_tool(tool)
    return tool, output


def test_config_parses_vision_provider_and_ignores_obsolete_image_input():
    config = Config.from_dict(
        {
            "provider": {
                "default": {"url": "http://main", "key": "k", "model": "m", "image_input": "off"},
                "v": {"url": "http://vision", "key": "vk", "model": "vm"},
            },
            "vision": {"provider": "v"},
        }
    )
    assert config.vision_provider == "v"
    assert not hasattr(config.provider, "image_input")


def test_config_rejects_unknown_vision_provider():
    with pytest.raises(ConfigError, match="vision.provider `nope` does not exist"):
        Config.from_dict({"provider": {"default": {"url": "u", "key": "k", "model": "m"}}, "vision": {"provider": "nope"}})


@pytest.mark.parametrize(
    ("api", "image_type"),
    [("chat", "image_url"), ("responses", "input_image"), ("anthropic", "image")],
)
def test_explicit_view_image_uses_configured_vision_protocol(tmp_path, monkeypatch, api, image_type):
    s = session(tmp_path, api=api)
    image_file(tmp_path / "shot.png")
    captured = {}

    def fake_api_request(self, messages, tools, **kwargs):
        captured.update(messages=messages, tools=tools, kwargs=kwargs)
        return {"role": "assistant", "content": OBSERVATION}, [], OBSERVATION

    monkeypatch.setattr(ModelClient, "api_request", fake_api_request)
    tool, output = call_view_image(s, ["shot.png", "exact error?"])

    assert output.endswith("\n" + OBSERVATION)
    assert tool.model_observation() is None
    assert captured["tools"] is None
    assert captured["kwargs"]["allow_stream"] is False
    assert captured["kwargs"]["provider"] is s.config.providers["v"]
    assert captured["messages"][0]["content"] == VISION_OBSERVE_PROMPT
    content = captured["messages"][1]["content"]
    assert [part["type"] for part in content] == [image_type, "input_text" if api == "responses" else "text"]
    assert content[-1]["text"] == "exact error?"
    assert IMAGE_REFS_KEY not in json.dumps(captured["messages"])


def test_explicit_view_image_uses_default_question(tmp_path, monkeypatch):
    s = session(tmp_path)
    image_file(tmp_path / "shot.png")
    captured = {}

    def fake_api_request(self, messages, tools, **kwargs):
        captured["question"] = messages[1]["content"][-1]["text"]
        return {}, [], OBSERVATION

    monkeypatch.setattr(ModelClient, "api_request", fake_api_request)
    call_view_image(s, ["shot.png"])
    assert captured["question"] == VISION_OBSERVE_DEFAULT_QUESTION


def test_view_image_without_vision_returns_direct_model_observation(tmp_path):
    s = session(tmp_path, vision=False)
    image_file(tmp_path / "shot.png")
    tool = ViewImageTool(s, ["shot.png", "what is shown?"])

    output = tool.call()
    observation = tool.model_observation()

    assert output.startswith("<ViewImage") and "vision=" not in output
    assert observation is not None
    assert ImageInputs.input_refs(observation)


def test_configured_vision_requires_runner_and_reports_errors(tmp_path, monkeypatch):
    s = session(tmp_path)
    image_file(tmp_path / "shot.png")
    with pytest.raises(ToolError, match="requires ToolRunner"):
        ViewImageTool(s, ["shot.png"]).call()

    def fail(*args, **kwargs):
        raise ModelError("boom")

    monkeypatch.setattr(ModelClient, "api_request", fail)
    with pytest.raises(ToolError, match="Vision observation failed: boom"):
        call_view_image(s, ["shot.png"])


def test_environment_advertises_configured_vision_once_and_schema_is_stable(tmp_path):
    with_vision = session(tmp_path / "with")
    without_vision = session(tmp_path / "without", vision=False)

    environment = ContextManager(with_vision).environment()
    assert environment.count("- vision: v/vision-model (available through ViewImage)") == 1
    assert "- vision:" not in ContextManager(without_vision).environment()

    def schema(s):
        return next(item for item in Tool.resolved_schemas(s) if item["function"]["name"] == "ViewImage")

    assert schema(with_vision) == schema(without_vision)
    assert "active model cannot consume images directly" in schema(with_vision)["function"]["description"]

    with_vision.tool_names = ("Read",)
    assert "- vision:" not in ContextManager(with_vision).environment()
