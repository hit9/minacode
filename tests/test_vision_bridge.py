"""Explicit ViewImage routing: main-first raw observation, [vision] only as a bounded fallback."""

import json

import pytest
from PIL import Image

from minacode.base import Billing, ConfigError, ModelError, ToolError
from minacode.config import Config, ProviderConfig
from minacode.context import ContextManager
from minacode.image import IMAGE_REFS_KEY, IMAGE_TEXT_ONLY_KEY, TOOL_IMAGE_OBSERVATION_PREFIX, ImageInputs
from minacode.model import ModelClient
from minacode.prompts import VISION_OBSERVE_DEFAULT_QUESTION, VISION_OBSERVE_PROMPT
from minacode.runner import ToolRunner
from minacode.session import Session
from minacode.tools import Tool, ViewImageTool
from minacode.vision import VisionObserver
from model_harness import _MockClientFactory

OBSERVATION = "The screenshot shows a terminal error."


def image_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), (12, 34, 56)).save(path, format="PNG")
    return path


def session(tmp_path, *, vision=True, model="main-model", vision_api="chat"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = Config(data_dir=str(tmp_path / "data"))
    config.providers = {"default": ProviderConfig(url="http://main.test", key="key", model=model)}
    if vision:
        config.providers["v"] = ProviderConfig(url="http://vision.test", key="vkey", model="vision-model", api=vision_api)
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


def test_view_image_on_unknown_route_keeps_raw_observation_even_with_vision(tmp_path):
    """Main-first: [vision] never intercepts an unknown route; the raw observation stays intact."""
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "shot.png")

    tool = ViewImageTool(s, ["shot.png", "exact error?"])
    output = tool.call()  # nothing is bridged, so no runner is required
    observation = tool.model_observation()

    assert output.startswith("<ViewImage") and "vision=" not in output
    assert observation is not None
    assert ImageInputs.input_refs(observation)
    assert observation.get(IMAGE_TEXT_ONLY_KEY) is not True


def test_view_image_without_vision_returns_direct_model_observation(tmp_path):
    s = session(tmp_path, vision=False)
    image_file(tmp_path / "shot.png")
    tool = ViewImageTool(s, ["shot.png", "what is shown?"])

    output = tool.call()
    observation = tool.model_observation()

    assert output.startswith("<ViewImage") and "vision=" not in output
    assert observation is not None
    assert ImageInputs.input_refs(observation)


@pytest.mark.parametrize(
    ("api", "image_type", "text_type"),
    [("chat", "image_url", "text"), ("responses", "input_image", "input_text"), ("anthropic", "image", "text")],
)
def test_vision_observe_wire_protocol_uses_configured_entry(tmp_path, monkeypatch, api, image_type, text_type):
    """The [vision] wire shape is protocol-specific, reached only when the route bridges."""
    s = session(tmp_path, model="deepseek-chat", vision=True, vision_api=api)
    image_file(tmp_path / "shot.png")
    captured = {}

    def fake_api_request(self, messages, tools, **kwargs):
        captured.update(messages=messages, tools=tools, kwargs=kwargs)
        return {}, [], OBSERVATION

    monkeypatch.setattr(ModelClient, "api_request", fake_api_request)
    model = ModelClient(s)
    observation = VisionObserver(model).observe((s.images.load(str(tmp_path / "shot.png")),), "exact error?")

    assert observation == OBSERVATION
    assert captured["tools"] is None
    assert captured["kwargs"]["allow_stream"] is False
    assert captured["kwargs"]["provider"] is s.config.providers["v"]
    assert captured["kwargs"]["billing"] is Billing.VISION
    assert captured["messages"][0]["content"] == VISION_OBSERVE_PROMPT
    content = captured["messages"][1]["content"]
    assert [part["type"] for part in content] == [image_type, text_type]
    assert content[-1]["text"] == "exact error?"
    assert IMAGE_REFS_KEY not in json.dumps(captured["messages"])


def test_static_text_only_view_image_bridges_with_default_question(tmp_path, monkeypatch):
    s = session(tmp_path, model="deepseek-chat", vision=True)
    image_file(tmp_path / "shot.png")
    captured = {}

    def fake_api_request(self, messages, tools, **kwargs):
        captured["question"] = messages[1]["content"][-1]["text"]
        return {}, [], OBSERVATION

    monkeypatch.setattr(ModelClient, "api_request", fake_api_request)
    tool, output = call_view_image(s, ["shot.png"])

    assert captured["question"] == VISION_OBSERVE_DEFAULT_QUESTION
    assert output.endswith("\n" + OBSERVATION)
    assert tool.vision_entry_label == "v/vision-model"
    observation = tool.model_observation()
    assert observation is not None
    assert observation[IMAGE_TEXT_ONLY_KEY] is True
    assert observation["content"] == f"{TOOL_IMAGE_OBSERVATION_PREFIX}\n{OBSERVATION}"
    assert not ImageInputs.input_refs(observation)


def test_vision_observe_joins_totals_but_keeps_main_last_snapshot(tmp_path, monkeypatch):
    """Vision usage joins the session totals but must not overwrite the last-request ctx/cache
    snapshot the status bar reads."""
    s = session(tmp_path, model="main-model", vision=True, vision_api="chat")
    s.usage.add({"prompt_tokens": 10_000, "completion_tokens": 500, "total_tokens": 10_500}, 200_000)
    image_file(tmp_path / "shot.png")
    model = ModelClient(s)
    monkeypatch.setattr(
        model,
        "client",
        _MockClientFactory(
            [
                (
                    200,
                    {
                        "id": "c",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "vision-model",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "the screen shows an error"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 300, "completion_tokens": 20, "total_tokens": 320},
                    },
                )
            ]
        ),
    )
    VisionObserver(model).observe((s.images.load(str(tmp_path / "shot.png")),), "")
    assert (s.usage.calls, s.usage.total_tokens) == (2, 10_820)
    assert (s.usage.last_prompt_tokens, s.usage.last_prompt_budget) == (10_000, 200_000)


def test_vision_bridge_requires_runner_and_reports_errors(tmp_path, monkeypatch):
    s = session(tmp_path, model="deepseek-chat", vision=True)
    image_file(tmp_path / "shot.png")
    with pytest.raises(ToolError, match="requires ToolRunner"):
        ViewImageTool(s, ["shot.png"]).call()

    def fail(*args, **kwargs):
        raise ModelError("boom")

    monkeypatch.setattr(ModelClient, "api_request", fail)
    with pytest.raises(ToolError, match="Vision observation failed: boom"):
        call_view_image(s, ["shot.png"])


def test_environment_advertises_vision_as_image_fallback_once_and_schema_is_stable(tmp_path):
    with_vision = session(tmp_path / "with", model="main-model", vision=True)
    without_vision = session(tmp_path / "without", model="main-model", vision=False)

    environment = ContextManager(with_vision).environment()
    assert environment.count("- vision: v/vision-model (available as image fallback)") == 1
    assert "- vision:" not in ContextManager(without_vision).environment()

    def schema(s):
        return next(item for item in Tool.resolved_schemas(s) if item["function"]["name"] == "ViewImage")

    assert schema(with_vision) == schema(without_vision)
    description = schema(with_vision)["function"]["description"]
    assert "session-owned path" in description
    assert "vision provider as fallback" in description

    with_vision.tool_names = ("Read",)
    assert "- vision:" not in ContextManager(with_vision).environment()
