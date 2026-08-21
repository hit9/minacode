"""Vision bridge: a dedicated [vision] provider observes images for a text-only main model.

Black-box acceptance per the bridge spec. Routing is a harness decision made from
ImageInputs.support() plus the [vision] config, never a model choice: main model supports
images -> direct attach (vision config ignored); support False or None + vision configured
-> bridge; support False + no vision -> today's disabled error. The vision request is a
tool-less, non-streaming api_request carrying the image by the vision entry's protocol, and
a bridged ViewImage call attaches no image to the next main-model request.
"""

import base64
import json

import pytest
from model_harness import _AnthropicMockClientFactory, _MockClientFactory
from PIL import Image

from minacode.base import ConfigError, ModelError, ToolCall, ToolError
from minacode.config import Config, ProviderConfig
from minacode.engine import Agent
from minacode.image import IMAGE_REFS_KEY, ImageInputs
from minacode.model import ModelClient
from minacode.prompts import VISION_OBSERVE_DEFAULT_QUESTION, VISION_OBSERVE_PROMPT
from minacode.session import Session
from minacode.tools import ViewImageTool

OBSERVATION_TEXT = "The screenshot shows a terminal with a red error line."


def image_file(path, *, size=(32, 24), image_format="PNG", color=(12, 34, 56)):
    Image.new("RGB", size, color).save(path, format=image_format)
    return path


def session(tmp_path, *, image_input="auto", vision=True, api="chat"):
    config = Config(data_dir=str(tmp_path / "data"))
    config.providers = {"default": ProviderConfig(url="http://main.test", key="key", model="main-model")}
    if vision:
        config.providers["v"] = ProviderConfig(url="http://vision.test", key="vkey", model="vision-model", api=api)
        config.vision_provider = "v"
    config.provider.image_input = image_input
    return Session(cwd=str(tmp_path), config=config)


# --- 配置（验收标准 1）---


def test_config_parses_vision_provider_block():
    config = Config.from_dict(
        {
            "provider": {
                "default": {"url": "http://main", "key": "k", "model": "m"},
                "v": {"url": "http://vision", "key": "vk", "model": "vm"},
            },
            "vision": {"provider": "v"},
        }
    )
    assert config.vision_provider == "v"
    assert config.providers["v"].model == "vm"


def test_config_rejects_vision_provider_that_does_not_exist():
    with pytest.raises(ConfigError, match="vision.provider `nope` does not exist"):
        Config.from_dict(
            {
                "provider": {"default": {"url": "http://main", "key": "k", "model": "m"}},
                "vision": {"provider": "nope"},
            }
        )


def test_config_without_vision_block_matches_current_behavior(tmp_path):
    config = Config.from_dict({"provider": {"default": {"url": "http://main", "key": "k", "model": "m"}}})
    assert config.vision_provider == ""
    # No [vision] block: ViewImage keeps today's behavior exactly -- a disabled main model still errors.
    config.provider.image_input = "off"
    s = Session(cwd=str(tmp_path), config=config)
    path = image_file(tmp_path / "shot.png")
    with pytest.raises(ToolError, match="Image input is disabled"):
        ViewImageTool(s, [path.name]).call()


# --- 桥接触发（验收标准 2、7）---


def test_view_image_bridges_observation_when_main_model_cannot_see(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="off")
    assert s.images.support() is False
    path = image_file(tmp_path / "shot.png")
    captured = {}

    def fake_api_request(self, messages, tools, *, allow_stream=True, response_timeout=None, provider=None, json_object=False):
        captured.update(messages=messages, tools=tools, allow_stream=allow_stream, provider=provider, response_timeout=response_timeout)
        return {"role": "assistant", "content": OBSERVATION_TEXT}, [], OBSERVATION_TEXT

    monkeypatch.setattr(ModelClient, "api_request", fake_api_request)

    tool = ViewImageTool(s, [path.name])
    output = tool.call()

    first, _, observation = output.partition("\n")
    assert first.startswith("<ViewImage")
    assert 'path="shot.png"' in first
    assert 'media_type="image/png"' in first
    assert 'vision="v/vision-model"' in first
    assert observation == OBSERVATION_TEXT
    # No image is attached to the next main-model request.
    assert tool.model_observation() is None

    assert captured["tools"] is None
    assert captured["allow_stream"] is False
    assert captured["response_timeout"] == s.config.providers["v"].response_timeout
    assert captured["provider"] is s.config.providers["v"]
    assert captured["messages"][0]["role"] == "system"
    assert VISION_OBSERVE_PROMPT in captured["messages"][0]["content"]
    user = captured["messages"][1]
    assert [part["type"] for part in user["content"]] == ["image_url", "text"]
    assert user["content"][-1]["text"] == VISION_OBSERVE_DEFAULT_QUESTION
    assert IMAGE_REFS_KEY not in json.dumps(captured["messages"])


def test_view_image_bridges_when_support_is_unknown(tmp_path, monkeypatch):
    s = session(tmp_path)  # auto, nothing learned yet -> support() is None
    assert s.images.support() is None
    path = image_file(tmp_path / "shot.png")
    monkeypatch.setattr(ModelClient, "api_request", lambda self, messages, tools, **kwargs: ({"role": "assistant", "content": "o"}, [], "o"))
    output = ViewImageTool(s, [path.name]).call()
    assert 'vision="v/vision-model"' in output.splitlines()[0]


def test_bridged_view_image_attaches_no_image_to_the_next_main_request(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="off")
    image_file(tmp_path / "shot.png")
    monkeypatch.setattr(
        ModelClient, "api_request", lambda self, messages, tools, **kwargs: ({"role": "assistant", "content": OBSERVATION_TEXT}, [], OBSERVATION_TEXT)
    )

    class Model:
        def __init__(self):
            self.requests = []

        def request(self, messages, tools=None):
            self.requests.append(messages)
            if len(self.requests) == 1:
                return {}, [ToolCall("image", "ViewImage", ["shot.png"])], ""
            return {"role": "assistant", "content": "done"}, [], "done"

    agent = Agent(s, output_fn=lambda _text: None)
    agent.model = Model()

    assert agent.run("inspect the screenshot") == "done"
    assert [message["role"] for message in s.messages] == ["user", "assistant", "tool", "assistant"]
    assert not any(IMAGE_REFS_KEY in message for message in agent.model.requests[1])


# --- 三条协议 wire 形状（验收标准 7）---


def test_bridged_wire_keeps_the_prebuilt_image_blocks_on_chat(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="off", api="chat")
    path = image_file(tmp_path / "shot.png")
    encoded = base64.b64encode(path.read_bytes()).decode()
    factory = _MockClientFactory(
        [
            (
                200,
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "vision-model",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": OBSERVATION_TEXT}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            )
        ]
    )
    monkeypatch.setattr(ModelClient, "client", factory)

    ViewImageTool(s, [path.name]).call()

    body = json.loads(factory.calls[0].content)
    assert body["model"] == "vision-model"
    assert body["stream"] is False
    assert "tools" not in body
    assert VISION_OBSERVE_PROMPT in body["messages"][0]["content"]
    assert body["messages"][1] == {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
            {"type": "text", "text": VISION_OBSERVE_DEFAULT_QUESTION},
        ],
    }


def test_bridged_wire_keeps_the_prebuilt_image_blocks_on_responses(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="off", api="responses")
    path = image_file(tmp_path / "shot.png")
    encoded = base64.b64encode(path.read_bytes()).decode()
    factory = _MockClientFactory(
        [
            (
                200,
                {
                    "id": "resp_1",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "vision-model",
                    "output": [
                        {
                            "id": "msg_1",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": OBSERVATION_TEXT, "annotations": [], "logprobs": []}],
                        }
                    ],
                    "usage": {"input_tokens": 10, "input_tokens_details": {}, "output_tokens": 5, "total_tokens": 15},
                },
            )
        ]
    )
    monkeypatch.setattr(ModelClient, "client", factory)

    ViewImageTool(s, [path.name]).call()

    body = json.loads(factory.calls[0].content)
    assert body["model"] == "vision-model"
    assert body["stream"] is False
    assert "tools" not in body
    assert body["input"][0]["role"] == "system"
    assert body["input"][-1] == {
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": "data:image/png;base64," + encoded},
            {"type": "input_text", "text": VISION_OBSERVE_DEFAULT_QUESTION},
        ],
    }


def test_bridged_wire_keeps_the_prebuilt_image_blocks_on_anthropic(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="off", api="anthropic")
    path = image_file(tmp_path / "shot.png")
    encoded = base64.b64encode(path.read_bytes()).decode()
    factory = _AnthropicMockClientFactory(
        [
            (
                200,
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "vision-model",
                    "content": [{"type": "text", "text": OBSERVATION_TEXT}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 8, "output_tokens": 4},
                },
            )
        ]
    )
    monkeypatch.setattr(ModelClient, "anthropic_client", factory)

    ViewImageTool(s, [path.name]).call()

    body = json.loads(factory.calls[0].content)
    assert body["model"] == "vision-model"
    assert "tools" not in body
    assert VISION_OBSERVE_PROMPT in json.dumps(body["system"])
    user = body["messages"][0]
    assert user["role"] == "user"
    assert user["content"][0] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": encoded},
    }
    assert user["content"][-1]["type"] == "text"
    assert user["content"][-1]["text"] == VISION_OBSERVE_DEFAULT_QUESTION


# --- 直塞不受影响（验收标准 3）---


def test_view_image_stays_direct_when_main_model_supports_images(tmp_path):
    s = session(tmp_path, image_input="on")
    assert s.images.support() is True
    path = image_file(tmp_path / "shot.png")
    tool = ViewImageTool(s, [path.name])

    output = tool.call()
    observation = tool.model_observation()

    assert "vision=" not in output
    assert observation is not None
    assert ImageInputs.is_tool_observation(observation)
    assert s.images.chat_content(observation)[0]["type"] == "image_url"


# --- 未配置时维持现状（验收标准 4）---


def test_view_image_stays_disabled_without_vision_config(tmp_path):
    s = session(tmp_path, image_input="off", vision=False)
    path = image_file(tmp_path / "shot.png")
    with pytest.raises(ToolError, match="Image input is disabled"):
        ViewImageTool(s, [path.name]).call()


# --- 可选 question 参数（验收标准 5）---


def test_question_reaches_the_vision_request(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="off")
    path = image_file(tmp_path / "shot.png")
    captured = {}

    def fake_api_request(self, messages, tools, **kwargs):
        captured["messages"] = messages
        return {"role": "assistant", "content": "o"}, [], "o"

    monkeypatch.setattr(ModelClient, "api_request", fake_api_request)

    ViewImageTool(s, [path.name, "截图里的报错原文是什么"]).call()

    user = captured["messages"][1]
    assert [part["type"] for part in user["content"]] == ["image_url", "text"]
    assert user["content"][-1]["text"] == "截图里的报错原文是什么"


def test_vision_request_is_sent_with_a_default_question_when_none_given(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="off")
    path = image_file(tmp_path / "shot.png")
    captured = {}

    def fake_api_request(self, messages, tools, **kwargs):
        captured["messages"] = messages
        return {"role": "assistant", "content": "o"}, [], "o"

    monkeypatch.setattr(ModelClient, "api_request", fake_api_request)

    ViewImageTool(s, [path.name]).call()

    assert captured["messages"][1]["content"][-1]["text"] == VISION_OBSERVE_DEFAULT_QUESTION


# --- vision 端错误可定位（验收标准 6）---


def test_missing_vision_fields_name_the_entry_and_fields(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="off")
    s.config.providers["v"] = ProviderConfig(url="http://vision.test", key="", model="vision-model")
    path = image_file(tmp_path / "shot.png")

    with pytest.raises(ToolError) as caught:
        ViewImageTool(s, [path.name]).call()

    message = str(caught.value)
    assert "vision provider `v` is missing key" in message
    assert "[vision]" in message
    assert "[provider.v]" in message


def test_vision_observe_raises_a_model_error_with_actionable_message(tmp_path):
    s = session(tmp_path)
    s.config.providers["v"] = ProviderConfig(url="http://vision.test", key="", model="")
    image = s.images.load(image_file(tmp_path / "shot.png"), force=True)

    with pytest.raises(ModelError, match=r"vision provider `v` is missing key, model; check \[vision\]"):
        ModelClient(s).vision_observe((image,), "")


def test_vision_request_failure_surfaces_as_a_tool_error(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="off")
    path = image_file(tmp_path / "shot.png")

    def failing(self, messages, tools, **kwargs):
        raise ModelError("upstream 502")

    monkeypatch.setattr(ModelClient, "api_request", failing)

    with pytest.raises(ToolError, match="Vision bridge failed: upstream 502"):
        ViewImageTool(s, [path.name]).call()
