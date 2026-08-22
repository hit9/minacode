"""Vision bridge: a dedicated [vision] provider observes images for a text-only main model.

Black-box acceptance per the bridge spec. Routing is a harness decision made from
ImageInputs.support() plus the [vision] config, never a model choice: main model supports
images -> direct attach (vision config ignored); support False + vision configured -> bridge;
support False + no vision -> today's disabled error. In auto mode an unknown provider routes
images to the main model first and learns from the outcome, so the bridge engages only once
the model is known to reject images. The vision request is a tool-less, non-streaming
api_request carrying the image by the vision entry's protocol, and a bridged ViewImage call
attaches no image to the next main-model request.
"""

import base64
import json
import threading

import pytest
from model_harness import _AnthropicMockClientFactory, _MockClientFactory
from PIL import Image

from minacode.base import ConfigError, ModelError, ToolCall, ToolError
from minacode.config import Config, ProviderConfig
from minacode.context import ContextManager
from minacode.engine import Agent
from minacode.image import IMAGE_MARKER, IMAGE_REFS_KEY, ImageInputs, UserInput
from minacode.model import ModelClient
from minacode.prompts import VISION_OBSERVE_DEFAULT_QUESTION, VISION_OBSERVE_PROMPT
from minacode.render import StatusBar
from minacode.runner import ToolRunner
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


def call_view_image(s, args):
    """Exercise ViewImage through its orchestration boundary, as production does."""
    tool = ViewImageTool(s, args)
    output = ToolRunner(s, ContextManager(s), output_fn=lambda _text: None).call_tool(tool)
    return tool, output


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

    tool, output = call_view_image(s, [path.name])

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


def test_bridged_view_image_requires_runner_owned_vision_client(tmp_path):
    s = session(tmp_path, image_input="off")
    path = image_file(tmp_path / "shot.png")

    with pytest.raises(ToolError, match="requires ToolRunner"):
        ViewImageTool(s, [path.name]).call()


def test_view_image_with_unknown_support_does_not_bridge(tmp_path):
    # auto + [vision] with unknown support routes to the main model (and learns from the
    # outcome) instead of bridging; only a known text-only model uses the bridge.
    s = session(tmp_path)  # auto, nothing learned yet -> support() is None
    assert s.images.support() is None
    path = image_file(tmp_path / "shot.png")
    tool = ViewImageTool(s, [path.name])
    output = tool.call()
    assert "vision=" not in output
    assert tool.model_observation() is not None  # the image rides the next main-model request


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


def test_bridged_attachment_turn_survives_a_stale_cancel_from_the_previous_turn(tmp_path, monkeypatch):
    # Ctrl-C on turn N leaves ModelClient.cancel_requested set. vision_observe() runs before the
    # turn's first request() and was the only client entry that never cleared the flag, so the next
    # bridged attachment turn raised KeyboardInterrupt before its first request and lost the input
    # (checkpoint_turn never ran). A fresh observation must start with a clean flag, like request().
    s = session(tmp_path, image_input="off")  # known text-only main model -> bridge
    path = image_file(tmp_path / "shot.png")
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
            ),
            (
                200,
                {
                    "id": "chatcmpl-2",
                    "object": "chat.completion",
                    "created": 2,
                    "model": "main-model",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
                },
            ),
        ]
    )
    monkeypatch.setattr(ModelClient, "client", factory)

    agent = Agent(s, output_fn=lambda _text: None)
    agent.model.cancel_requested.set()  # what model.cancel() leaves behind after a Ctrl-C
    image = s.images.load(str(path), force=True)

    assert agent.run(UserInput(f"what is shown {IMAGE_MARKER}", (image,))) == "done"
    # The turn checkpointed normally: the input and its observation reached history.
    assert any(OBSERVATION_TEXT in str(message) for message in s.messages)


def _observation_response(text: str) -> tuple[int, dict]:
    return 200, {
        "id": "chatcmpl-v",
        "object": "chat.completion",
        "created": 1,
        "model": "vision-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _main_answer_response(text: str) -> tuple[int, dict]:
    return 200, {
        "id": "chatcmpl-a",
        "object": "chat.completion",
        "created": 2,
        "model": "main-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    }


class _BlockingMockClientFactory(_MockClientFactory):
    """A mock factory whose wire handler blocks until released, simulating a slow provider read."""

    def __init__(self, response, entered: threading.Event, release: threading.Event):
        super().__init__([response])
        self._entered = entered
        self._release = release

    def _next_response(self, request):
        self._entered.set()
        self._release.wait()
        return super()._next_response(request)


def test_bridged_observation_text_is_not_scanned_for_mentions(tmp_path, monkeypatch):
    # The observation is the vision model's output, not the user's typing: an @file: reference
    # shown in a screenshot must not inline that file into the turn's context.
    s = session(tmp_path, image_input="off")
    (tmp_path / "secrets.toml").write_text("token = 'super-secret'")
    path = image_file(tmp_path / "shot.png")
    factory = _MockClientFactory([_observation_response("The screenshot shows @file:secrets.toml"), _main_answer_response("done")])
    monkeypatch.setattr(ModelClient, "client", factory)

    agent = Agent(s, output_fn=lambda _text: None)
    image = s.images.load(str(path), force=True)

    assert agent.run(UserInput(f"what is shown {IMAGE_MARKER}", (image,))) == "done"
    assert not any("super-secret" in str(message) or "FILE MENTIONS" in str(message) for message in s.messages)


def test_bridged_typed_text_mentions_still_resolve(tmp_path, monkeypatch):
    # The fix must not quiet mentions the user actually typed.
    s = session(tmp_path, image_input="off")
    (tmp_path / "secrets.toml").write_text("token = 'super-secret'")
    path = image_file(tmp_path / "shot.png")
    factory = _MockClientFactory([_observation_response("plain description"), _main_answer_response("done")])
    monkeypatch.setattr(ModelClient, "client", factory)

    agent = Agent(s, output_fn=lambda _text: None)
    image = s.images.load(str(path), force=True)

    assert agent.run(UserInput(f"read @file:secrets.toml {IMAGE_MARKER}", (image,))) == "done"
    assert any("FILE MENTIONS" in str(message) and "super-secret" in str(message) for message in s.messages)


def test_bridged_view_image_aborts_when_the_agent_is_cancelled(tmp_path, monkeypatch):
    # The bridged observation runs on the runner-owned vision client, so Agent.cancel() reaches
    # the in-flight request and the tool aborts instead of waiting out the provider timeout.
    s = session(tmp_path, image_input="off")
    path = image_file(tmp_path / "shot.png")
    entered = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(ModelClient, "client", _BlockingMockClientFactory(_observation_response(OBSERVATION_TEXT), entered, release))

    agent = Agent(s, output_fn=lambda _text: None)
    outcome: list[str] = []

    def run_tool() -> None:
        try:
            agent.tools.run([ToolCall("image", "ViewImage", [path.name])])
            outcome.append("returned")
        except KeyboardInterrupt:
            outcome.append("cancelled")

    thread = threading.Thread(target=run_tool, daemon=True)
    thread.start()
    try:
        assert entered.wait(5), "the bridged vision request never reached the wire"
        agent.cancel()
        thread.join(5)
        assert not thread.is_alive(), "Agent.cancel did not abort the in-flight bridged ViewImage"
        assert outcome == ["cancelled"]
    finally:
        release.set()  # let the blocked mock read return so the worker thread drains


def test_auto_unknown_with_vision_direct_attaches_and_learns_support(tmp_path, monkeypatch):
    # auto + [vision] with unknown support must route to the main model once, so a vision-capable
    # model is never permanently hijacked into the bridge: the old rule bridged on None, and the
    # bridge starved the learning that could have disproved it.
    s = session(tmp_path)  # auto, support unknown
    path = image_file(tmp_path / "shot.png")
    factory = _MockClientFactory([_main_answer_response("done")])
    monkeypatch.setattr(ModelClient, "client", factory)
    agent = Agent(s, output_fn=lambda _text: None)
    image = s.images.load(str(path), force=True)

    assert s.images.bridging() is False
    assert agent.run(UserInput(f"what is shown {IMAGE_MARKER}", (image,))) == "done"
    assert len(factory.calls) == 1  # the vision entry was never asked
    assert s.images.support() is True  # learned from the successful main-model request
    assert s.images.bridging() is False


def test_auto_unknown_learns_false_and_then_bridges(tmp_path, monkeypatch):
    # A main model that explicitly rejects images teaches support False and recovers the same turn
    # through the bridge instead of requiring the user to submit or ViewImage again.
    s = session(tmp_path)  # auto, support unknown
    path = image_file(tmp_path / "shot.png")
    factory = _MockClientFactory(
        [
            (
                400,
                {
                    "error": {
                        "code": "InvalidParameter",
                        "message": "Model only support text input",
                        "type": "BadRequest",
                    }
                },
            ),
            _observation_response(OBSERVATION_TEXT),
            _main_answer_response("done"),
        ]
    )
    monkeypatch.setattr(ModelClient, "client", factory)
    agent = Agent(s, output_fn=lambda _text: None)
    image = s.images.load(str(path), force=True)

    assert agent.run(UserInput(f"what is shown {IMAGE_MARKER}", (image,))) == "done"
    assert s.images.support() is False
    assert s.images.bridging() is True
    assert len(factory.calls) == 3
    assert OBSERVATION_TEXT in str(s.messages)


def test_ambiguous_image_400_learns_only_after_text_only_retry_succeeds(tmp_path, monkeypatch):
    s = session(tmp_path)
    path = image_file(tmp_path / "shot.png")
    factory = _MockClientFactory(
        [
            (422, {"error": {"code": "InvalidParameter", "message": "Invalid request", "type": "BadRequest"}}),
            _observation_response(OBSERVATION_TEXT),
            _main_answer_response("recovered"),
        ]
    )
    monkeypatch.setattr(ModelClient, "client", factory)
    agent = Agent(s, output_fn=lambda _text: None)
    image = s.images.load(str(path), force=True)

    assert agent.run(UserInput(f"what is shown {IMAGE_MARKER}", (image,))) == "recovered"
    assert s.images.support() is False
    assert len(factory.calls) == 3


def test_ambiguous_image_400_does_not_learn_when_text_only_retry_fails(tmp_path, monkeypatch):
    s = session(tmp_path)
    path = image_file(tmp_path / "shot.png")
    factory = _MockClientFactory(
        [
            (400, {"error": {"code": "InvalidParameter", "message": "Invalid request"}}),
            _observation_response(OBSERVATION_TEXT),
            (400, {"error": {"code": "InvalidParameter", "message": "Still invalid"}}),
        ]
    )
    monkeypatch.setattr(ModelClient, "client", factory)
    agent = Agent(s, output_fn=lambda _text: None)
    image = s.images.load(str(path), force=True)

    with pytest.raises(ModelError, match="Still invalid"):
        agent.run(UserInput(f"what is shown {IMAGE_MARKER}", (image,)))
    assert s.images.support() is None
    assert len(factory.calls) == 3


def test_non_image_candidate_error_does_not_probe_the_vision_bridge(tmp_path, monkeypatch):
    s = session(tmp_path)
    path = image_file(tmp_path / "shot.png")
    factory = _MockClientFactory([(401, {"error": {"message": "Invalid API key"}})])
    monkeypatch.setattr(ModelClient, "client", factory)
    agent = Agent(s, output_fn=lambda _text: None)
    image = s.images.load(str(path), force=True)

    with pytest.raises(ModelError, match="Invalid API key"):
        agent.run(UserInput(f"what is shown {IMAGE_MARKER}", (image,)))
    assert s.images.support() is None
    assert len(factory.calls) == 1


def test_unknown_view_image_rejection_recovers_through_bridge_with_question(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "shot.png")

    class Model:
        def __init__(self):
            self.requests = []
            self.observations = []

        def request(self, messages, tools=None):
            self.requests.append(messages)
            if len(self.requests) == 1:
                return {}, [ToolCall("image", "ViewImage", ["shot.png", "read the error"])], ""
            if len(self.requests) == 2:
                raise ModelError("Error code: 400 - Invalid request")
            return {"role": "assistant", "content": "recovered"}, [], "recovered"

        def vision_observe(self, images, question=""):
            self.observations.append((images, question))
            return OBSERVATION_TEXT

    agent = Agent(s, output_fn=lambda _text: None)
    model = Model()
    agent.model = model

    assert agent.run("inspect the screenshot") == "recovered"
    assert s.images.support() is False
    assert len(model.requests) == 3
    assert model.observations[0][1] == "read the error"
    assert not ImageInputs.has_images(model.requests[2])
    assert OBSERVATION_TEXT in str(model.requests[2])


def test_learned_support_survives_a_snapshot_reload(tmp_path, monkeypatch):
    # Persistence is what keeps a text-only main model from re-taking the failed first turn on
    # every new session: the verdict rides the snapshot like the rest of the session state.
    s = session(tmp_path)  # auto, support unknown
    path = image_file(tmp_path / "shot.png")
    factory = _MockClientFactory([_main_answer_response("done")])
    monkeypatch.setattr(ModelClient, "client", factory)
    agent = Agent(s, output_fn=lambda _text: None)
    image = s.images.load(str(path), force=True)
    assert agent.run(UserInput(f"what is shown {IMAGE_MARKER}", (image,))) == "done"
    assert s.images.support() is True

    s.save_snapshot()
    resumed = Session.load_snapshot(s.uid, config=s.config, cwd=s.cwd)
    assert resumed.images.support() is True
    assert resumed.images.bridging() is False


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

    call_view_image(s, [path.name])

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

    call_view_image(s, [path.name])

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

    call_view_image(s, [path.name])

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


# --- 观察请求不污染主模型的 cache/ctx 读数（回归）---


def test_observation_keeps_the_main_request_cache_snapshot(tmp_path, monkeypatch):
    """A vision observation is billed to the session totals but is not a main-model request, so it
    must not overwrite the last-request snapshot the status bar reads. Regression: an inline-image
    follow-up ran an observation with no prefix reuse (cached_tokens=0) and the status bar's cache%
    dropped to 0 until the next main-model request re-recorded it."""
    s = session(tmp_path, image_input="off", api="chat")
    path = image_file(tmp_path / "shot.png")
    factory = _MockClientFactory(
        [
            (  # the main-model request rides the warm conversation prefix
                200,
                {
                    "id": "chatcmpl-main",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "main-model",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105, "prompt_tokens_details": {"cached_tokens": 80}},
                },
            ),
            (  # the vision observation has no prefix reuse at all
                200,
                {
                    "id": "chatcmpl-vision",
                    "object": "chat.completion",
                    "created": 2,
                    "model": "vision-model",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": OBSERVATION_TEXT}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            ),
        ]
    )
    monkeypatch.setattr(ModelClient, "client", factory)

    model = ModelClient(s)
    model.request([{"role": "user", "content": "hello"}])
    assert s.usage.last_prompt_tokens == 100
    assert s.usage.last_cached_prompt_tokens == 80

    call_view_image(s, [path.name])  # one bridged observation through the real api_request path

    # The last-request snapshot still describes the main-model request, not the observation.
    assert s.usage.last_prompt_tokens == 100
    assert s.usage.last_cached_prompt_tokens == 80
    # The observation is still billed to the session totals exactly as before.
    assert s.usage.prompt_tokens == 110
    assert s.usage.cached_prompt_tokens == 80
    assert s.usage.calls == 2
    # And the status bar keeps reading the main-model cache ratio.
    bar = StatusBar(s)
    ctx_text = next(text for text, role in bar.entries(show_elapsed=False) if role == "ctx")
    assert ctx_text.endswith("· cache 80%")


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

    call_view_image(s, [path.name, "截图里的报错原文是什么"])

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

    call_view_image(s, [path.name])

    assert captured["messages"][1]["content"][-1]["text"] == VISION_OBSERVE_DEFAULT_QUESTION


# --- vision 端错误可定位（验收标准 6）---


def test_missing_vision_fields_name_the_entry_and_fields(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="off")
    s.config.providers["v"] = ProviderConfig(url="http://vision.test", key="", model="vision-model")
    path = image_file(tmp_path / "shot.png")

    with pytest.raises(ToolError) as caught:
        call_view_image(s, [path.name])

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
        call_view_image(s, [path.name])


# --- 观察请求后的主请求快照恢复（旗标 finally 清理，回归补充）---


def _usage_main_response(prompt=100, cached=80):
    return (
        200,
        {
            "id": "m",
            "object": "chat.completion",
            "created": 1,
            "model": "main",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": 5, "total_tokens": prompt + 5, "prompt_tokens_details": {"cached_tokens": cached}},
        },
    )


def _usage_obs_response():
    return (
        200,
        {
            "id": "v",
            "object": "chat.completion",
            "created": 2,
            "model": "vision",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": OBSERVATION_TEXT}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )


def test_main_request_after_observation_re_touches_the_snapshot(tmp_path, monkeypatch):
    """The flag must not linger: a main-model request after the observation still updates the
    last-request ctx/cache snapshot the status bar reads (a stale `touch_last=False` would freeze
    the readout on the pre-observation main request)."""
    s = session(tmp_path, image_input="off", api="chat")
    path = image_file(tmp_path / "shot.png")
    factory = _MockClientFactory([_usage_main_response(), _usage_obs_response(), _usage_main_response(prompt=200, cached=150)])
    monkeypatch.setattr(ModelClient, "client", factory)

    model = ModelClient(s)
    model.request([{"role": "user", "content": "hello"}])
    call_view_image(s, [path.name])
    assert s.state.vision_observe_active is False
    model.request([{"role": "user", "content": "again"}])

    assert s.usage.last_prompt_tokens == 200
    assert s.usage.last_cached_prompt_tokens == 150


def test_failed_observation_clears_the_flag_and_next_main_request_touches(tmp_path, monkeypatch):
    """An observation that fails mid-request (finally) must clear the flag, so the next main-model
    request is not mislabeled as an observation and keeps updating the snapshot."""
    s = session(tmp_path, image_input="off", api="chat")
    path = image_file(tmp_path / "shot.png")
    factory = _MockClientFactory([_usage_main_response(), 500, _usage_main_response(prompt=200, cached=150)])
    monkeypatch.setattr(ModelClient, "client", factory)

    model = ModelClient(s)
    model.request([{"role": "user", "content": "hello"}])
    with pytest.raises(ToolError, match="Vision bridge failed"):
        call_view_image(s, [path.name])
    assert s.state.vision_observe_active is False
    model.request([{"role": "user", "content": "again"}])

    assert s.usage.last_prompt_tokens == 200
    assert s.usage.last_cached_prompt_tokens == 150
