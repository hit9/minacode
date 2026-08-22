"""Main-first image routing: static text-only catalog, session-learned 400s, bounded vision fallback."""

import json

import pytest
from PIL import Image

from minacode.base import ImageRouteNotice, ModelError, ToolCall
from minacode.config import Config, ProviderConfig
from minacode.engine import Agent
from minacode.image import (
    ATTACHMENT_VISION_OBSERVATION_PREFIX,
    FAILED_IMAGE_CONTEXT_PREFIX,
    IMAGE_TEXT_ONLY_KEY,
    TOOL_IMAGE_OBSERVATION_KEY,
    TOOL_IMAGE_OBSERVATION_PREFIX,
    ImageInputs,
)
from minacode.prompts import VISION_OBSERVE_DEFAULT_QUESTION
from minacode.providers.compat import is_text_only_model
from minacode.session import Session

VISION_TEXT = "vision observation text"


def image_file(path, *, color=(12, 34, 56)):
    Image.new("RGB", (32, 24), color).save(path, format="PNG")
    return path


def session(tmp_path, *, vision=True, model="main-model"):
    config = Config(data_dir=str(tmp_path / "data"))
    config.providers = {"default": ProviderConfig(url="http://main.test", key="key", model=model)}
    if vision:
        config.providers["v"] = ProviderConfig(url="http://vision.test", key="vkey", model="vision-model")
        config.vision_provider = "v"
    return Session(cwd=str(tmp_path), config=config)


class FallbackModel:
    """Stands in for ModelClient: records main requests and vision observations.

    Outcomes are either an exception to raise or a ``(content, tool_calls)`` pair.
    """

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.requests = []
        self.vision_calls = []

    def request(self, messages, tools=None):
        self.requests.append(messages)
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        content, calls = outcome
        return {"role": "assistant", "content": content}, calls, content

    def vision_observe(self, images, question=""):
        # Mirrors ModelClient.vision_observe: an empty attachment question falls back to the
        # bounded default perception question; only ViewImage carries its own.
        self.vision_calls.append((tuple(image.name for image in images), question.strip() or VISION_OBSERVE_DEFAULT_QUESTION))
        return VISION_TEXT

    def cancel(self):
        pass


def run_with(s, model):
    agent = Agent(s, output_fn=lambda _text: None)
    agent.model = model
    notices = []
    agent.on_image_route_notice = notices.append
    return agent, notices


# --- static catalog ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # documented text families
        ("deepseek-chat", True),
        ("deepseek-reasoner", True),
        ("deepseek-v4-flash", True),
        ("deepseek-v4-pro", True),
        ("glm-5", True),
        ("glm-5-turbo", True),
        ("glm-5.1", True),
        ("glm-4.6", True),
        ("glm-4.7", True),
        ("kimi-k3", True),
        ("kimi-k2.7-code", True),
        ("kimi-k2", True),
        ("k3", True),
        ("kimi-for-coding", True),
        ("moonshot-v1-8k", True),
        ("moonshot-v1-32k", True),
        ("moonshot-v1-128k", True),
        ("minimax-m2.5", True),
        ("gpt-oss-20b", True),
        ("gpt-oss-120b", True),
        ("qwen3-max-2026-01-23", True),
        ("qwen3-coder-next", True),
        # vision variants and unknown models stay main-first (never in the negative list)
        ("deepseek-v4-flash-vision-exp", False),
        ("deepseek-v4-pro-vision", False),
        ("glm-5v", False),
        ("glm-4.6v", False),
        ("glm-4.5v", False),
        ("glm-ocr", False),
        ("kimi-k2.5", False),
        ("kimi-k2.6", False),
        ("moonshot-v1-32k-vision-preview", False),
        ("moonshot-v1-8k-vision-preview", False),
        ("qwen-vl-max", False),
        ("deepseek vision", False),
        ("gpt-4o", False),
        ("gpt-4o-mini", False),
        ("gpt-5", False),
        ("claude-sonnet-4-5", False),
        ("gemini-2.0-flash", False),
        ("", False),
    ],
)
def test_static_text_only_catalog_positives_and_negatives(model, expected):
    assert is_text_only_model(model) is expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("deepseek/deepseek-chat", True),
        ("z-ai/glm-5", True),
        ("bigmodel/glm-4.6", True),
        ("qwen/qwen3-coder-next", True),
        ("moonshotai/moonshot-v1-32k", True),
        ("openai/gpt-oss-120b", True),
        # a non-canonical vendor prefix keeps the ID unknown, so it is probed on the main model
        ("my-gateway/deepseek-chat", False),
        ("openrouter/deepseek-chat", False),
        ("deepseek/glm-5v", False),
    ],
)
def test_gateway_vendor_forms_match_only_canonical_vendors(model, expected):
    assert is_text_only_model(model) is expected


def test_resolve_folds_static_text_only_evidence():
    def resolved(model):
        return ProviderConfig(url="http://main.test", key="key", model=model).resolve()

    assert resolved("deepseek-chat").text_only is True
    assert resolved("deepseek/deepseek-chat").text_only is True
    assert resolved("glm-5v").text_only is False
    assert resolved("gpt-4o").text_only is False
    assert resolved("my-gateway/deepseek-chat").text_only is False


def test_route_identity_keys_learned_evidence_per_main_route(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    resolved = s.config.provider.resolve()
    assert s.image_route.identity() == ("default", resolved.api, resolved.base_url, "main-model")
    assert s.image_route.state() == "unknown"

    s.image_route.learn_text_only()
    assert s.image_route.state() == "text_only_learned"
    assert s.image_route.delivery() == "vision"

    other = session(tmp_path / "other", model="other-model", vision=True)
    assert other.image_route.identity() != s.image_route.identity()
    assert other.image_route.state() == "unknown"


# --- eligible 400 fallback --------------------------------------------------------------------


def test_eligible_400_attachment_falls_back_through_vision_once(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "shot.png")
    model = FallbackModel([ModelError("Error code: 400 - image input not supported"), ("described", [])])
    agent, notices = run_with(s, model)

    assert agent.run(s.images.recognize("inspect shot.png")) == "described"

    assert len(model.requests) == 2
    # the first request carries the raw image block on the main model
    assert any(ImageInputs.input_refs(message) for message in model.requests[0])
    # the retry projects no raw image at all: the converted occurrence is its plain text content
    assert "image_url" not in json.dumps(model.requests[1])
    converted = next(message for message in model.requests[1] if ImageInputs.refs(message))
    assert converted["content"] == f"inspect [Image #1 \u00b7 shot.png]\n\n{ATTACHMENT_VISION_OBSERVATION_PREFIX}\n{VISION_TEXT}"
    # exactly one vision observation of the current attachment, with the bounded default question
    assert model.vision_calls == [(("shot.png",), VISION_OBSERVE_DEFAULT_QUESTION)]
    assert s.image_route.state() == "text_only_learned"
    assert notices == [ImageRouteNotice("main model rejected image input (400); using v/vision-model", described_by="v/vision-model")]

    # durable text observation replaces the failed raw occurrence; refs stay for asset ownership
    message = s.messages[0]
    assert message[IMAGE_TEXT_ONLY_KEY] is True
    assert ImageInputs.refs(message) and not ImageInputs.input_refs(message)
    assert message["content"].startswith("inspect [Image #1")
    assert message["content"] == f"inspect [Image #1 \u00b7 shot.png]\n\n{ATTACHMENT_VISION_OBSERVATION_PREFIX}\n{VISION_TEXT}"
    assert FAILED_IMAGE_CONTEXT_PREFIX not in message["content"]


def test_static_text_only_attachment_goes_directly_to_vision(tmp_path):
    s = session(tmp_path, model="deepseek-chat", vision=True)
    image_file(tmp_path / "shot.png")
    model = FallbackModel([("done", [])])
    agent, notices = run_with(s, model)

    assert agent.run(s.images.recognize("inspect shot.png")) == "done"

    assert len(model.requests) == 1
    assert not any(ImageInputs.input_refs(message) for message in model.requests[0])
    assert model.vision_calls == [(("shot.png",), VISION_OBSERVE_DEFAULT_QUESTION)]
    assert s.image_route.state() == "text_only_static"
    # one gray routing notice with a described-by child, like the ViewImage tree rendering
    assert notices == [ImageRouteNotice("main model is text-only; image described through v/vision-model", described_by="v/vision-model")]


def test_static_text_only_without_vision_keeps_raw_attempt_and_original_error(tmp_path):
    s = session(tmp_path, model="deepseek-chat", vision=False)
    image_file(tmp_path / "shot.png")
    model = FallbackModel([ModelError("Error code: 400 - no vision configured")])
    agent, _notices = run_with(s, model)

    with pytest.raises(ModelError, match="no vision configured"):
        agent.run(s.images.recognize("inspect shot.png"))

    assert len(model.requests) == 1
    assert any(ImageInputs.input_refs(message) for message in model.requests[0])
    assert model.vision_calls == []
    assert s.image_route.state() == "text_only_static"
    assert s.messages[0][IMAGE_TEXT_ONLY_KEY] is True


def test_400_without_vision_keeps_original_error_and_notices(tmp_path):
    s = session(tmp_path, model="main-model", vision=False)
    image_file(tmp_path / "shot.png")
    model = FallbackModel([ModelError("Error code: 400 - image input not supported")])
    agent, notices = run_with(s, model)

    with pytest.raises(ModelError, match="image input not supported"):
        agent.run(s.images.recognize("inspect shot.png"))

    assert model.vision_calls == []
    assert notices == [ImageRouteNotice("main model rejected image input (400); no vision provider configured")]
    # evidence is recorded, but without [vision] delivery still raw-attempts
    assert s.image_route.state() == "text_only_learned"
    assert s.image_route.delivery() == "raw"


def test_multi_image_attachment_is_observed_once_with_both_images(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "one.png")
    image_file(tmp_path / "two.png", color=(65, 43, 21))
    model = FallbackModel([ModelError("Error code: 400 - boom"), ("ok", [])])
    agent, _notices = run_with(s, model)

    assert agent.run(s.images.recognize("compare one.png two.png")) == "ok"

    assert model.vision_calls == [(("one.png", "two.png"), VISION_OBSERVE_DEFAULT_QUESTION)]
    assert s.image_route.state() == "text_only_learned"


def test_view_image_observation_400_falls_back_with_the_tool_question(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "shot.png")
    model = FallbackModel(
        [
            ("", [ToolCall("view_image", "ViewImage", ["shot.png", "exact error?"])]),
            ModelError("Error code: 400 - image input not supported"),
            ("resolved", []),
        ]
    )
    agent, _notices = run_with(s, model)

    assert agent.run(s.images.recognize("debug the screenshot")) == "resolved"

    # raw ViewImage observation first, rejected with 400, then one vision observation with its question
    assert any(ImageInputs.input_refs(message) for message in model.requests[1])
    assert not any(ImageInputs.input_refs(message) for message in model.requests[2])
    assert model.vision_calls == [(("shot.png",), "exact error?")]
    assert s.image_route.state() == "text_only_learned"

    stored = next(message for message in s.messages if message.get(TOOL_IMAGE_OBSERVATION_KEY))
    assert stored[IMAGE_TEXT_ONLY_KEY] is True
    assert stored["content"] == f"{TOOL_IMAGE_OBSERVATION_PREFIX}\n{VISION_TEXT}"


def test_old_image_and_new_attachment_observe_only_the_new_one(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "first.png")
    image_file(tmp_path / "second.png", color=(65, 43, 21))
    model = FallbackModel([("first ok", []), ModelError("Error code: 400 - boom"), ("second ok", [])])
    agent, _notices = run_with(s, model)

    assert agent.run(s.images.recognize("look at first.png")) == "first ok"
    assert agent.run(s.images.recognize("now second.png")) == "second ok"

    assert model.vision_calls == [(("second.png",), VISION_OBSERVE_DEFAULT_QUESTION)]
    assert s.image_route.state() == "text_only_learned"
    # the retry projects no raw image block anywhere: the learned route suppresses the older
    # accepted history too, not only the failed occurrence (semantic refs stay, wire is clean)
    assert "image_url" not in json.dumps(model.requests[2])


def test_400_with_only_historical_images_does_not_learn(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "shot.png")
    model = FallbackModel([("first ok", []), ModelError("Error code: 400 - boom")])
    agent, notices = run_with(s, model)

    assert agent.run(s.images.recognize("inspect shot.png")) == "first ok"
    with pytest.raises(ModelError, match="boom"):
        agent.run("continue without new images")

    # the historical image is still resent raw (unknown route), but no current occurrence -> no fallback
    assert any(ImageInputs.input_refs(message) for message in model.requests[1])
    assert s.image_route.state() == "unknown"
    assert model.vision_calls == []
    assert notices == []


def test_queued_attachment_400_commits_once_after_fallback(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "queued.png")

    class QueueingModel(FallbackModel):
        def __init__(self):
            self.requests = []
            self.vision_calls = []
            self.step = 0

        def request(self, messages, tools=None):
            self.requests.append(messages)
            self.step += 1
            if self.step == 1:
                s.enqueue_user_input(s.images.recognize("look queued.png"))
                return {"role": "assistant", "content": ""}, [ToolCall("read", "Read", ["missing.txt"])], ""
            if self.step == 2:
                raise ModelError("Error code: 400 - boom")
            return {"role": "assistant", "content": "queued ok"}, [], "queued ok"

        def cancel(self):
            pass

    agent, _notices = run_with(s, QueueingModel())
    assert agent.run("start") == "queued ok"

    assert s.pending_user_inputs == []
    assert agent.model.vision_calls == [(("queued.png",), VISION_OBSERVE_DEFAULT_QUESTION)]
    assert s.image_route.state() == "text_only_learned"


# --- ineligible failures ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        "Error code: 401 - unauthorized",
        "Error code: 403 - forbidden",
        "Error code: 415 - unsupported media type",
        "Error code: 422 - unprocessable",
        "Error code: 429 - rate limited",
        "Error code: 500 - internal error",
        "connection timed out",
        "connection reset by peer",
    ],
)
def test_ineligible_failures_never_learn_or_call_vision(tmp_path, error):
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "shot.png")
    model = FallbackModel([ModelError(error)])
    agent, notices = run_with(s, model)

    with pytest.raises(ModelError, match=error.split(" - ")[-1] if " - " in error else error.split()[0]):
        agent.run(s.images.recognize("inspect shot.png"))

    assert s.image_route.state() == "unknown"
    assert not s.learned_text_only_routes
    assert model.vision_calls == []
    assert notices == []


def test_cancelled_request_does_not_learn(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "shot.png")
    model = FallbackModel([KeyboardInterrupt()])
    agent, _notices = run_with(s, model)

    with pytest.raises(KeyboardInterrupt):
        agent.run(s.images.recognize("inspect shot.png"))

    assert s.image_route.state() == "unknown"
    assert model.vision_calls == []


def test_vision_failure_propagates_without_retry_loop(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "shot.png")

    class FailingVision(FallbackModel):
        def vision_observe(self, images, question=""):
            self.vision_calls.append((tuple(image.name for image in images), question))
            raise ModelError("vision provider unreachable")

    model = FailingVision([ModelError("Error code: 400 - boom")])
    agent, _notices = run_with(s, model)

    with pytest.raises(ModelError, match="vision provider unreachable"):
        agent.run(s.images.recognize("inspect shot.png"))

    assert len(model.requests) == 1  # one raw attempt; no loop
    assert s.image_route.state() == "text_only_learned"
    # the failed occurrence is settled replay-safe
    assert FAILED_IMAGE_CONTEXT_PREFIX in s.messages[0]["content"]


# --- persistence ------------------------------------------------------------------------------


def test_learned_evidence_not_serialized_and_observation_survives_resume(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "shot.png")
    model = FallbackModel([ModelError("Error code: 400 - image input unsupported"), ("recovered", [])])
    agent, _notices = run_with(s, model)

    assert agent.run(s.images.recognize("inspect shot.png")) == "recovered"
    assert s.image_route.state() == "text_only_learned"
    s.save_snapshot()

    resumed = Session.load_snapshot(s.uid, config=s.config)
    # learned evidence is runtime-only: a resumed session starts unknown again
    assert resumed.image_route.state() == "unknown"
    assert not resumed.learned_text_only_routes
    # the durable observation text survived the resume
    assert ATTACHMENT_VISION_OBSERVATION_PREFIX in resumed.messages[0]["content"]
    assert resumed.messages[0][IMAGE_TEXT_ONLY_KEY] is True
    assert not ImageInputs.input_refs(resumed.messages[0])
