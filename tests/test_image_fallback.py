"""Main-first image routing: static text-only catalog, session-learned 400s, bounded vision fallback."""

import json

import pytest
from catalog_harness import resolve
from PIL import Image

from minacode.base import ImageRouteNotice, ModelError, ModelRequestRetry, ToolCall
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
from minacode.providers.compat import bundled_policy
from minacode.session import Session

VISION_TEXT = "vision observation text"


def is_text_only_model(model: str) -> bool:
    """Test the bundled generic catalog without restoring a production-only wrapper."""
    return bundled_policy().text_only(ProviderConfig(model=model), model)


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
    agent.vision_observe = model.vision_observe
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
        ("deepseek-v3", True),
        ("deepseek-v3.1", True),
        ("deepseek-v3.2", True),
        ("deepseek-r1", True),
        ("glm-5", True),
        ("glm-5-turbo", True),
            ("glm-5.1", True),
            ("glm-4.6", True),
            ("glm-4.7", True),
            ("kimi-k2", True),
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
            # Current Kimi K3 and K2.7 Code endpoints accept images; they must stay main-first.
            ("kimi-k3", False),
            ("kimi-k2.7-code", False),
            ("k3", False),
            ("kimi-for-coding", False),
        # documented DeepSeek vision families share only the `deepseek-` prefix
        ("deepseek-vl", False),
        ("deepseek-vl2", False),
        ("deepseek-ocr", False),
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
        ("deepseek/deepseek-v3.2", True),
        ("deepseek/deepseek-r1", True),
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
        return resolve(ProviderConfig(url="http://main.test", key="key", model=model))

    assert resolved("deepseek-chat").text_only is True
    assert resolved("deepseek/deepseek-chat").text_only is True
    assert resolved("glm-5v").text_only is False
    assert resolved("gpt-4o").text_only is False
    assert resolved("my-gateway/deepseek-chat").text_only is False


def test_route_identity_keys_learned_evidence_per_main_route(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    resolved = resolve(s.config.provider)
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
    assert notices == [ImageRouteNotice("main model rejected image input (400)", described_by="v/vision-model", images=("shot.png",))]

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
    assert notices == [ImageRouteNotice("main model is text-only", described_by="v/vision-model", images=("shot.png",))]


def test_static_text_only_without_vision_keeps_raw_attempt_and_original_error(tmp_path):
    s = session(tmp_path, model="deepseek-chat", vision=False)
    image_file(tmp_path / "shot.png")
    model = FallbackModel([ModelError("Error code: 400 - no vision configured")])
    agent, _ = run_with(s, model)

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
    assert notices == [ImageRouteNotice("main model rejected image input (400); no vision provider configured", images=("shot.png",))]
    # evidence is recorded, but without [vision] delivery still raw-attempts
    assert s.image_route.state() == "text_only_learned"
    assert s.image_route.delivery() == "raw"


def test_multi_image_attachment_is_observed_once_with_both_images(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "one.png")
    image_file(tmp_path / "two.png", color=(65, 43, 21))
    model = FallbackModel([ModelError("Error code: 400 - boom"), ("ok", [])])
    agent, _ = run_with(s, model)

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
    agent, _ = run_with(s, model)

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
    agent, _ = run_with(s, model)

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

    agent, _ = run_with(s, QueueingModel())
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
    agent, _ = run_with(s, model)

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
    agent, notices = run_with(s, model)

    with pytest.raises(ModelError, match="vision provider unreachable"):
        agent.run(s.images.recognize("inspect shot.png"))

    assert len(model.requests) == 1  # one raw attempt; no loop
    assert s.image_route.state() == "text_only_learned"
    # the notice names the entry only after the observation succeeds, so a failed vision call
    # never shows a fake described-by success
    assert notices == []
    # the failed occurrence is settled replay-safe
    assert FAILED_IMAGE_CONTEXT_PREFIX in s.messages[0]["content"]


# --- persistence ------------------------------------------------------------------------------


def test_learned_evidence_not_serialized_and_observation_survives_resume(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "shot.png")
    model = FallbackModel([ModelError("Error code: 400 - image input unsupported"), ("recovered", [])])
    agent, _ = run_with(s, model)

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


# --- notice honesty and transaction boundaries --------------------------------------------------


def test_static_vision_failure_emits_no_notice(tmp_path):
    s = session(tmp_path, model="deepseek-chat", vision=True)
    image_file(tmp_path / "shot.png")

    class FailingVision(FallbackModel):
        def vision_observe(self, images, question=""):
            self.vision_calls.append((tuple(image.name for image in images), question))
            raise ModelError("vision provider unreachable")

    model = FailingVision([("never reached", [])])
    agent, notices = run_with(s, model)

    with pytest.raises(ModelError, match="vision provider unreachable"):
        agent.run(s.images.recognize("inspect shot.png"))

    # the static-route notice would claim a described-by success; a failed observation must not
    assert notices == []
    assert len(model.requests) == 0  # no doomed raw request was ever sent


def test_learned_route_observes_follow_up_attachment_directly(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "first.png")
    image_file(tmp_path / "second.png", color=(65, 43, 21))
    model = FallbackModel([ModelError("Error code: 400 - boom"), ("learned ok", []), ("direct ok", [])])
    agent, notices = run_with(s, model)

    assert agent.run(s.images.recognize("look at first.png")) == "learned ok"
    assert agent.run(s.images.recognize("now second.png")) == "direct ok"

    assert s.image_route.state() == "text_only_learned"
    # the follow-up attachment is observed directly through [vision], never re-sent raw
    assert model.vision_calls == [(("first.png",), VISION_OBSERVE_DEFAULT_QUESTION), (("second.png",), VISION_OBSERVE_DEFAULT_QUESTION)]
    assert len(model.requests) == 3
    assert "image_url" not in json.dumps(model.requests[2])
    # the learned-route reason is truthful: runtime evidence is a rejected 400, not architecture
    assert notices == [
        ImageRouteNotice("main model rejected image input (400)", described_by="v/vision-model", images=("first.png",)),
        ImageRouteNotice("main model rejected image input (400)", described_by="v/vision-model", images=("second.png",)),
    ]


def test_learned_evidence_does_not_leak_across_provider_switch(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    s.config.providers["other"] = ProviderConfig(url="http://main2.test", key="k2", model="other-model")
    assert s.image_route.state() == "unknown"

    s.image_route.learn_text_only()
    assert s.image_route.state() == "text_only_learned"

    # switching the active main route is a different route: evidence does not apply
    s.config.active_provider = "other"
    assert s.image_route.state() == "unknown"
    # switching back restores the session-local learned evidence
    s.config.active_provider = "default"
    assert s.image_route.state() == "text_only_learned"


def test_queued_image_400_fallback_cancelled_keeps_pending_and_no_history_duplicate(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "queued.png")

    class CancelDuringFallback(FallbackModel):
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
            raise KeyboardInterrupt()

        def cancel(self):
            pass

    agent, _ = run_with(s, CancelDuringFallback())
    with pytest.raises(KeyboardInterrupt):
        agent.run("start")

    # the paid observation stays out of history on cancel: the queued image is released back to
    # the queue instead of appearing in both history and the queue (a next-turn duplicate)
    assert s.pending_user_inputs and "queued.png" in s.pending_user_inputs[0].text
    assert "queued.png" not in json.dumps(s.messages)
    assert agent.model.vision_calls == [(("queued.png",), VISION_OBSERVE_DEFAULT_QUESTION)]

    # a fresh turn re-submits the still-queued image, observes it once, and commits it once
    second = FallbackModel([("replayed ok", [])])
    agent2, _ = run_with(s, second)
    assert agent2.run("start") == "replayed ok"
    assert second.vision_calls == [(("queued.png",), VISION_OBSERVE_DEFAULT_QUESTION)]
    observations = [m for m in s.messages if ImageInputs.refs(m) and "queued.png" in json.dumps(m)]
    assert len(observations) == 1
    assert ATTACHMENT_VISION_OBSERVATION_PREFIX in observations[0]["content"]


def test_queued_image_400_fallback_manual_retry_observes_once(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "queued.png")

    class RetryDuringFallback(FallbackModel):
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
            if self.step == 3:
                raise ModelRequestRetry()
            return {"role": "assistant", "content": "queued ok"}, [], "queued ok"

        def cancel(self):
            pass

    agent, _ = run_with(s, RetryDuringFallback())
    assert agent.run("start") == "queued ok"

    # the manual retry re-sends the converted observation: the queued image is observed exactly
    # once and committed once, never duplicated between the request and history
    assert len(agent.model.requests) == 4
    assert agent.model.vision_calls == [(("queued.png",), VISION_OBSERVE_DEFAULT_QUESTION)]
    assert s.pending_user_inputs == []
    observations = [m for m in s.messages if ImageInputs.refs(m) and "queued.png" in json.dumps(m)]
    assert len(observations) == 1
    assert "image_url" not in json.dumps(agent.model.requests[3])


def test_view_image_400_fallback_cancel_keeps_paid_observation(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "shot.png")

    class CancelAfterViewImage(FallbackModel):
        def __init__(self):
            self.requests = []
            self.vision_calls = []
            self.step = 0

        def request(self, messages, tools=None):
            self.requests.append(messages)
            self.step += 1
            if self.step == 1:
                return {"role": "assistant", "content": ""}, [ToolCall("view", "ViewImage", ["shot.png", "what is shown?"])], ""
            if self.step == 2:
                raise ModelError("Error code: 400 - boom")
            raise KeyboardInterrupt()

        def cancel(self):
            pass

    agent, _ = run_with(s, CancelAfterViewImage())
    with pytest.raises(KeyboardInterrupt):
        agent.run("inspect")

    observations = [message for message in s.messages if message.get(TOOL_IMAGE_OBSERVATION_KEY)]
    assert len(observations) == 1
    assert observations[0][IMAGE_TEXT_ONLY_KEY] is True
    assert observations[0]["content"] == f"{TOOL_IMAGE_OBSERVATION_PREFIX}\n{VISION_TEXT}"
    assert agent.model.vision_calls == [(("shot.png",), "what is shown?")]


def test_queued_image_400_manual_retry_then_cancel_releases_pending(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "queued.png")

    class RetryThenCancel(FallbackModel):
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
            if self.step == 3:
                raise ModelRequestRetry()
            raise KeyboardInterrupt()

        def cancel(self):
            pass

    agent, _ = run_with(s, RetryThenCancel())
    with pytest.raises(KeyboardInterrupt):
        agent.run("start")

    assert s.pending_user_inputs and "queued.png" in s.pending_user_inputs[0].text
    assert "queued.png" not in json.dumps(s.messages)
    assert agent.model.vision_calls == [(("queued.png",), VISION_OBSERVE_DEFAULT_QUESTION)]


def test_queued_image_400_fallback_failure_keeps_paid_observation(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "queued.png")

    class FailingDuringFallback(FallbackModel):
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
            raise ModelError("fallback main request failed")

        def cancel(self):
            pass

    agent, _ = run_with(s, FailingDuringFallback())
    with pytest.raises(ModelError, match="fallback main request failed"):
        agent.run("start")

    # the fallback failed, but the vision observation was already paid for: it is committed as
    # durable text (with its queued input acknowledged) instead of being overwritten by the raw
    # request's settlement
    assert agent.model.vision_calls == [(("queued.png",), VISION_OBSERVE_DEFAULT_QUESTION)]
    assert s.pending_user_inputs == []
    observations = [m for m in s.messages if ImageInputs.refs(m) and "queued.png" in json.dumps(m)]
    assert len(observations) == 1
    assert ATTACHMENT_VISION_OBSERVATION_PREFIX in observations[0]["content"]
    assert FAILED_IMAGE_CONTEXT_PREFIX not in observations[0]["content"]


def test_two_view_image_questions_keep_order_and_cardinality(tmp_path):
    s = session(tmp_path, model="main-model", vision=True)
    image_file(tmp_path / "a.png")
    image_file(tmp_path / "b.png", color=(65, 43, 21))
    model = FallbackModel(
        [
            ("", [ToolCall("view_image", "ViewImage", ["a.png", "what is in a?"]), ToolCall("view_image", "ViewImage", ["b.png", "what is in b?"])]),
            ModelError("Error code: 400 - image input not supported"),
            ("both described", []),
        ]
    )
    agent, _ = run_with(s, model)

    assert agent.run(s.images.recognize("compare the two")) == "both described"

    # two ViewImage calls keep their individual questions and their replay ordering
    assert model.vision_calls == [(("a.png",), "what is in a?"), (("b.png",), "what is in b?")]
    assert len(model.requests) == 3
    stored = [m for m in s.messages if m.get(TOOL_IMAGE_OBSERVATION_KEY)]
    assert len(stored) == 2
    assert all(m[IMAGE_TEXT_ONLY_KEY] is True for m in stored)
    assert all(m["content"] == f"{TOOL_IMAGE_OBSERVATION_PREFIX}\n{VISION_TEXT}" for m in stored)
    assert [ImageInputs.tool_observation_question(m) for m in stored] == ["what is in a?", "what is in b?"]
