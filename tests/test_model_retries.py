"""Request resilience: retries, response deadlines, and compaction calls staying off the UI."""

import contextlib
import email.utils
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import anthropic
import httpx
import openai
import pytest
from model_harness import _MockClientFactory, _session

from minacode.base import (
    MODEL_REQUEST_RETRIES,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    ModelError,
    ModelOutputTruncated,
    ModelRequestRetry,
    ModelResponseTimeout,
)
from minacode.config import (
    Config,
)
from minacode.model import ModelClient, resilience


def test_compaction_does_not_publish_internal_model_output(tmp_path, monkeypatch):
    s = _session(tmp_path)
    model = ModelClient(s)
    factory = _MockClientFactory(
        [
            (
                200,
                {
                    "id": "chatcmpl-compact",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": '{"summary":"short","goal":"","plan":[],"known":[],"check":""}'},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        ]
    )
    streamed = []
    model.on_stream = lambda kind, delta: streamed.append((kind, delta))
    monkeypatch.setattr(model, "client", factory)

    result = model.compact("long context")

    body = json.loads(factory.calls[0].content)
    assert body["stream"] is False
    assert "stream_options" not in body
    assert streamed == []
    assert result["summary"] == "short"


def test_request_retries_then_succeeds(tmp_path, monkeypatch):
    s = _session(tmp_path)
    model = ModelClient(s)
    factory = _MockClientFactory(
        [
            (429, {"error": {"message": "rate limited", "type": "rate_limit_error"}}),
            (
                200,
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            ),
        ]
    )
    monkeypatch.setattr(model, "client", factory)
    _patch_fast_clock(monkeypatch)

    _assistant, _calls, content = model.request([{"role": "user", "content": "hi"}], None)

    assert content == "ok"
    assert len(factory.calls) == 2
    assert s.usage.calls == 1


def test_request_retry_exhausted(tmp_path, monkeypatch):
    s = _session(tmp_path)
    model = ModelClient(s)
    factory = _MockClientFactory([(500, {"error": {"message": "server error", "type": "internal_server_error"}})] * 6)
    monkeypatch.setattr(model, "client", factory)
    _patch_fast_clock(monkeypatch)

    with pytest.raises(ModelError, match="after 6 attempts"):
        model.request([{"role": "user", "content": "hi"}], None)

    assert len(factory.calls) == 6
    assert s.usage.calls == 0


def test_total_response_timeout_closes_client_and_does_not_retry(tmp_path, monkeypatch):
    class Client:
        def __init__(self):
            self.close_count = 0

        def close(self):
            self.close_count += 1

    s = _session(tmp_path, response_timeout=0.01)
    model = ModelClient(s)
    client = Client()
    started = threading.Event()
    release = threading.Event()

    def blocked_request():
        started.set()
        release.wait()
        return "completed after deadline"

    try:
        with pytest.raises(ModelResponseTimeout, match=r"provider\.response_timeout=0\.01s") as caught:
            model.call_client(client, blocked_request)
    finally:
        release.set()

    assert started.is_set()
    assert client.close_count == 1
    assert resilience.retryable_error(caught.value) is False

    calls = 0

    def expired(_messages, _tools):
        nonlocal calls
        calls += 1
        raise caught.value

    monkeypatch.setattr(model, "api_request", expired)
    with pytest.raises(ModelResponseTimeout):
        model.request([{"role": "user", "content": "hi"}], [])
    assert calls == 1
    assert s.state.model_retry_count == 0


def test_timed_out_request_cannot_emit_after_a_new_request_starts(tmp_path):
    class Client:
        def close(self):
            pass  # Deliberately ignore close, like the SDK behavior this boundary must contain.

    s = _session(tmp_path, response_timeout=0.01)
    model = ModelClient(s)
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    stream = []
    builtins = []
    model.on_stream = lambda kind, text: stream.append((kind, text))
    model.on_builtin_call = lambda label, detail: builtins.append((label, detail))

    def stale_request():
        started.set()
        release.wait()
        try:
            model._emit_stream("output", "stale")
            model.report_builtin_call("web_search_call", "stale query")
            model._emit_stream("", "")
            return "stale"
        finally:
            finished.set()

    with pytest.raises(ModelResponseTimeout):
        model.call_client(Client(), stale_request)
    assert started.is_set()

    def current_request():
        model._emit_stream("output", "current")
        return "current"

    assert model.call_client(Client(), current_request, response_timeout=1) == "current"
    release.set()
    assert finished.wait(1)
    assert stream == [("output", "current")]
    assert builtins == []


def test_cancelled_request_stays_stale_after_next_request_clears_cancel(tmp_path):
    class Client:
        def close(self):
            pass

    s = _session(tmp_path, response_timeout=1)
    model = ModelClient(s)
    started = threading.Event()
    release = threading.Event()
    caller_finished = threading.Event()
    worker_finished = threading.Event()
    stream = []
    errors = []
    model.on_stream = lambda kind, text: stream.append((kind, text))

    def stale_request():
        started.set()
        release.wait()
        try:
            model._emit_stream("output", "stale")
            return "stale"
        finally:
            worker_finished.set()

    def call_stale_request():
        try:
            model.call_client(Client(), stale_request)
        except BaseException as error:  # noqa: BLE001 - The cancellation contract is KeyboardInterrupt.
            errors.append(error)
        finally:
            caller_finished.set()

    caller = threading.Thread(target=call_stale_request)
    caller.start()
    assert started.wait(1)
    model.cancel()
    assert caller_finished.wait(1)
    assert isinstance(errors[0], KeyboardInterrupt)

    model.cancel_requested.clear()  # ModelClient.request does this before the next request.
    assert model.call_client(Client(), lambda: model._emit_stream("output", "current") or "current") == "current"
    release.set()
    assert worker_finished.wait(1)
    caller.join()
    assert stream == [("output", "current")]


def test_zero_response_timeout_does_not_start_deadline_timer(tmp_path, monkeypatch):
    class Client:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    s = _session(tmp_path, response_timeout=0)
    model = ModelClient(s)
    client = Client()

    assert model.call_client(client, lambda: "complete") == "complete"
    assert client.closed is True


@pytest.mark.parametrize(("configured", "expected"), [(600, 600), (30, 30), (0, 0)])
def test_compaction_follows_the_configured_response_deadline(tmp_path, monkeypatch, configured, expected):
    """No hidden cap: the summary call gets exactly the configured total-generation limit, and 0
    disables the deadline for it like for every other request."""
    s = _session(tmp_path)
    s.config.provider.response_timeout = configured
    model = ModelClient(s)
    seen = []

    def api_request(_messages, _tools, **kwargs):
        seen.append(kwargs)
        return {}, [], '{"summary":"short"}'

    monkeypatch.setattr(model, "api_request", api_request)

    assert model.compact("long context") == {"summary": "short"}
    assert len(seen) == 1
    assert seen[0]["allow_stream"] is False
    assert seen[0]["response_timeout"] == expected
    assert seen[0]["provider"].response_timeout == configured


def test_compaction_timeout_error_names_the_summary(tmp_path, monkeypatch):
    """The fallback log must say the summary timed out, not echo the generic request wording, and
    name the entry that served it -- compaction can run on its own `[compaction]` provider."""
    s = _session(tmp_path)
    s.config.provider.response_timeout = 600
    model = ModelClient(s)

    def api_request(_messages, _tools, **kwargs):
        raise ModelResponseTimeout("Model response exceeded provider.response_timeout=600s")

    monkeypatch.setattr(model, "api_request", api_request)

    with pytest.raises(ModelResponseTimeout, match=r"compaction summary on `default/gpt-4` exceeded provider.response_timeout=600s"):
        model.compact("long context")


def _retry_wait_recorder(monkeypatch, factory):
    """Replace time.sleep with a recorder that never actually sleeps and never blocks, bucketing
    each retry's total requested wait by provider-call index (sleeps happen between calls, so all
    slices of one wait share the same calls count). A fake monotonic clock is advanced by the
    recorder so the slice loop finishes instantly; call the returned function to read the waits."""
    buckets: dict[int, float] = {}
    clock = {"now": 0.0}

    def monotonic():
        return clock["now"]

    def sleeper(seconds):
        buckets[len(factory.calls)] = buckets.get(len(factory.calls), 0.0) + seconds
        clock["now"] += seconds

    monkeypatch.setattr(time, "sleep", sleeper)
    monkeypatch.setattr(time, "monotonic", monotonic)
    return lambda: [buckets[k] for k in sorted(buckets)]


def _patch_fast_clock(monkeypatch):
    """Advance the clock by each requested sleep so backoff waits complete instantly."""
    clock = {"now": 0.0}
    monkeypatch.setattr(time, "sleep", lambda seconds: clock.__setitem__("now", clock["now"] + seconds))
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])


_OVERLOADED = {"error": {"message": "overloaded", "type": "server_error"}}
_OK = {
    "id": "chatcmpl-ok",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def test_retry_backoff_sequence_within_jitter_bands(tmp_path, monkeypatch):
    """Each retry waits base*2**attempt (jitter pinned to exactly 1.0x), never more than the ceiling."""
    s = _session(tmp_path)
    model = ModelClient(s)
    factory = _MockClientFactory([(503, _OVERLOADED)] * MODEL_REQUEST_RETRIES + [(200, _OK)])
    monkeypatch.setattr(model, "client", factory)
    monkeypatch.setattr(resilience.random, "random", lambda: 0.5)  # jitter factor exactly 1.0
    waits = _retry_wait_recorder(monkeypatch, factory)

    _assistant, _calls, content = model.request([{"role": "user", "content": "hi"}], None)

    assert content == "ok"
    assert len(waits()) == MODEL_REQUEST_RETRIES
    for attempt, wait in enumerate(waits()):
        expected = min(RETRY_MAX_DELAY, RETRY_BASE_DELAY * 2**attempt)
        # 0.1s slice granularity keeps the recorded total within ~0.1s of the requested delay.
        assert expected - 0.2 <= wait <= expected + 0.2, f"attempt {attempt}: waited {wait}, expected ~{expected}"


def test_retry_after_seconds_preferred_over_backoff(tmp_path, monkeypatch):
    """A numeric Retry-After header wins over the algorithm (7s, not the 1-3s first-band)."""
    s = _session(tmp_path)
    model = ModelClient(s)
    factory = _MockClientFactory([httpx.Response(503, json=_OVERLOADED, headers={"retry-after": "7"}), (200, _OK)])
    monkeypatch.setattr(model, "client", factory)
    waits = _retry_wait_recorder(monkeypatch, factory)

    model.request([{"role": "user", "content": "hi"}], None)

    assert len(waits()) == 1
    assert 6.9 <= waits()[0] <= 7.1


def test_retry_after_clamped_to_max_delay(tmp_path, monkeypatch):
    """A provider claim beyond the ceiling is truncated, so one aberrant header cannot stall the CLI."""
    s = _session(tmp_path)
    model = ModelClient(s)
    factory = _MockClientFactory([httpx.Response(503, json=_OVERLOADED, headers={"retry-after": "300"}), (200, _OK)])
    monkeypatch.setattr(model, "client", factory)
    waits = _retry_wait_recorder(monkeypatch, factory)

    model.request([{"role": "user", "content": "hi"}], None)

    assert len(waits()) == 1
    assert RETRY_MAX_DELAY - 0.2 <= waits()[0] <= RETRY_MAX_DELAY + 0.2


def test_retry_after_http_date_respected(tmp_path, monkeypatch):
    """The HTTP-date form is parsed and the remaining delta is waited, not the raw date text."""
    s = _session(tmp_path)
    model = ModelClient(s)
    stamp = email.utils.format_datetime(datetime.now(UTC) + timedelta(seconds=5), usegmt=True)
    factory = _MockClientFactory([httpx.Response(503, json=_OVERLOADED, headers={"retry-after": stamp}), (200, _OK)])
    monkeypatch.setattr(model, "client", factory)
    waits = _retry_wait_recorder(monkeypatch, factory)

    model.request([{"role": "user", "content": "hi"}], None)

    assert len(waits()) == 1
    # format_datetime truncates to whole seconds, so the remaining delta can be ~1s short of 5.
    assert 3.8 <= waits()[0] <= 5.5


@pytest.mark.parametrize("value", ["", "   ", "-5", "garbage"])
def test_retry_after_invalid_falls_back_to_backoff(tmp_path, monkeypatch, value):
    """Empty, negative, and unparseable headers are silently ignored and the algorithm is used."""
    s = _session(tmp_path)
    model = ModelClient(s)
    factory = _MockClientFactory([httpx.Response(503, json=_OVERLOADED, headers={"retry-after": value}), (200, _OK)])
    monkeypatch.setattr(model, "client", factory)
    monkeypatch.setattr(resilience.random, "random", lambda: 0.5)
    waits = _retry_wait_recorder(monkeypatch, factory)

    model.request([{"role": "user", "content": "hi"}], None)

    assert len(waits()) == 1
    assert RETRY_BASE_DELAY - 0.2 <= waits()[0] <= RETRY_BASE_DELAY + 0.2


def test_retry_after_absurd_value_does_not_stall(tmp_path, monkeypatch):
    """A huge header value is clamped rather than waited out; the request still succeeds."""
    s = _session(tmp_path)
    model = ModelClient(s)
    factory = _MockClientFactory([httpx.Response(503, json=_OVERLOADED, headers={"retry-after": "999999999"}), (200, _OK)])
    monkeypatch.setattr(model, "client", factory)
    waits = _retry_wait_recorder(monkeypatch, factory)

    _assistant, _calls, content = model.request([{"role": "user", "content": "hi"}], None)

    assert content == "ok"
    assert len(waits()) == 1
    # Clamped to the ceiling like any out-of-band value: never waited out, never stalls the CLI.
    assert RETRY_MAX_DELAY - 0.2 <= waits()[0] <= RETRY_MAX_DELAY + 0.2


def test_retry_wait_cancels_promptly(tmp_path, monkeypatch):
    """Setting cancel during the wait aborts within one sleep slice instead of sleeping the delay out."""
    s = _session(tmp_path)
    model = ModelClient(s)
    factory = _MockClientFactory([(503, _OVERLOADED), (503, _OVERLOADED)])
    monkeypatch.setattr(model, "client", factory)
    slept: list[float] = []

    def sleeper(seconds):
        slept.append(seconds)
        model.cancel_requested.set()

    monkeypatch.setattr(time, "sleep", sleeper)

    with pytest.raises(KeyboardInterrupt):
        model.request([{"role": "user", "content": "hi"}], None)
    # The wait must abort within one 0.1s slice, far short of the first full backoff step (>=0.5s).
    assert sum(slept) < 0.3, f"cancelled wait slept {sum(slept)}s instead of aborting within a slice"
    assert s.state.model_retry_count == 1


def test_non_retryable_errors_skip_retry_path(tmp_path, monkeypatch):
    """Deterministic failures and validation errors never enter the backoff path: one attempt, no sleep."""
    s = _session(tmp_path)
    model = ModelClient(s)
    calls = {"n": 0}

    def truncated(_messages, _tools):
        calls["n"] += 1
        raise ModelOutputTruncated("cut off at the output cap")

    monkeypatch.setattr(model, "api_request", truncated)
    monkeypatch.setattr(time, "sleep", lambda _seconds: pytest.fail("non-retryable error must not sleep"))
    with pytest.raises(ModelOutputTruncated):
        model.request([{"role": "user", "content": "hi"}], [])
    assert calls["n"] == 1
    assert s.state.model_retry_count == 0

    # Validation error over the SDK wire (400): one attempt, no retry budget spent.
    s2 = _session(tmp_path)
    model2 = ModelClient(s2)
    factory = _MockClientFactory([(400, {"error": {"message": "bad request", "type": "invalid_request_error"}})])
    monkeypatch.setattr(model2, "client", factory)
    with pytest.raises(ModelError, match=r"400"):
        model2.request([{"role": "user", "content": "hi"}], None)
    assert len(factory.calls) == 1
    assert len(factory.calls) == 1
    assert s2.state.model_retry_count == 0


def test_retry_wait_phase_hook_pairs(tmp_path, monkeypatch):
    """The on_retry_wait hook fires True on entering the wait and False when it ends, and the
    deadline fact is reset, so the UI never sticks on the retrying phase."""
    s = _session(tmp_path)
    model = ModelClient(s)
    factory = _MockClientFactory([(503, _OVERLOADED), (200, _OK)])
    monkeypatch.setattr(model, "client", factory)
    phases: list[bool] = []
    model.on_retry_wait = phases.append
    _retry_wait_recorder(monkeypatch, factory)

    _assistant, _calls, content = model.request([{"role": "user", "content": "hi"}], None)

    assert content == "ok"
    assert phases == [True, False]
    assert s.state.model_retry_until == 0.0


def test_retry_wait_phase_hook_resets_on_cancel(tmp_path, monkeypatch):
    """Cancelling mid-wait still emits the closing False in a finally block: the live phase label
    cannot be left stuck on "retrying" by an interrupted wait."""
    s = _session(tmp_path)
    model = ModelClient(s)
    factory = _MockClientFactory([(503, _OVERLOADED), (503, _OVERLOADED)])
    monkeypatch.setattr(model, "client", factory)
    phases: list[bool] = []
    model.on_retry_wait = phases.append

    def sleeper(_seconds):
        model.cancel_requested.set()

    monkeypatch.setattr(time, "sleep", sleeper)

    with pytest.raises(KeyboardInterrupt):
        model.request([{"role": "user", "content": "hi"}], None)
    assert phases == [True, False]
    assert s.state.model_retry_until == 0.0


def test_manual_resend_racing_retry_wait_remains_a_retry(tmp_path, monkeypatch):
    s = _session(tmp_path)
    model = ModelClient(s)
    factory = _MockClientFactory([(503, _OVERLOADED)])
    monkeypatch.setattr(model, "client", factory)

    def resend(_seconds):
        s.state.manual_model_retry_requested = True
        model.cancel()

    monkeypatch.setattr(time, "sleep", resend)

    with pytest.raises(ModelRequestRetry):
        model.request([{"role": "user", "content": "hi"}], None)
    assert s.state.manual_model_retry_requested is False
    assert s.state.model_retry_until == 0.0


def test_streamed_httpx_transport_error_is_retryable():
    """httpx transport errors raised transparently during streaming (the provider SDK's
    Stream.__stream__ does not wrap them as APIConnectionError) are the same class of transient
    failure and must retry. Regression for "Error: peer closed connection without sending
    complete message body (incomplete chunked read)" surfacing on the first attempt."""
    cause = httpx.RemoteProtocolError("peer closed connection without sending complete message body (incomplete chunked read)")
    error = ModelError(str(cause))
    error.__cause__ = cause
    assert resilience.retryable_error(error) is True
    assert resilience.retry_reason(error) == "connection"


def test_streamed_httpx_read_error_is_retryable():
    """A connection dropped mid-stream (httpx.ReadError) is transient, like ConnectionResetError."""
    cause = httpx.ReadError("peer closed connection without sending bytes")
    error = ModelError(str(cause))
    error.__cause__ = cause
    assert resilience.retryable_error(error) is True


def test_streamed_httpx_error_retries_then_succeeds(tmp_path, monkeypatch):
    """A streaming httpx transport error (raised unwrapped by the SDK's Stream.__stream__ and
    wrapped by call_client as ModelError(cause)) is retried to success, not surfaced on the
    first attempt. Regression for "Error: peer closed connection without sending complete
    message body (incomplete chunked read)"."""
    s = _session(tmp_path)
    model = ModelClient(s)
    _patch_fast_clock(monkeypatch)

    cause = httpx.RemoteProtocolError("peer closed connection without sending complete message body (incomplete chunked read)")
    calls = {"n": 0}

    def api_request(_messages, _tools, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ModelError(str(cause)) from cause
        return {}, [], "ok"

    monkeypatch.setattr(model, "api_request", api_request)

    _assistant, _tool_calls, content = model.request([{"role": "user", "content": "hi"}], None)

    assert content == "ok"
    assert calls["n"] == 2
    assert s.state.model_retry_count == 1


def _error_with_cause(cause: Exception) -> ModelError:
    error = ModelError(str(cause))
    error.__cause__ = cause
    return error


def _openai_429(body: dict) -> openai.RateLimitError:
    request = httpx.Request("POST", "http://test/v1/chat/completions")
    return openai.RateLimitError(
        message="Error code: 429",
        response=httpx.Response(429, request=request),
        body=body,
    )


def _anthropic_429(body: dict) -> anthropic.RateLimitError:
    request = httpx.Request("POST", "http://test/v1/messages")
    return anthropic.RateLimitError(
        message="Error code: 429",
        response=httpx.Response(429, request=request),
        body=body,
    )


def test_429_quota_body_not_retryable_openai_style():
    """OpenAI insufficient_quota is a permanent billing failure: fail immediately, no backoff."""
    cause = _openai_429(
        {
            "message": "You exceeded your current quota, please check your plan and billing details.",
            "type": "insufficient_quota",
            "code": "insufficient_quota",
            "param": None,
        }
    )
    assert resilience.retryable_error(_error_with_cause(cause)) is False


def test_429_quota_body_not_retryable_kimi_style():
    """Kimi/Moonshot exceeds-quota errors carry the same billing wording in error.type."""
    cause = _openai_429(
        {
            "message": "Current quota exceeded. Please check your plan and billing details.",
            "type": "exceeded_current_quota_error",
            "code": "exceeded_current_quota_error",
        }
    )
    assert resilience.retryable_error(_error_with_cause(cause)) is False


def test_429_quota_body_not_retryable_zai_style():
    """z.ai/bigmodel uses numeric string codes (1113) but the message carries billing wording."""
    cause = _openai_429(
        {
            "message": "Insufficient balance or no resource package. Please recharge.",
            "type": "insufficient_balance",
            "code": "1113",
        }
    )
    assert resilience.retryable_error(_error_with_cause(cause)) is False


def test_429_anthropic_rate_limit_still_retryable():
    """Anthropic 429s are transient rate limiting (billing failures arrive as 400 there), so a
    marker hit on this body would be a false positive. Guard against misclassifying it."""
    cause = _anthropic_429(
        {
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "This request would exceed your organization's rate limit. Retry in a few minutes.",
            },
        }
    )
    assert resilience.retryable_error(_error_with_cause(cause)) is True


def test_429_openai_transient_rate_limit_still_retryable():
    """Transient rate-limit wording must not trip the billing-marker heuristic."""
    cause = _openai_429(
        {
            "message": "Rate limit reached for requests, retry after a few seconds.",
            "type": "rate_limit_exceeded",
            "code": "rate_limit_exceeded",
        }
    )
    assert resilience.retryable_error(_error_with_cause(cause)) is True


def test_429_text_fallback_with_billing_marker_not_retryable():
    """The text fallback honors the same markers: a message combining a 429 status pattern with
    account/billing wording must not be rescued by the status-code regex and retried."""
    error = ModelError("Error code: 429 - insufficient balance, please recharge your account")
    assert resilience.retryable_error(error) is False


def test_429_rate_limits_phrased_as_quota_still_retry():
    """Google/Vertex and DashScope phrase per-minute limits with the same nouns a billing failure
    uses. These clear on their own, so failing them at once is the outcome the rule exists to
    prevent — the markers have to be specific enough to tell the two apart."""
    vertex = _openai_429({"message": "Quota exceeded for quota metric 'Generate requests per minute'.", "code": 429})
    dashscope = _openai_429({"message": "Requests throttling triggered.", "code": "Throttling.RateQuota"})

    assert resilience.retryable_error(_error_with_cause(vertex)) is True
    assert resilience.retryable_error(_error_with_cause(dashscope)) is True


def test_429_allocated_quota_exhaustion_is_still_permanent():
    """DashScope's allocation quota is the permanent half of the same vocabulary."""
    cause = _openai_429({"message": "Free allocated quota exceeded.", "code": "Throttling.AllocationQuota"})
    assert resilience.retryable_error(_error_with_cause(cause)) is False


def test_billing_wording_outside_a_429_still_retries():
    """The billing rule is about 429 only. A 5xx is transient whatever its text mentions: an
    expired certificate or a failing credit service upstream says nothing about the account."""
    assert resilience.retryable_error(ModelError("Error code: 503 - upstream TLS certificate expired")) is True
    assert resilience.retryable_error(ModelError("Error code: 500 - internal error in credit service")) is True


def test_compaction_uses_effective_provider(tmp_path, monkeypatch):
    """[compaction] overrides reach the summary request: model/reasoning/api come from the resolved
    entry, never the shared parent object, and an empty [compaction] inherits the active entry."""
    s = _session(tmp_path)
    s.config = Config.from_dict(
        {
            "compaction": {"model": "compactor-1", "reasoning": "off", "api": "chat"},
            "provider": {"active": "default", "default": {"model": "main-1", "url": "http://test", "key": "sk-test"}},
        }
    )
    model = ModelClient(s)
    calls = []

    def api_request(_messages, _tools, **kwargs):
        calls.append(kwargs.get("provider"))
        return {}, [], '{"summary":"short"}'

    monkeypatch.setattr(model, "api_request", api_request)

    assert model.compact("long context") == {"summary": "short"}
    provider = calls[0]
    assert provider.model == "compactor-1"
    assert provider.reasoning == "off"
    assert provider.api == "chat"
    assert provider is not s.config.provider
    assert provider is not s.config.providers["default"]
    assert model.last_compaction_model == "compactor-1"


def test_compaction_response_timeout_follows_base_entry(tmp_path, monkeypatch):
    """The summary's total-generation deadline comes from the compaction base entry, not the active
    provider: base 30, active 600 -> the summary call carries 30."""
    s = _session(tmp_path)
    s.config = Config.from_dict(
        {
            "compaction": {"provider": "base"},
            "provider": {
                "active": "default",
                "default": {"model": "main-1", "url": "http://test", "key": "sk-test", "response_timeout": 600},
                "base": {"model": "base-1", "url": "http://test", "key": "sk-test", "response_timeout": 30},
            },
        }
    )
    model = ModelClient(s)
    seen = []

    def api_request(_messages, _tools, **kwargs):
        seen.append(kwargs)
        return {}, [], '{"summary":"short"}'

    monkeypatch.setattr(model, "api_request", api_request)

    model.compact("long context")
    assert seen[0]["response_timeout"] == 30
    assert seen[0]["provider"].model == "base-1"


def test_compaction_override_reaches_wire_params(tmp_path, monkeypatch):
    """The resolved entry drives the wire: the summary request body carries the [compaction] model."""
    s = _session(tmp_path)
    s.config = Config.from_dict(
        {
            "compaction": {"model": "compactor-2"},
            "provider": {"active": "default", "default": {"model": "main-2", "url": "http://test", "key": "sk-test"}},
        }
    )
    model = ModelClient(s)
    factory = _MockClientFactory(
        [
            (
                200,
                {
                    "id": "chatcmpl-compact",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "compactor-2",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"summary":"short"}'}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        ]
    )
    monkeypatch.setattr(model, "client", factory)

    assert model.compact("long context") == {"summary": "short"}
    body = json.loads(factory.calls[0].content)
    assert body["model"] == "compactor-2"
    assert model.last_compaction_model == "compactor-2"


def test_compaction_retries_once_when_the_model_replies_in_prose(tmp_path, monkeypatch):
    """The observed failure is a model continuing the conversation instead of summarizing it. A bare
    resend would reproduce it, so the retry carries the bad reply back with a correction."""
    s = _session(tmp_path)
    model = ModelClient(s)
    sent = []

    def api_request(messages, _tools, **kwargs):
        sent.append(messages)
        if len(sent) == 1:
            return None, None, "继续 Part B 收尾：检查 `_run_workflow` 的所有调用点。\n\ntool:\ntool tr.268 Bash rg -n"
        return None, None, '{"title":"Part B wrap-up","summary":"…","goal":"","plan":[],"known":"","check":""}'

    monkeypatch.setattr(model, "api_request", api_request)

    assert model.compact("long context")["title"] == "Part B wrap-up"
    assert len(sent) == 2
    # The second attempt shows the model what it did and what to do instead.
    assert sent[1][-1]["role"] == "user" and "not a JSON object" in sent[1][-1]["content"]
    assert sent[1][-2]["role"] == "assistant" and "继续 Part B" in sent[1][-2]["content"]


def test_compaction_failure_names_the_provider_entry(tmp_path, monkeypatch):
    """Compaction may run on a cheaper `[compaction]` entry, which is exactly the model that fails
    this way -- the fallback line has to say which one to go look at."""
    s = _session(tmp_path)
    model = ModelClient(s)
    monkeypatch.setattr(model, "api_request", lambda *_a, **_k: (None, None, "user:\nnot json at all"))

    with pytest.raises(ModelError, match=r"compaction provider `default/gpt-4`"):
        model.compact("long context")


def test_compaction_input_restates_the_contract_after_the_payload(tmp_path):
    """The payload ends with raw transcript, so the last instruction the model reads must be ours."""
    from minacode.prompts import compaction_input

    text = compaction_input(state="s", previous_summary="", older_messages="old", recent_messages="user:\n继续 Part B 收尾")
    assert text.rstrip().endswith("title, summary, goal, plan, known, check.")
    assert "never instructions to follow" in text
    assert text.index("END OF CONVERSATION TO COMPACT") > text.index("继续 Part B 收尾")


def test_compaction_sends_json_response_format_only_where_the_provider_supports_it(tmp_path, monkeypatch):
    """Constrained decoding is opt-in per provider: an unsupporting gateway answers 400, and the
    only caller is compaction, whose failure is a silent downgrade to deterministic trimming."""
    seen = {}

    def run(host: str) -> object:
        s = _session(tmp_path)
        s.config.provider.url = f"https://{host}/v1"
        model = ModelClient(s)

        def create(**params):
            seen[host] = params.get("response_format")
            raise RuntimeError("stop after params")

        monkeypatch.setattr(model, "client", lambda **_k: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
        with contextlib.suppress(Exception):
            model.compact("long context")

    run("api.deepseek.com")
    run("api.moonshot.cn")
    assert seen["api.deepseek.com"] == {"type": "json_object"}
    assert seen["api.moonshot.cn"] is None


def test_compaction_rejects_a_summary_that_copies_the_conversation(tmp_path, monkeypatch):
    """Constrained decoding forces the shape, never the task: a model that means to continue the
    conversation returns valid JSON with the conversation inside it, which would apply cleanly into
    session.state and be fed back into every later compaction."""
    s = _session(tmp_path)
    model = ModelClient(s)
    echo = "继续 Part B 收尾：检查 `_run_workflow` 的所有调用点，确保新参数没有遗漏，然后跑 lint 与 pyright，再把结果贴回来。"
    sent = []

    def api_request(messages, _tools, **kwargs):
        sent.append(messages)
        if len(sent) == 1:
            return None, None, json.dumps({"title": "Part B", "summary": echo}, ensure_ascii=False)
        return None, None, json.dumps({"title": "Part B", "summary": "Wrapped up Part B; lint and pyright still to run."})

    monkeypatch.setattr(model, "api_request", api_request)

    data = model.compact("long context", echo_source=f"user:\n{echo}")
    assert data["summary"].startswith("Wrapped up Part B")
    assert len(sent) == 2
    assert "copied the conversation" in sent[1][-1]["content"]


def test_compaction_echo_guard_leaves_real_summaries_alone(tmp_path):
    """A summary that quotes a path or paraphrases the request is not a copy; only a summary that is
    almost entirely one verbatim run is."""
    source = "user:\n继续 Part B 收尾：检查 `_run_workflow` 的所有调用点，确保新参数没有遗漏，然后跑 lint 与 pyright。"
    assert ModelClient.echoes_source("继续 Part B 收尾：检查 `_run_workflow` 的所有调用点，确保新参数没有遗漏，然后跑 lint 与 pyright。", source)
    assert not ModelClient.echoes_source(
        "用户要求收尾 Part B。已核对 `_run_workflow` 的调用点，新参数补齐；lint 与 pyright 尚未运行，是下一步。", source
    )
    assert not ModelClient.echoes_source("short", source)  # below the length floor
    assert not ModelClient.echoes_source("a" * 200, "")  # nothing to copy from
