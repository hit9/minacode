"""Request resilience: retries, response deadlines, and compaction calls staying off the UI."""

import email.utils
import json
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from model_harness import _MockClientFactory, _session

import minacode.model as model_module
from minacode.base import (
    MODEL_REQUEST_RETRIES,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    ModelError,
    ModelOutputTruncated,
    ModelResponseTimeout,
)
from minacode.model import ModelClient


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

    assistant, calls, content = model.request([{"role": "user", "content": "hi"}], None)

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
    class ImmediateTimer:
        def __init__(self, interval, callback):
            assert interval == 600
            self.callback = callback
            self.daemon = False

        def start(self):
            self.callback()

        def cancel(self):
            pass

    class Client:
        def __init__(self):
            self.close_count = 0

        def close(self):
            self.close_count += 1

    s = _session(tmp_path)
    model = ModelClient(s)
    client = Client()
    monkeypatch.setattr(model_module.threading, "Timer", ImmediateTimer)

    with pytest.raises(ModelResponseTimeout, match=r"provider\.response_timeout=600s") as caught:
        model.call_client(client, lambda: "completed after deadline")

    assert client.close_count == 2
    assert model.retryable_error(caught.value) is False

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


def test_zero_response_timeout_does_not_start_deadline_timer(tmp_path, monkeypatch):
    class Client:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    s = _session(tmp_path, response_timeout=0)
    model = ModelClient(s)
    client = Client()

    def unexpected_timer(*_args, **_kwargs):
        raise AssertionError("response_timeout=0 must not create a timer")

    monkeypatch.setattr(model_module.threading, "Timer", unexpected_timer)

    assert model.call_client(client, lambda: "complete") == "complete"
    assert client.closed is True


def test_total_response_timeout_relabels_interrupted_transport(tmp_path, monkeypatch):
    class ImmediateTimer:
        def __init__(self, _interval, callback):
            self.callback = callback
            self.daemon = False

        def start(self):
            self.callback()

        def cancel(self):
            pass

    class Client:
        def close(self):
            pass

    s = _session(tmp_path)
    model = ModelClient(s)
    monkeypatch.setattr(model_module.threading, "Timer", ImmediateTimer)

    def interrupted_request():
        raise RuntimeError("connection closed")

    with pytest.raises(ModelResponseTimeout, match=r"provider\.response_timeout=600s") as caught:
        model.call_client(Client(), interrupted_request)

    assert isinstance(caught.value.__cause__, RuntimeError)


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
    monkeypatch.setattr(model_module.random, "random", lambda: 0.5)  # jitter factor exactly 1.0
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
    monkeypatch.setattr(model_module.random, "random", lambda: 0.5)
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