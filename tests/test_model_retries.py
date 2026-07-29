"""Request resilience: retries, response deadlines, and compaction calls staying off the UI."""

import json
import time

import pytest
from model_harness import _MockClientFactory, _session

import minacode.model as model_module
from minacode.base import ModelError, ModelResponseTimeout
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
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    assistant, calls, content = model.request([{"role": "user", "content": "hi"}], None)

    assert content == "ok"
    assert len(factory.calls) == 2
    assert s.usage.calls == 1


def test_request_retry_exhausted(tmp_path, monkeypatch):
    s = _session(tmp_path)
    model = ModelClient(s)
    factory = _MockClientFactory([(500, {"error": {"message": "server error", "type": "internal_server_error"}})] * 6)
    monkeypatch.setattr(model, "client", factory)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

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
