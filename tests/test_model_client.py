"""Mock-transport tests for ModelClient.

These tests intercept OpenAI/Anthropic SDK HTTP calls with httpx.MockTransport so we can
exercise chat_request, anthropic_request, retry logic, and usage accounting without
hitting real providers.
"""

import httpx
import pytest
from anthropic import Anthropic
from openai import OpenAI

import minacode as n


class _MockClientFactory:
    """Factory that returns a fresh OpenAI client on each call, all sharing one request log."""

    def __init__(self, responses: list, base_url: str = "http://test"):
        self.responses = list(responses)
        self.calls: list[httpx.Request] = []
        self.base_url = base_url

    def _next_response(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        response = self.responses.pop(0)
        if isinstance(response, int):
            return httpx.Response(response)
        status, body = response
        return httpx.Response(status, json=body)

    def __call__(self) -> OpenAI:
        transport = httpx.MockTransport(self._next_response)
        http_client = httpx.Client(transport=transport)
        return OpenAI(api_key="sk-test", base_url=self.base_url, http_client=http_client, max_retries=0)


class _AnthropicMockClientFactory(_MockClientFactory):
    """Factory that returns a fresh Anthropic client on each call."""

    def __call__(self) -> Anthropic:
        transport = httpx.MockTransport(self._next_response)
        http_client = httpx.Client(transport=transport)
        return Anthropic(api_key="sk-test", base_url=self.base_url, http_client=http_client, max_retries=0)


def _session(tmp_path, **provider_kwargs):
    config = n.Config()
    config.data_dir = str(tmp_path / "data")
    provider_kwargs.setdefault("model", "gpt-4")
    config.providers = {
        "default": n.ProviderConfig(url="http://test", key="sk-test", **provider_kwargs)
    }
    return n.Session(cwd=str(tmp_path), config=config)


def test_chat_request_success(tmp_path, monkeypatch):
    s = _session(tmp_path)
    model = n.ModelClient(s)
    factory = _MockClientFactory([
        (
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )
    ])
    monkeypatch.setattr(model, "client", factory)

    assistant, calls, content = model.chat_request([{"role": "user", "content": "hi"}], None)

    assert content == "hello"
    assert assistant == {"role": "assistant", "content": "hello"}
    assert calls == []
    assert s.usage.prompt_tokens == 10
    assert s.usage.completion_tokens == 5
    assert s.usage.total_tokens == 15
    assert s.usage.calls == 1


def test_chat_request_with_tool_calls(tmp_path, monkeypatch):
    s = _session(tmp_path)
    model = n.ModelClient(s)
    factory = _MockClientFactory([
        (
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "Bash", "arguments": '{"command": "echo hi"}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            },
        )
    ])
    monkeypatch.setattr(model, "client", factory)

    assistant, calls, content = model.chat_request([{"role": "user", "content": "run"}], [])

    assert content == ""
    assert assistant["role"] == "assistant"
    assert len(assistant["tool_calls"]) == 1
    assert assistant["tool_calls"][0]["function"]["name"] == "Bash"
    assert len(calls) == 1
    assert calls[0].name == "Bash"
    assert calls[0].args == ["echo hi"]


def test_anthropic_request_success(tmp_path, monkeypatch):
    s = _session(tmp_path, model="claude-3", api="anthropic")
    model = n.ModelClient(s)
    factory = _AnthropicMockClientFactory([
        (
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-3",
                "content": [{"type": "text", "text": "hello from claude"}],
                "usage": {"input_tokens": 8, "output_tokens": 4},
            },
        )
    ])
    monkeypatch.setattr(model, "anthropic_client", factory)

    assistant, calls, content = model.anthropic_request([{"role": "user", "content": "hi"}], None)

    assert content == "hello from claude"
    assert assistant == {"role": "assistant", "content": "hello from claude"}
    assert calls == []
    assert s.usage.prompt_tokens == 8
    assert s.usage.completion_tokens == 4
    assert s.usage.total_tokens == 12


def test_request_retries_then_succeeds(tmp_path, monkeypatch):
    s = _session(tmp_path)
    model = n.ModelClient(s)
    factory = _MockClientFactory([
        (429, {"error": {"message": "rate limited", "type": "rate_limit_error"}}),
        (
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ),
    ])
    monkeypatch.setattr(model, "client", factory)
    monkeypatch.setattr(n.time, "sleep", lambda _seconds: None)

    assistant, calls, content = model.request([{"role": "user", "content": "hi"}], None)

    assert content == "ok"
    assert len(factory.calls) == 2
    assert s.usage.calls == 1


def test_request_retry_exhausted(tmp_path, monkeypatch):
    s = _session(tmp_path)
    model = n.ModelClient(s)
    factory = _MockClientFactory([
        (500, {"error": {"message": "server error", "type": "internal_server_error"}}),
        (500, {"error": {"message": "server error", "type": "internal_server_error"}}),
        (500, {"error": {"message": "server error", "type": "internal_server_error"}}),
    ])
    monkeypatch.setattr(model, "client", factory)
    monkeypatch.setattr(n.time, "sleep", lambda _seconds: None)

    with pytest.raises(n.ModelError, match="after 3 attempts"):
        model.request([{"role": "user", "content": "hi"}], None)

    assert len(factory.calls) == 3
    assert s.usage.calls == 0
