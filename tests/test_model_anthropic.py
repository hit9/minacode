"""Anthropic Messages requests: block streaming, thinking, and tool blocks."""

import json
from types import SimpleNamespace

import httpx
from anthropic import Anthropic
from model_harness import _MockClientFactory, _session

from minacode.base import ToolCall
from minacode.model import ModelClient


class _AnthropicStreamClientFactory:
    def __init__(self, events: list[tuple[str, dict]], base_url: str = "http://test"):
        self.events = events
        self.calls: list[httpx.Request] = []
        self.base_url = base_url

    def __call__(self) -> Anthropic:
        def respond(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            body = "".join(f"event: {name}\ndata: {json.dumps(event)}\n\n" for name, event in self.events)
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        http_client = httpx.Client(transport=httpx.MockTransport(respond))
        return Anthropic(api_key="sk-test", base_url=self.base_url, http_client=http_client, max_retries=0)


class _AnthropicMockClientFactory(_MockClientFactory):
    """Factory that returns a fresh Anthropic client on each call."""

    def __call__(self) -> Anthropic:
        transport = httpx.MockTransport(self._next_response)
        http_client = httpx.Client(transport=transport)
        return Anthropic(api_key="sk-test", base_url=self.base_url, http_client=http_client, max_retries=0)


def test_anthropic_request_success(tmp_path, monkeypatch):
    s = _session(tmp_path, model="claude-3", api="anthropic", stream=False)
    model = ModelClient(s)
    streamed = []
    model.on_stream = lambda kind, delta: streamed.append((kind, delta))
    factory = _AnthropicMockClientFactory(
        [
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
        ]
    )
    monkeypatch.setattr(model, "anthropic_client", factory)

    assistant, calls, content = model.anthropic_request([{"role": "user", "content": "hi"}], None)

    assert content == "hello from claude"
    assert assistant == {"role": "assistant", "content": "hello from claude", "_anthropic_content": [{"type": "text", "text": "hello from claude"}]}
    assert calls == []
    assert s.usage.prompt_tokens == 8
    assert s.usage.completion_tokens == 4
    assert s.usage.total_tokens == 12
    assert factory.calls[0].url.path.endswith("/messages")
    assert json.loads(factory.calls[0].content).get("stream") is not True
    assert streamed == []


def test_anthropic_stream_reports_thinking_and_text(tmp_path, monkeypatch):
    s = _session(tmp_path, model="claude-3", api="anthropic")
    model = ModelClient(s)
    factory = _AnthropicStreamClientFactory(
        [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-3",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 10, "output_tokens": 0},
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                },
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "check"}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "content_block_start",
                {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": "", "citations": None}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "hello"}},
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 1}),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 2,
                    "content_block": {"type": "tool_use", "id": "tool_1", "name": "Bash", "input": {}},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {"type": "input_json_delta", "partial_json": '{"command":"echo hi"}'},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 2}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                    "usage": {"output_tokens": 5},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    )
    streamed = []
    model.on_stream = lambda kind, delta: streamed.append((kind, delta))
    monkeypatch.setattr(model, "anthropic_client", factory)

    assistant, calls, content = model.anthropic_request([{"role": "user", "content": "hi"}], None)

    body = json.loads(factory.calls[0].content)
    assert factory.calls[0].url.path.endswith("/messages")
    assert body["stream"] is True
    assert streamed == [("reasoning", "check"), ("output", "hello"), ("output_done", "hello"), ("", "")]
    assert content == "hello"
    assert calls == [ToolCall("tool_1", "Bash", ["echo hi"])]
    assert assistant["_anthropic_content"] == [
        {"type": "thinking", "thinking": "check", "signature": "sig"},
        {"type": "text", "text": "hello"},
        {"type": "tool_use", "id": "tool_1", "name": "Bash", "input": {"command": "echo hi"}},
    ]
    assert s.usage.prompt_tokens == 10
    assert s.usage.completion_tokens == 5


def test_anthropic_stream_promotes_when_tool_precedes_completed_text(tmp_path):
    model = ModelClient(_session(tmp_path, model="claude-3", api="anthropic"))
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "hello"}},
        {"type": "content_block_stop", "index": 1},
    ]

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __iter__(self):
            return iter(events)

        def get_final_message(self):
            return {"content": []}

    streamed = []
    model.on_stream = lambda kind, delta: streamed.append((kind, delta))
    client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **_params: Stream()))

    model._anthropic_stream(client, {})

    assert streamed == [("output", "hello"), ("output_done", "hello"), ("", "")]
