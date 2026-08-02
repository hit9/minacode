"""Anthropic Messages requests: block streaming, thinking, and tool blocks."""

import json
from types import SimpleNamespace

from model_harness import _AnthropicMockClientFactory, _AnthropicStreamClientFactory, _session

from minacode.base import ToolCall
from minacode.model import ModelClient


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


def test_anthropic_terminal_tool_split_replays_text_once(tmp_path):
    model = ModelClient(_session(tmp_path, model="claude-3", api="anthropic"))
    converted = model.anthropic_messages(
        [
            {"role": "user", "content": "finish"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{}],
                "_anthropic_content": [
                    {"type": "thinking", "thinking": "reasoning", "signature": "signature"},
                    {"type": "text", "text": "done"},
                    {"type": "tool_use", "id": "call_1", "name": "NextHints", "input": {}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
            {"role": "assistant", "content": "done"},
        ]
    )

    assert [message["role"] for message in converted] == ["user", "assistant", "user", "assistant"]
    assert [block["type"] for block in converted[1]["content"]] == ["thinking", "tool_use"]
    assert converted[-1]["content"] == [{"type": "text", "text": "done"}]


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


def test_anthropic_no_tools_result_strips_tool_use_preserves_thinking_and_server_tool(tmp_path):
    """An explicit tools=[] request that returns tool_use blocks: the sanitizer removes them from
    the effective calls and _anthropic_content, while preserving thinking and server_tool_use."""
    s = _session(tmp_path, model="claude-3", api="anthropic", stream=False)
    model = ModelClient(s)
    result = SimpleNamespace(
        content=[
            {"type": "thinking", "thinking": "let me check", "signature": "sig"},
            {"type": "text", "text": "here is the answer"},
            {"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"command": "ls"}},
            {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "test"}},
        ],
        usage={},
    )
    assistant, calls, text = model.anthropic_result(result)

    # Before sanitization: calls exist
    assert len(calls) == 1
    assert calls[0].name == "Bash"
    assert "tool_calls" in assistant

    # Apply the sanitizer (as request() would for tools=[])
    sanitized_assistant, sanitized_calls, sanitized_text = ModelClient.sanitize_no_tools_result((assistant, calls, text))

    # Returned effective local calls are empty
    assert sanitized_calls == []
    # top-level tool_calls is absent
    assert "tool_calls" not in sanitized_assistant
    # _anthropic_content contains no tool_use block
    saved = sanitized_assistant["_anthropic_content"]
    assert all(block.get("type") != "tool_use" for block in saved)
    # thinking and server_tool_use are preserved
    assert any(block.get("type") == "thinking" for block in saved)
    assert any(block.get("type") == "server_tool_use" for block in saved)
    # text is preserved
    assert sanitized_text == "here is the answer"
    assert sanitized_assistant["content"] == "here is the answer"

    # anthropic_messages()/anthropic_assistant_blocks() cannot replay the discarded local call
    blocks = model.anthropic_assistant_blocks(sanitized_assistant)
    assert all(block.get("type") != "tool_use" for block in blocks)
