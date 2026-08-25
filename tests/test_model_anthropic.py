"""Anthropic Messages requests: block streaming, thinking, and tool blocks."""

import json
from types import SimpleNamespace

import pytest
from model_harness import _AnthropicMockClientFactory, _AnthropicStreamClientFactory, _session

from minacode.base import ModelOutputTruncated, ToolCall
from minacode.model import ModelClient, resilience


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

    assistant, calls, content = model.wire(model.session.config.provider).request([{"role": "user", "content": "hi"}], None)

    assert content == "hello from claude"
    assert assistant == {
        "role": "assistant",
        "content": "hello from claude",
        "_anthropic_content": [{"type": "text", "text": "hello from claude"}],
        "_provider_origin": model.provider_origin(),
    }
    assert calls == []
    assert s.usage.prompt_tokens == 8
    assert s.usage.completion_tokens == 4
    assert s.usage.total_tokens == 12
    assert factory.calls[0].url.path.endswith("/messages")
    assert json.loads(factory.calls[0].content).get("stream") is not True
    assert streamed == []


def test_anthropic_terminal_tool_split_replays_text_once(tmp_path):
    model = ModelClient(_session(tmp_path, model="claude-3", api="anthropic"))
    converted = model.wire(model.session.config.provider).messages(
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

    assistant, calls, content = model.wire(model.session.config.provider).request([{"role": "user", "content": "hi"}], None)

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

    model.wire(model.session.config.provider)._stream(client, {})

    assert streamed == [("output", "hello"), ("output_done", "hello"), ("", "")]


def test_anthropic_stream_promotes_completed_text_before_server_tool(tmp_path):
    """A completed text block followed by a provider-side server_tool_use must hand off the answer
    exactly once, before the durable builtin report and the stream teardown."""
    model = ModelClient(_session(tmp_path, model="claude-3", api="anthropic"))
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "the answer"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "q"}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"query": "q"}'}},
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

    timeline = []
    model.on_stream = lambda kind, delta: timeline.append((kind, delta))
    model.on_builtin_call = lambda label, detail: timeline.append(("builtin", label, detail))
    client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **_params: Stream()))

    model.wire(model.session.config.provider)._stream(client, {})

    promoted = ("output_done", "the answer")
    builtin = ("builtin", "Web Search", "q")
    assert timeline.count(promoted) == 1
    assert timeline.index(promoted) < timeline.index(("Web Search", "")) < timeline.index(builtin) < timeline.index(("", ""))


def test_anthropic_stream_promotes_server_tool_first_text_at_block_completion(tmp_path):
    """A server_tool_use before the text block must still promote exactly once, at text block
    completion, with no duplicate at message completion."""
    model = ModelClient(_session(tmp_path, model="claude-3", api="anthropic"))
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "q"}}},
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

    model.wire(model.session.config.provider)._stream(client, {})

    assert streamed == [("Web Search", ""), ("output", "hello"), ("output_done", "hello"), ("", "")]
    assert streamed.count(("output_done", "hello")) == 1


def test_anthropic_max_tokens_stop_reason_names_the_cap_only_when_nothing_was_generated(tmp_path):
    """Thinking spends the same budget as text, so a capped step can end with no content at all."""
    model = ModelClient(_session(tmp_path, api="anthropic", model="claude-sonnet-4-5"))
    empty = {"stop_reason": "max_tokens", "content": [], "usage": {"output_tokens": 16384}}

    with pytest.raises(ModelOutputTruncated) as error:
        model.wire(model.session.config.provider).result(empty)

    assert "provider.max_tokens" in str(error.value)
    assert resilience.retryable_error(error.value) is False

    partial = {"stop_reason": "max_tokens", "content": [{"type": "text", "text": "half a sen"}]}
    _, calls, content = model.wire(model.session.config.provider).result(partial)

    assert content == "half a sen"
    assert calls == []


def test_another_hosts_thinking_signature_is_not_replayed(tmp_path):
    """A gateway and the first-party endpoint both speak Messages, but a thinking block is verified
    against the key that signed it, so an echo from elsewhere is rebuilt rather than replayed."""
    issuer = ModelClient(_session(tmp_path, api="anthropic", url="https://api.anthropic.com/v1", model="claude-x"))
    model = ModelClient(_session(tmp_path, api="anthropic", url="https://opencode.ai/zen/v1", model="claude-x"))
    history = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "answer",
            "_anthropic_content": [{"type": "thinking", "thinking": "why", "signature": "sig"}, {"type": "text", "text": "answer"}],
            "_provider_origin": issuer.provider_origin(),
        },
    ]

    rebuilt = model.wire(model.session.config.provider).messages(history)

    assert rebuilt[1]["content"] == [{"type": "text", "text": "answer"}]

    echoed = issuer.wire(issuer.session.config.provider).messages(history)
    assert echoed[1]["content"][0] == {"type": "thinking", "thinking": "why", "signature": "sig"}


def test_prompt_cache_breakpoint_rolls_with_the_conversation(tmp_path):
    """Anthropic writes cache only at a breakpoint. The system block covers tools+system; without a
    second, rolling one the conversation body -- the part that grows -- is re-read at full price
    every turn. It has to land on the tail, skip thinking blocks, and leave saved state alone."""
    model = ModelClient(_session(tmp_path, api="anthropic", url="https://api.anthropic.com/v1", model="claude-x"))
    saved = [{"type": "thinking", "thinking": "why", "signature": "sig"}, {"type": "tool_use", "id": "tc.1", "name": "Read", "input": {}}]
    history = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "_anthropic_content": saved, "_provider_origin": model.provider_origin()},
        {"role": "tool", "tool_call_id": "tc.1", "content": "output"},
    ]

    messages = model.wire(model.session.config.provider).params(history, None)["messages"]

    # Only the tail is marked, and the earlier turns keep the exact bytes they were cached with.
    assert messages[0] == {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    assert messages[1]["content"] == saved
    assert messages[2]["content"] == [{"type": "tool_result", "tool_use_id": "tc.1", "content": "output", "cache_control": {"type": "ephemeral"}}]
    # The marker is a wire-only field: the session's replayed blocks must not pick it up.
    assert saved == [{"type": "thinking", "thinking": "why", "signature": "sig"}, {"type": "tool_use", "id": "tc.1", "name": "Read", "input": {}}]

    # A turn ending in thinking marks the last block that may carry it -- a signed thinking block
    # is verified byte-for-byte, so it is echoed untouched.
    trailing = model.wire(model.session.config.provider).params(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "answer",
                "_anthropic_content": [{"type": "text", "text": "answer"}, {"type": "thinking", "thinking": "why", "signature": "sig"}],
                "_provider_origin": model.provider_origin(),
            },
        ],
        None,
    )["messages"]
    assert trailing[1]["content"] == [
        {"type": "text", "text": "answer", "cache_control": {"type": "ephemeral"}},
        {"type": "thinking", "thinking": "why", "signature": "sig"},
    ]
