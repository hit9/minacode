"""Responses API requests: output items, streaming, replay, and reasoning parameters."""

import json
import time
from types import SimpleNamespace

import httpx
import pytest
from model_harness import _MockClientFactory, _session, _StreamClientFactory
from openai import OpenAI

from wizolt.base import SESSION_EVENT_KEY, ModelError, ModelOutputTruncated, ModelStreamIncomplete, ToolCall
from wizolt.config import ProviderConfig
from wizolt.model import ModelClient, resilience
from wizolt.tools import BashTool


def test_responses_request_preserves_output_items_and_uses_responses_shape(tmp_path, monkeypatch):
    s = _session(tmp_path, api="responses", model="gpt-5", stream=False)
    model = ModelClient(s)
    streamed = []
    model.on_stream = lambda kind, delta: streamed.append((kind, delta))
    factory = _MockClientFactory(
        [
            (
                200,
                {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-5",
                    "output": [
                        {"id": "rs_test", "type": "reasoning", "encrypted_content": "encrypted", "summary": []},
                        {
                            "id": "msg_test",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "hello", "annotations": [], "logprobs": []}],
                        },
                    ],
                    "usage": {
                        "input_tokens": 10,
                        "input_tokens_details": {"cached_tokens": 7},
                        "output_tokens": 5,
                        "output_tokens_details": {"reasoning_tokens": 2},
                        "total_tokens": 15,
                    },
                },
            )
        ]
    )
    monkeypatch.setattr(model, "client", factory)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Run a command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        }
    ]

    assistant, calls, content = model.request([{"role": "user", "content": "hi"}], tools)

    assert content == "hello"
    assert calls == []
    assert assistant["content"] == "hello"
    assert [item["type"] for item in assistant["_responses_output"]] == ["reasoning", "message"]
    assert assistant["_responses_output"][0]["encrypted_content"] == "encrypted"
    request = factory.calls[0]
    assert request.url.path.endswith("/responses")
    assert request.url.path == "/responses"
    body = json.loads(request.content)
    assert body["input"] == [{"role": "user", "content": "hi"}]
    assert body["store"] is False
    assert body["stream"] is False
    assert body["reasoning"] == {"effort": "medium"}
    assert body["tools"] == [
        {
            "type": "function",
            "name": "Bash",
            "description": "Run a command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            "strict": False,
        }
    ]
    assert s.usage.prompt_tokens == 10
    assert s.usage.cached_prompt_tokens == 7
    assert s.usage.last_cached_prompt_tokens == 7
    assert s.usage.completion_tokens == 5
    assert streamed == []


def test_responses_input_strips_session_event_metadata(tmp_path):
    s = _session(tmp_path, api="responses", model="gpt-5", stream=False)

    converted = ModelClient(s).wire(s.config.provider).messages([{"role": "user", "content": "<session_event />", SESSION_EVENT_KEY: "resumed"}])

    assert converted == [{"role": "user", "content": "<session_event />"}]


def test_responses_stream_reports_deltas_and_uses_terminal_response(tmp_path, monkeypatch):
    s = _session(tmp_path, api="responses", model="gpt-5")
    model = ModelClient(s)
    terminal = {
        "id": "resp_stream",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-5",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {"id": "rs_stream", "type": "reasoning", "summary": [{"type": "summary_text", "text": "checking"}]},
            {
                "id": "msg_stream",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}],
            },
        ],
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 7},
            "output_tokens": 5,
            "total_tokens": 15,
        },
    }
    events = [
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_stream",
            "output_index": 0,
            "summary_index": 0,
            "delta": "check",
            "sequence_number": 1,
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_stream",
            "output_index": 1,
            "content_index": 0,
            "delta": "hel",
            "logprobs": [],
            "sequence_number": 2,
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_stream",
            "output_index": 1,
            "content_index": 0,
            "delta": "lo",
            "logprobs": [],
            "sequence_number": 3,
        },
        {"type": "response.completed", "response": terminal, "sequence_number": 4},
    ]
    factory = _StreamClientFactory(events)
    streamed = []
    model.on_stream = lambda kind, delta: streamed.append((kind, delta))
    monkeypatch.setattr(model, "client", factory)

    assistant, calls, content = model.request([{"role": "user", "content": "hi"}], [])

    assert json.loads(factory.calls[0].content)["stream"] is True
    assert factory.calls[0].url.path.endswith("/responses")
    assert streamed == [("reasoning", "check"), ("output", "hel"), ("output", "lo"), ("", "")]
    assert content == "hello"
    assert assistant["content"] == "hello"
    assert calls == []
    assert s.usage.prompt_tokens == 10
    assert s.usage.cached_prompt_tokens == 7


class _SequenceStreamFactory:
    """Streams the nth event list on the nth call, so a test can drop the terminal event once."""

    def __init__(self, streams: list[list[dict]]):
        self.streams = streams
        self.calls: list[httpx.Request] = []

    def __call__(self, **kwargs) -> OpenAI:
        def respond(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            events = self.streams[min(len(self.calls) - 1, len(self.streams) - 1)]
            body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        http_client = httpx.Client(transport=httpx.MockTransport(respond))
        return OpenAI(api_key="sk-test", base_url="http://test", http_client=http_client, max_retries=0)


def _patch_fast_clock(monkeypatch):
    """Advance the clock by each requested sleep so backoff waits complete instantly."""
    clock = {"now": 0.0}
    monkeypatch.setattr(time, "sleep", lambda seconds: clock.__setitem__("now", clock["now"] + seconds))
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])


def test_responses_stream_incomplete_is_retryable():
    """A stream without a terminal event is a transient server-side drop, not a deterministic
    failure: it must retry, and the status bar labels it "stream" (matching the "connection"
    label for httpx transport errors)."""
    error = ModelStreamIncomplete("Responses stream ended without a terminal response")
    assert resilience.retryable_error(error) is True
    assert resilience.retry_reason(error) == "stream"


def test_responses_stream_without_terminal_event_retries_then_succeeds(tmp_path, monkeypatch):
    """A stream that ends with deltas but no terminal event is a server-side drop detected after
    the fact by reassemble_stream: the request retries instead of ending the turn. Regression for
    "Responses stream ended without a terminal response"."""
    s = _session(tmp_path, api="responses", model="gpt-5")
    model = ModelClient(s)
    _patch_fast_clock(monkeypatch)

    terminal = {
        "id": "resp_stream",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-5",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": "msg_stream",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 7},
            "output_tokens": 5,
            "total_tokens": 15,
        },
    }
    delta = {
        "type": "response.output_text.delta",
        "item_id": "msg_stream",
        "output_index": 1,
        "content_index": 0,
        "delta": "hel",
        "logprobs": [],
        "sequence_number": 2,
    }
    factory = _SequenceStreamFactory([[delta], [delta, {"type": "response.completed", "response": terminal, "sequence_number": 4}]])
    streamed = []
    model.on_stream = lambda kind, delta: streamed.append((kind, delta))
    monkeypatch.setattr(model, "client", factory)

    assistant, calls, content = model.request([{"role": "user", "content": "hi"}], [])

    assert len(factory.calls) == 2
    assert streamed == [("output", "hel"), ("", ""), ("output", "hel"), ("", "")]
    assert content == "hello"
    assert calls == []
    assert assistant["content"] == "hello"
    assert s.state.model_retry_count == 1


@pytest.mark.parametrize("event_type", ["response.reasoning_summary_text.delta", "response.reasoning_text.delta"])
def test_responses_stream_previews_reasoning_in_either_spelling(tmp_path, monkeypatch, event_type):
    """Hosts that summarize reasoning stream the summary; hosts that expose the raw chain stream the
    text. DeepSeek only sends the text spelling and generates no summary, so accepting one spelling
    left a thinking model with no preview at all. Both must keep working."""
    model = ModelClient(_session(tmp_path, api="responses", model="gpt-5"))
    terminal = {"id": "resp", "object": "response", "created_at": 1, "status": "completed", "output": []}
    events = [
        {"type": event_type, "delta": "weigh", "sequence_number": 1},
        {"type": event_type, "delta": "ing", "sequence_number": 2},
        {"type": "response.output_text.delta", "delta": "done", "sequence_number": 3},
        {"type": "response.completed", "response": terminal, "sequence_number": 4},
    ]
    streamed = []
    model.on_stream = lambda kind, delta: streamed.append((kind, delta))
    monkeypatch.setattr(model, "client", _StreamClientFactory(events))

    model.request([{"role": "user", "content": "hi"}], [])

    assert streamed == [("reasoning", "weigh"), ("reasoning", "ing"), ("output", "done"), ("", "")]


@pytest.mark.parametrize("order", ["text-first", "tool-first"])
def test_responses_stream_promotes_completed_text_before_tool_arguments_finish(tmp_path, monkeypatch, order):
    model = ModelClient(_session(tmp_path, api="responses", model="gpt-5"))
    text_delta = {"type": "response.output_text.delta", "delta": "I am editing the files."}
    text_done = {"type": "response.output_text.done"}
    tool_added = {"type": "response.output_item.added", "item": {"type": "function_call"}}
    prefix = [text_delta, text_done, tool_added] if order == "text-first" else [tool_added, text_delta, text_done]
    terminal = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I am editing the files."}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "Bash",
                "arguments": '{"command":"echo hi"}',
            },
        ],
    }
    timeline = []

    def events():
        yield from prefix
        timeline.append(("wire", "tool arguments"))
        yield {"type": "response.function_call_arguments.delta", "delta": '{"args"'}
        yield {"type": "response.completed", "response": terminal}

    responses = SimpleNamespace(create=lambda **_params: events())
    monkeypatch.setattr(model, "client", lambda **kwargs: SimpleNamespace(responses=responses))
    model.on_stream = lambda kind, delta: timeline.append((kind, delta))

    _, calls, content = model.request([{"role": "user", "content": "make the change"}], None)

    promoted = ("output_done", "I am editing the files.")
    assert timeline.index(promoted) < timeline.index(("wire", "tool arguments"))
    assert timeline.count(promoted) == 1
    assert calls == [ToolCall("call_1", "Bash", ["echo hi"])]
    assert content == "I am editing the files."


@pytest.mark.parametrize("order", ["text-first", "tool-first"])
def test_responses_stream_promotes_completed_text_across_provider_call(tmp_path, monkeypatch, order):
    """A provider-side `*_call` item is the same promotion boundary as a local function call.

    Text completing before or after the call must both produce exactly one `output_done` before the
    final empty stream callback clears the preview."""
    model = ModelClient(_session(tmp_path, api="responses", model="gpt-5"))
    text_delta = {"type": "response.output_text.delta", "delta": "The answer is sunny."}
    text_done = {"type": "response.output_text.done"}
    call_added = {"type": "response.output_item.added", "item": {"id": "ws_1", "type": "web_search_call", "status": "in_progress"}}
    call_done = {
        "type": "response.output_item.done",
        "item": {"id": "ws_1", "type": "web_search_call", "status": "completed", "action": {"type": "search", "query": "weather"}},
    }
    prefix = [text_delta, text_done, call_added, call_done] if order == "text-first" else [call_added, call_done, text_delta, text_done]
    terminal = {
        "status": "completed",
        "output": [
            {"id": "m", "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": "The answer is sunny."}]},
            {"id": "ws_1", "type": "web_search_call", "status": "completed", "action": {"type": "search", "query": "weather"}},
        ],
    }
    timeline = []

    def events():
        yield from prefix
        yield {"type": "response.completed", "response": terminal}

    responses = SimpleNamespace(create=lambda **_params: events())
    monkeypatch.setattr(model, "client", lambda **kwargs: SimpleNamespace(responses=responses))
    model.on_stream = lambda kind, delta: timeline.append((kind, delta))
    model.on_builtin_call = lambda label, detail: timeline.append(("builtin", label, detail))

    _, calls, content = model.request([{"role": "user", "content": "weather?"}], None)

    promoted = ("output_done", "The answer is sunny.")
    clear = ("", "")
    assert timeline.count(promoted) == 1
    assert timeline.index(promoted) < timeline.index(clear)
    if order == "text-first":
        builtin = ("builtin", "Web Search", "weather")
        assert timeline.index(promoted) < timeline.index(("Web Search", "")) < timeline.index(builtin) < timeline.index(clear)
    else:
        # The durable builtin report may precede the answer; the handoff happens at text completion.
        assert timeline.index(("Web Search", "")) < timeline.index(promoted) < timeline.index(clear)
    assert calls == []
    assert content == "The answer is sunny."


def test_responses_stream_promotes_when_output_item_added_is_missing(tmp_path, monkeypatch):
    """Some compatible providers emit output_item.done without the matching added event; the
    defensive boundary on the durable report must still permit promotion."""
    model = ModelClient(_session(tmp_path, api="responses", model="gpt-5"))
    terminal = {
        "status": "completed",
        "output": [
            {"id": "m", "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": "searched"}]},
            {"id": "ws_1", "type": "web_search_call", "status": "completed", "action": {"type": "search", "query": "q"}},
        ],
    }
    events = [
        {"type": "response.output_text.delta", "delta": "searched"},
        {"type": "response.output_text.done"},
        {
            "type": "response.output_item.done",
            "item": {"id": "ws_1", "type": "web_search_call", "status": "completed", "action": {"type": "search", "query": "q"}},
        },
        {"type": "response.completed", "response": terminal},
    ]
    timeline = []
    responses = SimpleNamespace(create=lambda **_params: iter(events))
    monkeypatch.setattr(model, "client", lambda **kwargs: SimpleNamespace(responses=responses))
    model.on_stream = lambda kind, delta: timeline.append((kind, delta))

    _, calls, content = model.request([{"role": "user", "content": "hi"}], None)

    promoted = ("output_done", "searched")
    assert timeline.count(promoted) == 1
    assert timeline.index(promoted) < timeline.index(("", ""))
    assert calls == []
    assert content == "searched"


def test_responses_stream_returns_incomplete_terminal_response_and_clears_preview(tmp_path, monkeypatch):
    s = _session(tmp_path, api="responses", model="gpt-5")
    model = ModelClient(s)
    factory = _StreamClientFactory(
        [
            {
                "type": "response.output_text.delta",
                "item_id": "msg_stream",
                "output_index": 0,
                "content_index": 0,
                "delta": "partial",
                "logprobs": [],
                "sequence_number": 1,
            },
            {
                "type": "response.incomplete",
                "response": {
                    "id": "resp_stream",
                    "object": "response",
                    "created_at": 1,
                    "status": "incomplete",
                    "model": "gpt-5",
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                    "output": [
                        {
                            "id": "msg_stream",
                            "type": "message",
                            "status": "incomplete",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "partial", "annotations": []}],
                        }
                    ],
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
                "sequence_number": 2,
            },
        ]
    )
    streamed = []
    model.on_stream = lambda kind, delta: streamed.append((kind, delta))
    monkeypatch.setattr(model, "client", factory)

    assistant, calls, content = model.request([{"role": "user", "content": "hi"}], [])

    assert streamed == [("output", "partial"), ("", "")]
    assert content == "partial"
    assert assistant["content"] == "partial"
    assert calls == []


def test_responses_failed_result_raises_for_streaming_and_non_streaming_paths(tmp_path):
    model = ModelClient(_session(tmp_path, api="responses"))

    with pytest.raises(ModelError, match="Responses request failed"):
        model.wire(model.session.config.provider).result({"status": "failed", "error": {"message": "bad request"}, "output": []})


def test_responses_incomplete_output_reports_the_cap_instead_of_an_empty_answer(tmp_path):
    """A high-effort step can spend the whole output cap on reasoning and return nothing else.

    Without this the turn fails as "empty final response", which names neither the cause nor the
    setting to change."""
    model = ModelClient(_session(tmp_path, api="responses"))
    truncated = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"type": "reasoning", "summary": []}],
        "usage": {"output_tokens": 16384, "output_tokens_details": {"reasoning_tokens": 16384}},
    }

    with pytest.raises(ModelOutputTruncated) as error:
        model.wire(model.session.config.provider).result(truncated)

    assert "provider.max_tokens" in str(error.value)
    assert "16384" in str(error.value)
    assert "reasoning" in str(error.value)
    # Deterministic: the same request hits the same cap again, so it must not consume a retry.
    assert resilience.retryable_error(error.value) is False


def test_responses_incomplete_for_another_reason_still_returns_its_output(tmp_path):
    model = ModelClient(_session(tmp_path, api="responses"))
    result = {
        "status": "incomplete",
        "incomplete_details": {"reason": "content_filter"},
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "partial"}]}],
    }

    _, calls, content = model.wire(model.session.config.provider).result(result)

    assert content == "partial"
    assert calls == []


def test_responses_failed_mock_servers_match_across_stream_modes(tmp_path, monkeypatch):
    terminal = {
        "id": "resp_failed",
        "object": "response",
        "created_at": 1,
        "status": "failed",
        "model": "gpt-5",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": [],
        "error": {"code": "server_error", "message": "provider failed"},
    }

    streaming = ModelClient(_session(tmp_path / "stream", api="responses", model="gpt-5"))
    stream_factory = _StreamClientFactory([{"type": "response.failed", "response": terminal, "sequence_number": 1}])
    streamed = []
    streaming.on_stream = lambda kind, delta: streamed.append((kind, delta))
    monkeypatch.setattr(streaming, "client", stream_factory)

    with pytest.raises(ModelError, match="Responses request failed"):
        streaming.request([{"role": "user", "content": "hi"}], [])

    assert stream_factory.calls[0].url.path.endswith("/responses")
    assert json.loads(stream_factory.calls[0].content)["stream"] is True
    assert streamed == [("", "")]

    non_streaming = ModelClient(_session(tmp_path / "plain", api="responses", model="gpt-5", stream=False))
    plain_factory = _MockClientFactory([(200, terminal)])
    non_streaming.on_stream = lambda _kind, _delta: pytest.fail("disabled stream callback was called")
    monkeypatch.setattr(non_streaming, "client", plain_factory)

    with pytest.raises(ModelError, match="Responses request failed"):
        non_streaming.request([{"role": "user", "content": "hi"}], [])

    assert plain_factory.calls[0].url.path.endswith("/responses")
    assert json.loads(plain_factory.calls[0].content)["stream"] is False


def test_responses_tool_items_are_converted_and_replayed(tmp_path):
    model = ModelClient(_session(tmp_path, api="responses"))
    result = {
        "output": [
            {"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque", "summary": []},
            {
                "id": "fc_1",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_1",
                "name": "Bash",
                "arguments": '{"command":"echo hi"}',
            },
        ]
    }

    assistant, calls, content = model.wire(model.session.config.provider).result(result)

    assert content == ""
    assert calls == [ToolCall("call_1", "Bash", ["echo hi"])]
    assert assistant["tool_calls"][0]["id"] == "call_1"
    converted = model.wire(model.session.config.provider).messages(
        [
            {"role": "user", "content": "run it"},
            assistant,
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        ]
    )
    assert converted == [
        {"role": "user", "content": "run it"},
        {"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque", "summary": []},
        {
            "id": "fc_1",
            "type": "function_call",
            "status": "completed",
            "call_id": "call_1",
            "name": "Bash",
            "arguments": '{"command":"echo hi"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "done"},
    ]


def test_responses_replay_repairs_duplicated_terminal_tool_reply(tmp_path):
    model = ModelClient(_session(tmp_path, api="responses"))
    saved_output = [
        {"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque"},
        {"id": "msg_1", "type": "message", "content": [{"type": "output_text", "text": "done"}]},
        {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "NextHints", "arguments": "{}"},
    ]
    converted = model.wire(model.session.config.provider).messages(
        [
            {"role": "user", "content": "finish"},
            {"role": "assistant", "content": None, "tool_calls": [{}], "_responses_output": saved_output},
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
            {"role": "assistant", "content": "done", "_responses_output": saved_output},
        ]
    )

    assert [item.get("id") for item in converted if item.get("id")] == ["rs_1", "fc_1", "msg_1"]
    assert [item.get("type", "message") for item in converted] == ["message", "reasoning", "function_call", "function_call_output", "message"]


def test_responses_function_call_round_trip_over_sdk_transport(tmp_path, monkeypatch):
    s = _session(tmp_path, api="responses")
    model = ModelClient(s)
    factory = _MockClientFactory(
        [
            (
                200,
                {
                    "id": "resp_tool",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-4",
                    "output": [
                        {"id": "rs_tool", "type": "reasoning", "encrypted_content": "opaque", "summary": []},
                        {
                            "id": "fc_tool",
                            "type": "function_call",
                            "status": "completed",
                            "call_id": "call_1",
                            "name": "Bash",
                            "arguments": '{"command":"echo hi"}',
                        },
                    ],
                },
            ),
            (
                200,
                {
                    "id": "resp_final",
                    "object": "response",
                    "created_at": 2,
                    "status": "completed",
                    "model": "gpt-4",
                    "output": [
                        {
                            "id": "msg_final",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "done", "annotations": [], "logprobs": []}],
                        }
                    ],
                },
            ),
        ]
    )
    monkeypatch.setattr(model, "client", factory)
    tools = [BashTool.schema(False)]

    assistant, calls, content = model.request([{"role": "user", "content": "run it"}], tools)
    assert content == ""
    assert calls == [ToolCall("call_1", "Bash", ["echo hi"])]

    final, final_calls, final_content = model.request(
        [
            {"role": "user", "content": "run it"},
            assistant,
            {"role": "tool", "tool_call_id": "call_1", "content": "hi"},
        ],
        tools,
    )

    assert final_content == "done"
    assert final_calls == []
    assert final["content"] == "done"
    second_body = json.loads(factory.calls[1].content)
    assert [item.get("type", "message") for item in second_body["input"]] == [
        "message",
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    assert second_body["input"][1]["encrypted_content"] == "opaque"
    assert second_body["input"][2]["call_id"] == "call_1"
    assert second_body["input"][3] == {"type": "function_call_output", "call_id": "call_1", "output": "hi"}


def test_responses_request_folds_effort_and_drops_rejected_temperature(tmp_path, monkeypatch):
    """The Responses path shares the chat path's compatibility handling: effort goes through the
    host's fold, and OpenAI reasoning models reject temperature outright."""
    s = _session(tmp_path, api="responses", model="gpt-5")
    s.config.provider.url = "https://api.openai.com/v1"
    s.config.provider.reasoning = "high"
    s.config.provider.temperature = 0.7
    model = ModelClient(s)
    empty = {"id": "r", "object": "response", "created_at": 1, "status": "completed", "model": "gpt-5", "output": []}
    factory = _MockClientFactory([(200, empty), (200, empty)])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], None)
    body = json.loads(factory.calls[0].content)
    assert body["reasoning"] == {"effort": "high"}
    assert "temperature" not in body

    # A host that documents an explicit spelling for "no thinking" still gets it when reasoning
    # is off, instead of falling back to the model's default behaviour.
    s.config.provider.url = "https://api.kimi.com/coding/v1"
    s.config.provider.model = "k3"
    s.config.provider.reasoning = "off"
    model.request([{"role": "user", "content": "hi"}], None)
    body = json.loads(factory.calls[1].content)
    assert body["reasoning"] == {"effort": "none"}
    assert body["temperature"] == 0.7


def test_openai_responses_reasoning_off_is_not_silently_replaced_by_the_model_default(tmp_path, monkeypatch):
    s = _session(tmp_path, api="responses", model="gpt-5.5")
    s.config.provider.url = "https://api.openai.com/v1"
    s.config.provider.reasoning = "off"
    model = ModelClient(s)
    empty = {"id": "r", "object": "response", "created_at": 1, "status": "completed", "model": "gpt-5.5", "output": []}
    factory = _MockClientFactory([(200, empty)])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], None)

    assert json.loads(factory.calls[0].content)["reasoning"] == {"effort": "none"}


@pytest.mark.parametrize(
    ("url", "model_name", "reasoning", "expected"),
    (
        ("https://api.openai.com/v1", "gpt-5.6-sol", "max", "max"),
        ("https://api.openai.com/v1", "gpt-5.5", "max", "xhigh"),
        ("https://api.openai.com/v1", "gpt-5.5-pro", "low", "medium"),
        ("https://api.openai.com/v1", "gpt-5.7", "max", "max"),
        # Grok 4.5 documents low/medium/high, so a stored `max` lands on the top level it has.
        ("https://opencode.ai/zen/v1", "grok-4.5", "max", "high"),
        ("https://models.example/v1", "future-reasoner", "max", "max"),
    ),
)
def test_responses_sends_the_resolved_reasoning_effort(tmp_path, monkeypatch, url, model_name, reasoning, expected):
    s = _session(tmp_path, url=url, api="responses", model=model_name, reasoning=reasoning, stream=False)
    model = ModelClient(s)
    empty = {"id": "r", "object": "response", "created_at": 1, "status": "completed", "model": model_name, "output": []}
    factory = _MockClientFactory([(200, empty)])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], None)

    assert json.loads(factory.calls[0].content)["reasoning"] == {"effort": expected}


@pytest.mark.parametrize("reasoning", ("medium", "off"))
def test_openai_responses_non_reasoning_model_omits_reasoning(tmp_path, monkeypatch, reasoning):
    """GPT-4.1 supports Responses but is not a reasoning model, so the optional reasoning object
    must be absent for both the default effort and an explicit off setting."""
    s = _session(tmp_path, api="responses", model="gpt-4.1", stream=False)
    s.config.provider.url = "https://api.openai.com/v1"
    s.config.provider.reasoning = reasoning
    model = ModelClient(s)
    empty = {"id": "r", "object": "response", "created_at": 1, "status": "completed", "model": "gpt-4.1", "output": []}
    factory = _MockClientFactory([(200, empty)])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], None)

    assert "reasoning" not in json.loads(factory.calls[0].content)


@pytest.mark.parametrize("model", ("custom-model", "gpt-5", "gpt-5-mini"))
def test_responses_reports_unsupported_reasoning_off_instead_of_guessing(tmp_path, model):
    s = _session(tmp_path, api="responses", model=model)
    s.config.provider.reasoning = "off"
    if model.startswith("gpt-5"):
        s.config.provider.url = "https://api.openai.com/v1"

    with pytest.raises(ModelError, match="reasoning off is not defined"):
        ModelClient(s).wire(s.config.provider).request([{"role": "user", "content": "hi"}], None)


def test_responses_replay_drops_reasoning_items_that_carry_no_payload(tmp_path):
    """Stateless reasoning travels in the encrypted payload; an id alone cannot stand in for it
    once the response was never stored, so an empty shell is dropped rather than replayed."""
    model = ModelClient(_session(tmp_path, api="responses"))
    assistant, _, _ = model.wire(model.session.config.provider).result(
        {
            "output": [
                {"id": "rs_bare", "type": "reasoning", "summary": []},
                {"id": "rs_kept", "type": "reasoning", "encrypted_content": "opaque", "summary": []},
                {"id": "rs_text", "type": "reasoning", "summary": [{"type": "summary_text", "text": "thought"}]},
                {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "Bash", "arguments": "{}"},
            ]
        }
    )

    replayed = model.wire(model.session.config.provider).messages([assistant])

    assert [item["id"] for item in replayed] == ["rs_kept", "rs_text", "fc_1"]


def test_responses_reasoning_history_filters_the_actual_replayed_items(tmp_path):
    s = _session(tmp_path, api="responses", model="gpt-5", reasoning_history="current_turn")
    model = ModelClient(s)
    origin = model.provider_origin()
    history = [
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": "done",
            "_provider_origin": origin,
            "_responses_output": [
                {"id": "rs_old", "type": "reasoning", "encrypted_content": "old"},
                {"id": "msg_old", "type": "message", "content": [{"type": "output_text", "text": "done"}]},
            ],
        },
        {"role": "user", "content": "next"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}],
            "_provider_origin": origin,
            "_responses_output": [
                {"id": "rs_current", "type": "reasoning", "encrypted_content": "current"},
                {"id": "fc_current", "type": "function_call", "call_id": "call_1", "name": "Read", "arguments": "{}"},
            ],
        },
    ]

    replayed = model.wire(s.config.provider).messages(history)
    replayed_ids = {item.get("id") for item in replayed}

    assert "rs_old" not in replayed_ids
    assert "rs_current" in replayed_ids


def test_no_protocol_sends_another_protocols_saved_reply(tmp_path, monkeypatch):
    """`/provider` can switch protocols mid-session, so history holds assistant turns produced by
    a protocol other than the one now in use. Each protocol replays only its own saved reply and
    never puts wizolt's bookkeeping keys on the wire."""
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "from responses", "_responses_output": [{"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque"}]},
        {"role": "assistant", "content": "from anthropic", "_anthropic_content": [{"type": "thinking", "thinking": "", "signature": "sig"}]},
    ]
    s = _session(tmp_path, model="claude-x")
    model = ModelClient(s)

    # Consecutive assistant turns merge into one message, as the Messages API requires roles to
    # alternate. The responses-only turn is rebuilt as text; only the Anthropic turn is echoed.
    anthropic_params = model.wire(ProviderConfig(api="anthropic", model="claude")).params(history, None)
    # The cache breakpoint skips the thinking block -- the API checks those against the signature
    # it issued -- and lands on the last block that may carry it.
    assert anthropic_params["messages"][1]["content"] == [
        {"type": "text", "text": "from responses", "cache_control": {"type": "ephemeral"}},
        {"type": "thinking", "thinking": "", "signature": "sig"},
    ]

    responses_input = model.wire(ProviderConfig(api="responses", model="gpt-5")).messages(history)
    assert responses_input == [
        {"role": "user", "content": "hi"},
        {"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque"},
        {"role": "assistant", "content": "from anthropic"},
    ]

    factory = _MockClientFactory(
        [
            (
                200,
                {
                    "id": "c",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "m",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                },
            )
        ]
    )
    monkeypatch.setattr(model, "client", factory)
    model.wire(ProviderConfig(api="chat", model="gpt-4o")).request(history, None)
    body = factory.calls[0].content.decode()
    assert "_responses_output" not in body
    assert "_anthropic_content" not in body


def test_responses_normal_tool_path_preserves_calls_and_replays(tmp_path, monkeypatch):
    """When non-empty schemas were offered and the provider returns a valid local tool call:
    preserve the normalized call and provider echo, then replay a matching result."""
    s = _session(tmp_path, api="responses", model="gpt-5", stream=False)
    model = ModelClient(s)
    factory = _MockClientFactory(
        [
            (
                200,
                {
                    "id": "resp_1",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-5",
                    "output": [
                        {"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque", "summary": []},
                        {
                            "id": "fc_1",
                            "type": "function_call",
                            "status": "completed",
                            "call_id": "call_1",
                            "name": "Bash",
                            "arguments": '{"command":"echo hi"}',
                        },
                    ],
                    "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                },
            ),
            (
                200,
                {
                    "id": "resp_2",
                    "object": "response",
                    "created_at": 2,
                    "status": "completed",
                    "model": "gpt-5",
                    "output": [
                        {
                            "id": "msg_2",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "done", "annotations": []}],
                        },
                    ],
                    "usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
                },
            ),
        ]
    )
    monkeypatch.setattr(model, "client", factory)
    tools = [BashTool.schema(False)]

    # First request: non-empty tools, provider returns a valid call
    assistant, calls, _ = model.request([{"role": "user", "content": "run it"}], tools)
    assert calls == [ToolCall("call_1", "Bash", ["echo hi"])]
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert any(item.get("type") == "function_call" for item in assistant["_responses_output"])

    # Second request: replay the call/result pair
    history = [
        {"role": "user", "content": "run it"},
        assistant,
        {"role": "tool", "tool_call_id": "call_1", "content": "hi"},
    ]
    _, final_calls, final_content = model.request(history, tools)
    assert final_content == "done"
    assert final_calls == []

    # The second request body contains the function_call and function_call_output
    second_body = json.loads(factory.calls[1].content)
    input_types = [item.get("type", "message") for item in second_body["input"]]
    assert "function_call" in input_types
    assert "function_call_output" in input_types


def test_another_hosts_encrypted_reasoning_is_not_replayed(tmp_path):
    """Two Responses hosts share a protocol but not a key: only the issuer can verify its own
    ciphertext, so a turn recorded elsewhere is rebuilt from text and calls instead of echoed."""
    issuer = ModelClient(_session(tmp_path, api="responses", url="https://api.openai.com/v1", model="gpt-5.6"))
    model = ModelClient(_session(tmp_path, api="responses", url="https://openrouter.ai/api/v1", model="vendor/model"))
    history = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "from openai",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "Bash", "arguments": "{}"}}],
            "_responses_output": [
                {"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque"},
                {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "Bash", "arguments": "{}"},
            ],
            "_provider_origin": issuer.provider_origin(),
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]

    replayed = model.wire(model.session.config.provider).messages(history)

    assert not any(item.get("type") == "reasoning" for item in replayed)
    assert [item.get("type", "message") for item in replayed] == ["message", "message", "function_call", "function_call_output"]
    # The tool pairing survives the downgrade: dropping the echo must not orphan a call result.
    assert [item["call_id"] for item in replayed if item.get("type") in ("function_call", "function_call_output")] == ["call_1", "call_1"]

    # The same history on the host that issued it replays untouched.
    assert any(item.get("encrypted_content") == "opaque" for item in issuer.wire(issuer.session.config.provider).messages(history))


def test_unmarked_history_stays_replayable(tmp_path):
    """Sessions recorded before origins were stamped carry no identity; refusing them would drop
    thinking blocks in the middle of a resumed tool loop."""
    s = _session(tmp_path, api="responses", url="https://api.openai.com/v1", model="gpt-5.6")
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "old", "_responses_output": [{"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque"}]},
    ]

    assert any(item.get("encrypted_content") == "opaque" for item in ModelClient(s).wire(s.config.provider).messages(history))


def test_configured_reasoning_fields_merge_into_the_managed_object(tmp_path, monkeypatch):
    """`extra_body` is merged over the body, so a whole `reasoning` object configured there used to
    take the resolved effort down with it. Documented extras stay reachable; `/reason` still wins."""
    s = _session(
        tmp_path,
        api="responses",
        url="https://api.openai.com/v1",
        model="gpt-5.6",
        reasoning="high",
        stream=False,
        extra_body={"reasoning": {"context": "current_turn", "effort": "low"}, "safety_identifier": "u1"},
    )
    model = ModelClient(s)
    factory = _MockClientFactory(
        [(200, {"id": "r", "object": "response", "created_at": 1, "status": "completed", "model": "gpt-5.6", "output": []})],
        base_url="https://api.openai.com/v1",
    )
    monkeypatch.setattr(model, "client", factory)

    model.wire(model.session.config.provider).request([{"role": "user", "content": "hi"}], None)

    body = json.loads(factory.calls[0].content)
    assert body["reasoning"] == {"effort": "high", "context": "current_turn"}
    assert body["safety_identifier"] == "u1"


def test_catalog_extra_body_paths_do_not_replace_user_extensions(tmp_path, monkeypatch):
    """The Responses adapter must expose user extra_body before the catalog recipe runs."""
    s = _session(
        tmp_path,
        api="responses",
        model="gpt-4.1",
        stream=False,
        extra_body={"user_extension": "kept"},
    )
    model = ModelClient(s)
    factory = _MockClientFactory([(200, {"id": "r", "object": "response", "created_at": 1, "status": "completed", "model": "gpt-4.1", "output": []})])
    monkeypatch.setattr(model, "client", factory)

    def apply_catalog_recipe(params, _provider, _resolved, *, wire):
        assert wire == "responses"
        params.setdefault("extra_body", {})["catalog_extension"] = "kept"
        return params

    monkeypatch.setattr(model, "apply_request", apply_catalog_recipe)

    model.wire(s.config.provider).request([{"role": "user", "content": "hi"}], None)

    body = json.loads(factory.calls[0].content)
    assert body["user_extension"] == "kept"
    assert body["catalog_extension"] == "kept"


def test_configured_reasoning_survives_a_model_that_manages_none(tmp_path, monkeypatch):
    """Nothing to fold into on a non-reasoning model: the configured object passes through whole."""
    s = _session(
        tmp_path,
        api="responses",
        url="https://api.openai.com/v1",
        model="gpt-4.1",
        stream=False,
        extra_body={"reasoning": {"context": "all_turns"}},
    )
    model = ModelClient(s)
    factory = _MockClientFactory(
        [(200, {"id": "r", "object": "response", "created_at": 1, "status": "completed", "model": "gpt-4.1", "output": []})],
        base_url="https://api.openai.com/v1",
    )
    monkeypatch.setattr(model, "client", factory)

    model.wire(model.session.config.provider).request([{"role": "user", "content": "hi"}], None)

    assert json.loads(factory.calls[0].content)["reasoning"] == {"context": "all_turns"}


def test_provider_origin_separates_issuers_a_hostname_cannot(tmp_path):
    """Whoever can verify the ciphertext is decided by the whole endpoint and the credential, not
    by the domain: same-host entries on different ports, paths, or organization keys are distinct."""

    def origin(**kwargs) -> str:
        base = {"url": "http://localhost:8000/v1", "model": "gpt-oss", "key": "sk-a"}
        return ModelClient(_session(tmp_path, **(base | kwargs))).provider_origin()

    assert origin() != origin(url="http://localhost:9000/v1")
    assert origin() != origin(url="http://localhost:8000/upstream-b/v1")
    assert origin() != origin(key="sk-b")
    assert origin() != origin(model="gpt-oss-mini")
    assert origin(headers={"x-tenant": "team-a"}) != origin(headers={"x-tenant": "team-b"})
    assert origin() == origin()
    # A credential is fingerprinted, never carried into a session snapshot verbatim.
    assert "sk-a" not in origin()
