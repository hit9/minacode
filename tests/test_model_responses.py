"""Responses API requests: output items, streaming, replay, and reasoning parameters."""

import json
from types import SimpleNamespace

import pytest
from model_harness import _MockClientFactory, _session, _StreamClientFactory

from minacode.base import SESSION_EVENT_KEY, ModelError, ToolCall
from minacode.model import ModelClient
from minacode.tools import BashTool


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

    converted = ModelClient(s).responses_input([{"role": "user", "content": "<session_event />", SESSION_EVENT_KEY: "resumed"}])

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
    monkeypatch.setattr(model, "client", lambda: SimpleNamespace(responses=responses))
    model.on_stream = lambda kind, delta: timeline.append((kind, delta))

    _assistant, calls, content = model.request([{"role": "user", "content": "make the change"}], [])

    promoted = ("output_done", "I am editing the files.")
    assert timeline.index(promoted) < timeline.index(("wire", "tool arguments"))
    assert timeline.count(promoted) == 1
    assert calls == [ToolCall("call_1", "Bash", ["echo hi"])]
    assert content == "I am editing the files."


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
        model.responses_result({"status": "failed", "error": {"message": "bad request"}, "output": []})


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

    assistant, calls, content = model.responses_result(result)

    assert content == ""
    assert calls == [ToolCall("call_1", "Bash", ["echo hi"])]
    assert assistant["tool_calls"][0]["id"] == "call_1"
    converted = model.responses_input(
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
    converted = model.responses_input(
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
        ModelClient(s).responses_request([{"role": "user", "content": "hi"}], None)


def test_responses_replay_drops_reasoning_items_that_carry_no_payload(tmp_path):
    """Stateless reasoning travels in the encrypted payload; an id alone cannot stand in for it
    once the response was never stored, so an empty shell is dropped rather than replayed."""
    model = ModelClient(_session(tmp_path, api="responses"))
    assistant, _, _ = model.responses_result(
        {
            "output": [
                {"id": "rs_bare", "type": "reasoning", "summary": []},
                {"id": "rs_kept", "type": "reasoning", "encrypted_content": "opaque", "summary": []},
                {"id": "rs_text", "type": "reasoning", "summary": [{"type": "summary_text", "text": "thought"}]},
                {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "Bash", "arguments": "{}"},
            ]
        }
    )

    replayed = model.responses_input([assistant])

    assert [item["id"] for item in replayed] == ["rs_kept", "rs_text", "fc_1"]


def test_no_protocol_sends_another_protocols_saved_reply(tmp_path, monkeypatch):
    """`/provider` can switch protocols mid-session, so history holds assistant turns produced by
    a protocol other than the one now in use. Each protocol replays only its own saved reply and
    never puts minacode's bookkeeping keys on the wire."""
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "from responses", "_responses_output": [{"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque"}]},
        {"role": "assistant", "content": "from anthropic", "_anthropic_content": [{"type": "thinking", "thinking": "", "signature": "sig"}]},
    ]
    s = _session(tmp_path, model="claude-x")
    model = ModelClient(s)

    # Consecutive assistant turns merge into one message, as the Messages API requires roles to
    # alternate. The responses-only turn is rebuilt as text; only the Anthropic turn is echoed.
    anthropic_params = model.anthropic_params(history, None)
    assert anthropic_params["messages"][1]["content"] == [
        {"type": "text", "text": "from responses"},
        {"type": "thinking", "thinking": "", "signature": "sig"},
    ]

    responses_input = model.responses_input(history)
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
    model.chat_request(history, None)
    body = factory.calls[0].content.decode()
    assert "_responses_output" not in body
    assert "_anthropic_content" not in body
