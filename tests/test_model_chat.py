"""Chat Completions requests: streaming reassembly, tool-call deltas, and reasoning history."""

import json
import time
from types import SimpleNamespace

import pytest
from model_harness import _MockClientFactory, _session, _StreamClientFactory

from minacode.base import SESSION_EVENT_KEY, ModelError, ModelOutputTruncated, ToolCall
from minacode.model import ModelClient, resilience
from minacode.model.protocol import ChatWire


def _chat_completion(content, finish_reason, completion_tokens=16384):
    return (
        200,
        {
            "id": "chatcmpl-truncated",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": completion_tokens, "total_tokens": 10 + completion_tokens},
        },
    )


def test_chat_output_cap_reached_with_nothing_generated_names_the_cap(tmp_path, monkeypatch):
    """Reasoning spends the same budget as text, so a capped step can return nothing at all.

    Without this the turn dies as "empty final response", naming neither the cause nor the setting
    to change."""
    s = _session(tmp_path, stream=False)
    s.config.provider.max_tokens = 16_384  # a configured cap makes the reached-cap case verifiable
    model = ModelClient(s)
    factory = _MockClientFactory([_chat_completion("", "length")])
    monkeypatch.setattr(model, "client", factory)

    with pytest.raises(ModelOutputTruncated) as error:
        model.request([{"role": "user", "content": "hi"}], None)

    assert "provider.max_tokens" in str(error.value)
    assert "16384" in str(error.value)
    # Deterministic: the same request hits the same cap again, so it must not consume a retry.
    assert resilience.retryable_error(error.value) is False
    # The call reached the provider and was billed, so it belongs in usage regardless of the failure.
    assert s.usage.completion_tokens == 16384


def test_chat_length_with_output_below_the_cap_names_both_settings(tmp_path, monkeypatch):
    """Some OpenAI-compatible providers report `finish_reason=length` when the input exceeds the
    model's context window, not only when the output cap was hit. Only the cap case is provable from
    usage, so anything else must name both settings instead of pushing max_tokens blindly."""
    s = _session(tmp_path, stream=False)
    s.config.provider.max_tokens = 16_384
    model = ModelClient(s)
    monkeypatch.setattr(model, "client", _MockClientFactory([_chat_completion("", "length", completion_tokens=4_096)]))

    with pytest.raises(ModelError) as error:
        model.request([{"role": "user", "content": "hi"}], None)

    assert "provider.max_tokens" in str(error.value)
    assert "runtime.max_context_tokens" in str(error.value)
    assert not isinstance(error.value, ModelOutputTruncated)
    # Deterministic: the same request stops the same way, so it must not consume a retry.
    assert resilience.retryable_error(error.value) is False


def test_chat_length_without_a_configured_cap_names_both_settings(tmp_path, monkeypatch):
    """With max_tokens unset the provider's default cap is unknown, so `length` stays ambiguous
    even when the output is large."""
    s = _session(tmp_path, stream=False)
    model = ModelClient(s)
    monkeypatch.setattr(model, "client", _MockClientFactory([_chat_completion("", "length", completion_tokens=16_384)]))

    with pytest.raises(ModelError, match="context window"):
        model.request([{"role": "user", "content": "hi"}], None)

    assert s.usage.completion_tokens == 16_384


def test_chat_output_cap_reached_after_text_keeps_the_partial_answer(tmp_path, monkeypatch):
    """A visible partial answer is its own evidence of the cut; only an empty one needs explaining."""
    model = ModelClient(_session(tmp_path, stream=False))
    monkeypatch.setattr(model, "client", _MockClientFactory([_chat_completion("half a sen", "length")]))

    _, calls, content = model.request([{"role": "user", "content": "hi"}], None)

    assert content == "half a sen"
    assert calls == []


def test_chat_stream_reports_the_cap_when_the_stream_produced_nothing(tmp_path, monkeypatch):
    s = _session(tmp_path)
    s.config.provider.max_tokens = 16_384
    model = ModelClient(s)
    model.on_stream = lambda _kind, _delta: None
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}], "usage": {"prompt_tokens": 10, "completion_tokens": 16384}},
    ]
    monkeypatch.setattr(model, "client", _StreamClientFactory(chunks))

    with pytest.raises(ModelOutputTruncated, match="provider.max_tokens"):
        model.request([{"role": "user", "content": "hi"}], None)

    assert s.usage.completion_tokens == 16384


def test_chat_request_success(tmp_path, monkeypatch):
    s = _session(tmp_path, stream=False)
    model = ModelClient(s)
    streamed = []
    model.on_stream = lambda kind, delta: streamed.append((kind, delta))
    factory = _MockClientFactory(
        [
            (
                200,
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            )
        ]
    )
    monkeypatch.setattr(model, "client", factory)

    assistant, calls, content = model.request([{"role": "user", "content": "hi"}], None)

    assert content == "hello"
    assert assistant == {"role": "assistant", "content": "hello"}
    assert calls == []
    assert s.usage.prompt_tokens == 10
    assert s.usage.completion_tokens == 5
    assert s.usage.total_tokens == 15
    assert s.usage.calls == 1
    body = json.loads(factory.calls[0].content)
    assert factory.calls[0].url.path.endswith("/chat/completions")
    assert body["stream"] is False
    assert "stream_options" not in body
    assert streamed == []


def test_chat_request_strips_session_event_metadata(tmp_path, monkeypatch):
    s = _session(tmp_path, stream=False)
    model = ModelClient(s)
    factory = _MockClientFactory(
        [
            (
                200,
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {},
                },
            )
        ]
    )
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "<session_event />", SESSION_EVENT_KEY: "resumed"}])

    assert json.loads(factory.calls[0].content)["messages"] == [{"role": "user", "content": "<session_event />"}]


def test_chat_request_with_tool_calls(tmp_path, monkeypatch):
    s = _session(tmp_path)
    model = ModelClient(s)
    factory = _MockClientFactory(
        [
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
        ]
    )
    monkeypatch.setattr(model, "client", factory)

    assistant, calls, content = model.request([{"role": "user", "content": "run"}], [])

    assert content == ""
    assert assistant["role"] == "assistant"
    assert len(assistant["tool_calls"]) == 1
    assert assistant["tool_calls"][0]["function"]["name"] == "Bash"
    assert len(calls) == 1
    assert calls[0].name == "Bash"
    assert calls[0].args == ["echo hi"]


def test_chat_appends_declared_builtin_function_to_the_local_tools(tmp_path, monkeypatch):
    builtin = {"type": "builtin_function", "function": {"name": "$web_search"}}
    s = _session(
        tmp_path,
        url="https://api.moonshot.ai/v1",
        api="chat",
        model="kimi-k3",
        stream=False,
        builtin_tools=(builtin,),
    )
    model = ModelClient(s)
    factory = _MockClientFactory(
        [
            (
                200,
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "kimi-k3",
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
                                        "function": {"name": "$web_search", "arguments": '{"search_query":"minacode"}'},
                                    },
                                    {"id": "call_2", "type": "function", "function": {"name": "Bash", "arguments": '{"command":"ls"}'}},
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            )
        ]
    )
    monkeypatch.setattr(model, "client", factory)

    local = {"type": "function", "function": {"name": "Bash", "parameters": {}}}
    assistant, calls, content = model.request([{"role": "user", "content": "search"}], [local])

    assert content == ""
    assert calls == [ToolCall("call_1", "$web_search", [{"search_query": "minacode"}]), ToolCall("call_2", "Bash", ["ls"])]
    assert [call["function"]["name"] for call in assistant["tool_calls"]] == ["$web_search", "Bash"]
    # The builtin is appended to the local tools rather than replacing them: one stable tool block.
    assert json.loads(factory.calls[0].content)["tools"] == [local, builtin]


def test_chat_stream_reports_reasoning_text_and_complete_tool_calls(tmp_path, monkeypatch):
    s = _session(tmp_path)
    model = ModelClient(s)
    chunks = [
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "check"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "Bash", "arguments": '{"command":"echo'}}]},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"name": "Bash", "arguments": ' hi"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    ]
    factory = _StreamClientFactory(chunks)
    streamed = []
    model.on_stream = lambda kind, delta: streamed.append((kind, delta))
    monkeypatch.setattr(model, "client", factory)

    assistant, calls, content = model.request([{"role": "user", "content": "run"}], [])

    body = json.loads(factory.calls[0].content)
    assert factory.calls[0].url.path.endswith("/chat/completions")
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert streamed == [("reasoning", "check"), ("output", "hello"), ("output_done", "hello"), ("", "")]
    assert content == "hello"
    assert assistant["reasoning_content"] == "check"
    assert assistant["tool_calls"][0]["function"] == {"name": "Bash", "arguments": '{"command":"echo hi"}'}
    assert calls == [ToolCall("call_1", "Bash", ["echo hi"])]
    assert s.usage.total_tokens == 15


def test_chat_stream_waits_for_tool_finish_before_promoting_text(tmp_path):
    model = ModelClient(_session(tmp_path))
    timeline = []

    def chunks():
        yield {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]}
        yield {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "Bash", "arguments": '{"command":"echo'}}]},
                    "finish_reason": None,
                }
            ]
        }
        timeline.append(("wire", "remaining tool arguments"))
        yield {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ' hi"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        }

    completions = SimpleNamespace(create=lambda **_params: chunks())
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    model.on_stream = lambda kind, delta: timeline.append((kind, delta))

    message, _, _ = ChatWire(model)._stream(client, {})

    assert timeline == [
        ("output", "hello"),
        ("wire", "remaining tool arguments"),
        ("output_done", "hello"),
        ("", ""),
    ]
    assert message["tool_calls"][0]["function"]["arguments"] == '{"command":"echo hi"}'


def test_chat_stream_preserves_openrouter_reasoning_alias_and_details(tmp_path, monkeypatch):
    s = _session(tmp_path, url="https://openrouter.ai/api/v1", model="anthropic/claude-sonnet-4")
    model = ModelClient(s)
    chunks = [
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": s.config.provider.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning": "think ",
                        "reasoning_details": [
                            {"type": "reasoning.text", "text": "think ", "signature": None, "id": "r1", "format": "anthropic-claude-v1", "index": 0}
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": s.config.provider.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning": "more",
                        "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque", "id": "r2", "format": "anthropic-claude-v1", "index": 1}],
                        "content": "done",
                    },
                    "finish_reason": "stop",
                }
            ],
        },
    ]
    factory = _StreamClientFactory(chunks, base_url="https://openrouter.ai/api/v1")
    streamed = []
    model.on_stream = lambda kind, delta: streamed.append((kind, delta))
    monkeypatch.setattr(model, "client", factory)

    assistant, calls, content = model.request([{"role": "user", "content": "go"}], None)

    assert calls == []
    assert content == "done"
    assert assistant["reasoning"] == "think more"
    assert assistant["reasoning_details"] == [
        {"type": "reasoning.text", "text": "think ", "signature": None, "id": "r1", "format": "anthropic-claude-v1", "index": 0},
        {"type": "reasoning.encrypted", "data": "opaque", "id": "r2", "format": "anthropic-claude-v1", "index": 1},
    ]
    assert streamed == [("reasoning", "think "), ("reasoning", "more"), ("output", "done"), ("", "")]


def test_non_streaming_chat_preserves_all_reasoning_shapes(tmp_path):
    model = ModelClient(_session(tmp_path))
    details = [{"type": "reasoning.summary", "summary": "short", "id": "r", "format": "openai-responses-v1", "index": 0}]

    assert model.assistant_message({"content": "answer", "reasoning_content": "native", "reasoning": "alias", "reasoning_details": details}) == {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "native",
        "reasoning": "alias",
        "reasoning_details": details,
    }


@pytest.mark.parametrize(
    ("url", "model", "keeps_final"),
    [
        ("https://api.deepseek.com/v1", "deepseek-chat", False),
        ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3.8-max-preview", False),
        ("https://api.z.ai/api/paas/v4", "glm-5.1", False),
        ("https://api.moonshot.cn/v1", "kimi-k2.6", False),
        ("https://api.moonshot.cn/v1", "kimi-k3", True),
        ("https://api.moonshot.cn/v1", "kimi-k2.7-code", True),
        ("https://api.kimi.com/coding/v1", "k3", True),
        ("https://openrouter.ai/api/v1", "vendor/model", True),
        ("https://gateway.example/v1", "vendor/model", True),
    ],
)
def test_chat_reasoning_history_follows_provider_contract(tmp_path, url, model, keeps_final):
    client = ModelClient(_session(tmp_path, url=url, model=model))
    reasoning = {"reasoning_content": "native", "reasoning": "alias", "reasoning_details": [{"type": "reasoning.text", "text": "detail"}]}
    history = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "final", **reasoning},
        {"role": "user", "content": "next"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}],
            **reasoning,
        },
    ]

    converted = client.wire(client.session.config.provider).messages(history)

    assert ("reasoning_content" in converted[1]) is keeps_final
    assert ("reasoning" in converted[1]) is keeps_final
    assert ("reasoning_details" in converted[1]) is keeps_final
    assert converted[3]["reasoning_content"] == "native"
    assert converted[3]["reasoning"] == "alias"
    assert converted[3]["reasoning_details"] == reasoning["reasoning_details"]


def test_only_deepseek_keeps_completed_tool_reasoning_across_user_turns(tmp_path):
    reasoning_tool_call = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "reasoning",
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}],
    }
    history = [
        {"role": "user", "content": "first"},
        reasoning_tool_call,
        {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        {"role": "user", "content": "second"},
    ]

    deepseek = ModelClient(_session(tmp_path / "deepseek", url="https://api.deepseek.com/v1", model="deepseek-chat"))
    qwen = ModelClient(_session(tmp_path / "qwen", url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen3.8-max-preview"))

    assert deepseek.wire(deepseek.session.config.provider).messages(history)[1]["reasoning_content"] == "reasoning"
    assert "reasoning_content" not in qwen.wire(qwen.session.config.provider).messages(history)[1]


@pytest.mark.parametrize(
    "extra_body",
    [
        {"preserve_thinking": True},
        {"thinking": {"keep": "all"}},
        {"thinking": {"clear_thinking": False}},
    ],
)
def test_explicit_preserved_thinking_keeps_final_reasoning(tmp_path, extra_body):
    s = _session(tmp_path, url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen3.8-max-preview", extra_body=extra_body)
    message = {"role": "assistant", "content": "answer", "reasoning_content": "reasoning"}

    assert ModelClient(s).wire(s.config.provider).messages([message]) == [message]


def test_chat_stream_keeps_sequential_tool_calls_without_indexes_distinct(tmp_path, monkeypatch):
    s = _session(tmp_path)
    model = ModelClient(s)
    chunks = [
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "Read", "arguments": '{"path":"a"}'}}]},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "Read", "arguments": '{"path":"'}}]},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"function": {"arguments": 'b"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]
    factory = _StreamClientFactory(chunks)
    streamed = []
    model.on_stream = lambda kind, delta: streamed.append((kind, delta))
    monkeypatch.setattr(model, "client", factory)

    assistant, calls, _ = model.request([{"role": "user", "content": "read"}], [])

    assert [call["id"] for call in assistant["tool_calls"]] == ["call_1", "call_2"]
    assert [call.id for call in calls] == ["call_1", "call_2"]
    assert [call.name for call in calls] == ["Read", "Read"]
    assert [call["function"]["arguments"] for call in assistant["tool_calls"]] == ['{"path":"a"}', '{"path":"b"}']
    assert streamed == [("", "")]


def test_chat_stream_rejects_ambiguous_tool_fragments_without_indexes_or_ids(tmp_path, monkeypatch):
    s = _session(tmp_path)
    model = ModelClient(s)
    chunks = [
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {"name": "Read", "arguments": '{"path":"a"}'}},
                            {"id": "call_2", "type": "function", "function": {"name": "Bash", "arguments": '{"command":"echo'}},
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"function": {"arguments": ' hi"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]
    factory = _StreamClientFactory(chunks)
    model.on_stream = lambda _kind, _delta: None
    monkeypatch.setattr(model, "client", factory)

    with pytest.raises(ModelError, match="cannot associate it safely"):
        model.request([{"role": "user", "content": "run"}], [])


def test_chat_stream_clears_failed_attempt_before_retry(tmp_path, monkeypatch):
    s = _session(tmp_path)
    model = ModelClient(s)
    factory = _StreamClientFactory(
        [
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4",
                "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
            },
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4",
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ],
        failures=1,
    )
    streamed = []
    model.on_stream = lambda kind, delta: streamed.append((kind, delta))
    monkeypatch.setattr(model, "client", factory)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    _, _, content = model.request([{"role": "user", "content": "hi"}], [])

    assert content == "ok"
    assert len(factory.calls) == 2
    assert streamed == [("", ""), ("output", "ok"), ("", "")]
    assert s.state.model_retry_count == 1
    assert s.usage.total_tokens == 2


def test_chat_request_drops_responses_only_metadata(tmp_path, monkeypatch):
    s = _session(tmp_path)
    model = ModelClient(s)
    factory = _MockClientFactory(
        [
            (
                200,
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                },
            )
        ]
    )
    monkeypatch.setattr(model, "client", factory)

    model.request(
        [{"role": "assistant", "content": "old", "_responses_output": [{"type": "reasoning", "id": "rs_1", "summary": []}]}],
        None,
    )

    body = json.loads(factory.calls[0].content)
    assert body["messages"] == [{"role": "assistant", "content": "old"}]
