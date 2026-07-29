"""Chat Completions requests: streaming reassembly, tool-call deltas, and reasoning history."""

import json
import time
from types import SimpleNamespace

import pytest
from model_harness import _MockClientFactory, _session, _StreamClientFactory

from minacode.base import ModelError, ToolCall
from minacode.model import ModelClient


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

    assistant, calls, content = model.chat_request([{"role": "user", "content": "hi"}], None)

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

    assistant, calls, content = model.chat_request([{"role": "user", "content": "run"}], [])

    assert content == ""
    assert assistant["role"] == "assistant"
    assert len(assistant["tool_calls"]) == 1
    assert assistant["tool_calls"][0]["function"]["name"] == "Bash"
    assert len(calls) == 1
    assert calls[0].name == "Bash"
    assert calls[0].args == ["echo hi"]


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

    assistant, calls, content = model.chat_request([{"role": "user", "content": "run"}], [])

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

    message, _usage = model._chat_stream(client, {})

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

    assistant, calls, content = model.chat_request([{"role": "user", "content": "go"}], None)

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

    converted = client.chat_messages(history)

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

    assert deepseek.chat_messages(history)[1]["reasoning_content"] == "reasoning"
    assert "reasoning_content" not in qwen.chat_messages(history)[1]


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

    assert ModelClient(s).chat_messages([message]) == [message]


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

    assistant, calls, _content = model.chat_request([{"role": "user", "content": "read"}], [])

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
        model.chat_request([{"role": "user", "content": "run"}], [])


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

    _assistant, _calls, content = model.request([{"role": "user", "content": "hi"}], [])

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

    model.chat_request(
        [{"role": "assistant", "content": "old", "_responses_output": [{"type": "reasoning", "id": "rs_1", "summary": []}]}],
        None,
    )

    body = json.loads(factory.calls[0].content)
    assert body["messages"] == [{"role": "assistant", "content": "old"}]
