"""Mock-transport tests for ModelClient.

These tests intercept OpenAI/Anthropic SDK HTTP calls with httpx.MockTransport so we can
exercise chat_request, responses_request, anthropic_request, retry logic, and usage accounting without
hitting real providers.
"""

import json
import time

import httpx
import pytest
from anthropic import Anthropic
from openai import OpenAI

import minacode.engine as engine_module
from minacode.base import Config, ModelError, ModelResponseTimeout, ProviderConfig, ToolCall
from minacode.engine import ModelClient
from minacode.session import Session
from minacode.tools import BashTool


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


class _StreamClientFactory:
    def __init__(self, events: list[dict], base_url: str = "http://test", failures: int = 0):
        self.events = events
        self.calls: list[httpx.Request] = []
        self.base_url = base_url
        self.failures = failures

    def __call__(self) -> OpenAI:
        def respond(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            if self.failures:
                self.failures -= 1
                return httpx.Response(500, json={"error": {"message": "temporary failure", "type": "server_error"}})
            body = "".join(f"data: {json.dumps(event)}\n\n" for event in self.events) + "data: [DONE]\n\n"
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        http_client = httpx.Client(transport=httpx.MockTransport(respond))
        return OpenAI(api_key="sk-test", base_url=self.base_url, http_client=http_client, max_retries=0)


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


def _session(tmp_path, **provider_kwargs):
    config = Config()
    config.data_dir = str(tmp_path / "data")
    provider_kwargs.setdefault("model", "gpt-4")
    provider_kwargs.setdefault("url", "http://test")
    provider_kwargs.setdefault("key", "sk-test")
    config.providers = {"default": ProviderConfig(**provider_kwargs)}
    return Session(cwd=str(tmp_path), config=config)


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
    assert streamed == [("reasoning", "check"), ("output", "hello"), ("", "")]
    assert content == "hello"
    assert assistant["reasoning_content"] == "check"
    assert assistant["tool_calls"][0]["function"] == {"name": "Bash", "arguments": '{"command":"echo hi"}'}
    assert calls == [ToolCall("call_1", "Bash", ["echo hi"])]
    assert s.usage.total_tokens == 15


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
    model.on_stream = lambda _kind, _delta: None
    monkeypatch.setattr(model, "client", factory)

    assistant, calls, _content = model.chat_request([{"role": "user", "content": "read"}], [])

    assert [call["id"] for call in assistant["tool_calls"]] == ["call_1", "call_2"]
    assert [call.id for call in calls] == ["call_1", "call_2"]
    assert [call.name for call in calls] == ["Read", "Read"]
    assert [call["function"]["arguments"] for call in assistant["tool_calls"]] == ['{"path":"a"}', '{"path":"b"}']


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
    assert streamed == [("reasoning", "check"), ("output", "hello"), ("", "")]
    assert content == "hello"
    assert calls == [ToolCall("tool_1", "Bash", ["echo hi"])]
    assert assistant["_anthropic_content"] == [
        {"type": "thinking", "thinking": "check", "signature": "sig"},
        {"type": "text", "text": "hello"},
        {"type": "tool_use", "id": "tool_1", "name": "Bash", "input": {"command": "echo hi"}},
    ]
    assert s.usage.prompt_tokens == 10
    assert s.usage.completion_tokens == 5


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
    monkeypatch.setattr(engine_module.threading, "Timer", ImmediateTimer)

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

    monkeypatch.setattr(engine_module.threading, "Timer", unexpected_timer)

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
    monkeypatch.setattr(engine_module.threading, "Timer", ImmediateTimer)

    def interrupted_request():
        raise RuntimeError("connection closed")

    with pytest.raises(ModelResponseTimeout, match=r"provider\.response_timeout=600s") as caught:
        model.call_client(Client(), interrupted_request)

    assert isinstance(caught.value.__cause__, RuntimeError)
