"""Mock-transport tests for ModelClient.

These tests intercept OpenAI/Anthropic SDK HTTP calls with httpx.MockTransport so we can
exercise chat_request, responses_request, anthropic_request, retry logic, and usage accounting without
hitting real providers.
"""

import json

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
    config.providers = {"default": n.ProviderConfig(url="http://test", key="sk-test", **provider_kwargs)}
    return n.Session(cwd=str(tmp_path), config=config)


def test_chat_request_success(tmp_path, monkeypatch):
    s = _session(tmp_path)
    model = n.ModelClient(s)
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


def test_chat_request_with_tool_calls(tmp_path, monkeypatch):
    s = _session(tmp_path)
    model = n.ModelClient(s)
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


def test_responses_request_preserves_output_items_and_uses_responses_shape(tmp_path, monkeypatch):
    s = _session(tmp_path, api="responses", model="gpt-5")
    model = n.ModelClient(s)
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
    assert request.url.path == "/responses"
    body = json.loads(request.content)
    assert body["input"] == [{"role": "user", "content": "hi"}]
    assert body["store"] is False
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


def test_responses_tool_items_are_converted_and_replayed(tmp_path):
    model = n.ModelClient(_session(tmp_path, api="responses"))
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
    assert calls == [n.ToolCall("call_1", "Bash", ["echo hi"])]
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
    model = n.ModelClient(s)
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
    tools = [n.BashTool.schema(False)]

    assistant, calls, content = model.request([{"role": "user", "content": "run it"}], tools)
    assert content == ""
    assert calls == [n.ToolCall("call_1", "Bash", ["echo hi"])]

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
    model = n.ModelClient(s)
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
    model = n.ModelClient(s)
    empty = {"id": "r", "object": "response", "created_at": 1, "status": "completed", "model": "gpt-5.5", "output": []}
    factory = _MockClientFactory([(200, empty)])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], None)

    assert json.loads(factory.calls[0].content)["reasoning"] == {"effort": "none"}


@pytest.mark.parametrize("model", ("custom-model", "gpt-5", "gpt-5-mini"))
def test_responses_reports_unsupported_reasoning_off_instead_of_guessing(tmp_path, model):
    s = _session(tmp_path, api="responses", model=model)
    s.config.provider.reasoning = "off"
    if model.startswith("gpt-5"):
        s.config.provider.url = "https://api.openai.com/v1"

    with pytest.raises(n.ModelError, match="reasoning off is not defined"):
        n.ModelClient(s).responses_request([{"role": "user", "content": "hi"}], None)


def test_responses_replay_drops_reasoning_items_that_carry_no_payload(tmp_path):
    """Stateless reasoning travels in the encrypted payload; an id alone cannot stand in for it
    once the response was never stored, so an empty shell is dropped rather than replayed."""
    model = n.ModelClient(_session(tmp_path, api="responses"))
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
    model = n.ModelClient(s)

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
    model = n.ModelClient(s)
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
    s = _session(tmp_path, model="claude-3", api="anthropic")
    model = n.ModelClient(s)
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


def test_request_retries_then_succeeds(tmp_path, monkeypatch):
    s = _session(tmp_path)
    model = n.ModelClient(s)
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
    monkeypatch.setattr(n.time, "sleep", lambda _seconds: None)

    assistant, calls, content = model.request([{"role": "user", "content": "hi"}], None)

    assert content == "ok"
    assert len(factory.calls) == 2
    assert s.usage.calls == 1


def test_request_retry_exhausted(tmp_path, monkeypatch):
    s = _session(tmp_path)
    model = n.ModelClient(s)
    factory = _MockClientFactory([(500, {"error": {"message": "server error", "type": "internal_server_error"}})] * 6)
    monkeypatch.setattr(model, "client", factory)
    monkeypatch.setattr(n.time, "sleep", lambda _seconds: None)

    with pytest.raises(n.ModelError, match="after 6 attempts"):
        model.request([{"role": "user", "content": "hi"}], None)

    assert len(factory.calls) == 6
    assert s.usage.calls == 0
