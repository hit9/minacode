"""builtin tools requests (split from tests/test_builtin_tools.py)."""
import json

import pytest
from model_harness import _AnthropicMockClientFactory, _MockClientFactory, _session
from test_builtin_tools import FUNCTION_TOOL, WEB_SEARCH, _responses_body

from minacode.base import (
    SEARCH_SOURCES_KEY,
    ConfigError,
)
from minacode.config import (
    ProviderConfig,
)
from minacode.model import ModelClient


def test_builtin_tools_parse_as_tables_with_a_type():
    provider = ProviderConfig.from_dict({"builtin_tools": [{"type": "web_search", "search_context_size": "high"}]})

    assert provider.builtin_tools == ({"type": "web_search", "search_context_size": "high"},)

def test_builtin_tools_default_to_empty():
    assert ProviderConfig.from_dict({}).builtin_tools == ()

@pytest.mark.parametrize(
    "value",
    [
        {"type": "web_search"},  # a bare table, not a list of them
        [{"name": "web_search"}],  # every documented builtin tool carries a type
        [{"type": ""}],
        ["web_search"],
    ],
)
def test_builtin_tools_reject_shapes_no_provider_accepts(value):
    with pytest.raises(ConfigError):
        ProviderConfig.from_dict({"builtin_tools": value})

def test_builtin_tools_are_not_shared_with_the_loaded_config(tmp_path):
    """A request must not be able to mutate config that outlives it."""
    s = _session(tmp_path, api="responses", model="gpt-5", builtin_tools=(dict(WEB_SEARCH),))

    ModelClient(s).builtin_tools()[0]["type"] = "mutated"

    assert s.config.provider.builtin_tools == ({"type": "web_search"},)

def test_responses_request_appends_builtin_tools_after_function_schemas(tmp_path, monkeypatch):
    s = _session(tmp_path, url="https://api.openai.com/v1", api="responses", model="gpt-5", stream=False, builtin_tools=(WEB_SEARCH,))
    model = ModelClient(s)
    factory = _MockClientFactory([(200, _responses_body())])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

    body = json.loads(factory.calls[0].content)
    assert [tool.get("name") or tool["type"] for tool in body["tools"]] == ["Bash", "web_search"]
    assert body["tools"][1] == {"type": "web_search"}

def test_chat_request_appends_builtin_tools(tmp_path, monkeypatch):
    """Z.AI and Kimi express builtin tools in the Chat tools array, not the request body."""
    zai_search = {"type": "web_search", "web_search": {"enable": "True"}}
    s = _session(tmp_path, model="glm-5", stream=False, builtin_tools=(zai_search,))
    model = ModelClient(s)
    factory = _MockClientFactory(
        [
            (
                200,
                {
                    "id": "c",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "glm-5",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                },
            )
        ]
    )
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

    body = json.loads(factory.calls[0].content)
    assert body["tools"] == [FUNCTION_TOOL, zai_search]

def test_anthropic_request_appends_builtin_tools(tmp_path, monkeypatch):
    search = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
    s = _session(tmp_path, model="claude-3", api="anthropic", stream=False, builtin_tools=(search,))
    model = ModelClient(s)
    factory = _AnthropicMockClientFactory(
        [
            (
                200,
                {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3",
                    "content": [{"type": "text", "text": "hi"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        ]
    )
    monkeypatch.setattr(model, "anthropic_client", factory)

    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

    body = json.loads(factory.calls[0].content)
    assert [tool["name"] for tool in body["tools"]] == ["Bash", "web_search"]
    assert body["tools"][1] == search

def test_builtin_tools_are_sent_without_any_function_tools(tmp_path, monkeypatch):
    """Compaction and live follow-ups request with no function tools; search must still be offered."""
    s = _session(tmp_path, api="responses", model="gpt-5", stream=False, builtin_tools=(WEB_SEARCH,))
    model = ModelClient(s)
    factory = _MockClientFactory([(200, _responses_body())])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], [])

    assert json.loads(factory.calls[0].content)["tools"] == [{"type": "web_search"}]

def test_builtin_tools_change_the_prompt_cache_key(tmp_path):
    """Enabling search changes the provider-rendered tool prefix, so the cached prefix differs."""
    plain = _session(tmp_path, api="responses", model="gpt-5")
    searching = _session(tmp_path, api="responses", model="gpt-5", builtin_tools=(WEB_SEARCH,))

    plain_key = ModelClient(plain).prompt_cache_key(plain.config.provider, [FUNCTION_TOOL])
    searching_key = ModelClient(searching).prompt_cache_key(searching.config.provider, [FUNCTION_TOOL])

    assert plain_key and searching_key and plain_key != searching_key

def test_inactive_builtin_tools_do_not_change_the_prompt_cache_key(tmp_path):
    """The cache key describes the projected request, not inactive configuration."""
    plain = _session(tmp_path, url="https://dashscope.aliyuncs.com/compatible-mode/v1", api="chat", model="glm-5.2")
    searching = _session(
        tmp_path,
        url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api="chat",
        model="glm-5.2",
        builtin_tools=(WEB_SEARCH,),
    )

    plain_key = ModelClient(plain).prompt_cache_key(plain.config.provider, [FUNCTION_TOOL])
    searching_key = ModelClient(searching).prompt_cache_key(searching.config.provider, [FUNCTION_TOOL])

    assert plain_key == searching_key

def test_responses_result_collects_openai_citations_and_qwen_sources(tmp_path, monkeypatch):
    """OpenAI cites inline; Qwen reports sources only on the search call. Both must be read."""
    s = _session(tmp_path, api="responses", model="gpt-5", stream=False, builtin_tools=(WEB_SEARCH,))
    model = ModelClient(s)
    output = [
        {
            "id": "ws_1",
            "type": "web_search_call",
            "status": "completed",
            "action": {"type": "search", "query": "httpx timeout", "sources": [{"url": "https://qwen.example/a", "title": "A"}]},
        },
        {
            "id": "msg_1",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "sunny",
                    "annotations": [{"type": "url_citation", "url": "https://openai.example/b", "title": "B"}],
                }
            ],
        },
    ]
    factory = _MockClientFactory([(200, _responses_body(output=output))])
    monkeypatch.setattr(model, "client", factory)

    assistant, _, content = model.request([{"role": "user", "content": "hi"}], [])

    assert content == "sunny"
    assert assistant[SEARCH_SOURCES_KEY] == [
        {"url": "https://qwen.example/a", "title": "A"},
        {"url": "https://openai.example/b", "title": "B"},
    ]

def test_anthropic_result_collects_cited_and_raw_search_results(tmp_path, monkeypatch):
    s = _session(tmp_path, model="claude-3", api="anthropic", stream=False)
    model = ModelClient(s)
    content_blocks = [
        {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "shannon"}},
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srv_1",
            "content": [{"type": "web_search_result", "url": "https://wiki.example/s", "title": "Shannon", "encrypted_content": "x"}],
        },
        {
            "type": "text",
            "text": "born 1916",
            "citations": [{"type": "web_search_result_location", "url": "https://wiki.example/s", "title": "Shannon", "cited_text": "…"}],
        },
    ]
    factory = _AnthropicMockClientFactory(
        [
            (
                200,
                {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3",
                    "content": content_blocks,
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        ]
    )
    monkeypatch.setattr(model, "anthropic_client", factory)

    assistant, _, _ = model.request([{"role": "user", "content": "hi"}], [])

    # The same URL is both a raw result and a citation; it is reported once.
    assert assistant[SEARCH_SOURCES_KEY] == [{"url": "https://wiki.example/s", "title": "Shannon"}]

def test_anthropic_search_error_reports_no_sources(tmp_path, monkeypatch):
    """A failed search returns an error object where results normally are, and cites nothing."""
    s = _session(tmp_path, model="claude-3", api="anthropic", stream=False)
    model = ModelClient(s)
    blocks = [
        {"type": "web_search_tool_result", "tool_use_id": "srv_1", "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}},
        {"type": "text", "text": "could not search"},
    ]
    factory = _AnthropicMockClientFactory(
        [
            (
                200,
                {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3",
                    "content": blocks,
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        ]
    )
    monkeypatch.setattr(model, "anthropic_client", factory)

    assistant, _, _ = model.request([{"role": "user", "content": "hi"}], [])

    assert SEARCH_SOURCES_KEY not in assistant

def test_chat_result_collects_message_annotations(tmp_path, monkeypatch):
    """OpenRouter's web-search server tool cites through message annotations."""
    s = _session(
        tmp_path,
        url="https://openrouter.ai/api/v1",
        model="openai/gpt-5",
        stream=False,
        builtin_tools=({"type": "openrouter:web_search"},),
    )
    model = ModelClient(s)
    message = {
        "role": "assistant",
        "content": "hi",
        "annotations": [{"type": "url_citation", "url_citation": {"url": "https://router.example/c", "title": "C"}}],
    }
    factory = _MockClientFactory(
        [
            (
                200,
                {
                    "id": "c",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "openai/gpt-5",
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                },
            )
        ]
    )
    monkeypatch.setattr(model, "client", factory)

    assistant, _, _ = model.request([{"role": "user", "content": "hi"}], [])

    assert assistant[SEARCH_SOURCES_KEY] == [{"url": "https://router.example/c", "title": "C"}]

def test_stored_sources_never_replay_to_the_provider(tmp_path, monkeypatch):
    """Sources are presentation state: they persist, but no protocol sends them back."""
    s = _session(tmp_path, model="gpt-4", stream=False)
    model = ModelClient(s)
    history = [{"role": "assistant", "content": "hi", SEARCH_SOURCES_KEY: [{"url": "https://example.com", "title": "T"}]}]
    factory = _MockClientFactory(
        [
            (
                200,
                {
                    "id": "c",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                },
            )
        ]
    )
    monkeypatch.setattr(model, "client", factory)

    model.request(history, [])

    sent = json.loads(factory.calls[0].content)["messages"]
    assert sent == [{"role": "assistant", "content": "hi"}]
    assert ModelClient(s).wire(ProviderConfig(api="responses", model="gpt-5")).messages(history) == [{"role": "assistant", "content": "hi"}]
