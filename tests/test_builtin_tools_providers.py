"""builtin tools providers (split from tests/test_builtin_tools.py)."""
import json

import pytest
from model_harness import _AnthropicMockClientFactory, _MockClientFactory, _session
from test_builtin_tools import FUNCTION_TOOL, WEB_SEARCH, _chat_body, _responses_body

from minacode.base import (
    ModelError,
)
from minacode.context import ContextManager
from minacode.model import ModelClient, responses


def test_aliyun_chat_keeps_responses_builtin_tools_inactive(tmp_path, monkeypatch):
    """A model on the same gateway may use Chat while configured Responses tools stay dormant."""
    s = _session(
        tmp_path,
        url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        model="glm-5.2",
        api="chat",
        stream=False,
        builtin_tools=({"type": "web_search"}, {"type": "web_extractor"}),
    )
    model = ModelClient(s)
    factory = _MockClientFactory([(200, _chat_body("glm-5.2"))])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

    assert json.loads(factory.calls[0].content)["tools"] == [FUNCTION_TOOL]
    assert s.config.provider.builtin_tools == ({"type": "web_search"}, {"type": "web_extractor"})

def test_qwen_responses_keeps_builtin_tools_unchanged(tmp_path, monkeypatch):
    """The same configuration stays valid on the Responses wire, untouched by any Chat wrapper."""
    s = _session(
        tmp_path,
        url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen3.8-max-preview",
        api="responses",
        stream=False,
        builtin_tools=({"type": "web_search"}, {"type": "web_extractor"}),
    )
    model = ModelClient(s)
    factory = _MockClientFactory([(200, _responses_body())])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

    body = json.loads(factory.calls[0].content)
    tools = body["tools"]
    # Local function tools use Responses' flat function schema; provider entries follow unchanged.
    assert [tool["type"] for tool in tools] == ["function", "web_search", "web_extractor"]
    assert tools[1:] == [{"type": "web_search"}, {"type": "web_extractor"}]

def test_unknown_provider_keeps_generic_builtin_tools_pass_through(tmp_path, monkeypatch):
    """Unmatched hosts keep the pass-through path for private and future providers."""
    entry = {"type": "web_search", "custom_field": "kept"}
    s = _session(tmp_path, model="made-up-model", api="chat", stream=False, builtin_tools=(entry,))
    model = ModelClient(s)
    factory = _MockClientFactory([(200, _chat_body("made-up-model"))])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

    assert json.loads(factory.calls[0].content)["tools"] == [FUNCTION_TOOL, entry]

@pytest.mark.parametrize("url", ["https://api.z.ai/api/paas/v4", "https://open.bigmodel.cn/api/paas/v4"])
def test_zai_and_bigmodel_chat_accept_their_documented_builtin_tool(tmp_path, monkeypatch, url):
    """Both GLM hosts place provider-native web_search inside the Chat tools array."""
    zai_search = {"type": "web_search", "web_search": {"enable": "True"}}
    s = _session(tmp_path, url=url, model="glm-5", api="chat", stream=False, builtin_tools=(zai_search,))
    model = ModelClient(s)
    factory = _MockClientFactory([(200, _chat_body("glm-5"))])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

    assert json.loads(factory.calls[0].content)["tools"] == [FUNCTION_TOOL, zai_search]

def test_kimi_chat_accepts_builtin_function_unchanged(tmp_path, monkeypatch):
    """Kimi declares $web_search as a Chat builtin_function that keeps its handshake."""
    kimi_search = {"type": "builtin_function", "function": {"name": "$web_search"}}
    s = _session(tmp_path, url="https://api.moonshot.ai/v1", model="kimi-k3", api="chat", stream=False, builtin_tools=(kimi_search,))
    model = ModelClient(s)
    factory = _MockClientFactory([(200, _chat_body("kimi-k3"))])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

    assert json.loads(factory.calls[0].content)["tools"] == [FUNCTION_TOOL, kimi_search]

def test_anthropic_builtin_tools_are_protocol_scoped(tmp_path, monkeypatch):
    """Anthropic server tools travel over Messages and stay inactive on Chat."""
    search = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
    s = _session(tmp_path, url="https://api.anthropic.com/v1", model="claude-3", api="anthropic", stream=False, builtin_tools=(search,))
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
    assert [tool["name"] for tool in json.loads(factory.calls[0].content)["tools"]] == ["Bash", "web_search"]

    # The same known-provider configuration forced onto Chat is retained but not projected.
    s.config.provider.api = "chat"
    chat_factory = _MockClientFactory([(200, _chat_body("claude-3"))])
    monkeypatch.setattr(model, "client", chat_factory)
    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])
    assert json.loads(chat_factory.calls[0].content)["tools"] == [FUNCTION_TOOL]
    assert s.config.provider.builtin_tools == (search,)

@pytest.mark.parametrize("api", ["chat", "responses"])
def test_openrouter_sends_supported_server_tools_unchanged_on_both_wires(tmp_path, monkeypatch, api):
    """OpenRouter documents these server tools on both Chat and Responses."""
    server_tools = (
        {"type": "openrouter:web_search", "parameters": {"max_results": 5}},
        {"type": "openrouter:web_fetch"},
        {"type": "openrouter:datetime", "parameters": {"timezone": "Asia/Shanghai"}},
    )
    s = _session(tmp_path, url="https://openrouter.ai/api/v1", model="vendor/model", api=api, stream=False, builtin_tools=server_tools)
    model = ModelClient(s)
    response = _chat_body("vendor/model") if api == "chat" else _responses_body()
    factory = _MockClientFactory([(200, response)])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

    body = json.loads(factory.calls[0].content)
    expected_local = FUNCTION_TOOL if api == "chat" else responses.responses_tool_schemas([FUNCTION_TOOL])[0]
    assert body["tools"] == [expected_local, *server_tools]

@pytest.mark.parametrize(
    ("url", "model", "api", "entry", "supported"),
    [
        ("https://api.moonshot.ai/v1", "kimi-k3", "chat", {"type": "builtin_function"}, "builtin_function/$web_search"),
        (
            "https://api.moonshot.ai/v1",
            "kimi-k3",
            "chat",
            {"type": "builtin_function", "function": {"name": "$other"}},
            "builtin_function/$web_search",
        ),
        ("https://api.z.ai/api/paas/v4", "glm-5", "chat", {"type": "web_search"}, "web_search object"),
        ("https://api.anthropic.com/v1", "claude-3", "anthropic", {"type": "web_search_20250305"}, "name=web_search"),
    ],
)
def test_known_provider_rejects_incomplete_or_different_supported_type_shapes(tmp_path, url, model, api, entry, supported):
    """A matching type alone must not claim support for a different provider lifecycle."""
    s = _session(tmp_path, url=url, model=model, api=api, builtin_tools=(entry,))

    with pytest.raises(ModelError) as excinfo:
        ModelClient(s).builtin_tools()

    message = str(excinfo.value)
    assert "not supported" in message
    assert supported in message

def test_known_provider_rejects_unsupported_builtin_tool_types(tmp_path):
    """Unsupported provider-side tools fail locally instead of leaking lifecycle gaps."""
    cases = [
        ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3.8-max-preview", "responses", {"type": "code_interpreter"}, "web_search, web_extractor"),
        ("https://api.openai.com/v1", "gpt-5", "responses", {"type": "file_search"}, "web_search"),
        ("https://api.anthropic.com/v1", "claude-3", "anthropic", {"type": "web_fetch_20250628"}, "web_search_20250305"),
        ("https://api.z.ai/api/paas/v4", "glm-5", "chat", {"type": "retrieval"}, "web_search"),
        ("https://api.moonshot.ai/v1", "kimi-k3", "chat", {"type": "web_search"}, "builtin_function"),
    ]
    for url, model, api, entry, supported in cases:
        s = _session(tmp_path, url=url, model=model, api=api, builtin_tools=(entry,))
        with pytest.raises(ModelError) as excinfo:
            ModelClient(s).builtin_tools()
        message = str(excinfo.value)
        assert entry["type"] in message
        assert "not supported" in message
        assert supported in message

def test_known_providers_without_server_tools_keep_builtin_tools_inactive(tmp_path):
    """A shared config remains usable when the selected provider has no builtin-tool wire."""
    cases = [
        ("https://api.deepseek.com/v1", "deepseek-chat", "chat"),
        ("https://api.kimi.com/coding/v1", "k3", "chat"),
        ("https://opencode.ai/zen/v1", "gpt-5.5", "responses"),
    ]
    for url, model, api in cases:
        s = _session(tmp_path, url=url, model=model, api=api, builtin_tools=(WEB_SEARCH,))
        assert ModelClient(s).builtin_tools() == []
        assert s.config.provider.builtin_tools == (WEB_SEARCH,)

def test_an_unsupported_entry_fails_the_request_without_breaking_read_only_paths(tmp_path):
    """Refusing an entry belongs to the request that would send it, not to measuring the payload.

    `/status`, the status bar, and session resume all estimate the request, and raising there took
    down the whole frontend over config a request has not tried to use yet."""
    s = _session(tmp_path, url="https://api.openai.com/v1", model="gpt-5", api="responses", builtin_tools=({"type": "file_search"},))
    model = ModelClient(s)

    assert model.estimated_request_tokens([{"role": "user", "content": "hi"}], [FUNCTION_TOOL]) > 0
    assert ContextManager(s, model).update_current_tokens("system") > 0
    with pytest.raises(ModelError):
        model.builtin_tools()

def test_estimation_and_send_share_the_builtin_tools_policy(tmp_path, monkeypatch):
    """The estimator and sender project the same active subset on each wire."""
    s = _session(
        tmp_path, url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen3.8-max-preview", api="chat", stream=False, builtin_tools=(WEB_SEARCH,)
    )
    model = ModelClient(s)

    # Chat ignores the configured Responses entry in both paths.
    inactive = model.estimated_request_tokens([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])
    s.config.provider.builtin_tools = ()
    without_builtin = model.estimated_request_tokens([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])
    assert inactive == without_builtin
    s.config.provider.builtin_tools = (WEB_SEARCH,)
    chat_factory = _MockClientFactory([(200, _chat_body())])
    monkeypatch.setattr(model, "client", chat_factory)
    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])
    assert json.loads(chat_factory.calls[0].content)["tools"] == [FUNCTION_TOOL]

    # On the valid wire, the estimator consumes the same builtin entry the request sends.
    s.config.provider.api = "responses"
    factory = _MockClientFactory([(200, _responses_body())])
    monkeypatch.setattr(model, "client", factory)
    with_builtin = model.estimated_request_tokens([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])
    s.config.provider.builtin_tools = ()
    without_builtin = model.estimated_request_tokens([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])
    assert with_builtin > without_builtin
    s.config.provider.builtin_tools = (WEB_SEARCH,)
    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])
    assert json.loads(factory.calls[0].content)["tools"][-1] == WEB_SEARCH
