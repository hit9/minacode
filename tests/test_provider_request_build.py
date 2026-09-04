"""provider request build (split from tests/test_core_logic.py)."""

import json
import time
from types import SimpleNamespace

import pytest
from catalog_harness import resolve
from model_harness import record_backoff
from test_core_logic import session

from wizolt.base import (
    ModelError,
    ModelUsage,
    ToolCall,
    drop_nulls,
)
from wizolt.config import (
    ProviderConfig,
)
from wizolt.context import ContextManager
from wizolt.model import ModelClient, resilience
from wizolt.tools import TOOL_REGISTRY, Tool


def test_explicit_manual_thinking_maps_max_to_the_largest_budget(tmp_path):
    provider = ProviderConfig(url="https://gateway.example/v1", model="custom-model", chat_reasoning="enable_thinking", reasoning="max")
    params = {}

    ModelClient(session(tmp_path)).apply_provider_params(params, provider)

    assert params == {"extra_body": {"enable_thinking": True, "thinking_budget": 32768}}


def test_explicit_manual_thinking_budget_stays_under_the_configured_output_cap(tmp_path):
    """These hosts fold max_tokens into max_completion_tokens and reject a budget that reaches it."""
    client = ModelClient(session(tmp_path))

    for max_tokens, reasoning, expected in ((16_384, "xhigh", 15_360), (16_384, "max", 15_360), (2_048, "high", 1_024), (0, "max", 32_768)):
        provider = ProviderConfig(
            url="https://gateway.example/v1", model="custom-model", chat_reasoning="enable_thinking", reasoning=reasoning, max_tokens=max_tokens
        )
        params: dict = {}

        client.apply_provider_params(params, provider)

        assert params["extra_body"]["thinking_budget"] == expected


def test_chat_provider_extra_body_passthrough(tmp_path):
    client = ModelClient(session(tmp_path))

    # Vendor extensions (e.g. Qianwen web search) pass through verbatim into extra_body.
    params = {}
    search = {"enable_search": True, "search_options": {"forced_search": True, "search_strategy": "max"}}
    provider = ProviderConfig(url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen3-max", reasoning="off", extra_body=search)
    client.apply_provider_params(params, provider)
    assert params["extra_body"] == search

    # Configured extra_body merges with wizolt-managed reasoning fields...
    params = {}
    client.apply_provider_params(params, ProviderConfig(url="https://openrouter.ai/api/v1", model="x", reasoning="high", extra_body={"enable_search": True}))
    assert params["extra_body"] == {"enable_search": True, "reasoning": {"effort": "high"}}

    # ...and reasoning wins on key conflict so wizolt stays in control of its own fields.
    params = {}
    client.apply_provider_params(
        params, ProviderConfig(url="https://openrouter.ai/api/v1", model="x", reasoning="high", extra_body={"reasoning": {"effort": "low"}})
    )
    assert params["extra_body"] == {"reasoning": {"effort": "high"}}

    # Managed thinking.type remains authoritative without discarding documented history options.
    params = {}
    client.apply_provider_params(
        params,
        ProviderConfig(
            url="https://api.z.ai/api/paas/v4",
            model="glm-5.1",
            reasoning="high",
            extra_body={"thinking": {"clear_thinking": False}},
        ),
    )
    assert params["extra_body"] == {"thinking": {"clear_thinking": False, "type": "enabled"}}

    # extra_body round-trips through config; non-object values are ignored.
    assert ProviderConfig.from_dict({"extra_body": search}).extra_body == search
    assert ProviderConfig.from_dict({"extra_body": "nope"}).extra_body == {}
    assert ProviderConfig().extra_body == {}


def _strict_check(node, path="root"):
    if isinstance(node, dict):
        for key in ("minItems", "maxItems", "minLength", "maxLength"):
            assert key not in node, f"{path}: leftover {key}"
        kind = node.get("type")
        if isinstance(kind, list):
            # DeepSeek strict rejects object/array inside a type union; only scalars + null allowed.
            assert all(item in ("string", "number", "integer", "boolean", "null") for item in kind), f"{path}: non-scalar in type union {kind}"
        if isinstance(node.get("properties"), dict):
            assert node.get("additionalProperties") is False, f"{path}: additionalProperties"
            assert set(node["required"]) == set(node["properties"]), f"{path}: required != properties"
            for key, sub in node["properties"].items():
                _strict_check(sub, f"{path}.{key}")
        if "items" in node:
            _strict_check(node["items"], f"{path}[]")
        for combiner in ("anyOf", "oneOf", "allOf"):
            for index, sub in enumerate(node.get(combiner, [])):
                _strict_check(sub, f"{path}.{combiner}[{index}]")
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _strict_check(item, f"{path}[{index}]")


def test_strict_tools_off_path_emits_non_strict_schema():
    for tool in TOOL_REGISTRY.values():
        legacy = {
            "type": "function",
            "function": {
                "name": tool.NAME,
                "description": "\n".join([tool.DESCRIPTION, *(("- " + item) for item in tool.EXAMPLE if item)]),
                "parameters": tool.params_schema(),
            },
        }
        assert tool.schema(False) == legacy
        assert "strict" not in tool.schema(False)["function"]


def test_strict_tools_gating_and_beta_routing():
    def resolved(url, strict=False):
        return resolve(ProviderConfig(url=url, strict_tools=strict))

    # Unsupported hosts never activate strict, even when requested, and stay on their endpoint.
    for url in ("https://openrouter.ai/api/v1", "https://api.together.xyz/v1", "http://localhost:1234/v1"):
        assert resolved(url, strict=True).strict_tools_active is False
        assert resolved(url, strict=True).base_url == url

    # DeepSeek: off keeps the stable endpoint; on activates strict and routes to /beta (idempotently).
    assert resolved("https://api.deepseek.com").strict_tools_active is False
    assert resolved("https://api.deepseek.com").base_url == "https://api.deepseek.com"
    assert resolved("https://api.deepseek.com", strict=True).strict_tools_active is True
    assert resolved("https://api.deepseek.com", strict=True).base_url == "https://api.deepseek.com/beta"
    assert resolved("https://api.deepseek.com/beta", strict=True).base_url == "https://api.deepseek.com/beta"

    # OpenAI supports strict but not the beta endpoint, so it stays on the normal URL.
    assert resolved("https://api.openai.com/v1", strict=True).strict_tools_active is True
    assert resolved("https://api.openai.com/v1", strict=True).base_url == "https://api.openai.com/v1"


def test_resolved_base_url_removes_known_protocol_suffixes():
    def p(url):
        return resolve(ProviderConfig(url=url)).base_url

    assert p("https://api.openai.com/v1/chat/completions") == "https://api.openai.com/v1"
    assert p("https://api.openai.com/v1/responses") == "https://api.openai.com/v1"
    assert p("https://api.openai.com/v1/messages") == "https://api.openai.com/v1"
    assert p("https://api.openai.com/v1") == "https://api.openai.com/v1"
    assert p("https://api.openai.com/v1/") == "https://api.openai.com/v1"
    assert p("https://api.openai.com/v1/chat/completions/") == "https://api.openai.com/v1"


def test_provider_api_auto_recognizes_explicit_endpoint_suffixes():
    assert ProviderConfig.from_dict({"api": "responses"}).api == "responses"
    assert resolve(ProviderConfig(url="https://api.openai.com/v1/responses")).api == "responses"
    assert resolve(ProviderConfig(url="https://api.openai.com/v1/chat/completions")).api == "chat"
    assert resolve(ProviderConfig(url="https://api.anthropic.com/v1/messages")).api == "anthropic"
    assert resolve(ProviderConfig(url="https://api.openai.com/v1")).api == "chat"
    assert resolve(ProviderConfig(url="https://api.openai.com/v1/responses", api="chat")).api == "chat"


def test_openai_responses_path_supports_strict_tools():
    provider = ProviderConfig(url="https://api.openai.com/v1", api="responses", strict_tools=True)
    assert resolve(provider).strict_tools_active is True


def test_strict_tools_schema_is_valid_and_does_not_mutate_classvars():
    before = {name: json.dumps(tool.params_schema()) for name, tool in TOOL_REGISTRY.items()}
    for name, tool in TOOL_REGISTRY.items():
        function = tool.schema(True)["function"]
        if function.get("strict"):
            _strict_check(function["parameters"], name)
        else:
            # Only free-form schemas (open objects) may skip strict; they stay untransformed.
            assert Tool._strictifiable(tool.params_schema()) is False, name
            assert function["parameters"] == tool.params_schema()
    after = {name: json.dumps(tool.params_schema()) for name, tool in TOOL_REGISTRY.items()}
    assert before == after  # deepcopy keeps shared ClassVar schemas intact

    search_context = TOOL_REGISTRY["Search"].schema(True)["function"]["parameters"]["properties"]["context"]
    assert "null" in search_context["type"]
    # Optional array/object params use anyOf (never object/array inside a type union).
    search_queries = TOOL_REGISTRY["Search"].schema(True)["function"]["parameters"]["properties"]["queries"]
    assert search_queries["anyOf"][1] == {"type": "null"}


def test_strict_tools_skips_free_form_object_schemas():
    # MCP.arguments is a free-form object; strict cannot close it, so MCP stays non-strict.
    mcp = TOOL_REGISTRY["MCP"].schema(True)["function"]
    assert "strict" not in mcp
    assert Tool._strictifiable(TOOL_REGISTRY["MCP"].params_schema()) is False
    assert Tool._strictifiable(TOOL_REGISTRY["Read"].params_schema()) is True


def test_drop_nulls_strips_omitted_strict_arguments():
    assert drop_nulls({"a": 1, "b": None, "c": {"d": None, "e": 2}, "f": [{"g": None, "h": 3}]}) == {"a": 1, "c": {"e": 2}, "f": [{"h": 3}]}


def test_chat_tool_call_parsing_handles_valid_invalid_and_non_object_payloads(tmp_path):
    client = ModelClient(session(tmp_path))
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(id="ok", function=SimpleNamespace(name="Bash", arguments=json.dumps({"command": "pwd"}))),
            SimpleNamespace(id="second", function=SimpleNamespace(name="Bash", arguments=json.dumps({"command": "whoami"}))),
            SimpleNamespace(id="bad-json", function=SimpleNamespace(name="Read", arguments="{")),
            SimpleNamespace(id="list-payload", function=SimpleNamespace(name="Recall", arguments=json.dumps(["tr.1"]))),
        ]
    )

    calls = client.tool_calls(message)

    assert calls[0] == ToolCall(id="ok", name="Bash", args=["pwd"])
    assert calls[1] == ToolCall(id="second", name="Bash", args=["whoami"])
    assert calls[2].id == "bad-json"
    assert calls[2].name == "Read"
    assert calls[2].args == []
    assert calls[3] == ToolCall(id="list-payload", name="Recall", args=[["tr.1"]])


def test_model_request_retries_retryable_errors_and_reports_attempts(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.config.provider.url = "https://example.test/v1"
    s.config.provider.key = "key"
    s.config.provider.model = "model"
    client = ModelClient(s)
    calls = []

    async def fail(_messages, _tools, **_kwargs):
        calls.append(1)
        raise ModelError("Error code: 500 - provider failed")

    monkeypatch.setattr(client.wire(s.config.provider), "request", fail)
    seen: dict[int, str] = {}
    # Record each attempt's published phase as its backoff begins; the status facts, not the
    # pacing, are what this test is about, so no wait is actually spent.
    record_backoff(monkeypatch, on_wait=lambda: seen.__setitem__(s.state.current_model_attempt, s.state.model_retry_reason))

    with pytest.raises(ModelError, match="after 6 attempts"):
        client.request([{"role": "user", "content": "hi"}])

    assert len(calls) == 6
    assert seen == {2: "500", 3: "500", 4: "500", 5: "500", 6: "500"}
    assert s.state.model_retry_count == 5
    assert s.state.current_model_attempt == 0
    assert s.state.model_retry_reason == ""


def test_retryable_error_detects_status_codes_in_text(tmp_path):

    assert resilience.retryable_error(ModelError("Error code: 500 - provider failed"))
    assert resilience.retryable_error(ModelError("{'error': {'code': 503, 'message': 'busy'}}"))
    assert not resilience.retryable_error(ModelError("Error code: 400 - bad request"))


def test_retry_reason_is_short_and_safe(tmp_path):

    assert resilience.retry_reason(ModelError("Error code: 429 - secret provider payload")) == "429"
    assert resilience.retry_reason(ModelError("request timed out with secret provider payload")) == "timeout"
    assert resilience.retry_reason(ModelError("connection reset by peer")) == "connection"


def test_model_usage_counts_cached_tokens_from_multiple_shapes():
    usage = ModelUsage()

    usage.add(
        SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=20,
            prompt_tokens_details=SimpleNamespace(cached_tokens=4, cache_write_tokens=6),
        )
    )
    usage.add({"input_tokens": 7, "output_tokens": 3, "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 5}})

    assert usage.calls == 2
    assert usage.prompt_tokens == 17
    assert usage.completion_tokens == 8
    assert usage.total_tokens == 30
    assert usage.cached_prompt_tokens == 6
    assert usage.last_cached_prompt_tokens == 2
    assert usage.cache_write_prompt_tokens == 11
    assert usage.last_cache_write_prompt_tokens == 5


def test_model_usage_folds_anthropic_cache_legs_into_prompt_tokens():
    usage = ModelUsage()

    # Anthropic reports input_tokens without the cached legs, so a cache hit must not read as a
    # ratio above 100% or shrink the request's token total to the uncached remainder.
    usage.add(SimpleNamespace(input_tokens=20, output_tokens=5, cache_read_input_tokens=30_000, cache_creation_input_tokens=1_000))

    assert usage.last_prompt_tokens == 31_020
    assert usage.last_cached_prompt_tokens == 30_000
    assert usage.last_cache_write_prompt_tokens == 1_000
    assert usage.last_cached_prompt_tokens * 100 // usage.last_prompt_tokens == 96
    assert usage.prompt_tokens == 31_020
    assert usage.total_tokens == 31_025


def test_model_usage_records_the_request_budget_beside_the_last_tokens():
    usage = ModelUsage()
    usage.add({"prompt_tokens": 10, "completion_tokens": 5}, budget=85_904)
    assert usage.last_prompt_tokens == 10
    assert usage.last_prompt_budget == 85_904

    # A later request without a budget keeps the previous one rather than zeroing it.
    usage.add({"prompt_tokens": 20, "completion_tokens": 5})
    assert usage.last_prompt_tokens == 20
    assert usage.last_prompt_budget == 85_904


def test_context_cleans_surrogate_text(tmp_path):
    bad = "bad \udce5 text"
    s = session(tmp_path)
    s.store_tool_result("Bash", [bad], bad)
    s.record_tool_error("tr.1", "Bash", [bad], bad)

    messages = ContextManager(s).model_messages("sys", [{"role": "user", "content": bad}])

    json.dumps(messages, ensure_ascii=False).encode("utf-8")
    assert "\udce5" not in str(messages)


def _session_for(tmp_path, provider):
    """A session whose active entry is the one under test: the client builders check the session's
    own config for completeness before honouring the entry they are handed."""
    built = session(tmp_path)
    built.config.providers[built.config.active_provider] = provider
    return built


def test_configured_headers_reach_both_wire_clients(tmp_path):
    """`extra_body` cannot express a header, so a provider feature documented as one -- Command
    Code's zero-retention `x-cmd-zdr` -- has to ride on the client's default headers instead."""
    from wizolt.base import HTTP_USER_AGENT

    provider = ProviderConfig.from_dict(
        {"url": "https://api.commandcode.ai/provider/v1", "key": "k", "model": "deepseek/deepseek-v4-flash", "headers": {"x-cmd-zdr": 1, "x-tenant": "team"}}
    )
    assert provider.headers == {"x-cmd-zdr": "1", "x-tenant": "team"}

    client = ModelClient(_session_for(tmp_path, provider))
    for built in (client.client(provider), client.anthropic_client(provider)):
        assert built.default_headers["x-cmd-zdr"] == "1"
        assert built.default_headers["x-tenant"] == "team"
        assert built.default_headers["User-Agent"] == HTTP_USER_AGENT


def test_configured_headers_may_replace_wizolt_defaults(tmp_path):
    provider = ProviderConfig.from_dict({"url": "https://gateway.example/v1", "key": "k", "model": "m", "headers": {"User-Agent": "fleet/1"}})

    assert ModelClient(_session_for(tmp_path, provider)).client(provider).default_headers["User-Agent"] == "fleet/1"


def test_unsendable_headers_are_a_config_error_not_a_request_failure():
    for headers in (
        "x-flag",
        {1: "one"},
        {"x-flag": True},
        {"x-flag": 1.5},
        {"x-flag": ["a"]},
        {"x-flag": "two\nlines"},
        {"bad\theader": "1"},
        {"x-unicode": "é"},
        {"x-é": "one"},
        {"X-Tenant": "one", "x-tenant": "two"},
    ):
        with pytest.raises(Exception) as error:
            ProviderConfig.from_dict({"headers": headers})
        assert "headers" in str(error.value)

    assert ProviderConfig.from_dict({}).headers == {}
