import json
import time
from types import SimpleNamespace

import pytest

import minacode as n


def session(tmp_path):
    return n.Session(cwd=str(tmp_path))


def data_session(tmp_path):
    return n.Session(cwd=str(tmp_path), config=n.Config(data_dir=str(tmp_path / ".data")))


@pytest.mark.parametrize("flag", ["-c", "--last", "--latest"])
def test_continue_flags_resume_latest_session_in_current_project(tmp_path, monkeypatch, flag):
    config = n.Config(data_dir=str(tmp_path / "data"))
    settings = n.RuntimeSettings()
    resumed = SimpleNamespace(settings=settings, mcp=None)
    selected = []

    monkeypatch.setattr(n.ConfigFile, "load", lambda _path: {})
    monkeypatch.setattr(n.Config, "from_dict", classmethod(lambda _cls, _data: config))
    monkeypatch.setattr(n.RuntimeSettings, "from_dict", classmethod(lambda _cls, _data, **_kwargs: settings))
    monkeypatch.setattr(
        n.Session,
        "load_snapshot",
        classmethod(lambda _cls, uid, config=None, settings=None, cwd="": selected.append((uid, config, settings, cwd)) or resumed),
    )
    monkeypatch.setattr(n.tui, "Agent", lambda session: session)

    class Loop:
        def run(self):
            return 0

        def close_background_output(self):
            pass

    monkeypatch.setattr(n.tui, "CommandLoop", lambda _agent: Loop())
    monkeypatch.chdir(tmp_path)

    assert n.main([flag]) == 0
    # The alias is resolved against the current project, not a global pointer.
    assert selected == [("latest", config, settings, str(tmp_path))]


def test_runtime_settings_reads_limits_and_yolo_override():
    settings = n.RuntimeSettings.from_dict(
        {"runtime": {"shell_timeout": 7, "max_agent_steps": 0, "max_context_tokens": 0, "yolo": False}},
        yolo=True,
    )

    assert settings.shell_timeout == 7
    assert settings.max_steps == 1
    assert settings.max_context_tokens == 1
    assert settings.yolo is True


def test_runtime_settings_default_context_budget_is_240k():
    assert n.RuntimeSettings().max_context_tokens == 240 * 1024
    assert n.RuntimeSettings.from_dict({}).max_context_tokens == 240 * 1024


def test_provider_default_timeout_is_120_seconds():
    assert n.ProviderConfig().timeout == 120
    assert n.Config.from_dict({}).provider.timeout == 120


def test_runtime_settings_reads_theme_from_config():
    settings = n.RuntimeSettings.from_dict(
        {"runtime": {"theme": "light"}},
    )
    assert settings.theme == "light"

    # default when not set
    settings = n.RuntimeSettings.from_dict({})
    assert settings.theme == "auto"

    # override via keyword
    settings = n.RuntimeSettings.from_dict({"runtime": {"theme": "light"}}, theme="dark")
    assert settings.theme == "dark"

    # keyword override even when config is absent
    settings = n.RuntimeSettings.from_dict({}, theme="light")
    assert settings.theme == "light"


def test_config_validates_provider_selection_and_provider_fields():
    config = n.Config.from_dict(
        {
            "provider": {
                "active": "main",
                "main": {"url": "https://example.test/v1", "key": "k", "model": "m", "available_models": "a,b", "temperature": "off"},
            },
            "paths": {"data_dir": ".data"},
        }
    )
    assert config.active_provider == "main"
    assert config.provider.available_models == ("a", "b")
    assert config.provider.temperature is None
    assert config.data_dir == ".data"

    with pytest.raises(n.ConfigError):
        n.Config.from_dict({"provider": {"active": "missing", "main": {}}})
    with pytest.raises(n.ConfigError):
        n.ProviderConfig.from_dict({"api": "bad"})
    with pytest.raises(n.ConfigError):
        n.ProviderConfig.from_dict({"reasoning": "bad"})
    with pytest.raises(n.ConfigError):
        n.ProviderConfig.from_dict({"chat_reasoning": "bad"})
    with pytest.raises(n.ConfigError):
        n.ProviderConfig.from_dict({"prompt_cache_key": "not stable"})


def test_chat_provider_params_cover_reasoning_variants(tmp_path):
    client = n.ModelClient(session(tmp_path))

    params = {}
    client.apply_provider_params(params, n.ProviderConfig(url="https://openrouter.ai/api/v1", model="x", reasoning="high"))
    assert params["extra_body"] == {"reasoning": {"effort": "high"}}

    params = {}
    client.apply_provider_params(params, n.ProviderConfig(url="https://api.openai.com/v1", model="gpt-5-mini", reasoning="low"))
    assert params["reasoning_effort"] == "low"

    params = {}
    client.apply_provider_params(params, n.ProviderConfig(url="https://api.deepseek.com/v1", model="deepseek-chat", reasoning="off"))
    assert params["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in params

    deepseek = n.ProviderConfig(url="https://api.deepseek.com/v1", model="deepseek-v4-flash")
    assert deepseek.resolved_max_tokens() == 0


@pytest.mark.parametrize("model", ("o3", "o4-mini", "gpt-5.6"))
def test_openai_compatibility_recognizes_reasoning_model_families(model):
    provider = n.ProviderConfig(url="https://api.openai.com/v1", model=model)
    assert provider.resolved_chat_reasoning() == "reasoning_effort"


def test_openai_compatibility_leaves_non_reasoning_chat_models_off():
    provider = n.ProviderConfig(url="https://api.openai.com/v1", model="gpt-4o")
    assert provider.resolved_chat_reasoning() == "off"


def test_qwen_token_plan_compatibility_uses_reasoning_effort(tmp_path):
    client = n.ModelClient(session(tmp_path))
    provider = n.ProviderConfig.from_dict(
        {
            "url": "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            "model": "qwen3.8-max-preview",
            "reasoning": "medium",
        }
    )
    assert provider.resolved_chat_reasoning() == "reasoning_effort"

    for reasoning in ("minimal", "low", "medium", "high", "xhigh"):
        provider.reasoning = reasoning
        params = {}
        client.apply_provider_params(params, provider)
        assert params == {"reasoning_effort": reasoning}

    provider.reasoning = "off"
    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"reasoning_effort": "none"}

    provider.url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert provider.resolved_chat_reasoning() == "reasoning_effort"

    provider.url = "https://notaliyuncs.com/compatible-mode/v1"
    assert provider.resolved_chat_reasoning() == "off"

    provider.model = "other-model"
    assert provider.resolved_chat_reasoning() == "off"


def test_kimi_compatibility_uses_model_native_reasoning_controls(tmp_path):
    client = n.ModelClient(session(tmp_path))
    provider = n.ProviderConfig(url="https://api.moonshot.ai/v1", model="kimi-k3", reasoning="medium", temperature=0.2)
    assert provider.resolved_chat_reasoning() == "reasoning_effort"
    assert provider.supports_prompt_cache_key() is True
    assert provider.supports_strict_tools() is True
    assert client.prompt_cache_key(provider, None).startswith("minacode-")

    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"reasoning_effort": "high"}

    provider.reasoning = "off"
    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"reasoning_effort": "low"}

    provider.model = "kimi-k2.6"
    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"extra_body": {"thinking": {"type": "disabled"}}}

    provider.reasoning = "low"
    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"extra_body": {"thinking": {"type": "enabled"}}}

    provider.model = "kimi-k2.7-code-highspeed"
    params = {}
    client.apply_provider_params(params, provider)
    assert params == {}

    provider.url = "https://api.moonshot.cn/v1"
    assert provider.resolved_chat_reasoning() == "mandatory_thinking"


def test_kimi_code_compatibility_is_distinct_from_open_platform(tmp_path):
    client = n.ModelClient(session(tmp_path))
    provider = n.ProviderConfig(url="https://api.kimi.com/coding/v1", model="k3", reasoning="medium", temperature=0.2)
    assert provider.resolved_chat_reasoning() == "reasoning_effort"
    assert provider.supports_prompt_cache_key() is True
    assert client.prompt_cache_key(provider, None).startswith("minacode-")

    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"temperature": 0.2, "reasoning_effort": "high"}

    provider.reasoning = "off"
    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"temperature": 0.2, "reasoning_effort": "none"}

    provider.model = "kimi-for-coding-highspeed"
    provider.reasoning = "high"
    params = {}
    client.apply_provider_params(params, provider)
    assert provider.resolved_chat_reasoning() == "mandatory_thinking"
    assert params == {"temperature": 0.2}


@pytest.mark.parametrize("url", ("https://api.z.ai/api/paas/v4", "https://open.bigmodel.cn/api/paas/v4"))
def test_zai_regional_endpoints_share_documented_reasoning_effort(url, tmp_path):
    client = n.ModelClient(session(tmp_path))
    provider = n.ProviderConfig(url=url, model="glm-5.2", reasoning="xhigh", temperature=0.6)
    assert provider.resolved_chat_reasoning() == "thinking_effort"
    assert provider.supports_prompt_cache_key() is False
    assert client.prompt_cache_key(provider, None) == ""

    params = {}
    client.apply_provider_params(params, provider)
    assert params == {
        "temperature": 0.6,
        "reasoning_effort": "xhigh",
        "extra_body": {"thinking": {"type": "enabled"}},
    }

    provider.reasoning = "off"
    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"temperature": 0.6, "extra_body": {"thinking": {"type": "disabled"}}}


@pytest.mark.parametrize("url", ("https://api.z.ai/api/paas/v4", "https://open.bigmodel.cn/api/paas/v4"))
def test_zai_older_reasoning_families_use_only_thinking_toggle(url, tmp_path):
    client = n.ModelClient(session(tmp_path))
    provider = n.ProviderConfig(url=url, model="glm-5.1", reasoning="high", temperature=0.6)
    assert provider.resolved_chat_reasoning() == "thinking_toggle"

    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"temperature": 0.6, "extra_body": {"thinking": {"type": "enabled"}}}


@pytest.mark.parametrize(
    ("url", "model"),
    (
        ("https://api.moonshot.ai.evil.test/v1", "kimi-k3"),
        ("https://notmoonshot.cn/v1", "kimi-k3"),
        ("https://api.kimi.com.evil.test/coding/v1", "k3"),
        ("https://notz.ai/api/paas/v4", "glm-5.2"),
        ("https://notbigmodel.cn/api/paas/v4", "glm-5.2"),
    ),
)
def test_provider_compatibility_requires_a_real_domain_boundary(url, model, tmp_path):
    client = n.ModelClient(session(tmp_path))
    provider = n.ProviderConfig(url=url, model=model, reasoning="high", temperature=0.4)
    assert provider.resolved_chat_reasoning() == "off"
    assert provider.supports_prompt_cache_key() is True

    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"temperature": 0.4}


def test_chat_provider_extra_body_passthrough(tmp_path):
    client = n.ModelClient(session(tmp_path))

    # Vendor extensions (e.g. Qianwen web search) pass through verbatim into extra_body.
    params = {}
    search = {"enable_search": True, "search_options": {"forced_search": True, "search_strategy": "max"}}
    provider = n.ProviderConfig(url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen3-max", reasoning="off", extra_body=search)
    client.apply_provider_params(params, provider)
    assert params["extra_body"] == search

    # Configured extra_body merges with minacode-managed reasoning fields...
    params = {}
    client.apply_provider_params(params, n.ProviderConfig(url="https://openrouter.ai/api/v1", model="x", reasoning="high", extra_body={"enable_search": True}))
    assert params["extra_body"] == {"enable_search": True, "reasoning": {"effort": "high"}}

    # ...and reasoning wins on key conflict so minacode stays in control of its own fields.
    params = {}
    client.apply_provider_params(
        params, n.ProviderConfig(url="https://openrouter.ai/api/v1", model="x", reasoning="high", extra_body={"reasoning": {"effort": "low"}})
    )
    assert params["extra_body"] == {"reasoning": {"effort": "high"}}

    # extra_body round-trips through config; non-object values are ignored.
    assert n.ProviderConfig.from_dict({"extra_body": search}).extra_body == search
    assert n.ProviderConfig.from_dict({"extra_body": "nope"}).extra_body == {}
    assert n.ProviderConfig().extra_body == {}


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


def test_strict_tools_off_path_emits_legacy_schema_unchanged():
    for tool in n.TOOL_REGISTRY.values():
        legacy = {
            "type": "function",
            "function": {
                "name": tool.NAME,
                "description": "\n".join([tool.DESCRIPTION, "Signature: " + tool.SIGNATURE, *(("- " + item) for item in tool.EXAMPLE if item)]),
                "parameters": tool.params_schema(),
            },
        }
        assert tool.schema(False) == legacy
        assert "strict" not in tool.schema(False)["function"]


def test_strict_tools_gating_and_beta_routing():
    def provider(url, strict=False):
        return n.ProviderConfig(url=url, strict_tools=strict)

    # Unsupported hosts never activate strict, even when requested, and stay on their endpoint.
    for url in ("https://openrouter.ai/api/v1", "https://api.together.xyz/v1", "http://localhost:1234/v1"):
        assert provider(url, strict=True).resolved_strict_tools() is False
        assert provider(url, strict=True).base_url() == url

    # DeepSeek: off keeps the stable endpoint; on activates strict and routes to /beta (idempotently).
    assert provider("https://api.deepseek.com").resolved_strict_tools() is False
    assert provider("https://api.deepseek.com").base_url() == "https://api.deepseek.com"
    assert provider("https://api.deepseek.com", strict=True).resolved_strict_tools() is True
    assert provider("https://api.deepseek.com", strict=True).base_url() == "https://api.deepseek.com/beta"
    assert provider("https://api.deepseek.com/beta", strict=True).base_url() == "https://api.deepseek.com/beta"

    # OpenAI supports strict but not the beta endpoint, so it stays on the normal URL.
    assert provider("https://api.openai.com/v1", strict=True).resolved_strict_tools() is True
    assert provider("https://api.openai.com/v1", strict=True).base_url() == "https://api.openai.com/v1"


def test_stripped_url_removes_known_suffixes():
    def p(url):
        return n.ProviderConfig(url=url)._stripped_url()

    assert p("https://api.openai.com/v1/chat/completions") == "https://api.openai.com/v1"
    assert p("https://api.openai.com/v1/responses") == "https://api.openai.com/v1"
    assert p("https://api.openai.com/v1/messages") == "https://api.openai.com/v1"
    assert p("https://api.openai.com/v1") == "https://api.openai.com/v1"
    assert p("https://api.openai.com/v1/") == "https://api.openai.com/v1"
    assert p("https://api.openai.com/v1/chat/completions/") == "https://api.openai.com/v1"


def test_provider_api_auto_recognizes_explicit_endpoint_suffixes():
    assert n.ProviderConfig.from_dict({"api": "responses"}).api == "responses"
    assert n.ProviderConfig(url="https://api.openai.com/v1/responses").resolved_api() == "responses"
    assert n.ProviderConfig(url="https://api.openai.com/v1/chat/completions").resolved_api() == "chat"
    assert n.ProviderConfig(url="https://api.anthropic.com/v1/messages").resolved_api() == "anthropic"
    assert n.ProviderConfig(url="https://api.openai.com/v1").resolved_api() == "chat"
    assert n.ProviderConfig(url="https://api.openai.com/v1/responses", api="chat").resolved_api() == "chat"


def test_openai_responses_path_supports_strict_tools():
    provider = n.ProviderConfig(url="https://api.openai.com/v1", api="responses", strict_tools=True)
    assert provider.resolved_strict_tools() is True


def test_strict_tools_schema_is_valid_and_does_not_mutate_classvars():
    before = {name: json.dumps(tool.params_schema()) for name, tool in n.TOOL_REGISTRY.items()}
    for name, tool in n.TOOL_REGISTRY.items():
        function = tool.schema(True)["function"]
        if function.get("strict"):
            _strict_check(function["parameters"], name)
        else:
            # Only free-form schemas (open objects) may skip strict; they stay untransformed.
            assert n.Tool._strictifiable(tool.params_schema()) is False, name
            assert function["parameters"] == tool.params_schema()
    after = {name: json.dumps(tool.params_schema()) for name, tool in n.TOOL_REGISTRY.items()}
    assert before == after  # deepcopy keeps shared ClassVar schemas intact

    search_context = n.TOOL_REGISTRY["Search"].schema(True)["function"]["parameters"]["properties"]["context"]
    assert "null" in search_context["type"]
    # Optional array/object params use anyOf (never object/array inside a type union).
    search_queries = n.TOOL_REGISTRY["Search"].schema(True)["function"]["parameters"]["properties"]["queries"]
    assert search_queries["anyOf"][1] == {"type": "null"}


def test_strict_tools_skips_free_form_object_schemas():
    # MCP.arguments is a free-form object; strict cannot close it, so MCP stays non-strict.
    mcp = n.TOOL_REGISTRY["MCP"].schema(True)["function"]
    assert "strict" not in mcp
    assert n.Tool._strictifiable(n.TOOL_REGISTRY["MCP"].params_schema()) is False
    assert n.Tool._strictifiable(n.TOOL_REGISTRY["Read"].params_schema()) is True


def test_drop_nulls_strips_omitted_strict_arguments():
    assert n.ModelClient.drop_nulls({"a": 1, "b": None, "c": {"d": None, "e": 2}, "f": [{"g": None, "h": 3}]}) == {"a": 1, "c": {"e": 2}, "f": [{"h": 3}]}


def test_chat_tool_call_parsing_handles_valid_invalid_and_non_object_payloads(tmp_path):
    client = n.ModelClient(session(tmp_path))
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(id="ok", function=SimpleNamespace(name="Bash", arguments=json.dumps({"command": "pwd"}))),
            SimpleNamespace(id="second", function=SimpleNamespace(name="Bash", arguments=json.dumps({"command": "whoami"}))),
            SimpleNamespace(id="bad-json", function=SimpleNamespace(name="Read", arguments="{")),
            SimpleNamespace(id="list-payload", function=SimpleNamespace(name="Recall", arguments=json.dumps(["tr.1"]))),
        ]
    )

    calls = client.tool_calls(message)

    assert calls[0] == n.ToolCall(id="ok", name="Bash", args=["pwd"])
    assert calls[1] == n.ToolCall(id="second", name="Bash", args=["whoami"])
    assert calls[2].id == "bad-json"
    assert calls[2].name == "Read"
    assert calls[2].args == []
    assert calls[3] == n.ToolCall(id="list-payload", name="Recall", args=[["tr.1"]])


def test_model_request_retries_retryable_errors_and_reports_attempts(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.config.provider.url = "https://example.test/v1"
    s.config.provider.key = "key"
    s.config.provider.model = "model"
    client = n.ModelClient(s)
    calls = []
    retries = []

    def fail(_messages, _tools):
        calls.append(1)
        raise n.ModelError("Error code: 500 - provider failed")

    monkeypatch.setattr(client, "chat_request", fail)
    monkeypatch.setattr(
        n.time,
        "sleep",
        lambda _seconds: retries.append((s.state.current_model_attempt, s.state.model_retry_reason)),
    )

    with pytest.raises(n.ModelError, match="after 6 attempts"):
        client.request([{"role": "user", "content": "hi"}])

    assert len(calls) == 6
    assert retries == [(2, "500"), (3, "500"), (4, "500"), (5, "500"), (6, "500")]
    assert s.state.model_retry_count == 5
    assert s.state.current_model_attempt == 0
    assert s.state.model_retry_reason == ""


def test_retryable_error_detects_status_codes_in_text(tmp_path):
    client = n.ModelClient(session(tmp_path))

    assert client.retryable_error(n.ModelError("Error code: 500 - provider failed"))
    assert client.retryable_error(n.ModelError("{'error': {'code': 503, 'message': 'busy'}}"))
    assert not client.retryable_error(n.ModelError("Error code: 400 - bad request"))


def test_retry_reason_is_short_and_safe(tmp_path):
    client = n.ModelClient(session(tmp_path))

    assert client.retry_reason(n.ModelError("Error code: 429 - secret provider payload")) == "429"
    assert client.retry_reason(n.ModelError("request timed out with secret provider payload")) == "timeout"
    assert client.retry_reason(n.ModelError("connection reset by peer")) == "connection"


def test_model_usage_counts_cached_tokens_from_multiple_shapes():
    usage = n.ModelUsage()

    usage.add(SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=20, prompt_tokens_details=SimpleNamespace(cached_tokens=4)))
    usage.add({"input_tokens": 7, "output_tokens": 3, "input_tokens_details": {"cached_tokens": 2}})

    assert usage.calls == 2
    assert usage.prompt_tokens == 17
    assert usage.completion_tokens == 8
    assert usage.total_tokens == 30
    assert usage.cached_prompt_tokens == 6
    assert usage.last_cached_prompt_tokens == 2


def test_context_cleans_surrogate_text(tmp_path):
    bad = "bad \udce5 text"
    s = session(tmp_path)
    s.store_tool_result("Bash", [bad], bad)
    s.record_tool_error("tr.1", "Bash", [bad], bad)

    messages = n.ContextManager(s).model_messages("sys", [{"role": "user", "content": bad}])

    json.dumps(messages, ensure_ascii=False).encode("utf-8")
    assert "\udce5" not in str(messages)


def test_code_index_update_paths_only_keeps_workspace_files(tmp_path):
    s = session(tmp_path)
    inside = tmp_path / "inside.py"
    outside = tmp_path.parent / "outside.py"
    directory = tmp_path / "pkg"
    inside.write_text("x = 1\n", encoding="utf-8")
    outside.write_text("x = 2\n", encoding="utf-8")
    directory.mkdir()

    paths = n.CodeIndex(s).update_paths([str(inside), str(outside), str(directory), str(tmp_path / "missing.py")])

    assert paths == [str(inside)]


def test_code_index_update_pending_updates_small_batches_and_skips_large_batches(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    updates = []

    def status(root, *, check=False, max_pending_files=20):
        if check:
            return SimpleNamespace(status="stale", message="", reason="changed", pending_changes=1, pending_files=("a.py",))
        return SimpleNamespace(status="ready", message="", reason="", pending_changes="unknown", pending_files=())

    monkeypatch.setattr(n.csi, "status", status)
    monkeypatch.setattr(n.csi, "update", lambda paths, *, root: updates.append((root, list(paths))))

    assert n.CodeIndex(session(tmp_path)).update_pending() == "updated 1 file(s)"
    assert updates == [(str(tmp_path), [str(tmp_path / "a.py")])]

    updates.clear()
    monkeypatch.setattr(
        n.csi,
        "status",
        lambda root, *, check=False, max_pending_files=20: SimpleNamespace(
            status="stale", message="", reason="changed", pending_changes=n.CodeIndex.AUTO_UPDATE_LIMIT + 1, pending_files=("a.py",) * 21
        ),
    )
    assert n.CodeIndex(session(tmp_path)).update_pending() == ""
    assert updates == []


def test_code_index_sync_uses_python_api_and_updates_status(tmp_path, monkeypatch):
    calls = []

    monkeypatch.setattr(n.csi, "clean", lambda root: calls.append(("clean", root)))
    monkeypatch.setattr(n.csi, "index", lambda root: calls.append(("index", root)))
    monkeypatch.setattr(
        n.csi,
        "status",
        lambda root, *, check=False, max_pending_files=20: SimpleNamespace(status="ready", message="", reason="", pending_changes=0, pending_files=()),
    )

    s = session(tmp_path)
    result = n.CodeIndex(s).sync(force=True)

    assert calls == [("clean", str(tmp_path)), ("index", str(tmp_path))]
    assert "code_index: rebuilt" in result
    assert s.state.code_index_status == "synced"


def test_code_index_refresh_existing_uses_library_async_refresh(tmp_path, monkeypatch):
    calls = []

    class Worker:
        def join(self):
            calls.append(("join",))

    monkeypatch.setattr(
        n.csi,
        "status",
        lambda root, *, check=False, max_pending_files=20: (
            calls.append(("status", check)) or SimpleNamespace(status="ready", message="", reason="", pending_changes=0, pending_files=())
        ),
    )
    monkeypatch.setattr(n.csi, "refresh_async", lambda root: calls.append(("refresh_async", root)) or Worker())

    s = session(tmp_path)
    assert n.CodeIndex(s).refresh_existing_async() is True
    for _ in range(50):
        if ("join",) in calls and not s.state.code_index_refreshing:
            break
        time.sleep(0.01)

    assert ("refresh_async", str(tmp_path)) in calls
    assert ("join",) in calls
    assert ("status", True) in calls
    assert s.state.code_index_refreshing is False
    assert s.state.code_index_status == "synced"


def test_status_bar_animates_refreshing_code_index(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.state.code_index_refreshing = True
    s.state.code_index_notice = "syncing"
    bar = n.StatusBar(s)

    monkeypatch.setattr(n.time, "monotonic", lambda: 0.0)
    first = bar.index_status()
    monkeypatch.setattr(n.time, "monotonic", lambda: n.StatusBar.INTERVAL)
    second = bar.index_status()

    assert first != second
    assert first in n.StatusBar.INDEX_SPINNER
    assert second in n.StatusBar.INDEX_SPINNER


def test_update_checker_start_spawns_daemon_thread(tmp_path, monkeypatch):
    started = []

    class FakeThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            started.append((self.target, self.daemon))

    s = data_session(tmp_path)
    monkeypatch.setattr(n.threading, "Thread", FakeThread)

    n.UpdateChecker(s).start()
    assert len(started) == 1
    assert started[0][1] is True  # daemon
    assert s.update.checking is True

    # start() is a no-op while a check is already in flight so we don't stack duplicates.
    n.UpdateChecker(s).start()
    assert len(started) == 1


def test_update_status_signals_newer_version_in_status_bar(tmp_path):
    s = data_session(tmp_path)
    s.update.latest = "99.0.0"
    assert n.UpdateStatus.version_tuple("1.2") == (1, 2, 0)
    assert s.update.newer_than(n.__version__)
    assert s.update.latest in n.StatusBar(s).update_status()


def test_update_checker_fetch_latest_uses_bounded_timeout(tmp_path, monkeypatch):
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'{"info":{"version":"9.8.7"}}'

    def fake_urlopen(request, timeout):
        seen["timeout"] = timeout
        seen["user_agent"] = request.get_header("User-agent")
        return Response()

    monkeypatch.setattr(n.engine, "urlopen", fake_urlopen)

    assert n.UpdateChecker(data_session(tmp_path)).fetch_latest() == "9.8.7"
    assert seen == {"timeout": n.UpdateChecker.TIMEOUT, "user_agent": n.HTTP_USER_AGENT}


def test_tool_runner_unknown_tool_records_concise_error(tmp_path):
    s = session(tmp_path)
    n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None).run([n.ToolCall("x", "MissingTool", [])])
    assert s.tool_records == []
    assert s.tool_results == {}
    assert len(s.tool_errors) == 1


def test_tool_runner_non_refusal_failures_do_not_stop_batch(tmp_path):
    s = session(tmp_path)
    s.settings.yolo = True
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    runner.run([n.ToolCall("bad", "Bash", []), n.ToolCall("create", "Edit", ["ok.txt", [{"op": "create", "content": "ok\n"}]])])

    assert len(s.tool_errors) == 1
    assert len(s.tool_records) == 1
    assert (tmp_path / "ok.txt").read_text(encoding="utf-8") == "ok\n"
