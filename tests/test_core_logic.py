import json
import time
from types import SimpleNamespace

import pytest

import nanocode as n


def session(tmp_path):
    return n.Session(cwd=str(tmp_path))


def test_runtime_settings_reads_limits_and_yolo_override():
    settings = n.RuntimeSettings.from_dict(
        {"runtime": {"shell_timeout": 7, "max_agent_steps": 0, "max_context_tokens": 0, "yolo": False}},
        yolo=True,
    )

    assert settings.shell_timeout == 7
    assert settings.max_steps == 1
    assert settings.max_context_tokens == 1
    assert settings.yolo is True


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

    params = {}
    client.apply_provider_params(params, n.ProviderConfig(url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen-plus", reasoning="medium"))
    assert params["extra_body"]["enable_thinking"] is True
    assert isinstance(params["extra_body"]["thinking_budget"], int)


def test_chat_tool_call_parsing_handles_valid_invalid_and_non_object_payloads(tmp_path):
    client = n.ModelClient(session(tmp_path))
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(id="ok", function=SimpleNamespace(name="Bash", arguments=json.dumps({"args": ["pwd"], "intention": "cwd"}))),
            SimpleNamespace(id="bad-json", function=SimpleNamespace(name="Read", arguments="{")),
            SimpleNamespace(id="list-payload", function=SimpleNamespace(name="Recall", arguments=json.dumps(["tr.1"]))),
        ]
    )

    calls = client.tool_calls(message)

    assert calls[0] == n.ToolCall(id="ok", name="Bash", args=["pwd"], intention="cwd")
    assert calls[1].id == "bad-json"
    assert calls[1].name == "Read"
    assert calls[1].args == []
    assert calls[2] == n.ToolCall(id="list-payload", name="Recall", args=[["tr.1"]], intention="")


def test_model_usage_counts_cached_tokens_from_multiple_shapes():
    usage = n.ModelUsage()

    usage.add(SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=20, prompt_tokens_details=SimpleNamespace(cached_tokens=4)))
    usage.add({"input_tokens": 7, "output_tokens": 3, "input_tokens_details": {"cached_tokens": 2}})

    assert usage.calls == 2
    assert usage.prompt_tokens == 17
    assert usage.completion_tokens == 8
    assert usage.total_tokens == 30
    assert usage.cached_prompt_tokens == 6


def test_context_and_debug_trace_clean_surrogate_text(tmp_path):
    bad = "bad \udce5 text"
    s = session(tmp_path)
    s.store_tool_result("Bash", [bad], bad, bad)
    s.record_tool_error("tr.1", "Bash", [bad], bad, bad)

    messages = n.ContextManager(s).model_messages("sys", bad)
    debug_payload = n.DebugTrace.value({"messages": messages, "raw": bad})

    json.dumps(messages, ensure_ascii=False).encode("utf-8")
    json.dumps(debug_payload, ensure_ascii=False).encode("utf-8")
    assert "\udce5" not in str(messages)
    assert "\udce5" not in str(debug_payload)


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
        lambda root, *, check=False, max_pending_files=20: calls.append(("status", check))
        or SimpleNamespace(status="ready", message="", reason="", pending_changes=0, pending_files=()),
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


def test_tool_runner_unknown_tool_debug_controls_result_storage(tmp_path):
    off = session(tmp_path)
    n.ToolRunner(off, n.ContextManager(off), output_fn=lambda text: None).run([n.ToolCall("x", "MissingTool", [], "missing")])
    assert off.tool_records == []
    assert len(off.tool_errors) == 1

    debug = session(tmp_path)
    debug.settings.debug = True
    n.ToolRunner(debug, n.ContextManager(debug), output_fn=lambda text: None).run([n.ToolCall("x", "MissingTool", [], "missing")])
    assert len(debug.tool_records) == 1
    assert len(debug.tool_results) == 1
    assert len(debug.tool_errors) == 1


def test_tool_runner_non_refusal_failures_do_not_stop_batch(tmp_path):
    s = session(tmp_path)
    s.settings.yolo = True
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    runner.run([n.ToolCall("bad", "Bash", [], "bad args"), n.ToolCall("create", "CreateFile", ["ok.txt", "ok\n"], "create")])

    assert len(s.tool_errors) == 1
    assert len(s.tool_records) == 1
    assert (tmp_path / "ok.txt").read_text(encoding="utf-8") == "ok\n"


def test_file_context_handles_deleted_files_and_newer_reads_overwrite_old_reads(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("first\n", encoding="utf-8")
    s = session(tmp_path)
    context = n.ContextManager(s)

    first_output = n.ReadTool(s, [{"path": "a.txt", "ranges": [[0, 1]]}]).call()
    first_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 1]]}], "read", first_output)
    assert "|first" in context.file_context()

    path.unlink()
    deleted = context.file_context()
    assert "|first" not in deleted
    assert first_key in deleted

    path.write_text("first\n", encoding="utf-8")
    s = session(tmp_path)
    context = n.ContextManager(s)
    old_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 1]]}], "read", n.ReadTool(s, [{"path": "a.txt", "ranges": [[0, 1]]}]).call())
    path.write_text("second\n", encoding="utf-8")
    new_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 1]]}], "read", n.ReadTool(s, [{"path": "a.txt", "ranges": [[0, 1]]}]).call())

    rendered = context.file_context()
    assert "|second" in rendered
    assert "|first" not in rendered
    assert f"source={new_key}" in rendered
    assert old_key in rendered
