"""code index and update (split from tests/test_core_logic.py)."""
import threading
import time
from types import SimpleNamespace

import code_symbol_index as csi
from test_core_logic import data_session, session

import minacode.cli.update as update_module
from minacode.base import (
    HTTP_USER_AGENT,
    ToolCall,
    UpdateStatus,
    __version__,
)
from minacode.cli import CommandLoop
from minacode.cli.update import UpdateChecker
from minacode.context import ContextManager
from minacode.engine import Agent
from minacode.render import StatusBar
from minacode.runner import ToolRunner
from minacode.session import SessionSnapshotStore
from minacode.tools import CodeIndex


def test_code_index_update_paths_only_keeps_workspace_files(tmp_path):
    s = session(tmp_path)
    inside = tmp_path / "inside.py"
    outside = tmp_path.parent / "outside.py"
    directory = tmp_path / "pkg"
    inside.write_text("x = 1\n", encoding="utf-8")
    outside.write_text("x = 2\n", encoding="utf-8")
    directory.mkdir()

    paths = CodeIndex(s).update_paths([str(inside), str(outside), str(directory), str(tmp_path / "missing.py")])

    assert paths == [str(inside)]

def test_code_index_update_pending_updates_small_batches_and_skips_large_batches(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    updates = []

    def status(root, *, check=False, max_pending_files=20):
        if check:
            return SimpleNamespace(status="stale", message="", reason="changed", pending_changes=1, pending_files=("a.py",))
        return SimpleNamespace(status="ready", message="", reason="", pending_changes="unknown", pending_files=())

    monkeypatch.setattr(csi, "status", status)
    monkeypatch.setattr(csi, "update", lambda paths, *, root: updates.append((root, list(paths))))

    assert CodeIndex(session(tmp_path)).update_pending() == "updated 1 file(s)"
    assert updates == [(str(tmp_path), [str(tmp_path / "a.py")])]

    updates.clear()
    monkeypatch.setattr(
        csi,
        "status",
        lambda root, *, check=False, max_pending_files=20: SimpleNamespace(
            status="stale", message="", reason="changed", pending_changes=CodeIndex.AUTO_UPDATE_LIMIT + 1, pending_files=("a.py",) * 21
        ),
    )
    assert CodeIndex(session(tmp_path)).update_pending() == ""
    assert updates == []

def test_code_index_sync_uses_python_api_and_updates_status(tmp_path, monkeypatch):
    calls = []

    monkeypatch.setattr(csi, "clean", lambda root: calls.append(("clean", root)))
    monkeypatch.setattr(csi, "index", lambda root: calls.append(("index", root)))
    monkeypatch.setattr(
        csi,
        "status",
        lambda root, *, check=False, max_pending_files=20: SimpleNamespace(status="ready", message="", reason="", pending_changes=0, pending_files=()),
    )

    s = session(tmp_path)
    result = CodeIndex(s).sync(force=True)

    assert calls == [("clean", str(tmp_path)), ("index", str(tmp_path))]
    assert "code_index: rebuilt" in result
    assert s.state.code_index_status == "synced"

def test_code_index_refresh_existing_uses_library_async_refresh(tmp_path, monkeypatch):
    calls = []

    class Worker:
        def join(self):
            calls.append(("join",))

    monkeypatch.setattr(
        csi,
        "status",
        lambda root, *, check=False, max_pending_files=20: (
            calls.append(("status", check)) or SimpleNamespace(status="ready", message="", reason="", pending_changes=0, pending_files=())
        ),
    )
    monkeypatch.setattr(csi, "refresh_async", lambda root: calls.append(("refresh_async", root)) or Worker())

    s = session(tmp_path)
    assert CodeIndex(s).refresh_existing_async() is True
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
    bar = StatusBar(s)

    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    first = bar.index_status()
    monkeypatch.setattr(time, "monotonic", lambda: StatusBar.INTERVAL)
    second = bar.index_status()

    assert first != second
    assert first in StatusBar.INDEX_SPINNER
    assert second in StatusBar.INDEX_SPINNER

def test_update_checker_start_spawns_daemon_thread(tmp_path, monkeypatch):
    started = []

    class FakeThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            started.append((self.target, self.daemon))

    s = data_session(tmp_path)
    monkeypatch.setattr(threading, "Thread", FakeThread)

    UpdateChecker(s).start()
    assert len(started) == 1
    assert started[0][1] is True  # daemon
    assert s.update.checking is True

    # start() is a no-op while a check is already in flight so we don't stack duplicates.
    UpdateChecker(s).start()
    assert len(started) == 1

def test_update_status_signals_newer_version_in_status_bar(tmp_path):
    s = data_session(tmp_path)
    s.update.latest = "99.0.0"
    assert UpdateStatus.version_tuple("1.2") == (1, 2, 0)
    assert s.update.newer_than(__version__)
    assert s.update.latest in StatusBar(s).update_status()

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

    monkeypatch.setattr(update_module, "urlopen", fake_urlopen)

    assert UpdateChecker(data_session(tmp_path)).fetch_latest() == "9.8.7"
    assert seen == {"timeout": UpdateChecker.TIMEOUT, "user_agent": HTTP_USER_AGENT}

def test_start_session_announces_detected_upgrade_command(tmp_path, monkeypatch):
    s = data_session(tmp_path)
    s.update.latest = "999.0.0"
    emitted = []
    monkeypatch.setattr(UpdateChecker, "start", lambda _checker: None)
    monkeypatch.setattr(UpdateChecker, "upgrade_command", lambda: ["uv", "tool", "upgrade", "minacode"])
    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 0)
    monkeypatch.setattr(CodeIndex, "refresh_existing_async", lambda _index: False)

    CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=emitted.append).start_session()

    assert any("upgrade with `uv tool upgrade minacode`" in line for line in emitted)

def test_tool_runner_unknown_tool_records_concise_error(tmp_path):
    s = session(tmp_path)
    ToolRunner(s, ContextManager(s), output_fn=lambda text: None).run([ToolCall("x", "MissingTool", [])])
    assert s.tool_records == []
    assert s.tool_results == {}
    assert len(s.tool_errors) == 1

def test_tool_runner_non_refusal_failures_do_not_stop_batch(tmp_path):
    s = session(tmp_path)
    s.settings.yolo = True
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run([ToolCall("bad", "Bash", []), ToolCall("create", "Edit", ["ok.txt", [{"op": "create", "content": "ok\n"}]])])

    assert len(s.tool_errors) == 1
    assert len(s.tool_records) == 1
    assert (tmp_path / "ok.txt").read_text(encoding="utf-8") == "ok\n"

def test_retry_status_renders_countdown_from_model_retry_until(tmp_path, monkeypatch):
    """The status bar formats the wait deadline published by the model; a passed deadline never
    renders a negative countdown."""
    s = session(tmp_path)
    bar = StatusBar(s)
    s.state.current_model_attempt = 3
    s.state.model_retry_count = 1
    s.state.model_retry_reason = "503"
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    s.state.model_retry_until = 112.0

    assert bar.retry_status() == "retrying 3/6 · 503 · 12s"

    # Deadline already passed: no negative countdown, no bare seconds fragment.
    s.state.model_retry_until = 99.0
    assert bar.retry_status() == "retrying 3/6 · 503"

    # Notice window expires: nothing at all.
    monkeypatch.setattr(time, "monotonic", lambda: 200.0)
    # Notice window expires: nothing at all.
    monkeypatch.setattr(time, "monotonic", lambda: 200.0)
    assert bar.retry_status() == ""
