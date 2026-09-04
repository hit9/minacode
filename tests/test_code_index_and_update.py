"""code index and update (split from tests/test_core_logic.py)."""

import asyncio
import threading
import time
from types import SimpleNamespace

import code_symbol_index as csi
import pytest
from test_core_logic import data_session, session

import wizolt.cli.update as update_module
from wizolt.base import (
    HTTP_USER_AGENT,
    ToolCall,
    UpdateStatus,
    __version__,
)
from wizolt.cli import CommandLoop
from wizolt.cli.update import UpdateChecker
from wizolt.context import ContextManager
from wizolt.engine import Agent
from wizolt.render import StatusBar
from wizolt.runner import ToolRunner
from wizolt.session import SessionSnapshotStore
from wizolt.tools import CodeIndex


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


async def test_code_index_update_pending_updates_small_batches_and_skips_large_batches(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    updates = []

    def status(root, *, check=False, max_pending_files=20):
        if check:
            return SimpleNamespace(status="stale", message="", reason="changed", pending_changes=1, pending_files=("a.py",))
        return SimpleNamespace(status="ready", message="", reason="", pending_changes="unknown", pending_files=())

    monkeypatch.setattr(csi, "status", status)
    monkeypatch.setattr(csi, "update", lambda paths, *, root: updates.append((root, list(paths))))

    assert await CodeIndex(session(tmp_path)).update_pending() == "updated 1 file(s)"
    assert updates == [(str(tmp_path), [str(tmp_path / "a.py")])]

    updates.clear()
    monkeypatch.setattr(
        csi,
        "status",
        lambda root, *, check=False, max_pending_files=20: SimpleNamespace(
            status="stale", message="", reason="changed", pending_changes=CodeIndex.AUTO_UPDATE_LIMIT + 1, pending_files=("a.py",) * 21
        ),
    )
    assert await CodeIndex(session(tmp_path)).update_pending() == ""
    assert updates == []


async def test_code_index_sync_uses_python_api_and_updates_status(tmp_path, monkeypatch):
    calls = []
    loop_thread = threading.get_ident()
    status_read_threads = []

    monkeypatch.setattr(csi, "clean", lambda root: calls.append(("clean", root)))
    monkeypatch.setattr(csi, "index", lambda root: calls.append(("index", root)))
    monkeypatch.setattr(
        csi,
        "status",
        lambda root, *, check=False, max_pending_files=20: (
            status_read_threads.append(threading.get_ident()) or SimpleNamespace(status="ready", message="", reason="", pending_changes=0, pending_files=())
        ),
    )

    s = session(tmp_path)
    index = CodeIndex(s)
    published_threads = []
    real_set_status = index.set_status
    monkeypatch.setattr(index, "set_status", lambda *args: published_threads.append(threading.get_ident()) or real_set_status(*args))
    result = await index.sync(force=True)

    assert calls == [("clean", str(tmp_path)), ("index", str(tmp_path))]
    assert "code_index: rebuilt" in result
    assert s.state.code_index_status == "synced"
    assert status_read_threads and status_read_threads[-1] != loop_thread
    assert published_threads and published_threads[-1] == loop_thread


async def test_cancelling_code_index_update_waits_then_clears_refreshing(tmp_path, monkeypatch):
    path = tmp_path / "a.py"
    path.write_text("x = 1\n", encoding="utf-8")
    entered = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(
        csi,
        "status",
        lambda root, *, check=False, max_pending_files=20: SimpleNamespace(status="ready", message="", reason="", pending_changes=0, pending_files=()),
    )

    def update(_paths, *, root):
        entered.set()
        release.wait(5)

    monkeypatch.setattr(csi, "update", update)
    index = CodeIndex(session(tmp_path))
    task = asyncio.create_task(index.update([str(path)]))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert index.session.state.code_index_refreshing is True
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert index.session.state.code_index_refreshing is False
    assert index.session.state.code_index_status == "synced"


async def test_cancelling_code_index_sync_waits_then_clears_refreshing(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def index_tree(_root):
        entered.set()
        release.wait(5)

    monkeypatch.setattr(csi, "index", index_tree)
    monkeypatch.setattr(
        csi,
        "status",
        lambda root, *, check=False, max_pending_files=20: SimpleNamespace(status="ready", message="", reason="", pending_changes=0, pending_files=()),
    )
    index = CodeIndex(session(tmp_path))
    task = asyncio.create_task(index.sync())
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert index.session.state.code_index_refreshing is True
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert index.session.state.code_index_refreshing is False
    assert index.session.state.code_index_status == "synced"


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


def test_loading_the_cache_claims_the_due_remote_check(tmp_path):
    """`load_cached` is the whole synchronous half: publish the cached version, say if a check is
    due, and claim it so a second caller does not stack a duplicate request."""
    s = data_session(tmp_path)

    assert UpdateChecker(s).load_cached() is True
    assert s.update.checking is True

    assert UpdateChecker(s).load_cached() is False


def test_update_status_signals_newer_version_in_status_bar(tmp_path):
    s = data_session(tmp_path)
    s.update.latest = "99.0.0"
    assert UpdateStatus.version_tuple("1.2") == (1, 2, 0)
    assert s.update.newer_than(__version__)
    assert s.update.latest in StatusBar(s).update_status()


def mock_pypi(monkeypatch, handler, seen: dict | None = None):
    """Route the checker's async client at `handler`, recording how it was constructed."""
    real = update_module.httpx2.AsyncClient

    def client(**kwargs):
        if seen is not None:
            seen.update(kwargs)
        return real(**kwargs, transport=update_module.httpx2.MockTransport(handler))

    monkeypatch.setattr(update_module.httpx2, "AsyncClient", client)


async def test_update_check_uses_the_bounded_timeout_and_user_agent(monkeypatch):
    seen: dict = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["user_agent"] = request.headers.get("user-agent")
        return update_module.httpx2.Response(200, content=b'{"info":{"version":"9.8.7"}}')

    mock_pypi(monkeypatch, handler, seen)

    assert await UpdateChecker.fetch_latest() == "9.8.7"
    assert seen["url"] == UpdateChecker.PYPI_URL
    assert seen["user_agent"] == HTTP_USER_AGENT
    assert seen["timeout"] == UpdateChecker.TIMEOUT


async def test_update_check_records_a_malformed_response_as_a_status_error(tmp_path, monkeypatch):
    """A proxy that answers with HTML is an expected failure: it leaves a status, not a crash."""
    s = data_session(tmp_path)
    mock_pypi(monkeypatch, lambda _request: update_module.httpx2.Response(200, content=b"<html>nope</html>"))

    await UpdateChecker(s).check()

    assert s.update.error and s.update.checking is False
    assert s.update.latest == ""


async def test_update_check_records_a_timeout_as_a_status_error(tmp_path, monkeypatch):
    s = data_session(tmp_path)

    def times_out(request):
        raise update_module.httpx2.ConnectTimeout("timed out", request=request)

    mock_pypi(monkeypatch, times_out)

    await UpdateChecker(s).check()

    assert "timed out" in s.update.error
    assert s.update.checking is False


async def test_cancelling_the_update_check_closes_the_client(tmp_path, monkeypatch):
    """The request is the runtime's, so cancelling it must close the pool rather than leave a
    socket to a finalizer -- which is the whole reason the client is used as a context manager."""
    s = data_session(tmp_path)
    entered = asyncio.Event()
    closed = []
    real = update_module.httpx2.AsyncClient

    class TrackedClient(real):
        async def get(self, *args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        async def __aexit__(self, *args):
            closed.append(True)
            return await super().__aexit__(*args)

    monkeypatch.setattr(update_module.httpx2, "AsyncClient", TrackedClient)

    check = asyncio.ensure_future(UpdateChecker(s).check())
    await entered.wait()
    check.cancel()
    with pytest.raises(asyncio.CancelledError):
        await check

    assert closed == [True]


def test_start_session_announces_detected_upgrade_command(tmp_path, monkeypatch):
    s = data_session(tmp_path)
    s.update.latest = "999.0.0"
    emitted = []
    monkeypatch.setattr(UpdateChecker, "load_cached", lambda _checker: False)
    monkeypatch.setattr(UpdateChecker, "upgrade_command", lambda: ["uv", "tool", "upgrade", "wizolt"])
    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda *_args: 0)
    CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=emitted.append).start_session()

    assert any("upgrade with `uv tool upgrade wizolt`" in line for line in emitted)


async def test_tool_runner_unknown_tool_records_concise_error(tmp_path):
    s = session(tmp_path)
    await ToolRunner(s, ContextManager(s), output_fn=lambda text: None).run([ToolCall("x", "MissingTool", [])])
    assert s.tool_records == []
    assert s.tool_results == {}
    assert len(s.tool_errors) == 1


async def test_tool_runner_non_refusal_failures_do_not_stop_batch(tmp_path):
    s = session(tmp_path)
    s.settings.yolo = True
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    await runner.run([ToolCall("bad", "Bash", []), ToolCall("create", "Edit", ["ok.txt", "", [{"op": "create", "content": "ok\n"}]])])

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


async def test_repeated_freshness_triggers_coalesce_onto_one_check(tmp_path, monkeypatch):
    """A turn end, /status, and a queued command can all ask within a moment. Only the first walks
    the tree; the rest see the check already claimed and return at once."""
    from tui_harness import loop as command_loop_for

    checks = 0
    release = threading.Event()

    def slow_status(root, *, check=False, max_pending_files=20):
        nonlocal checks
        if check:
            checks += 1
            release.wait(5)
        return SimpleNamespace(status="ready", message="", reason="", pending_changes=0, pending_files=())

    monkeypatch.setattr(csi, "status", slow_status)
    command_loop = command_loop_for(tmp_path)
    command_loop.open_background()
    try:
        command_loop.schedule_index_freshness()
        await asyncio.sleep(0.02)
        command_loop.schedule_index_freshness()
        command_loop.schedule_index_freshness()
        await asyncio.sleep(0.02)
        release.set()
        await asyncio.sleep(0.05)
    finally:
        release.set()
        await command_loop.close_background()

    assert checks == 1


async def test_post_turn_index_check_does_not_delay_the_answer(tmp_path, monkeypatch):
    """The check walks and hashes the tree; scheduling it must return immediately."""
    from tui_harness import loop as command_loop_for

    release = threading.Event()
    monkeypatch.setattr(
        csi,
        "status",
        lambda root, *, check=False, max_pending_files=20: (
            release.wait(5) if check else None,
            SimpleNamespace(status="ready", message="", reason="", pending_changes=0, pending_files=()),
        )[1],
    )
    command_loop = command_loop_for(tmp_path)
    command_loop.open_background()
    try:
        started = time.monotonic()
        command_loop.schedule_index_freshness()
        assert time.monotonic() - started < 0.5
        await asyncio.sleep(0.02)
        assert command_loop._background  # still running behind the answer
    finally:
        release.set()
        await command_loop.close_background()


async def test_index_command_keeps_the_loop_responsive(tmp_path, monkeypatch):
    """A manual full index is minutes of third-party work; the prompt must keep breathing."""
    release = threading.Event()
    monkeypatch.setattr(csi, "clean", lambda root: None)
    monkeypatch.setattr(csi, "index", lambda root: release.wait(5))
    monkeypatch.setattr(
        csi,
        "status",
        lambda root, *, check=False, max_pending_files=20: SimpleNamespace(status="ready", message="", reason="", pending_changes=0, pending_files=()),
    )
    beats = 0

    async def heartbeat():
        nonlocal beats
        while True:
            beats += 1
            await asyncio.sleep(0.005)

    pulse = asyncio.ensure_future(heartbeat())
    sync = asyncio.ensure_future(CodeIndex(session(tmp_path)).sync(force=True))
    await asyncio.sleep(0.05)
    assert beats > 3  # the loop kept running while csi.index blocked its worker
    release.set()
    assert "code_index: rebuilt" in await sync
    pulse.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pulse
