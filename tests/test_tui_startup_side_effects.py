"""tui startup side effects (split from tests/test_tui_runtime.py)."""
import asyncio
import os
import threading

import pytest
from prompt_toolkit.history import FileHistory
from test_tui_runtime import history_file
from tui_harness import loop

from wizolt.cli import CommandLoop
from wizolt.cli.update import UpdateChecker
from wizolt.config import (
    Config,
)
from wizolt.engine import Agent
from wizolt.session import Session, SessionSnapshotStore, bootstrap_features
from wizolt.tools import CodeIndex
from wizolt.tui import TuiApp


async def _refuses_refresh(_index) -> bool:
    """Stands in for the code index refresh: this scenario has no index to refresh."""
    return False



def test_background_output_is_closed_before_final_output(tmp_path):
    command_loop = loop(tmp_path)
    emitted = []
    command_loop.emit = lambda text="", indent=0: emitted.append(text)

    command_loop.close_background_output(lambda: emitted.append("final"))
    command_loop.emit_background("late worker output")

    assert emitted == ["final"]

def test_start_session_does_not_scan_or_refresh_code_index(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    status_checks = []
    monkeypatch.setattr(UpdateChecker, "load_cached", lambda _checker: False)
    monkeypatch.setattr(CommandLoop, "render_resumed_session", lambda _loop: None)
    monkeypatch.setattr(
        CodeIndex,
        "status",
        lambda _index, *, check=False, max_pending_files=20: status_checks.append(check) or ("ready", ""),
    )
    monkeypatch.setattr(CodeIndex, "refresh_existing", lambda _index: pytest.fail("startup refreshed the code index"))

    command_loop.start_session()

    assert status_checks == [False]

async def test_startup_discovers_mcp_without_blocking_the_prompt(tmp_path, monkeypatch):
    """MCP discovery is a task the runtime owns, never a wait on the startup path: an unreachable
    server would otherwise hold the prompt for the whole discovery timeout. Regression guard for
    the lifecycle refactor that had briefly made discovery a synchronous startup call."""
    config = Config.from_dict(
        {
            "provider": {"active": "d", "d": {"url": "u", "key": "k", "model": "m"}},
            "mcp": {"slow": {"url": "http://unreachable/mcp", "auto_connect": True}},
            "paths": {"data_dir": str(tmp_path / "data")},
        }
    )
    s = Session(cwd=str(tmp_path), config=config)
    bootstrap_features(s)
    command_loop = CommandLoop(
        Agent(s, output_fn=lambda text: None),
        input_fn=lambda prompt="": "",
        output_fn=lambda text: None,
    )

    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 0)
    monkeypatch.setattr(CodeIndex, "refresh_existing", _refuses_refresh)
    monkeypatch.setattr(UpdateChecker, "load_cached", lambda _checker: False)

    started = asyncio.Event()
    allow_finish = asyncio.Event()

    async def blocking_discover() -> None:
        started.set()
        await allow_finish.wait()

    monkeypatch.setattr(s.mcp, "discover_auto", blocking_discover)

    command_loop.start_session()  # startup itself never touches MCP
    discovery = asyncio.ensure_future(command_loop.discover_mcp())
    try:
        await asyncio.wait_for(started.wait(), 2)
        # Still blocked in discovery, and the caller is free: this is a task, not a wait.
        assert not discovery.done()
    finally:
        allow_finish.set()
        await discovery

def test_input_history_is_trimmed_to_a_bounded_size(tmp_path):
    path = history_file(tmp_path / "history.txt", 5000)
    assert os.path.getsize(path) > CommandLoop.INPUT_HISTORY_BYTES

    CommandLoop.trim_input_history(str(path))

    assert os.path.getsize(path) <= CommandLoop.INPUT_HISTORY_BYTES
    # The newest entries survive, the oldest are the ones dropped, and what remains still loads.
    kept = list(FileHistory(str(path)).load_history_strings())
    assert kept[0].startswith("4999-")
    assert not any(entry.startswith("0-") for entry in kept)
    assert all(entry.split("-")[0].isdigit() for entry in kept)

def test_input_history_trim_cuts_only_at_an_entry_boundary(tmp_path):
    path = history_file(tmp_path / "history.txt", 4000)

    CommandLoop.trim_input_history(str(path))

    # A cut inside an entry would leave a partial first line; the survivor must start with a header.
    with open(path, "rb") as file:
        assert file.read(2) == b"# "
    with open(path, encoding="utf-8") as file:
        text = file.read()
    assert all(line.startswith(("#", "+")) for line in text.splitlines() if line)

def test_input_history_under_the_cap_is_left_alone(tmp_path):
    path = history_file(tmp_path / "history.txt", 10)
    with open(path, "rb") as file:
        before = file.read()

    CommandLoop.trim_input_history(str(path))

    with open(path, "rb") as file:
        assert file.read() == before

def test_input_history_trim_survives_a_missing_or_odd_file(tmp_path):
    CommandLoop.trim_input_history(str(tmp_path / "absent.txt"))  # must not raise

    # One entry larger than the whole budget is kept rather than cut in half.
    path = tmp_path / "huge.txt"
    path.write_bytes(b"\n# 2026-01-01 00:00:00\n+" + b"y" * (CommandLoop.INPUT_HISTORY_BYTES + 1000) + b"\n")
    before = path.read_bytes()

    CommandLoop.trim_input_history(str(path))

    assert path.read_bytes() == before

async def test_expired_session_cleanup_reports_what_it_removed(monkeypatch, tmp_path):
    """The traversal runs off the loop; the notice is emitted here, once the count comes back."""
    command_loop = loop(tmp_path)
    command_loop.session.settings.session_retention_days = 7
    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 3)
    lines = []
    monkeypatch.setattr(command_loop, "emit", lambda text="", indent=0: lines.append(str(text)))

    await command_loop.clean_expired_sessions()

    assert len(lines) == 1
    # Says what was lost and which setting governs it, so the knob is discoverable when it acts.
    assert "removed 3 saved sessions" in lines[0]
    assert "7 days" in lines[0]
    assert "session_retention_days" in lines[0]

async def test_no_notice_when_nothing_expired(monkeypatch, tmp_path):
    command_loop = loop(tmp_path)
    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 0)
    lines = []
    monkeypatch.setattr(command_loop, "emit", lambda text="", indent=0: lines.append(str(text)))

    await command_loop.clean_expired_sessions()

    assert lines == []

async def test_expired_session_sweep_never_breaks_startup(monkeypatch, tmp_path):
    """A failing sweep must not escape the task; retention is not worth a broken session."""
    command_loop = loop(tmp_path)

    def boom(_session):
        raise OSError("data dir unreadable")

    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", boom)

    await command_loop.clean_expired_sessions()

async def test_cancelling_the_retention_sweep_finishes_the_deletion_pass(monkeypatch, tmp_path):
    """Retention removes unrecoverable work, so an accepted pass is never abandoned half-done."""
    command_loop = loop(tmp_path)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def slow_sweep(_session):
        entered.set()
        release.wait(5)
        finished.set()
        return 0

    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", slow_sweep)

    sweep = asyncio.ensure_future(command_loop.clean_expired_sessions())
    await asyncio.to_thread(entered.wait, 5)
    sweep.cancel()
    await asyncio.sleep(0.05)
    assert not finished.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await sweep
    assert finished.is_set()

def test_expired_session_notice_reads_correctly_when_singular(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.session.settings.session_retention_days = 1

    assert "removed 1 saved session inactive for over 1 day " in command_loop.expired_sessions_notice(1) + " "

def test_toolscript_phase_shows_on_divider_and_yields_to_compaction(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    command_loop.tui.set_running("working")
    # A stale stream phase from before the script must not relabel the script's own divider.
    command_loop.model_stream_output("output", "answering")

    command_loop.toolscript_run_status(True)
    running = "".join(text for _, text in command_loop.view.queue_divider_fragments())
    assert "running script" in running
    assert "responding" not in running

    # Compaction is the inner phase while it lasts, and the script phase returns underneath it.
    command_loop.model_stream_output("", "")
    command_loop.automatic_compaction_status(True)
    assert "compacting context" in "".join(text for _, text in command_loop.view.queue_divider_fragments())
    command_loop.automatic_compaction_status(False)
    assert "running script" in "".join(text for _, text in command_loop.view.queue_divider_fragments())

    command_loop.toolscript_run_status(False)
    assert "working" in "".join(text for _, text in command_loop.view.queue_divider_fragments())


async def test_startup_does_not_wait_for_a_blocked_maintenance_operation(tmp_path, monkeypatch):
    """The prompt comes first. A hung PyPI, an unreachable catalog remote, or a retention sweep on
    a slow network filesystem are all scheduled, never awaited, on the way to the first prompt."""
    command_loop = loop(tmp_path)
    monkeypatch.setattr(UpdateChecker, "load_cached", lambda _checker: True)
    monkeypatch.setattr(UpdateChecker, "fetch_latest", staticmethod(lambda: asyncio.Event().wait()))
    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 0)
    command_loop.open_background()
    try:
        command_loop.start_session()

        # start_session returned with the work merely admitted, not done.
        assert {task.get_name() for task in command_loop._background} >= {"update-check", "session-cleanup"}
    finally:
        await command_loop.close_background()


async def test_closing_right_after_startup_leaves_no_later_output_or_state_change(tmp_path, monkeypatch):
    """Shutdown means shutdown: a sweep or a check that was still scheduled must not write to the
    session or the terminal once close_background has returned."""
    command_loop = loop(tmp_path)
    lines: list[str] = []
    monkeypatch.setattr(command_loop, "emit", lambda text="", indent=0: lines.append(str(text)))
    monkeypatch.setattr(UpdateChecker, "load_cached", lambda _checker: True)

    async def never_answers():
        await asyncio.Event().wait()
        return "999.0.0"

    monkeypatch.setattr(UpdateChecker, "fetch_latest", staticmethod(never_answers))
    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 5)

    command_loop.open_background()
    command_loop.start_session()
    lines.clear()
    await command_loop.close_background()
    await asyncio.sleep(0.05)

    assert lines == []
    assert command_loop.session.update.latest == ""
    # Nothing is admitted after close, either: a later scheduler call is refused outright.
    assert command_loop.spawn_background(command_loop.clean_expired_sessions(), name="late") is None
