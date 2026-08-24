"""tui startup side effects (split from tests/test_tui_runtime.py)."""
import os
import threading
import time

import pytest
from prompt_toolkit.history import FileHistory
from test_tui_runtime import history_file
from tui_harness import loop

from minacode.cli import CommandLoop
from minacode.cli.update import UpdateChecker
from minacode.config import (
    Config,
)
from minacode.engine import Agent
from minacode.session import Session, SessionSnapshotStore
from minacode.tools import CodeIndex
from minacode.tui import TuiApp


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
    monkeypatch.setattr(UpdateChecker, "start", lambda _checker: None)
    monkeypatch.setattr(CommandLoop, "clean_expired_sessions_async", lambda _loop: None)
    monkeypatch.setattr(CommandLoop, "render_resumed_session", lambda _loop: None)
    monkeypatch.setattr(command_loop.session.mcp, "discover_auto", lambda: None)
    monkeypatch.setattr(
        CodeIndex,
        "status",
        lambda _index, *, check=False, max_pending_files=20: status_checks.append(check) or ("ready", ""),
    )
    monkeypatch.setattr(CodeIndex, "refresh_existing_async", lambda _index: pytest.fail("startup refreshed the code index"))

    command_loop.start_session()

    assert status_checks == [False]

def test_start_session_discovers_mcp_off_the_main_thread(tmp_path, monkeypatch):
    """start_session must dispatch auto_connect MCP discovery in the background: an unreachable
    server otherwise blocks the prompt for the whole discovery timeout. Regression guard for the
    lifecycle refactor that had briefly made discover_auto a synchronous startup call."""
    config = Config.from_dict(
        {
            "provider": {"active": "d", "d": {"url": "u", "key": "k", "model": "m"}},
            "mcp": {"slow": {"url": "http://unreachable/mcp", "auto_connect": True}},
            "paths": {"data_dir": str(tmp_path / "data")},
        }
    )
    s = Session(cwd=str(tmp_path), config=config)
    command_loop = CommandLoop(
        Agent(s, output_fn=lambda text: None),
        input_fn=lambda prompt="": "",
        output_fn=lambda text: None,
    )

    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 0)
    monkeypatch.setattr(CodeIndex, "refresh_existing_async", lambda _index: False)
    monkeypatch.setattr(UpdateChecker, "start", lambda _checker: None)

    discover_started = threading.Event()
    allow_finish = threading.Event()
    ran_on: list[threading.Thread] = []

    def blocking_discover() -> None:
        ran_on.append(threading.current_thread())
        discover_started.set()
        allow_finish.wait(timeout=5)

    monkeypatch.setattr(s.mcp, "discover_auto", blocking_discover)

    try:
        command_loop.start_session()
        # Discovery was dispatched, but start_session returned while it is still blocked —
        # i.e. it ran on a background thread rather than blocking the main (prompt) thread.
        assert discover_started.wait(timeout=2), "discover_auto was never dispatched"
        assert not allow_finish.is_set()
        assert ran_on and ran_on[0] is not threading.main_thread()
    finally:
        allow_finish.set()

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

def test_expired_session_cleanup_reports_without_blocking_startup(monkeypatch, tmp_path):
    """The sweep runs on a daemon thread, so the notice arrives through the background channel."""
    command_loop = loop(tmp_path)
    command_loop.session.settings.session_retention_days = 7
    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 3)
    lines = []
    monkeypatch.setattr(command_loop, "emit", lambda text="", indent=0: lines.append(str(text)))

    command_loop.clean_expired_sessions_async()
    for _ in range(200):
        if lines:
            break
        time.sleep(0.01)

    assert len(lines) == 1
    # Says what was lost and which setting governs it, so the knob is discoverable when it acts.
    assert "removed 3 saved sessions" in lines[0]
    assert "7 days" in lines[0]
    assert "session_retention_days" in lines[0]

def test_no_notice_when_nothing_expired(monkeypatch, tmp_path):
    command_loop = loop(tmp_path)
    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 0)
    lines = []
    monkeypatch.setattr(command_loop, "emit", lambda text="", indent=0: lines.append(str(text)))

    command_loop.clean_expired_sessions_async()
    time.sleep(0.1)

    assert lines == []

def test_expired_session_sweep_never_breaks_startup(monkeypatch, tmp_path):
    """A failing sweep must not escape the thread; retention is not worth a broken session."""
    command_loop = loop(tmp_path)

    def boom(_session):
        raise OSError("data dir unreadable")

    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", boom)

    command_loop.clean_expired_sessions_async()
    time.sleep(0.1)

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
