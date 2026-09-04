"""tui runtime startup (split from tests/test_tui_runtime.py)."""
import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from tui_harness import loop, session

import wizolt.render as render_module
import wizolt.tui.app as tui_module
from wizolt.base import (
    WizoltError,
)
from wizolt.cli import CommandLoop, TuiRuntime
from wizolt.cli.runtime import RESUME_STATUS_LABEL
from wizolt.cli.update import UpdateChecker
from wizolt.engine import Agent
from wizolt.session import SessionSnapshotStore
from wizolt.tools import CodeIndex
from wizolt.tui import TuiApp


async def _returns_immediately():
    """Stands in for the runtime's input loop when a test only exercises startup."""


def handled_command(exit_now=False, handled=True):
    """A stand-in for `CommandLoop.command`, which dispatch awaits like the real one."""

    async def command(_text):
        return handled, exit_now

    return command


def test_tui_emits_resumed_history_after_primary_screen_starts(tmp_path, monkeypatch):
    scenario_session = session(tmp_path)
    scenario_session.resumed = True
    scenario_session.messages.extend(
        [
            {"role": "user", "content": "restored question"},
            {"role": "assistant", "content": "restored answer"},
        ]
    )
    command_loop = CommandLoop(
        Agent(scenario_session, output_fn=lambda _text: None),
        input_fn=lambda prompt="": "",
        output_fn=lambda _text: None,
    )
    command_loop.ui.color = True
    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 0)
    monkeypatch.setattr(CodeIndex, "schedule_existing_refresh", lambda _index: False)
    monkeypatch.setattr(UpdateChecker, "start", lambda _checker: None)
    real_application = Application
    emitted_while_running = []
    history_emitted = threading.Event()

    def print_formatted(*values, **kwargs):
        # The batched resume replay arrives as one call with every fragment value as a positional
        # argument (print_formatted_text accepts *values); scan them all, not just the first.
        text = "".join(fragment_list_to_text(to_formatted_text(value)) for value in values)
        if "restored answer" in text:
            emitted_while_running.append(command_loop.tui is not None and command_loop.tui.app is not None and command_loop.tui.app.is_running)
            history_emitted.set()

    monkeypatch.setattr(render_module, "print_formatted_text", print_formatted)

    with create_pipe_input() as pipe_input:
        monkeypatch.setattr(tui_module, "Application", lambda **kwargs: real_application(input=pipe_input, **(kwargs | {"output": DummyOutput()})))

        def drive():
            assert history_emitted.wait(timeout=1)
            # The resuming status ends with set_idle; EOF only exits from chat mode, so wait for the
            # transition to finish before sending it.
            while command_loop.tui.input_mode != "chat":
                time.sleep(0.01)
            pipe_input.send_text("\x04")

        driver = threading.Thread(target=drive, daemon=True)
        driver.start()
        assert command_loop.run_tui() == 0
        driver.join(timeout=1)

    assert not driver.is_alive()
    assert emitted_while_running == [True]

def test_batched_emits_join_the_scrollback_queue_in_order(monkeypatch):
    """A batched block's exit flushes through the scrollback queue: its parts land after emits
    already queued for the live application instead of printing over them, so the scrollback
    cannot reorder."""
    ui = render_module.UiPrinter(output_fn=lambda _text: None)
    ui.color = True
    loop = asyncio.new_event_loop()
    app = SimpleNamespace(is_running=True, _running_in_terminal=False, loop=loop)
    monkeypatch.setattr(render_module, "get_app_or_none", lambda: app)
    printed = []

    def capture(*values, **kwargs):
        printed.append("".join(fragment_list_to_text(to_formatted_text(value)) for value in values))

    monkeypatch.setattr(render_module, "print_formatted_text", capture)
    thread = threading.Thread(
        target=lambda: (asyncio.set_event_loop(loop), loop.run_forever()), daemon=True
    )
    thread.start()
    try:
        ui.emit("queued first")
        with ui.batched():
            ui.emit("batched second")
            ui.emit("batched third")
        deadline = time.monotonic() + 2
        while not printed and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)

    assert printed, "the queued and batched parts never flushed"
    # One flush, in emit order: nothing landed ahead of the line queued before the batch.
    assert "".join(printed) == "queued first\nbatched second\nbatched third\n"

@pytest.mark.parametrize("entered", [" /help", "exit "])
async def test_tui_runtime_strips_input_before_command_dispatch(tmp_path, entered):
    command_loop = loop(tmp_path)
    dispatched = []
    async def record(text):
        dispatched.append(text)
        return True, False

    command_loop.command = record
    command_loop.tui = TuiApp()
    runtime = TuiRuntime(command_loop)

    assert await runtime.dispatch(entered)
    assert dispatched == [entered.strip()]

def test_tui_runtime_warms_file_mentions_after_startup(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    runtime = TuiRuntime(command_loop)
    warmed = []
    done = asyncio.Event()

    class FakeTui:
        """The application as the runtime now sees it: a task it awaits, not a thread it joins."""

        ready = threading.Event()

        async def run(self, style=None):
            del style
            self.on_ready()
            await done.wait()

        def exit(self):
            done.set()

        def write_to_scrollback(self, callback):
            raise AssertionError("this scenario writes nothing")

    fake_tui = FakeTui()
    monkeypatch.setattr(runtime, "build_tui", lambda: fake_tui)
    monkeypatch.setattr(runtime, "run_agent_loop", _returns_immediately)
    monkeypatch.setattr(command_loop, "start_session", lambda: None)
    monkeypatch.setattr(command_loop, "take_pending_inputs", list)
    monkeypatch.setattr(command_loop, "close_background_output", lambda: None)
    monkeypatch.setattr(command_loop.session.mentions, "schedule_refresh", lambda callback=None: warmed.append(callback))

    assert runtime.run_sync() == 0
    assert warmed == [None]

def test_tui_run_shows_resuming_status_while_restoring(tmp_path, monkeypatch):
    """While a resumed session's transcript is being restored the TUI shows a resuming status, and
    returns to idle the moment the replay is out."""
    scenario_session = session(tmp_path)
    scenario_session.resumed = True
    command_loop = CommandLoop(
        Agent(scenario_session, output_fn=lambda _text: None),
        input_fn=lambda prompt="": "",
        output_fn=lambda _text: None,
    )
    runtime = TuiRuntime(command_loop)
    calls = []

    class FakeTui:
        ready = threading.Event()

        def __init__(self):
            self.ready.set()

        async def run(self, style=None):
            del style
            self.on_ready()

        def exit(self):
            pass

        def write_to_scrollback(self, callback):
            raise AssertionError("this scenario writes nothing")

        def set_running(self, label):
            calls.append(("running", label))

        def set_idle(self):
            calls.append(("idle",))

    fake_tui = FakeTui()
    monkeypatch.setattr(runtime, "build_tui", lambda: fake_tui)
    monkeypatch.setattr(runtime, "run_agent_loop", _returns_immediately)
    monkeypatch.setattr(command_loop, "start_session", lambda: calls.append(("start_session",)))
    monkeypatch.setattr(command_loop, "take_pending_inputs", list)
    monkeypatch.setattr(command_loop, "close_background_output", lambda: None)
    monkeypatch.setattr(command_loop.session.mentions, "schedule_refresh", lambda callback=None: None)

    assert runtime.run_sync() == 0
    assert calls == [("running", RESUME_STATUS_LABEL), ("start_session",), ("idle",)]

async def test_tui_dispatch_compact_flushes_queued_followups(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    runtime = TuiRuntime(command_loop)
    command_loop.session.enqueue_user_input("followup A")
    command_loop.session.enqueue_user_input("followup B")

    # Empty history makes /compact return early (no model) yet still exercise the command path.
    assert await runtime.dispatch("/compact")

    # The queued follow-ups flush exactly as they do after a model turn: the first is ready to
    # run and the rest stay queued, instead of being stranded behind the command.
    assert runtime.pending.qsize() == 1
    assert runtime.pending.get_nowait() == "followup A"
    assert [item.text for item in command_loop.session.pending_user_inputs] == ["followup B"]

async def test_tui_dispatch_command_flushes_single_followup_completely(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.command = handled_command()
    command_loop.tui = TuiApp()
    runtime = TuiRuntime(command_loop)
    command_loop.session.enqueue_user_input("only followup")

    assert await runtime.dispatch("/compact")

    assert runtime.pending.qsize() == 1
    assert runtime.pending.get_nowait() == "only followup"
    assert command_loop.session.pending_user_inputs == []

async def test_tui_dispatch_failed_command_still_flushes_followup(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    runtime = TuiRuntime(command_loop)
    command_loop.session.enqueue_user_input("followup after error")
    async def fail(_text):
        raise WizoltError("command failed")

    command_loop.command = fail

    assert await runtime.dispatch("/broken")

    assert runtime.pending.get_nowait() == "followup after error"
    assert command_loop.session.pending_user_inputs == []

async def test_tui_dispatch_queues_older_followup_before_restoring_idle(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.command = handled_command()
    command_loop.tui = TuiApp()
    runtime = TuiRuntime(command_loop)
    command_loop.session.enqueue_user_input("older followup")
    events = []
    real_submit_next = runtime.submit_next
    real_reset_turn = runtime.reset_turn

    def submit_next(entered):
        events.append("submit")
        real_submit_next(entered)

    def reset_turn():
        events.append("idle")
        real_reset_turn()

    runtime.submit_next = submit_next
    runtime.reset_turn = reset_turn

    assert await runtime.dispatch("/slow-command")

    assert events == ["submit", "idle"]
    assert runtime.pending.get_nowait() == "older followup"

async def test_tui_dispatch_command_with_empty_queue_stays_idle(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.command = handled_command()
    command_loop.tui = TuiApp()
    runtime = TuiRuntime(command_loop)

    assert await runtime.dispatch("/help")

    assert runtime.pending.qsize() == 0
    assert command_loop.session.pending_user_inputs == []

async def test_tui_dispatch_non_command_leaves_followups_for_agent_turn(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    runtime = TuiRuntime(command_loop)
    command_loop.session.enqueue_user_input("followup A")

    # A plain message is not a command: dispatch returns False and must not flush the queue,
    # because run_agent_turn owns follow-up dispatch for model turns (no double dispatch).
    assert not await runtime.dispatch("answer me")

    assert runtime.pending.qsize() == 0
    assert [item.text for item in command_loop.session.pending_user_inputs] == ["followup A"]

async def test_tui_dispatch_exit_does_not_flush_queued_followups(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.command = handled_command(exit_now=True)
    command_loop.tui = TuiApp()
    command_loop.tui.exit = lambda: None
    runtime = TuiRuntime(command_loop)
    command_loop.session.enqueue_user_input("followup A")

    assert await runtime.dispatch("/exit")

    assert runtime.pending.qsize() == 0
    assert [item.text for item in command_loop.session.pending_user_inputs] == ["followup A"]
