"""TuiApp behavior: layout, input modes, key bindings, modals, and approval prompts."""

import asyncio
import multiprocessing
import os
import signal
import threading
import time
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Size
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput
from tui_harness import ResizableOutput, loop, rendered_screen_text, run_interactive_tui, session, wait_until

import minacode.tui.app as tui_module
from minacode.base import (
    SESSION_EVENT_KEY,
    LogBlock,
    LogEdge,
)
from minacode.cli import CommandCompleter, CommandLoop, TuiRuntime, hints
from minacode.cli.commands import select_choice
from minacode.cli.hints import HintPicker
from minacode.cli.update import UpdateChecker
from minacode.config import (
    Config,
)
from minacode.engine import Agent
from minacode.mentions import FilePick, active_mention
from minacode.prompts import LIVE_FOLLOWUP_PREFIX
from minacode.session import Session, SessionSnapshotStore
from minacode.tools import CodeIndex
from minacode.tui import TUI_MODAL_PENDING, CallbackPlaceholder, TuiApp


def ctrl_c_queue_scenario(cwd, results):
    config = Config(data_dir=cwd)
    scenario_session = Session(cwd=cwd, config=config)
    command_loop = CommandLoop(
        Agent(scenario_session, output_fn=lambda text: None),
        input_fn=lambda prompt="": "",
        output_fn=lambda text: None,
    )
    started = threading.Event()
    first_running = threading.Event()
    cancel_calls = []
    requests = []
    draft_after_ctrl_c = []
    elapsed = []
    driver_errors = []

    class RecordingModel:
        def request(self, messages, tools=None):
            requests.append([message.get("content") for message in messages if message.get("role") == "user"])
            if len(requests) > 1:
                return {"role": "assistant", "content": "next request complete"}, [], "next request complete"
            started.set()
            first_running.set()
            try:
                while True:
                    time.sleep(0.05)
            finally:
                first_running.clear()

        def cancel(self):
            cancel_calls.append(True)

    command_loop.agent.model = RecordingModel()
    SessionSnapshotStore.clean_expired = lambda _session: 0
    CodeIndex.refresh_existing_async = lambda _index: False
    CodeIndex.update_pending_async = lambda _index: None
    UpdateChecker.start = lambda _checker: None
    real_application = Application

    try:
        with create_pipe_input() as pipe_input:
            tui_module.Application = lambda **kwargs: real_application(input=pipe_input, **(kwargs | {"output": DummyOutput()}))

            def drive():
                try:
                    wait_until(lambda: command_loop.tui is not None and command_loop.tui.app is not None and command_loop.tui.app.is_running)
                    pipe_input.send_text("long request\r")
                    assert started.wait(timeout=1)
                    pipe_input.send_text("queued one\rqueued two\r")
                    wait_until(lambda: len(command_loop.session.pending_user_inputs) == 2)
                    pipe_input.send_text("unfinished draft")
                    wait_until(lambda: command_loop.tui.input_buffer.text == "unfinished draft")
                    began = time.monotonic()
                    pipe_input.send_text("\x03" * 10)
                    wait_until(lambda: not first_running.is_set())
                    wait_until(lambda: len(requests) == 2)
                    wait_until(lambda: command_loop.tui is not None and command_loop.tui.input_mode == "chat")
                    # The first Ctrl-C consumes the draft, the next interrupts the turn.
                    wait_until(lambda: command_loop.tui.input_buffer.text == "")
                    draft_after_ctrl_c.append(command_loop.tui.input_buffer.text)
                    elapsed.append(time.monotonic() - began)
                    command_loop.tui.input_buffer.reset(Document(""))
                    pipe_input.send_text("\x04")
                except BaseException as error:  # noqa: BLE001 - harness collects every driver-thread failure
                    driver_errors.append(repr(error))
                    if first_running.is_set():
                        os.kill(os.getpid(), signal.SIGINT)
                    if command_loop.tui is not None:
                        command_loop.tui.on_exit_request()
                        if command_loop.tui.app is not None:
                            command_loop.tui.app.loop.call_soon_threadsafe(command_loop.tui.app.exit)

            driver = threading.Thread(target=drive, daemon=True)
            driver.start()
            return_code = command_loop.run_tui()
            driver.join(timeout=1)
            if driver.is_alive():
                driver_errors.append("driver did not exit")
        restored_session = Session.load_snapshot(command_loop.session.uid, config=config)
        results.put(
            {
                "cancel_calls": len(cancel_calls),
                "driver_errors": driver_errors,
                "elapsed": elapsed,
                "draft_after_ctrl_c": draft_after_ctrl_c,
                "persisted_user_inputs": [
                    message.get("content") for message in restored_session.messages if message.get("role") == "user" and not message.get(SESSION_EVENT_KEY)
                ],
                "restored_queue": [item.text for item in restored_session.pending_user_inputs],
                "requests": requests,
                "return_code": return_code,
            }
        )
    except BaseException as error:  # noqa: BLE001 - surface every failure from the TUI thread onto the test
        results.put({"fatal": repr(error)})


def test_tui_app_build_layout_composes_input_and_status():
    app = TuiApp()
    layout = app.build_layout()
    focused = layout.current_window
    assert focused is not None
    # Layout is composable and the focused element accepts typed input via app.input_buffer.
    app.input_buffer.insert_text("hi")
    assert app.input_buffer.text == "hi"


def test_tui_approval_prompt_keeps_connector_style_and_spinner(monkeypatch):
    app = TuiApp()
    connector = LogBlock.prefix(2, LogEdge.CONTINUE)
    app.input_mode = "approval"
    app.input_prompt = connector + "[Y/n] "
    monkeypatch.setattr(time, "monotonic", lambda: 0.2)

    assert app.status_fragments() == [
        ("ansibrightblack", connector),
        ("class:approval", "[Y/n] "),
        ("class:approval.wait", "/ "),
    ]


def test_tui_loading_models_prompt_is_simple_and_dim():
    app = TuiApp()
    app.set_dispatching("Loading models...")

    assert app.status_fragments() == [("ansibrightblack", "Loading models...")]


def test_tui_non_editing_modes_clear_stale_input_errors():
    app = TuiApp()
    app.input_error = "stale image error"

    app.set_dispatching("Loading models...")
    assert app.input_error_fragments() == []

    app.input_error = "another stale image error"
    app._set_mode("approval", "Continue? ")
    assert app.input_error_fragments() == []


def test_stream_deltas_leave_the_frame_rate_to_the_animation_ticker(tmp_path):
    command_loop = loop(tmp_path)
    app = TuiApp()
    command_loop.tui = app
    frames = []
    app.invalidate = lambda: frames.append(True)

    # While the running region is up, the ticker already redraws at the frame rate; redrawing per
    # token on top of it only makes the animation's cadence swing with the model's pace.
    app.set_running("working")
    frames.clear()  # entering the mode redraws once; the deltas are what must not
    for token in ("thinking", " about", " it"):
        command_loop.model_stream_output("output", token)
    assert frames == []

    # Anywhere else there is no ticker, so a delta still has to ask for its own redraw.
    app.set_idle()
    frames.clear()
    command_loop.model_stream_output("output", "late token")
    assert frames == [True]


def test_animation_ticker_only_asks_for_frames_while_the_running_region_is_up():
    app = TuiApp()
    frames = []
    app.invalidate = lambda: frames.append(app.input_mode)

    async def run_ticker():
        ticker = asyncio.ensure_future(app.animate())
        app.set_running("working")
        await asyncio.sleep(app.ANIMATION_INTERVAL * 4)
        app.input_mode = "chat"
        running = len(frames)
        await asyncio.sleep(app.ANIMATION_INTERVAL * 4)
        ticker.cancel()
        return running

    running = asyncio.run(run_ticker())

    assert running >= 2  # the divider is animating: keep drawing it
    assert len(frames) == running  # the idle screen has nothing to animate: stop
    assert set(frames) == {"running"}


def test_interactive_tui_uses_cpr_again_after_resize_without_warning(monkeypatch):
    class CprOutput(ResizableOutput):
        def __init__(self):
            super().__init__()
            self.requests = 0

        @property
        def responds_to_cpr(self):
            return True

        def get_rows_below_cursor_position(self):
            raise NotImplementedError

        def ask_for_cpr(self):
            self.requests += 1

    output = CprOutput()
    app = TuiApp()

    def drive(_pipe_input):
        wait_until(lambda: app.app is not None and output.requests == 1)
        callback = app.app.renderer.cpr_not_supported_callback
        assert getattr(callback, "__self__", None) is None
        assert callback() is None
        app.app.loop.call_soon_threadsafe(app.app.renderer.report_absolute_cursor_row, 20)
        wait_until(lambda: not app.app.renderer.waiting_for_cpr)
        output.size = Size(rows=40, columns=120)
        app.app.loop.call_soon_threadsafe(app.app._on_resize)
        wait_until(lambda: output.requests == 2)
        app.app.loop.call_soon_threadsafe(app.app.renderer.report_absolute_cursor_row, 20)
        wait_until(lambda: not app.app.renderer.waiting_for_cpr)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive, output=output)

    assert output.requests == 2


def test_tui_app_accept_handler_fires_on_submit_and_clears_buffer():
    received: list[str] = []
    cleared_before_callback = []
    app = None

    def submit(text):
        received.append(text)
        cleared_before_callback.append(app.input_buffer.text)

    app = TuiApp(on_chat_submit=submit)
    app.input_buffer.insert_text("hello")
    app.input_buffer.validate_and_handle()
    assert received == ["hello"]
    assert cleared_before_callback == [""]
    assert app.input_buffer.text == ""


def test_tui_running_submit_clears_buffer_before_callback():
    received = []
    app = None

    def submit(text):
        received.append((text, app.input_buffer.text))

    app = TuiApp(on_running_submit=submit)
    app.set_running("working")
    app.input_buffer.insert_text("queued task")
    app.input_buffer.validate_and_handle()

    assert received == [("queued task", "")]
    assert app.input_buffer.text == ""


def test_interactive_tui_decodes_submit_and_eof(monkeypatch):
    received = []
    app = None

    def submit(text):
        received.append(text)
        app.set_idle()

    app = TuiApp(on_chat_submit=submit)

    run_interactive_tui(monkeypatch, app, text="hello from pipe\r\x04")

    assert received == ["hello from pipe"]
    assert app.app is None


@pytest.mark.parametrize(
    ("mode", "draft", "expected_interrupts"),
    [
        ("chat", "", []),
        ("chat", "unfinished draft", []),
        ("dispatch", "", ["interrupt"]),
        ("dispatch", "unfinished draft", []),
        ("running", "", ["interrupt"]),
        ("running", "unfinished draft", []),
    ],
)
def test_interactive_tui_ctrl_c_input_state_matrix(monkeypatch, tmp_path, mode, draft, expected_interrupts):
    command_loop = loop(tmp_path)
    output = []
    command_loop.emit = lambda text="", indent=0: output.append(text)
    runtime = TuiRuntime(command_loop)
    app = runtime.build_tui()
    command_loop.tui = app
    interrupts = []
    app.on_interrupt = lambda: interrupts.append("interrupt")

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        if mode == "chat":
            app.set_idle()
        elif mode == "dispatch":
            app.set_dispatching()
        else:
            app.set_running("working")
        pipe_input.send_text(draft + "\x03x")
        wait_until(lambda: app.input_buffer.text == "x")
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert interrupts == expected_interrupts
    assert output == []


def test_tui_ctrl_u_clears_the_idle_draft_without_cancelling(monkeypatch):
    """Ctrl-U discards the line. Unlike Ctrl-C it carries no other meaning, so nothing is
    cancelled."""
    interrupted = []
    app = TuiApp(on_interrupt=lambda: interrupted.append(True))

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("half typed")
        wait_until(lambda: app.input_buffer.text == "half typed")
        # Cursor into the middle: prompt_toolkit's stock Ctrl-U only discards to the left, so this
        # is what distinguishes clearing the line from clearing part of it.
        pipe_input.send_text("\x1b[D" * 5)
        wait_until(lambda: app.input_buffer.cursor_position == len("half typed") - 5)
        pipe_input.send_text("\x15")
        wait_until(lambda: app.input_buffer.text == "")
        pipe_input.send_text("\x04")

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert interrupted == []


def test_tui_ctrl_u_clears_the_running_draft_without_interrupting(monkeypatch):
    """In the queued-input editor Ctrl-C interrupts the turn, so clearing a draft there needs its
    own key."""
    interrupted = []
    app = TuiApp(on_interrupt=lambda: interrupted.append(True))

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        app.set_running("working")
        pipe_input.send_text("queued draft")
        wait_until(lambda: app.input_buffer.text == "queued draft")
        pipe_input.send_text("\x1b[D" * 6)
        wait_until(lambda: app.input_buffer.cursor_position == len("queued draft") - 6)
        pipe_input.send_text("\x15")
        wait_until(lambda: app.input_buffer.text == "")
        app.set_idle()
        pipe_input.send_text("\x04")

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert interrupted == []


def test_tui_ctrl_d_emits_resume_command_without_alternate_screen(tmp_path, monkeypatch):
    scenario_session = session(tmp_path)
    scenario_session.messages.append({"role": "user", "content": "persist me"})
    output = []
    command_loop = CommandLoop(
        Agent(scenario_session, output_fn=output.append),
        input_fn=lambda prompt="": "",
        output_fn=output.append,
    )
    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 0)
    monkeypatch.setattr(CodeIndex, "refresh_existing_async", lambda _index: False)
    monkeypatch.setattr(UpdateChecker, "start", lambda _checker: None)
    real_application = Application
    full_screen_modes = []
    tui_daemon = []

    with create_pipe_input() as pipe_input:

        def application(**kwargs):
            full_screen_modes.append(kwargs["full_screen"])
            return real_application(input=pipe_input, **(kwargs | {"output": DummyOutput()}))

        monkeypatch.setattr(tui_module, "Application", application)

        def drive():
            wait_until(lambda: command_loop.tui is not None and command_loop.tui.app is not None and command_loop.tui.app.is_running)
            tui_daemon.append(next(thread for thread in threading.enumerate() if thread.name == "tui").daemon)
            pipe_input.send_text("\x04")

        driver = threading.Thread(target=drive, daemon=True)
        driver.start()
        assert command_loop.run_tui() == 0
        driver.join(timeout=1)

    assert any(f"minacode --resume {scenario_session.uid}" in line for line in output)
    assert full_screen_modes == [False]
    assert tui_daemon == [False]


def test_interactive_tui_control_backslash_forces_exit(monkeypatch):
    forced = []
    app = None

    def force_exit():
        forced.append(True)
        app.app.exit()

    app = TuiApp(on_force_exit=force_exit)

    run_interactive_tui(monkeypatch, app, text="\x1c")

    assert forced == [True]


def test_interactive_tui_recalls_and_submits_queued_input(monkeypatch):
    received = []
    recalled = []
    app = None

    def recall():
        recalled.append(True)
        return "edit queued message"

    def submit(text):
        received.append(text)
        app.set_idle()

    app = TuiApp(on_running_submit=submit, on_recall=recall)
    app.set_running("working")

    run_interactive_tui(monkeypatch, app, text="\x1b[A\r\x04")

    assert recalled == [True]
    assert received == ["edit queued message"]


@pytest.mark.parametrize("history_key", ["\x10", "\x1b[A"])
def test_interactive_tui_history_keys_recall_when_queue_is_empty(monkeypatch, tmp_path, history_key):
    received = []
    recalled = []
    app = TuiApp(
        on_running_submit=received.append,
        history=FileHistory(str(tmp_path / "history.txt")),
    )
    app.set_running("working")

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("queued message\r")
        wait_until(lambda: received == ["queued message"])
        pipe_input.send_text(history_key)
        wait_until(lambda: app.input_buffer.text == "queued message")
        recalled.append(app.input_buffer.text)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert recalled == ["queued message"]


def test_interactive_tui_history_recall_wins_the_race_with_the_async_history_loader(monkeypatch, tmp_path):
    """Ctrl-P right after Enter must recall the entry the submit just appended.

    Every submit resets the buffer, which cancels prompt_toolkit's background task that copies
    history into the buffer's working lines; the copy only restarts at the next repaint. The
    recall key can arrive first. With the async loader pinned off, the entry must still land.
    """
    from prompt_toolkit.buffer import Buffer

    received = []
    app = TuiApp(
        on_running_submit=received.append,
        history=FileHistory(str(tmp_path / "history.txt")),
    )
    app.set_running("working")
    monkeypatch.setattr(Buffer, "load_history_if_not_yet_loaded", lambda self: None)

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("queued message\r")
        wait_until(lambda: received == ["queued message"])
        pipe_input.send_text("\x10")
        wait_until(lambda: app.input_buffer.text == "queued message")
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert app.input_buffer.text == "queued message"


def test_interactive_tui_manual_history_load_matches_the_async_loader(monkeypatch, tmp_path):
    """The synchronous history load must reproduce the native async loader exactly.

    It reaches into prompt_toolkit's private buffer state, so this pins the contract it relies
    on: the same working-lines layout the loader produces, the same recall order, and no
    duplication when a later repaint runs the real loader again.
    """
    from prompt_toolkit.buffer import Buffer

    received = []
    app = TuiApp(
        on_running_submit=received.append,
        history=FileHistory(str(tmp_path / "history.txt")),
    )
    app.set_running("working")

    native_loader = Buffer.load_history_if_not_yet_loaded
    native_loading = {"enabled": True}

    def toggleable_loader(self):
        if native_loading["enabled"]:
            native_loader(self)

    monkeypatch.setattr(Buffer, "load_history_if_not_yet_loaded", toggleable_loader)

    def drive(pipe_input):
        buffer = app.input_buffer

        def working_lines():
            try:
                return list(buffer._working_lines)
            except RuntimeError:  # The deque mutated between appends while the loader ran.
                return None

        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("first\r")
        wait_until(lambda: received == ["first"])
        wait_until(lambda: working_lines() == ["first", ""])
        pipe_input.send_text("second\r")
        wait_until(lambda: received == ["first", "second"])
        wait_until(lambda: working_lines() == ["first", "second", ""])
        assert buffer.working_index == 2  # The native layout: oldest..newest, then the editing line.

        native_loading["enabled"] = False  # The next recall runs on the manual load alone.
        pipe_input.send_text("third\r")
        wait_until(lambda: received == ["first", "second", "third"])
        pipe_input.send_text("\x10")
        wait_until(lambda: buffer.text == "third")
        assert working_lines() == ["first", "second", "third", ""]
        assert buffer.working_index == 2  # Sitting on the recalled entry, not the editing line.

        pipe_input.send_text("\x10")
        wait_until(lambda: buffer.text == "second")  # The walk order follows the native layout.

        native_loading["enabled"] = True
        pipe_input.send_text("\x10")
        wait_until(lambda: buffer.text == "first")
        time.sleep(0.3)  # Repaints ran with the real loader again; a duplicate copy would show here.
        assert working_lines() == ["first", "second", "third", ""]
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)


def test_interactive_tui_recall_over_a_draft_keeps_the_cursor(monkeypatch, tmp_path):
    """A recall over a draft that matches no history entry recalls nothing and leaves the cursor.

    The manual load moves the buffer's working index, whose setter parks the cursor at zero;
    the text is unchanged (the draft line just moved), so the cursor must stay where it was.
    """
    from prompt_toolkit.buffer import Buffer

    received = []
    app = TuiApp(
        on_running_submit=received.append,
        history=FileHistory(str(tmp_path / "history.txt")),
    )
    app.set_running("working")
    monkeypatch.setattr(Buffer, "load_history_if_not_yet_loaded", lambda self: None)

    def drive(pipe_input):
        buffer = app.input_buffer
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("submitted\r")
        wait_until(lambda: received == ["submitted"])
        pipe_input.send_text("draft")
        wait_until(lambda: buffer.text == "draft")
        pipe_input.send_text("\x10")
        wait_until(lambda: len(buffer._working_lines) == 2)  # The manual load ran.
        time.sleep(0.1)
        assert buffer.text == "draft"  # No entry starts with the draft, so nothing is recalled.
        assert buffer.cursor_position == len("draft")
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)


def test_interactive_tui_ctrl_r_search_enter_fills_input_without_submitting(monkeypatch, tmp_path):
    received = []
    app = None

    def submit(text):
        received.append(text)
        app.set_idle()

    app = TuiApp(
        on_chat_submit=submit,
        history=FileHistory(str(tmp_path / "history.txt")),
    )

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("earlier prompt\r")
        wait_until(lambda: received == ["earlier prompt"])
        pipe_input.send_text("\x12")
        wait_until(lambda: app.app.layout.current_control is app.search_toolbar.control)
        pipe_input.send_text("earlier")
        wait_until(lambda: app.search_toolbar.control.buffer.text == "earlier")
        # prompt_toolkit only applies the incremental search on another Ctrl-R (or up/down)
        # press; typing alone fills the search field and the UI preview, not the buffer.
        pipe_input.send_text("\x12")
        wait_until(lambda: app.input_buffer.text == "earlier prompt")
        # Enter accepts the match into the input box and ends the search without submitting.
        pipe_input.send_text("\r")
        wait_until(lambda: app.app.layout.current_control is not app.search_toolbar.control and app.input_buffer.text == "earlier prompt")
        assert len(received) == 1
        # The second Enter sends the accepted text.
        pipe_input.send_text("\r")
        wait_until(lambda: len(received) == 2)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert received == ["earlier prompt", "earlier prompt"]


@pytest.mark.parametrize("abort_key", ["\x03", "\x15"])
def test_interactive_tui_search_abort_keys_restore_pre_search_input(monkeypatch, tmp_path, abort_key):
    received = []
    app = None

    def submit(text):
        received.append(text)
        app.set_idle()

    app = TuiApp(
        on_chat_submit=submit,
        history=FileHistory(str(tmp_path / "history.txt")),
    )

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("earlier prompt\r")
        wait_until(lambda: received == ["earlier prompt"])
        pipe_input.send_text("draft text")
        wait_until(lambda: app.input_buffer.text == "draft text")
        pipe_input.send_text("\x12")
        wait_until(lambda: app.app.layout.current_control is app.search_toolbar.control)
        pipe_input.send_text("earlier")
        # prompt_toolkit only applies the incremental search on another Ctrl-R (or up/down)
        # press; typing alone fills the search field and the UI preview, not the buffer.
        pipe_input.send_text("\x12")
        wait_until(lambda: app.input_buffer.text == "earlier prompt")
        pipe_input.send_text(abort_key)
        wait_until(lambda: app.input_buffer.text == "draft text" and app.app.layout.current_control is not app.search_toolbar.control)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert received == ["earlier prompt"]


def test_interactive_tui_tab_inserts_single_completion_without_menu(monkeypatch):
    app = TuiApp(completer=CommandCompleter())

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("/pro\t")
        wait_until(lambda: app.input_buffer.text == "/provider")
        assert app.input_buffer.complete_state is None
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert app.input_buffer.text == "/provider"


def test_interactive_tui_bracketed_paste_displays_all_lines(monkeypatch):
    app = TuiApp()
    pasted = "\n".join(f"line {index}" for index in range(10))
    rendered = threading.Event()
    input_heights = []

    def capture(application):
        screen = application.renderer.last_rendered_screen
        if screen is None:
            return
        position = screen.visible_windows_to_write_positions.get(app.input_window)
        if position is not None and app.input_buffer.text == pasted:
            input_heights.append(position.height)
            rendered.set()

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text(f"\x1b[200~{pasted}\x1b[201~")
        wait_until(lambda: app.input_buffer.text == pasted)
        assert rendered.wait(timeout=1)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive, after_render=capture)

    assert app.input_buffer.text == pasted
    assert input_heights and input_heights[-1] == 10


def test_interactive_tui_keeps_legacy_padding_around_input(monkeypatch):
    app = TuiApp()
    frames = []
    rendered = threading.Event()

    def capture(application):
        screen = application.renderer.last_rendered_screen
        if screen is None:
            return
        positions = screen.visible_windows_to_write_positions
        if app.input_window in positions and app.status_window in positions:
            frames.append((positions[app.input_window], positions[app.status_window]))
        rendered.set()

    def drive(_pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        assert rendered.wait(timeout=1)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive, after_render=capture)

    assert frames
    prompt, status = frames[0]
    assert prompt.ypos == 1
    assert status.ypos == prompt.ypos + prompt.height + 1


def test_interactive_tui_keeps_padding_around_running_queue(monkeypatch):
    app = TuiApp(activity_fragments_fn=lambda: [("", "working\n+ queued")])
    app.set_running("working")
    frames = []
    rendered = threading.Event()

    def capture(application):
        screen = application.renderer.last_rendered_screen
        if screen is None:
            return
        positions = screen.visible_windows_to_write_positions
        windows = (app.activity_window, app.input_window, app.status_window)
        if all(window in positions for window in windows):
            frames.append(tuple(positions[window] for window in windows))
        rendered.set()

    def drive(_pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        assert rendered.wait(timeout=1)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive, after_render=capture)

    assert frames
    activity, prompt, status = frames[0]
    assert activity.ypos == 1
    assert prompt.ypos == activity.ypos + activity.height + 1
    assert status.ypos == prompt.ypos + prompt.height + 1


def test_interactive_tui_approval_has_no_leading_blank_row(monkeypatch):
    app = TuiApp()
    app._set_mode("approval", "    ├ [Y/n or reason] ")
    frames = []
    rendered = threading.Event()

    def capture(application):
        screen = application.renderer.last_rendered_screen
        if screen is None:
            return
        positions = screen.visible_windows_to_write_positions
        if app.input_window in positions and app.status_window in positions:
            frames.append((positions[app.input_window], positions[app.status_window]))
        rendered.set()

    def drive(_pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        assert rendered.wait(timeout=1)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive, after_render=capture)

    assert frames
    prompt, status = frames[0]
    assert prompt.ypos == 0
    assert status.ypos == prompt.ypos + prompt.height + 1


def test_tui_running_input_queues_one_multiline_message():
    received: list[str] = []
    app = TuiApp(on_running_submit=received.append)
    app.set_running("working")
    app.input_buffer.insert_text("first\nsecond\nthird")

    app.input_buffer.validate_and_handle()

    assert received == ["first\nsecond\nthird"]
    assert app.input_buffer.text == ""


def test_tui_running_input_drops_whitespace_only_draft():
    received: list[str] = []
    app = TuiApp(on_running_submit=received.append)
    app.set_running("working")
    app.input_buffer.insert_text("  \n ")

    app.input_buffer.validate_and_handle()

    assert received == []
    assert app.input_buffer.text == ""


def test_tui_running_input_shows_contextual_placeholder():
    hint = {"text": "Enter queues follow-up"}
    placeholder = CallbackPlaceholder(lambda: hint["text"])

    def transform(text):
        document = Document(text)
        ti = type(
            "TransformationInput",
            (),
            {
                "buffer_control": type("Control", (), {"buffer": type("Buffer", (), {"text": text})()})(),
                "document": document,
                "lineno": document.line_count - 1,
                "fragments": [],
            },
        )()
        return placeholder.apply_transformation(ti).fragments

    assert transform("") == [("class:queue.hint", "Enter queues follow-up")]
    assert transform("draft") == []


def test_tui_running_queue_hint_shows_recall_and_interrupt(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    command_loop.tui.set_running("working")
    command_loop.session.enqueue_user_input("queued")

    assert command_loop.view.tui_input_hint() == "↑ recalls queued · Ctrl-C interrupts"


def test_tui_chat_input_shows_random_idle_placeholder(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()

    hint = command_loop.view.tui_input_hint()
    assert hint in {entry.text for entry in hints.HINTS}
    assert command_loop.view.tui_input_hint() == hint  # stable within a situation (no flicker)


def test_tui_idle_hint_sessions_only_before_work(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    command_loop.view._hint_picker = HintPicker(choice=lambda pool: pool[-1])

    # Early session: the pool ends with the /sessions hint (the only early-only entry).
    assert command_loop.view.tui_input_hint() == "/sessions resumes a past session"

    # Once work exists the session is no longer early and /sessions leaves the pool.
    command_loop.session.store_tool_result("Bash", ["ls"], "ok")
    assert command_loop.view.tui_input_hint() == "Type / for commands"  # last technique hint


def test_tui_idle_hint_favors_diff_right_after_editing(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    command_loop.view._hint_picker = HintPicker(choice=lambda pool: pool[-1])
    command_loop.session.store_tool_result("Bash", ["ls"], "ok")  # mature phase
    command_loop.session.state.round_count = 1
    command_loop.session.store_turn_diff("tr.1", 1, "a.py", "diff", round=1)

    # The post-edit pool ends with the weighted /diff copies.
    assert command_loop.view.tui_input_hint() == "/diff reviews recent edits"

    # A later round without edits drops /diff back out of the pool.
    command_loop.session.state.round_count = 2
    assert command_loop.view.tui_input_hint() == "Type / for commands"


class _StubJob:
    def __init__(self, status):
        self.status = status


def test_tui_idle_hint_ps_while_jobs_running(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    command_loop.view._hint_picker = HintPicker(choice=lambda pool: pool[-1])
    command_loop.session.store_tool_result("Bash", ["ls"], "ok")  # mature

    assert command_loop.view.tui_input_hint() == "Type / for commands"  # no jobs yet

    command_loop.session.jobs["j1"] = _StubJob("running")
    assert command_loop.view.tui_input_hint() == "/ps lists background jobs"

    command_loop.session.jobs["j1"] = _StubJob("done")  # finished -> hint clears
    assert command_loop.view.tui_input_hint() == "Type / for commands"


def test_tui_hint_context_projects_availability(tmp_path):
    command_loop = loop(tmp_path)
    session = command_loop.session

    session.skills = None
    session.mcp = None
    session.jobs.clear()
    ctx = command_loop.view._hint_context()
    assert not ctx.skills_available and not ctx.mcp_connected and not ctx.jobs_running

    session.skills = SimpleNamespace(skills={"demo": object()})
    session.mcp = SimpleNamespace(tools={"srv": []})
    session.jobs["j1"] = _StubJob("running")
    ctx = command_loop.view._hint_context()
    assert ctx.skills_available and ctx.mcp_connected and ctx.jobs_running


def test_tui_idle_hint_rerolls_each_turn(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    picks = iter(["first", "second"])
    command_loop.view._hint_picker = HintPicker(choice=lambda pool: next(picks))
    command_loop.session.state.round_count = 1
    assert command_loop.view.tui_input_hint() == "first"
    assert command_loop.view.tui_input_hint() == "first"  # stable within the round
    command_loop.session.state.round_count = 2
    assert command_loop.view.tui_input_hint() == "second"  # a new turn re-rolls


def test_edit_delta_frames_minimal_edit():
    delta = tui_module._edit_delta("please refactor the auth module", "please refactor the auth")
    assert (delta.prefix, delta.removed, delta.inserted) == (24, " module", "")
    delta = tui_module._edit_delta("abc", "aXc")
    assert (delta.prefix, delta.removed, delta.inserted) == (1, "b", "X")
    delta = tui_module._edit_delta("same", "same")
    assert (delta.prefix, delta.removed, delta.inserted) == (4, "", "")
    delta = tui_module._edit_delta("", "added")
    assert (delta.prefix, delta.removed, delta.inserted) == (0, "", "added")


def test_tui_sigint_interrupts_dispatch_and_running_modes():
    interrupted = []
    app = TuiApp(on_interrupt=lambda: interrupted.append(True))
    bindings = app.make_bindings()
    handler = next(binding.handler for binding in bindings.bindings if binding.keys == (Keys.SIGINT,))
    event = type("Event", (), {})()

    app.set_dispatching()
    handler(event)
    app.set_running("working")
    handler(event)

    assert interrupted == [True, True]


def test_tui_ctrl_o_opens_latest_bash_output():
    expanded = []
    app = TuiApp(on_expand_output=lambda: expanded.append(True))
    binding = next(binding for binding in app.make_bindings().bindings if binding.keys == (Keys.ControlO,) and binding.filter())

    binding.handler(type("Event", (), {})())

    assert expanded == [True]


@pytest.mark.parametrize("mode", ["chat", "running"])
def test_tui_ctrl_d_deletes_at_cursor_when_input_is_nonempty(mode):
    app = TuiApp()
    app.input_buffer.reset(Document("abc", cursor_position=1))
    app.input_mode = mode
    binding = next(binding for binding in reversed(app.make_bindings().bindings) if binding.keys == (Keys.ControlD,) and binding.filter())
    event = type("Event", (), {"app": type("Application", (), {"exit": lambda self: None})()})()

    binding.handler(event)

    assert app.input_buffer.text == "ac"


ACTIONS = [("Approve", ""), ("View order", "v"), ("Worker config", "c"), ("Refuse", "n")]


def _approval_app():
    app = TuiApp()
    app._input_pending = threading.Event()
    app.input_mode = "approval"
    assert app.set_approval_form(ACTIONS) is True
    return app


def _active(app, key):
    return [binding for binding in reversed(app.make_bindings().bindings) if binding.keys == (key,) and binding.filter()]


def test_tui_approval_form_fires_the_focused_action_on_enter():
    # Enter submits the focused action's answer -- the same whole line ("", "v", "c", "n") the
    # approval loop already understands, so the form is a renderer, not a second protocol.
    for steps, expected in ((0, ""), (1, "v"), (2, "c"), (3, "n"), (4, "")):  # 4 wraps back to Approve
        app = _approval_app()
        for _ in range(steps):
            _active(app, Keys.Tab)[0].handler(type("Event", (), {})())
        app._accept(app.input_buffer)
        assert (app._input_result, app._input_pending.is_set()) == (expected, True)

    app = _approval_app()  # Shift-Tab from the default wraps backwards to the last action
    _active(app, Keys.BackTab)[0].handler(type("Event", (), {})())
    app._accept(app.input_buffer)
    assert app._input_result == "n"


def test_tui_approval_form_yields_the_keyboard_once_a_reason_is_typed():
    # The whole point of selecting actions instead of binding letters: nothing a reason might
    # contain is spent on a shortcut, so navigation keys go back to editing the moment there is text.
    app = _approval_app()

    def live(key):
        return [binding for binding in app.make_bindings().bindings if binding.keys == (key,) and binding.filter()]

    assert len(live(Keys.Tab)) == 2  # the action row, layered over Tab's usual completion
    assert len(live(Keys.Escape)) == 1
    assert len(live(Keys.Right)) == 1

    app.input_buffer.reset(Document("cost too high"))
    assert len(live(Keys.Tab)) == 1  # Tab completes again
    assert live(Keys.Right) == []  # ... and the arrows move the cursor
    assert len(live(Keys.Escape)) == 1  # Escape stays bound, but now clears back to the row

    app._accept(app.input_buffer)  # Enter sends the reason rather than firing an action
    assert app._input_result == "cost too high"


def test_tui_approval_form_escape_takes_back_a_reason_then_refuses():
    # Escape always undoes the current thing: with a reason typed it clears back to the action row,
    # and with nothing to take back it cancels -- which confirm() reads as a refusal with no reason.
    app = _approval_app()
    escape = _active(app, Keys.Escape)[0].handler
    app.input_buffer.reset(Document("cost too high"))

    escape(type("Event", (), {})())
    assert app.input_buffer.text == ""
    assert not app._input_pending.is_set()  # taken back, not submitted

    escape(type("Event", (), {})())
    assert (app._input_result, app._input_pending.is_set()) == (None, True)

    # A prompt with no form binds none of it.
    plain = TuiApp()
    plain._input_pending = threading.Event()
    plain.input_mode = "approval"
    assert [binding for binding in plain.make_bindings().bindings if binding.keys == (Keys.Escape,) and binding.filter()] == []


def test_tui_approval_form_row_shows_focus_and_dims_while_typing():
    app = _approval_app()

    def row():
        return "".join(text for _style, text in app.approval_form_fragments())

    def styles():
        return [style for style, _text in app.approval_form_fragments()]

    assert all(label in row() for label, _answer in ACTIONS)  # every action is visible, none memorized
    assert "class:approval.action.focused" in styles()
    assert "Tab to move" in row()

    _active(app, Keys.Tab)[0].handler(type("Event", (), {})())
    focused = [text for style, text in app.approval_form_fragments() if style == "class:approval.action.focused"]
    assert focused == [" View order "]

    # Typing disarms the row: Enter no longer fires the focused action, so it must stop looking armed.
    app.input_buffer.reset(Document("cost too high"))
    assert "class:approval.action.focused" not in styles()
    assert "Enter send · Esc back" in row()


def test_tui_ctrl_d_submits_multiline_approval_input():
    app = TuiApp()
    pending = threading.Event()
    app.input_mode = "approval"
    app._input_pending = pending
    app.input_buffer.reset(Document("first\nsecond"))
    binding = next(binding for binding in reversed(app.make_bindings().bindings) if binding.keys == (Keys.ControlD,) and binding.filter())
    event = type("Event", (), {"app": type("Application", (), {"exit": lambda self: None})()})()

    binding.handler(event)

    assert pending.is_set()
    assert app._input_result == "first\nsecond"


def test_tui_ctrl_g_and_ctrl_x_ctrl_e_open_editor():
    opened = []
    app = TuiApp()
    app.edit_input_in_editor = lambda: opened.append(True)
    bindings = app.make_bindings()
    event = type("Event", (), {})()

    # A fresh TuiApp is in chat mode, where the editor bindings are active.
    for keys in ((Keys.ControlG,), (Keys.ControlX, Keys.ControlE)):
        binding = next(binding for binding in bindings.bindings if binding.keys == keys)
        assert binding.filter()
        binding.handler(event)

    assert opened == [True, True]


@pytest.mark.parametrize("recall_key", [(Keys.Up,), (Keys.ControlP,)])
def test_tui_running_recall_removes_latest_pending_message(recall_key):
    pending = ["first", "second"]

    def recall():
        return pending.pop() if pending else ""

    app = TuiApp(on_recall=recall)
    app.set_running("working")
    bindings = app.make_bindings()
    event = type("Event", (), {"current_buffer": app.input_buffer})()
    handler = next(binding.handler for binding in reversed(bindings.bindings) if binding.keys == recall_key and binding.filter())

    handler(event)

    assert pending == ["first"]
    assert app.input_buffer.text == "second"


def test_tui_running_recall_with_draft_walks_history_when_nothing_queued(monkeypatch):
    recalled = []

    def recall():
        recalled.append(True)
        return ""

    app = TuiApp(on_recall=recall)
    app.set_running("working")
    app.input_buffer.reset(Document("draft text"))
    bindings = app.make_bindings()
    event = type("Event", (), {"current_buffer": app.input_buffer, "arg": 1})()
    handler = next(binding.handler for binding in reversed(bindings.bindings) if binding.keys == (Keys.Up,) and binding.filter())

    auto_up = []
    cursor_up = []
    monkeypatch.setattr(app.input_buffer, "auto_up", lambda count=1: auto_up.append(count))
    monkeypatch.setattr(app.input_buffer, "cursor_up", lambda: cursor_up.append(True))

    handler(event)

    # A draft must not short-circuit the recall handler: it still tries the queued
    # follow-up first, and walks history (auto_up) when none is queued.
    assert recalled == [True]
    assert auto_up == [1]
    assert cursor_up == []
    assert app.input_buffer.text == "draft text"


def test_interactive_tui_modal_uses_real_j_and_enter_keys(monkeypatch):
    app = TuiApp()
    selected = {"index": 0}
    result = []

    def key(key, _data):
        if key == "j":
            selected["index"] = 1
            return TUI_MODAL_PENDING
        if key == "enter":
            return selected["index"]
        return TUI_MODAL_PENDING

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        waiter = threading.Thread(target=lambda: result.append(app.show_modal(lambda: [("", "one\ntwo")], key)), daemon=True)
        waiter.start()
        wait_until(lambda: app.modal is not None)
        pipe_input.send_text("j\r")
        waiter.join(timeout=1)
        assert not waiter.is_alive()
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert result == [1]


@pytest.mark.parametrize("exclusive", [False, True])
def test_interactive_tui_modal_survives_repeated_resize(monkeypatch, exclusive):
    app = TuiApp()
    output = ResizableOutput()
    result = []
    rendered = threading.Event()

    def fragments():
        return [("", "\n".join(f"choice {index}" for index in range(40)))]

    def key(key, _data):
        return None if key == "q" else TUI_MODAL_PENDING

    def after_render(_application):
        rendered.set()

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        waiter = threading.Thread(target=lambda: result.append(app.show_modal(fragments, key, exclusive=exclusive)), daemon=True)
        waiter.start()
        wait_until(lambda: app.modal is not None)
        for rows, columns in ((10, 40), (35, 120), (8, 24), (24, 80)):
            rendered.clear()
            output.size = Size(rows=rows, columns=columns)
            app.app.loop.call_soon_threadsafe(app.app._on_resize)
            assert rendered.wait(timeout=1)
        pipe_input.send_text("q")
        waiter.join(timeout=1)
        assert not waiter.is_alive()
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive, output=output, after_render=after_render)

    assert result == [None]


def test_interactive_tui_renders_a_multi_line_question_as_rows_not_control_characters(monkeypatch):
    """The whole point of the split, on a real screen: every line of a multi-line Ask question is
    visible and no "^J" is drawn."""
    app = TuiApp()
    output = ResizableOutput(rows=12, columns=60)
    frames = []
    rendered = threading.Event()

    def after_render(application):
        text = rendered_screen_text(application, output)
        if "B) bar" in text:
            frames.append(text)
            rendered.set()

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        asking = threading.Thread(target=lambda: app.request_input("\nWhich one?\nA) foo\nB) bar"), daemon=True)
        asking.start()
        wait_until(lambda: app.input_mode == "approval")
        assert rendered.wait(timeout=1)
        pipe_input.send_text("x\r")
        asking.join(timeout=1)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive, output=output, after_render=after_render)

    assert "^J" not in frames[-1]
    for line in ("Which one?", "A) foo", "B) bar"):
        assert line in frames[-1]


@pytest.mark.parametrize("exclusive", [False, True])
def test_interactive_tui_modal_presentation_matches_legacy_scope(monkeypatch, exclusive):
    app = TuiApp(status_fragments_fn=lambda: [("", "status marker")])
    output = ResizableOutput(rows=12, columns=60)
    frames = []
    rendered = threading.Event()

    def after_render(application):
        text = rendered_screen_text(application, output)
        if "modal marker" in text:
            frames.append(text)
            rendered.set()

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        waiter = threading.Thread(
            target=lambda: app.show_modal(lambda: [("", "modal marker")], lambda key, _data: None if key == "q" else TUI_MODAL_PENDING, exclusive=exclusive),
            daemon=True,
        )
        waiter.start()
        wait_until(lambda: app.modal is not None)
        assert rendered.wait(timeout=1)
        pipe_input.send_text("q")
        waiter.join(timeout=1)
        assert not waiter.is_alive()
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive, output=output, after_render=after_render)

    assert "status marker" in frames[-1]
    if exclusive:
        assert frames[-1].splitlines()[-1] == "status marker"


def test_interactive_command_loop_ctrl_c_stops_llm_and_returns_to_input(tmp_path):
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    process = context.Process(target=ctrl_c_queue_scenario, args=(str(tmp_path), results))
    process.start()
    process.join(timeout=6)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
        pytest.fail("Ctrl-C TUI scenario did not exit within 6 seconds")

    assert process.exitcode == 0
    outcome = results.get(timeout=1)
    assert "fatal" not in outcome, outcome
    assert outcome["driver_errors"] == []
    assert outcome["return_code"] == 0
    assert outcome["elapsed"] and outcome["elapsed"][0] < 1.0
    assert outcome["cancel_calls"] == 1
    assert "long request" in outcome["requests"][0]
    queued_request = outcome["requests"][1]
    assert "queued one" in queued_request
    marked_followup = LIVE_FOLLOWUP_PREFIX + "queued two"
    assert marked_followup in queued_request
    assert queued_request.index("queued one") < queued_request.index(marked_followup)
    assert outcome["draft_after_ctrl_c"] == [""]
    # The interrupted first turn produced no output, so it is retracted: "long request" leaves no
    # trace in the persisted conversation, while the queued follow-ups become the next turn.
    # The follow-up keeps the marker it was sent with; only the transcript hides it.
    assert outcome["persisted_user_inputs"] == ["queued one", marked_followup]
    assert outcome["restored_queue"] == []


def test_tui_app_approval_mode_resolves_bridge_event():
    import threading as _threading

    app = TuiApp()
    result: list[str] = []
    ready = _threading.Event()

    def waiter():
        result.append(app.request_input("[Y/n] "))
        ready.set()

    thread = _threading.Thread(target=waiter, daemon=True)
    thread.start()
    # Wait until the requesting thread has switched us into approval mode.
    for _ in range(200):
        if app.input_mode == "approval":
            break
        import time as _time

        _time.sleep(0.005)
    assert app.input_mode == "approval"
    app.input_buffer.insert_text("y")
    app._accept(app.input_buffer)
    ready.wait(timeout=1.0)
    assert result == ["y"]
    assert app.input_mode == "chat"


def test_tui_approval_restores_half_typed_draft():
    app = TuiApp()
    app.set_running("working")
    app.input_buffer.insert_text("unfinished draft")
    result = []

    thread = threading.Thread(target=lambda: result.append(app.request_input("Approve? ")), daemon=True)
    thread.start()
    wait_until(lambda: app.input_mode == "approval")
    assert app.input_buffer.text == ""
    app.input_buffer.insert_text("y")
    app.input_buffer.validate_and_handle()
    thread.join(timeout=1)

    assert result == ["y"]
    assert app.input_mode == "running"
    assert app.input_buffer.text == "unfinished draft"


@pytest.mark.parametrize(
    ("prompt", "expected_above", "expected_prefix"),
    [
        ("\nPick?", [""], "Pick?"),  # Ask free-text: a blank line above the question
        ("Which one?\nA) foo\nB) bar", ["Which one?", "A) foo"], "B) bar"),  # a multi-line question
        ("\n\nStarts with its own newline", ["", ""], "Starts with its own newline"),
        ("[Y/n or reason] ", [], "[Y/n or reason] "),  # an ordinary tool approval is untouched
    ],
)
def test_tui_approval_prompt_never_renders_a_newline_as_a_control_character(prompt, expected_above, expected_prefix):
    """The input row's prefix is a single-line BeforeInput processor, and BufferControl does not
    split processor output on "\n" the way FormattedTextControl does -- a literal newline reaches
    the screen as "^J". Every line but the last is rendered as its own row above the input."""
    app = TuiApp()
    result = []
    thread = threading.Thread(target=lambda: result.append(app.request_input(prompt)), daemon=True)
    thread.start()
    wait_until(lambda: app.input_mode == "approval")

    assert app.input_prompt == expected_prefix
    assert app._input_prompt_above == expected_above
    assert all("\n" not in text for _style, text in app.status_fragments())
    assert app.full_input_prompt() == prompt  # nothing is dropped, only relocated

    app.input_buffer.insert_text("typed")
    app.input_buffer.validate_and_handle()
    thread.join(timeout=1)

    assert result == ["typed"]
    assert app._input_prompt_above == []  # the rows belong to one prompt, not the restored mode


def test_interactive_tui_ctrl_c_cancels_approval_without_interrupting_turn(monkeypatch):
    interrupted = []
    result = []
    app = TuiApp(on_interrupt=lambda: interrupted.append(True))

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        approval = threading.Thread(target=lambda: result.append(app.request_input("Approve? ")), daemon=True)
        approval.start()
        wait_until(lambda: app.input_mode == "approval")
        pipe_input.send_text("\x03")
        approval.join(timeout=1)
        assert not approval.is_alive()
        wait_until(lambda: app.input_mode == "chat")
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    # Ctrl-C cancels the approval: request_input resolves to None, which is neither "" (confirm()
    # reads that as the default approve) nor text the model could mistake for a typed answer.
    # The turn itself is not interrupted.
    assert result == [None]
    assert interrupted == []


def test_interactive_tui_ctrl_d_on_an_empty_approval_cancels_instead_of_approving(monkeypatch):
    # EOF on an empty approval line used to submit "", which confirm() reads as the default
    # approve -- the same trap Ctrl-C fell into. It must cancel; with text typed it still submits.
    result = []
    app = TuiApp()

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        for typed in ("", "too risky"):
            approval = threading.Thread(target=lambda: result.append(app.request_input("Approve? ")), daemon=True)
            approval.start()
            wait_until(lambda: app.input_mode == "approval")
            if typed:
                pipe_input.send_text(typed)
                wait_until(lambda text=typed: app.input_buffer.text == text)
            pipe_input.send_text("\x04")
            approval.join(timeout=1)
            assert not approval.is_alive()
            wait_until(lambda: app.input_mode == "chat")
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert result == [None, "too risky"]


def test_interactive_tui_exit_while_an_approval_is_pending_cancels_it(monkeypatch):
    # Shutting the app down unblocks the parked agent thread, but must not do so with "" -- that
    # would have confirm() grant the pending call on the way out.
    result = []
    app = TuiApp()

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        approval = threading.Thread(target=lambda: result.append(app.request_input("Approve? ")), daemon=True)
        approval.start()
        wait_until(lambda: app.input_mode == "approval")
        app.app.loop.call_soon_threadsafe(app.app.exit)
        approval.join(timeout=1)
        assert not approval.is_alive()

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert result == [None]


def test_interactive_tui_ctrl_c_closes_modal_and_restores_input_focus(monkeypatch):
    received = []
    app = None

    def submit(text):
        received.append(text)
        app.app.exit()

    app = TuiApp(on_chat_submit=submit)

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        waiter = threading.Thread(
            target=lambda: app.show_modal(lambda: [("", "selector")], lambda _key, _data: TUI_MODAL_PENDING),
            daemon=True,
        )
        waiter.start()
        wait_until(lambda: app.modal is not None)
        pipe_input.send_text("\x03")
        waiter.join(timeout=1)
        assert not waiter.is_alive()
        app.set_idle()
        wait_until(lambda: app.modal is None and app.app.layout.current_window is app.input_window)
        pipe_input.send_text("after cancel\r")

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert received == ["after cancel"]


def test_interactive_tui_resolved_modal_allows_followup_approval(monkeypatch):
    app = TuiApp()
    selected = []
    approved = []

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        selector = threading.Thread(
            target=lambda: selected.append(
                app.show_modal(
                    lambda: [("", "selector")],
                    lambda key, _data: "chosen" if key == "enter" else TUI_MODAL_PENDING,
                )
            ),
            daemon=True,
        )
        selector.start()
        wait_until(lambda: app.modal is not None)
        pipe_input.send_text("\r")
        selector.join(timeout=1)
        assert not selector.is_alive()
        wait_until(lambda: app.modal is None and app.app.layout.current_window is app.input_window)

        approval = threading.Thread(target=lambda: approved.append(app.request_input("Approve? ")), daemon=True)
        approval.start()
        wait_until(lambda: app.input_mode == "approval")
        pipe_input.send_text("y\r")
        approval.join(timeout=1)
        assert not approval.is_alive()
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert selected == ["chosen"]
    assert approved == ["y"]


def test_interactive_tui_choice_ctrl_c_reports_cancellation(monkeypatch, tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    output = []
    command_loop.emit = lambda text="", indent=0: output.append(text)
    app = TuiApp()
    command_loop.tui = app
    result = []

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        selector = threading.Thread(
            target=lambda: result.append(select_choice(command_loop, "Pick", ("a", "b"))),
            daemon=True,
        )
        selector.start()
        wait_until(lambda: app.modal is not None)
        pipe_input.send_text("\x03")
        selector.join(timeout=1)
        assert not selector.is_alive()
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert result == [None]
    assert output == ["Cancelled"]


def quick_hint_app(hints=("run the tests", "show the diff", "commit")):
    submitted = []
    app = TuiApp(on_chat_submit=submitted.append, quick_hints_fn=lambda: hints)
    app.set_idle()
    return app, submitted


def test_quick_hint_tab_cycles_focus_and_wraps():
    app, _ = quick_hint_app()
    assert app.quick_hint_focus == -1
    for expected in (0, 1, 2, -1):
        app.tab_or_complete(app.input_buffer, reverse=False)
        assert app.quick_hint_focus == expected


def test_quick_hint_shift_tab_cycles_focus_backwards_and_wraps():
    app, _ = quick_hint_app()
    assert app.quick_hint_focus == -1
    for expected in (2, 1, 0, -1):
        app.tab_or_complete(app.input_buffer, reverse=True)
        assert app.quick_hint_focus == expected


def test_quick_hint_tab_falls_through_to_completion_with_text():
    app, _ = quick_hint_app()
    app.input_buffer.insert_text("/mod")
    app.tab_or_complete(app.input_buffer, reverse=False)
    assert app.quick_hint_focus == -1


def test_quick_hint_tab_ignored_without_hints():
    app, _ = quick_hint_app(())
    app.tab_or_complete(app.input_buffer, reverse=False)
    assert app.quick_hint_focus == -1


def test_quick_hint_enter_picks_chip_and_returns_focus():
    app, submitted = quick_hint_app()
    app.quick_hint_focus = 1
    assert app._pick_quick_hint(app.input_buffer)
    assert submitted == []
    assert app.input_buffer.text == "show the diff"
    assert app.quick_hint_focus == -1  # Enter returns focus to the input line
    app._accept(app.input_buffer)  # a second Enter sends
    assert [str(value) for value in submitted] == ["show the diff"]


def test_quick_hint_pick_ignored_with_completion_menu_open():
    app, submitted = quick_hint_app()
    app.quick_hint_focus = 1
    app.input_buffer.complete_state = object()  # a completion menu is open
    assert app._pick_quick_hint(app.input_buffer) is False
    assert submitted == []
    assert app.input_buffer.text == ""


def test_quick_hint_pick_ignored_while_running():
    app, submitted = quick_hint_app()
    app.quick_hint_focus = 0
    app.set_running("working")
    assert app._pick_quick_hint(app.input_buffer) is False
    assert submitted == []


def test_quick_hint_enter_on_empty_unfocused_input_does_nothing():
    app, submitted = quick_hint_app()
    app._accept(app.input_buffer)
    assert submitted == []


def test_quick_hint_enter_picks_chips_one_per_tab_and_sends():
    """Enter picks the focused chip and returns to the prompt; Tab to the next chip and Enter
    again combines, and a final Enter sends the whole text."""
    app, submitted = quick_hint_app()
    app.quick_hint_focus = 0
    assert app._pick_quick_hint(app.input_buffer)  # Enter picks "run the tests"
    assert app.input_buffer.text == "run the tests"
    assert app.quick_hint_focus == -1  # focus is back on the input line
    app.tab_or_complete(app.input_buffer, reverse=False)  # resumes after picked chip 0 -> 1
    assert app._pick_quick_hint(app.input_buffer)  # Enter picks "show the diff"
    assert app.input_buffer.text == "run the tests\nshow the diff"
    assert app.quick_hint_picked == ["run the tests", "show the diff"]
    assert submitted == []
    assert app._accept(app.input_buffer)  # a final Enter sends
    assert [str(value) for value in submitted] == ["run the tests\nshow the diff"]


def test_quick_hint_enter_keys_pick_and_send_through_real_bindings(monkeypatch):
    """The wiring, not just the methods: Tab focuses, Enter picks and returns to the prompt, and
    only the Enter with no chip focused sends."""
    received = []
    app = None

    def submit(text):
        received.append(str(text))
        app.set_idle()

    app = TuiApp(on_chat_submit=submit, quick_hints_fn=lambda: ("run the tests", "show the diff", "commit"))
    app.set_idle()

    # Tab focuses chip 0, Enter picks it; one Tab reaches chip 1, Enter picks it; the final
    # Enter, with focus back on the input line, sends the combined text.
    run_interactive_tui(monkeypatch, app, text="\t\r\t\r\r\x04")

    assert received == ["run the tests\nshow the diff"]


def test_quick_hint_space_is_plain_through_real_bindings(monkeypatch):
    """Space no longer picks a focused chip: it reaches the buffer as an ordinary character."""
    received = []
    app = None

    def submit(text):
        received.append(str(text))
        app.set_idle()

    app = TuiApp(on_chat_submit=submit, quick_hints_fn=lambda: ("run the tests", "show the diff", "commit"))
    app.set_idle()

    # Tab focuses chip 0, but the space lands in the buffer instead of picking it.
    run_interactive_tui(monkeypatch, app, text="\t ok\r\x04")

    assert received == [" ok"]


def test_quick_hint_enter_again_unpicks_chip():
    app, _ = quick_hint_app()
    app.quick_hint_focus = 0
    app._pick_quick_hint(app.input_buffer)  # pick "run the tests", focus back on the input
    assert app.input_buffer.text == "run the tests"
    app.quick_hint_focus = 0  # cycling back to a picked chip makes Enter a toggle
    assert app._pick_quick_hint(app.input_buffer)  # Enter toggles it off
    assert app.quick_hint_picked == []
    assert app.input_buffer.text == ""


def test_quick_hint_pick_uses_one_hint_snapshot():
    states = iter((("old hint",), ()))
    app = TuiApp(quick_hints_fn=lambda: next(states))
    app.set_idle()
    app.quick_hint_focus = 0

    app._pick_quick_hint(app.input_buffer)
    assert app.input_buffer.text == "old hint"


def test_quick_hint_tab_cycles_while_picked():
    app, _ = quick_hint_app()
    app.quick_hint_focus = 0
    app._pick_quick_hint(app.input_buffer)  # pick -> focus back on the input
    for expected in (1, 2, -1, 0):
        app.tab_or_complete(app.input_buffer, reverse=False)
        assert app.quick_hint_focus == expected


def test_quick_hint_manual_edit_drops_picked_and_sends_edited_text():
    app, submitted = quick_hint_app()
    app.quick_hint_focus = 0
    app._pick_quick_hint(app.input_buffer)  # pick -> buffer = "run the tests"
    app.input_buffer.insert_text("!")
    assert app.quick_hint_picked == []
    assert app.quick_hint_focus == -1
    assert app.input_buffer.text == "run the tests!"
    app._accept(app.input_buffer)
    assert [str(value) for value in submitted] == ["run the tests!"]


def test_quick_hint_hints_change_resets_picked_keeps_text():
    hints = ["run the tests", "show the diff"]
    submitted = []
    app = TuiApp(on_chat_submit=submitted.append, quick_hints_fn=lambda: tuple(hints))
    app.set_idle()
    app.quick_hint_focus = 0
    app._pick_quick_hint(app.input_buffer)  # pick -> buffer = "run the tests"
    assert app.quick_hint_picked == ["run the tests"]
    hints[:] = ["commit", "push"]
    app.quick_hints()  # lazy comparison resets the picked state
    assert app.quick_hint_picked == []
    assert app.quick_hint_focus == -1
    assert app.input_buffer.text == "run the tests"


def test_quick_hint_tab_refreshes_hints_before_deciding_to_cycle():
    hints = ["old hint"]
    app, _ = quick_hint_app(tuple(hints))
    app.quick_hint_focus = 0
    app._pick_quick_hint(app.input_buffer)
    app.quick_hints_fn = lambda: tuple(hints)
    hints[:] = ["new hint"]

    app.tab_or_complete(app.input_buffer, reverse=False)

    assert app.quick_hint_picked == []
    assert app.quick_hint_focus == -1
    assert app.input_buffer.text == "old hint"


def test_quick_hint_external_edit_drops_picked_state(monkeypatch):
    app, _ = quick_hint_app()
    app.quick_hint_focus = 0
    app._pick_quick_hint(app.input_buffer)

    async def edit_in_terminal(callback, *, in_executor):
        del callback, in_executor
        return "edited text"

    monkeypatch.setattr("minacode.tui.app.run_in_terminal", edit_in_terminal)
    asyncio.run(app._run_input_editor())

    assert app.input_buffer.text == "edited text"
    assert app.quick_hint_picked == []
    assert app.quick_hint_focus == -1


def test_quick_hint_fragments_mark_picked_chips():
    app, _ = quick_hint_app(("a", "b"))
    app.quick_hint_focus = 0
    app._pick_quick_hint(app.input_buffer)  # pick "a"; the ✓ mark survives the pick
    fragments = app.quick_hint_fragments()
    assert ("class:quickhint", " \u2713 a ") in fragments
    assert ("class:quickhint", " b ") in fragments


def test_quick_hint_fragments_highlight_focused_chip():
    app, _ = quick_hint_app(("a", "b"))
    # With no terminal width (the app is not running) chips stay on one horizontal row,
    # separated by a bar, exactly as before the wrap-aware layout.
    assert app.quick_hint_fragments() == [("class:quickhint", " a "), ("class:quickhint.sep", " \u2502 "), ("class:quickhint", " b ")]
    app.quick_hint_focus = 0
    assert ("class:quickhint.focused", " a ") in app.quick_hint_fragments()


def test_quick_hints_all_visible_at_narrow_width(monkeypatch):
    """Chips flow left to right and wrap only between chips once the row is full, so all three
    hints stay fully visible — never truncated or split mid-text — at a narrow terminal.
    Exercises the real prompt_toolkit layout/render boundary, not just quick_hint_fragments()."""
    hints = ("run the tests and check the coverage", "构建文档并同步中文 locale 目录", "commit the work with a clear message")
    app = TuiApp(quick_hints_fn=lambda: hints)
    output = ResizableOutput(rows=14, columns=30)
    frames = []
    rendered = threading.Event()

    def after_render(application):
        frames.append(rendered_screen_text(application, output))
        rendered.set()

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        assert rendered.wait(timeout=1)
        pipe_input.send_text("\x04")
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive, output=output, after_render=after_render)

    assert frames, "the app rendered at least one frame"
    # A chip wider than the terminal wraps onto extra visual lines and eats the whitespace at
    # the break, so compare compact forms: one frame must show every hint's full text.
    compact_frames = ["".join(frame.split()) for frame in frames]
    assert any(all("".join(hint.split()) in screen for hint in hints) for screen in compact_frames), "no single rendered frame showed every hint"


def test_quick_hint_flow_lays_chips_horizontally_and_wraps_between_chips():
    """The wrap-aware layout keeps as many chips as fit on one row and only breaks between
    chips, never inside one; a narrow column pushes later chips to their own lines."""
    flow = TuiApp._flow_quick_hints

    # Wide enough: all three chips on one horizontal row, bar-separated, no newlines.
    wide = flow(("run", "show diff", "commit"), columns=100, focus=-1, picked=())
    assert wide == [
        ("class:quickhint", " run "),
        ("class:quickhint.sep", " \u2502 "),
        ("class:quickhint", " show diff "),
        ("class:quickhint.sep", " \u2502 "),
        ("class:quickhint", " commit "),
    ]

    # Unknown width (0) keeps the single horizontal row too, letting the window wrap as fallback.
    unknown = flow(("a", "b"), columns=0, focus=-1, picked=())
    assert "\n" not in [text for _style, text in unknown]

    # Narrow: " run the tests " (15) fits alone, but " b " would not fit the remaining row,
    # so it wraps to its own line; no chip is ever split and no bar appears mid-row.
    narrow = flow(("run the tests", "b"), columns=16, focus=-1, picked=())
    lines = "".join(text for _style, text in narrow)
    assert lines == " run the tests \n b "
    assert "\u2502" not in lines

    # A chip wider than the whole row still stays whole on its own line.
    lone = flow(("构建文档并同步中文 locale 目录", "x"), columns=12, focus=-1, picked=())
    lines = "".join(text for _style, text in lone)
    assert lines.startswith(" 构建文档并同步中文 locale 目录 ")
    assert " \u2502 " not in lines

    # Focus and picked marks keep working through the flow layout.
    picked = flow(("a", "b"), columns=100, focus=1, picked=("a",))
    assert ("class:quickhint", " \u2713 a ") in picked
    assert ("class:quickhint.focused", " b ") in picked

    # At most MAX_QUICK_HINTS_PER_ROW chips per row even when the terminal is wide: the fourth
    # short hint starts its own line instead of crowding one row.
    four = flow(("a", "b", "c", "d"), columns=100, focus=-1, picked=())
    lines = "".join(text for _style, text in four)
    assert lines == " a  \u2502  b  \u2502  c \n d "

    # Width and the per-row cap both end a row; the width check still wins on a narrow column.
    cap_and_width = flow(("a", "b", "c", "d"), columns=5, focus=-1, picked=())
    assert "".join(text for _style, text in cap_and_width) == " a \n b \n c \n d "


def test_quick_hint_pick_and_send_still_work_after_wrapping(monkeypatch):
    """Chips that wrap onto multiple visual lines keep Tab focus, Enter pick/unpick, and the
    final Enter submission."""
    received = []
    app = None

    def submit(text):
        received.append(str(text))
        app.set_idle()

    app = TuiApp(
        on_chat_submit=submit,
        quick_hints_fn=lambda: ("run the tests and check the coverage", "构建文档并同步中文 locale 目录", "commit the work"),
    )
    app.set_idle()

    # Tab focuses chip 0, Enter picks it; one Tab reaches chip 1, Enter picks it; the final
    # Enter, with focus back on the input line, sends the combined text.
    run_interactive_tui(monkeypatch, app, text="\t\r\t\r\r\x04")

    assert received == ["run the tests and check the coverage\n构建文档并同步中文 locale 目录"]


def test_quick_hint_placeholder_hints_keys_until_focused():
    app, _ = quick_hint_app()
    assert app.placeholder_text() == "Tab cycles suggestions \u00b7 Enter picks \u00b7 Enter sends"
    app.quick_hint_focus = 0
    assert app.placeholder_text() == ""


def test_quick_hint_pick_ignored_when_input_was_edited():
    app, _ = quick_hint_app()
    app.quick_hint_focus = 0
    app.input_buffer.insert_text("hello")
    assert app._pick_quick_hint(app.input_buffer) is False
    assert app.quick_hint_picked == []
    assert app.quick_hint_focus == 0  # unchanged, so Enter falls through to sending


def test_quick_hint_enter_without_focus_sends():
    """Enter with no focused chip sends; it never unpicks picked text."""
    app, submitted = quick_hint_app()
    app.quick_hint_focus = 0
    app._pick_quick_hint(app.input_buffer)  # pick -> buffer = "run the tests"
    assert app.quick_hint_focus == -1
    assert app._accept(app.input_buffer) is True  # Enter sends, it never unpicks
    assert [str(value) for value in submitted] == ["run the tests"]
    assert app.quick_hint_focus == -1  # sending clears the quick-hint state


def test_quick_hint_placeholder_falls_back_without_hints():
    app, _ = quick_hint_app(())
    assert app.placeholder_text() == app.input_hint_fn()


def test_quick_hint_mode_change_resets_focus_and_picked():
    app, _ = quick_hint_app()
    app.quick_hint_focus = 2
    app.quick_hint_picked = ["run the tests"]
    app.set_running("working")
    assert app.quick_hint_focus == -1
    assert app.quick_hint_picked == []


def test_model_retry_wait_status_labels_live_phase(tmp_path):
    """The model's retry-wait hook is wired to the live phase label: the wait shows as its own
    phase ("retrying") and returns to "working" when it ends."""
    command_loop = loop(tmp_path)
    transitions = []
    command_loop.tui = SimpleNamespace(set_running=transitions.append)
    assert command_loop.agent.model.on_retry_wait == command_loop.model_retry_wait_status

    command_loop.agent.model.on_retry_wait(True)
    command_loop.agent.model.on_retry_wait(False)
    assert transitions == ["retrying", "working"]


def test_mention_opens_completions_while_typing(monkeypatch):
    """`@`, `@kind:`, and `$` name something the completer knows, so the list opens as they are
    typed and narrows as more characters arrive - everything else in this prompt is prose and
    waits for Tab."""
    app = TuiApp(
        completer=CommandCompleter(
            mcp_servers=lambda: ("github", "gitlab", "playwright"),
            skills=lambda: ("release",),
            files=lambda: (("minacode/tui.py", "minacode/tui.py"), ("minacode/hints.py", "minacode/hints.py")),
        )
    )

    def completions():
        state = app.input_buffer.complete_state
        return None if state is None else [c.text for c in state.completions]

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)

        pipe_input.send_text("use @gi")
        wait_until(lambda: completions() == ["@mcp:github", "@mcp:gitlab"])

        pipe_input.send_text("th")
        wait_until(lambda: completions() == ["@mcp:github"])  # the list narrows as typing continues

        pipe_input.send_text(" and @")
        wait_until(lambda: completions() == ["@file:", "@mcp:", "@skill:"])

        pipe_input.send_text("mcp:")
        wait_until(lambda: completions() == ["@mcp:github", "@mcp:gitlab", "@mcp:playwright"])

        pipe_input.send_text("gi")
        wait_until(lambda: completions() == ["@mcp:github", "@mcp:gitlab"])

        pipe_input.send_text(" and @file:tu")
        wait_until(lambda: completions() == ["@file:minacode/tui.py"])

        pipe_input.send_text(" and $")
        wait_until(lambda: completions() == ["@skill:release"])

        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)


def test_selecting_mention_kind_opens_its_candidate_list(monkeypatch):
    app = TuiApp(completer=CommandCompleter(skills=lambda: ("release", "review")))

    def completions():
        state = app.input_buffer.complete_state
        return None if state is None else [c.text for c in state.completions]

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("@")
        wait_until(lambda: completions() == ["@file:", "@mcp:", "@skill:"])

        # Shift-Tab selects the last namespace row. Once that selection settles, its own candidates
        # replace the parent namespace menu without another key press.
        pipe_input.send_text("\x1b[Z")
        wait_until(lambda: app.input_buffer.text == "@skill:")
        wait_until(lambda: completions() == ["@skill:release", "@skill:review"])
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)


@pytest.mark.parametrize(
    ("typed", "namespace", "expected"),
    [
        ("@m", "@mcp:", ["@mcp:github", "@mcp:gitlab"]),
        ("@sk", "@skill:", ["@skill:release", "@skill:review"]),
    ],
)
def test_selecting_partially_typed_name_kind_opens_its_candidate_list(monkeypatch, typed, namespace, expected):
    app = TuiApp(completer=CommandCompleter(mcp_servers=lambda: ("github", "gitlab"), skills=lambda: ("release", "review")))

    def completions():
        state = app.input_buffer.complete_state
        return None if state is None else [c.text for c in state.completions]

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text(typed)
        wait_until(lambda: completions() == [namespace])
        pipe_input.send_text("\t")
        wait_until(lambda: app.input_buffer.text == namespace)
        wait_until(lambda: completions() == expected)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)


def test_prose_and_email_do_not_open_completions(monkeypatch):
    """A menu on every keystroke would be noise: only a mention at the cursor opens one, and an
    address is not a mention because the `@` follows a word character."""
    app = TuiApp(completer=CommandCompleter(mcp_servers=lambda: ("github",)))
    seen = []

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("mail me at hit9@icloud")
        wait_until(lambda: app.input_buffer.text == "mail me at hit9@icloud")
        seen.append(app.input_buffer.complete_state)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert seen == [None]


def test_mention_trigger_uses_canonical_scanner_spans():
    for text in (
        "use @file:tu",
        "use @",
        "use @skill:rel",
        "use @mcp:git",
    ):
        assert active_mention(text) is not None, text
    assert active_mention("mail me at hit9@icloud") is None
    assert active_mention("use file:notes here") is None
    assert active_mention("profile:x") is None


def test_file_picker_tab_replaces_only_active_span(monkeypatch):
    app = TuiApp(file_picker_available_fn=lambda: True, file_picker_fn=lambda query: FilePick("docs/中文 notes.txt") if query == "not" else FilePick())

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("inspect @file:not please")
        for _ in range(len(" please")):
            pipe_input.send_text("\x1b[D")
        pipe_input.send_text("\t")
        wait_until(lambda: app.input_buffer.text == 'inspect @file:"docs/中文 notes.txt" please')
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)


def test_file_picker_opens_after_typing_without_tab(monkeypatch):
    queries = []
    app = TuiApp(
        file_picker_available_fn=lambda: True,
        file_picker_fn=lambda query: (queries.append(query), FilePick("minacode/tui.py"))[1],
    )

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("inspect @file:")
        wait_until(lambda: app.input_buffer.text == "inspect @file:minacode/tui.py")
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert queries == [""]


@pytest.mark.parametrize("typed", ["@f", "@fi"])
def test_selecting_partially_typed_file_kind_opens_picker(monkeypatch, typed):
    queries = []
    app = TuiApp(
        completer=CommandCompleter(),
        file_picker_available_fn=lambda: True,
        file_picker_fn=lambda query: (queries.append(query), FilePick())[1],
    )

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text(typed)
        wait_until(lambda: app.input_buffer.complete_state is not None)
        pipe_input.send_text("\t")
        wait_until(lambda: queries == [""] and not app._file_picker_active)
        assert app.input_buffer.text == "@file:"
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)


def test_pasting_file_namespace_does_not_open_picker(monkeypatch):
    queries = []
    app = TuiApp(file_picker_available_fn=lambda: True, file_picker_fn=lambda query: (queries.append(query), FilePick())[1])

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("\x1b[200~@file:\x1b[201~")
        wait_until(lambda: app.input_buffer.text == "@file:")
        time.sleep(app.MENTION_TRANSITION_DELAY * 2)
        assert queries == []
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)


def test_file_picker_cancel_keeps_buffer(monkeypatch):
    queries = []
    app = TuiApp(file_picker_available_fn=lambda: True, file_picker_fn=lambda query: (queries.append(query), FilePick())[1])

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("@file:keep")
        wait_until(lambda: queries == ["keep"] and not app._file_picker_active)
        time.sleep(app.MENTION_TRANSITION_DELAY * 2)
        assert app.input_buffer.text == "@file:keep"
        assert queries == ["keep"]  # Cancel does not reopen until the input changes again.
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)
