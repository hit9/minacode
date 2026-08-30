"""tui input (split from tests/test_tui_app.py)."""
import asyncio
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
from tui_harness import ResizableOutput, loop, run_interactive_tui, session, wait_until

import wizolt.tui.app as tui_module
from wizolt.base import (
    SESSION_EVENT_KEY,
    LogBlock,
    LogEdge,
)
from wizolt.cli import CommandCompleter, CommandLoop, TuiRuntime
from wizolt.cli.update import UpdateChecker
from wizolt.config import (
    Config,
)
from wizolt.engine import Agent
from wizolt.session import Session, SessionSnapshotStore
from wizolt.tools import CodeIndex
from wizolt.tui import CallbackPlaceholder, TuiApp


def test_invalidate_ignores_redraw_that_loses_application_shutdown_race():
    app = TuiApp()
    prompt_app = SimpleNamespace(is_running=True)

    def invalidate():
        prompt_app.is_running = False
        raise RuntimeError("no running event loop")

    prompt_app.invalidate = invalidate
    app.app = prompt_app

    app.invalidate()


def test_invalidate_preserves_runtime_error_while_application_is_running():
    app = TuiApp()

    def invalidate():
        raise RuntimeError("redraw failed")

    app.app = SimpleNamespace(is_running=True, invalidate=invalidate)

    with pytest.raises(RuntimeError, match="redraw failed"):
        app.invalidate()


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

    assert any(f"wizolt --resume {scenario_session.uid}" in line for line in output)
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
