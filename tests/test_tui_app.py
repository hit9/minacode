"""TuiApp behavior: layout, input modes, key bindings, modals, and approval prompts."""

import asyncio
import multiprocessing
import os
import signal
import threading
import time

import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Size
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput
from tui_harness import ResizableOutput, loop, rendered_screen_text, run_interactive_tui, session, wait_until

import minacode.tui as tui_module
from minacode.base import (
    Config,
    LogBlock,
    LogEdge,
)
from minacode.engine import Agent
from minacode.loop import CommandCompleter, CommandLoop
from minacode.prompts import LIVE_FOLLOWUP_PREFIX
from minacode.session import Session, SessionSnapshotStore
from minacode.tools import CodeIndex
from minacode.tui import TUI_MODAL_PENDING, CallbackPlaceholder, TuiApp
from minacode.update import UpdateChecker


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
                "persisted_user_inputs": [message.get("content") for message in restored_session.messages if message.get("role") == "user"],
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


@pytest.mark.parametrize("draft", ["", "unfinished draft"])
def test_interactive_tui_ctrl_c_cancels_idle_input_like_master(monkeypatch, draft):
    cancelled = []
    app = TuiApp(on_input_cancel=lambda: cancelled.append(True))

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text(draft + "\x03")
        wait_until(lambda: cancelled == [True])
        assert app.input_buffer.text == ""
        pipe_input.send_text("\x04")

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert cancelled == [True]


def test_tui_ctrl_c_consumes_a_running_draft_before_interrupting(monkeypatch):
    """While the agent works, a draft absorbs the first Ctrl-C; the turn keeps running."""
    events = []
    app = TuiApp(on_interrupt=lambda: events.append("interrupt"))

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        app.set_running("working")
        pipe_input.send_text("queued draft")
        wait_until(lambda: app.input_buffer.text == "queued draft")
        pipe_input.send_text("\x03")
        wait_until(lambda: app.input_buffer.text == "")
        assert events == []
        # With the draft gone the next press interrupts.
        pipe_input.send_text("\x03")
        wait_until(lambda: events == ["interrupt"])
        app.set_idle()
        pipe_input.send_text("\x04")

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert events == ["interrupt"]


def test_tui_ctrl_c_interrupts_immediately_with_an_empty_running_input(monkeypatch):
    """The queue hint renders only on an empty buffer, so "Ctrl-C interrupts" is shown exactly
    when a single press interrupts."""
    events = []
    app = TuiApp(on_interrupt=lambda: events.append("interrupt"))

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        app.set_running("working")
        pipe_input.send_text("\x03")
        wait_until(lambda: events == ["interrupt"])
        app.set_idle()
        pipe_input.send_text("\x04")

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert events == ["interrupt"]


def test_tui_ctrl_u_clears_the_idle_draft_without_cancelling(monkeypatch):
    """Ctrl-U discards the line. Unlike Ctrl-C it carries no other meaning, so nothing is
    cancelled."""
    cancelled = []
    app = TuiApp(on_input_cancel=lambda: cancelled.append(True))

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

    assert cancelled == []


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

    assert command_loop.tui_input_hint() == "↑ recalls queued · Ctrl-C interrupts"


def test_tui_chat_input_shows_random_idle_placeholder(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()

    assert "Ctrl-X Ctrl-E opens $EDITOR" in CommandLoop.IDLE_HINTS
    hint = command_loop.tui_input_hint()
    assert hint in CommandLoop.IDLE_HINTS
    assert command_loop.tui_input_hint() == hint  # stable within a session (no flicker)


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
    assert outcome["persisted_user_inputs"] == ["queued one", "queued two"]
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
    command_loop.emit = output.append
    app = TuiApp()
    command_loop.tui = app
    result = []

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        selector = threading.Thread(
            target=lambda: result.append(command_loop.select_choice("Pick", ("a", "b"))),
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


def test_quick_hint_tab_falls_through_to_completion_with_text():
    app, _ = quick_hint_app()
    app.input_buffer.insert_text("/mod")
    app.tab_or_complete(app.input_buffer, reverse=False)
    assert app.quick_hint_focus == -1


def test_quick_hint_tab_ignored_without_hints():
    app, _ = quick_hint_app(())
    app.tab_or_complete(app.input_buffer, reverse=False)
    assert app.quick_hint_focus == -1


def test_quick_hint_enter_submits_focused_chip():
    app, submitted = quick_hint_app()
    app.quick_hint_focus = 1
    app._accept(app.input_buffer)
    assert [str(value) for value in submitted] == ["show the diff"]
    assert app.quick_hint_focus == -1


def test_quick_hint_enter_on_empty_unfocused_input_does_nothing():
    app, submitted = quick_hint_app()
    app._accept(app.input_buffer)
    assert submitted == []


def test_quick_hint_fragments_highlight_focused_chip():
    app, _ = quick_hint_app(("a", "b"))
    assert app.quick_hint_fragments() == [("class:quickhint", " a "), ("class:quickhint.sep", " │ "), ("class:quickhint", " b ")]
    app.quick_hint_focus = 0
    assert ("class:quickhint.focused", " a ") in app.quick_hint_fragments()


def test_quick_hint_placeholder_hints_keys_until_focused():
    app, _ = quick_hint_app()
    assert app.placeholder_text() == "Tab cycles suggestions \u00b7 Enter submits"
    app.quick_hint_focus = 0
    assert app.placeholder_text() == ""


def test_quick_hint_placeholder_falls_back_without_hints():
    app, _ = quick_hint_app(())
    assert app.placeholder_text() == app.input_hint_fn()


def test_quick_hint_mode_change_resets_focus():
    app, _ = quick_hint_app()
    app.quick_hint_focus = 2
    app.set_running("working")
    assert app.quick_hint_focus == -1
