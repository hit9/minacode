import multiprocessing
import threading
import time
from types import SimpleNamespace

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.history import FileHistory
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.utils import get_cwidth

import minacode as n


def session(tmp_path):
    config = n.Config()
    config.data_dir = str(tmp_path / "data")
    return n.Session(cwd=str(tmp_path), config=config)


def loop(tmp_path):
    return n.CommandLoop(n.Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt="": "", output_fn=lambda text: None)


class ResizableOutput(DummyOutput):
    def __init__(self, rows=24, columns=80):
        self.size = Size(rows=rows, columns=columns)

    def get_size(self):
        return self.size


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("interactive TUI condition was not reached")


def rendered_screen_text(application, output):
    screen = application.renderer.last_rendered_screen
    if screen is None:
        return ""
    return "\n".join("".join(screen.data_buffer[row][column].char for column in range(output.size.columns)).rstrip() for row in range(output.size.rows))


def run_interactive_tui(monkeypatch, tui, *, text="", drive=None, output=None, after_render=None):
    real_application = n.Application
    output = output or DummyOutput()
    driver_errors = []
    with create_pipe_input() as pipe_input:

        def application(**kwargs):
            return real_application(input=pipe_input, after_render=after_render, **(kwargs | {"output": output}))

        monkeypatch.setattr(n.tui, "Application", application)
        if text:
            pipe_input.send_text(text)
        driver = None
        if drive is not None:

            def run_driver():
                try:
                    drive(pipe_input)
                except BaseException as error:
                    driver_errors.append(error)
                    if tui.app is not None:
                        tui.app.loop.call_soon_threadsafe(tui.app.exit)

            driver = threading.Thread(target=run_driver, daemon=True)
            driver.start()
        tui.run()
        if driver is not None:
            driver.join(timeout=1)
            assert not driver.is_alive()
    if driver_errors:
        raise driver_errors[0]


def ctrl_c_queue_scenario(cwd, results):
    config = n.Config(data_dir=cwd)
    scenario_session = n.Session(cwd=cwd, config=config)
    command_loop = n.CommandLoop(
        n.Agent(scenario_session, output_fn=lambda text: None),
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
    n.SessionSnapshotStore.clean_expired = lambda _session: 0
    n.CodeIndex.refresh_existing_async = lambda _index: False
    n.CodeIndex.update_pending_async = lambda _index: None
    n.UpdateChecker.start = lambda _checker: None
    real_application = n.Application

    try:
        with create_pipe_input() as pipe_input:
            n.tui.Application = lambda **kwargs: real_application(input=pipe_input, **(kwargs | {"output": DummyOutput()}))

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
                    command_loop.tui.input_buffer.reset(n.Document(""))
                    pipe_input.send_text("\x04")
                except BaseException as error:
                    driver_errors.append(repr(error))
                    if first_running.is_set():
                        n.os.kill(n.os.getpid(), n.signal.SIGINT)
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
        restored_session = n.Session.load_snapshot(command_loop.session.uid, config=config)
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
    except BaseException as error:
        results.put({"fatal": repr(error)})


def test_theme_palettes_have_identical_complete_keys():
    assert n.Theme.DARK.keys() == n.Theme.LIGHT.keys()
    assert all(n.Theme.DARK.values())
    assert all(n.Theme.LIGHT.values())


def test_editor_command_prefers_visual_then_editor_then_vim(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    assert n.TuiApp.editor_command() == ["vim"]

    monkeypatch.setenv("EDITOR", "code --wait")
    assert n.TuiApp.editor_command() == ["code", "--wait"]

    monkeypatch.setenv("VISUAL", "nvim")
    assert n.TuiApp.editor_command() == ["nvim"]


def test_edit_text_in_editor_roundtrips_edited_content(tmp_path, monkeypatch):
    # A fake $EDITOR that appends a marker to whatever file it is given.
    editor = tmp_path / "fake_editor.sh"
    editor.write_text('#!/bin/sh\nprintf " EDITED" >> "$1"\n')
    editor.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(editor))
    monkeypatch.delenv("VISUAL", raising=False)

    assert n.TuiApp()._edit_text_in_editor("hello") == "hello EDITED"


def test_edit_text_in_editor_leaves_input_untouched_when_editor_missing(monkeypatch):
    monkeypatch.setenv("EDITOR", "definitely-not-an-editor-binary")
    monkeypatch.delenv("VISUAL", raising=False)

    assert n.TuiApp()._edit_text_in_editor("hello") is None


def test_edit_text_in_editor_leaves_input_untouched_on_nonzero_exit(monkeypatch):
    monkeypatch.setenv("EDITOR", "false")
    monkeypatch.delenv("VISUAL", raising=False)

    assert n.TuiApp()._edit_text_in_editor("hello") is None


def test_editor_text_compose_and_strip_roundtrip():
    # The editor receives the draft plus the agent's reply below a scissors line; stripping
    # drops the reference context and returns exactly the (possibly edited) draft.
    composed, marker = n.TuiApp._compose_editor_text("my draft", "reply line one\nline two")
    assert "my draft" in composed
    assert n.TuiApp.EDITOR_CONTEXT_MARKER in composed
    assert marker and marker in composed
    assert "reply line one" in composed
    assert n.TuiApp._strip_editor_context(composed, marker) == "my draft"
    # Editing above the scissors line survives; everything below it is dropped.
    assert n.TuiApp._strip_editor_context(composed.replace("my draft", "edited draft"), marker) == "edited draft"


def test_editor_text_compose_without_context_is_identity():
    assert n.TuiApp._compose_editor_text("draft", "") == ("draft", "")
    assert n.TuiApp._compose_editor_text("draft", "   ") == ("draft", "")
    assert n.TuiApp._strip_editor_context("plain text\n", "") == "plain text"


def test_editor_strip_preserves_a_scissors_line_the_user_typed():
    # Only the marker this composition added is stripped; a scissors line already in the draft
    # (pasted Markdown or code) survives, whether or not reference context was appended.
    draft = f"before\n{n.TuiApp.EDITOR_CONTEXT_MARKER}\nafter"
    assert n.TuiApp._strip_editor_context(draft, "") == draft
    composed, marker = n.TuiApp._compose_editor_text(draft, "reply")
    assert n.TuiApp._strip_editor_context(composed, marker) == draft


def test_editor_context_returns_last_assistant_reply(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.session.messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
        {"role": "assistant", "content": None},  # a tool-call turn carries no text
    ]
    assert command_loop.editor_context() == "second answer"

    command_loop.session.messages = [{"role": "user", "content": "only a question"}]
    assert command_loop.editor_context() == ""


def test_editor_context_caps_long_replies_to_recent_lines(tmp_path):
    command_loop = loop(tmp_path)
    total = command_loop.EDITOR_CONTEXT_MAX_LINES + 50
    reply = "\n".join(f"line {index}" for index in range(total))
    command_loop.session.messages = [{"role": "assistant", "content": reply}]
    lines = command_loop.editor_context().splitlines()
    assert len(lines) == command_loop.EDITOR_CONTEXT_MAX_LINES + 1  # omission note + kept lines
    assert lines[0].startswith("# [...")
    assert lines[1] == "line 50"
    assert lines[-1] == f"line {total - 1}"


def test_tui_app_build_layout_composes_input_and_status():
    app = n.TuiApp()
    layout = app.build_layout()
    focused = layout.current_window
    assert focused is not None
    # Layout is composable and the focused element accepts typed input via app.input_buffer.
    app.input_buffer.insert_text("hi")
    assert app.input_buffer.text == "hi"


def test_tui_approval_prompt_keeps_connector_style_and_spinner(monkeypatch):
    app = n.TuiApp()
    connector = n.LogBlock.prefix(2, n.LogEdge.CONTINUE)
    app.input_mode = "approval"
    app.input_prompt = connector + "[Y/n] "
    monkeypatch.setattr(n.time, "monotonic", lambda: 0.2)

    assert app.status_fragments() == [
        ("ansibrightblack", connector),
        ("class:approval", "[Y/n] "),
        ("class:approval.wait", "/ "),
    ]


def test_tui_loading_models_prompt_is_simple_and_dim():
    app = n.TuiApp()
    app.set_dispatching("Loading models...")

    assert app.status_fragments() == [("ansibrightblack", "Loading models...")]


def test_tui_non_editing_modes_clear_stale_input_errors():
    app = n.TuiApp()
    app.input_error = "stale image error"

    app.set_dispatching("Loading models...")
    assert app.input_error_fragments() == []

    app.input_error = "another stale image error"
    app._set_mode("approval", "Continue? ")
    assert app.input_error_fragments() == []


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
    app = n.TuiApp()

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

    app = n.TuiApp(on_chat_submit=submit)
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

    app = n.TuiApp(on_running_submit=submit)
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

    app = n.TuiApp(on_chat_submit=submit)

    run_interactive_tui(monkeypatch, app, text="hello from pipe\r\x04")

    assert received == ["hello from pipe"]
    assert app.app is None


@pytest.mark.parametrize("draft", ["", "unfinished draft"])
def test_interactive_tui_ctrl_c_cancels_idle_input_like_master(monkeypatch, draft):
    cancelled = []
    app = n.TuiApp(on_input_cancel=lambda: cancelled.append(True))

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
    app = n.TuiApp(on_interrupt=lambda: events.append("interrupt"))

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
    app = n.TuiApp(on_interrupt=lambda: events.append("interrupt"))

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
    app = n.TuiApp(on_input_cancel=lambda: cancelled.append(True))

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
    app = n.TuiApp(on_interrupt=lambda: interrupted.append(True))

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
    command_loop = n.CommandLoop(
        n.Agent(scenario_session, output_fn=output.append),
        input_fn=lambda prompt="": "",
        output_fn=output.append,
    )
    monkeypatch.setattr(n.SessionSnapshotStore, "clean_expired", lambda _session: 0)
    monkeypatch.setattr(n.CodeIndex, "refresh_existing_async", lambda _index: False)
    monkeypatch.setattr(n.UpdateChecker, "start", lambda _checker: None)
    real_application = n.Application
    full_screen_modes = []
    tui_daemon = []

    with create_pipe_input() as pipe_input:

        def application(**kwargs):
            full_screen_modes.append(kwargs["full_screen"])
            return real_application(input=pipe_input, **(kwargs | {"output": DummyOutput()}))

        monkeypatch.setattr(n.tui, "Application", application)

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


def test_tui_emits_resumed_history_after_primary_screen_starts(tmp_path, monkeypatch):
    scenario_session = session(tmp_path)
    scenario_session.resumed = True
    scenario_session.messages.extend(
        [
            {"role": "user", "content": "restored question"},
            {"role": "assistant", "content": "restored answer"},
        ]
    )
    command_loop = n.CommandLoop(
        n.Agent(scenario_session, output_fn=lambda _text: None),
        input_fn=lambda prompt="": "",
        output_fn=lambda _text: None,
    )
    command_loop.ui.color = True
    monkeypatch.setattr(n.SessionSnapshotStore, "clean_expired", lambda _session: 0)
    monkeypatch.setattr(n.CodeIndex, "refresh_existing_async", lambda _index: False)
    monkeypatch.setattr(n.UpdateChecker, "start", lambda _checker: None)
    real_application = n.Application
    emitted_while_running = []
    history_emitted = threading.Event()

    def print_formatted(value, *args, **kwargs):
        text = fragment_list_to_text(to_formatted_text(value))
        if "restored answer" in text:
            emitted_while_running.append(command_loop.tui is not None and command_loop.tui.app is not None and command_loop.tui.app.is_running)
            history_emitted.set()

    monkeypatch.setattr(n.render, "print_formatted_text", print_formatted)

    with create_pipe_input() as pipe_input:
        monkeypatch.setattr(n.tui, "Application", lambda **kwargs: real_application(input=pipe_input, **(kwargs | {"output": DummyOutput()})))

        def drive():
            assert history_emitted.wait(timeout=1)
            pipe_input.send_text("\x04")

        driver = threading.Thread(target=drive, daemon=True)
        driver.start()
        assert command_loop.run_tui() == 0
        driver.join(timeout=1)

    assert not driver.is_alive()
    assert emitted_while_running == [True]


@pytest.mark.parametrize("entered", [" /help", "exit "])
def test_tui_runtime_strips_input_before_command_dispatch(tmp_path, entered):
    command_loop = loop(tmp_path)
    dispatched = []
    command_loop.command = lambda text: dispatched.append(text) or (True, False)
    command_loop.tui = n.TuiApp()
    runtime = n.TuiRuntime(command_loop)

    assert runtime.dispatch(entered)
    assert dispatched == [entered.strip()]


def test_tui_runtime_keeps_space_around_user_input_before_working(tmp_path, monkeypatch):
    output = []
    scenario_session = session(tmp_path)
    command_loop = n.CommandLoop(
        n.Agent(scenario_session, output_fn=output.append),
        input_fn=lambda prompt="": "",
        output_fn=output.append,
    )
    runtime = n.TuiRuntime(command_loop)
    command_loop.tui = n.TuiApp()
    command_loop.tui.set_running = lambda label: output.append("set_running:" + label)
    command_loop.command = lambda _text: (False, False)
    command_loop.agent.run = lambda _text: "done"
    monkeypatch.setattr(n.CodeIndex, "update_pending_async", lambda _index: None)

    assert not runtime.dispatch("answer me")
    runtime.run_agent_turn("answer me")

    assert output[:3] == ["\n• answer me", "", "set_running:working"]


def test_tui_runtime_clears_thinking_before_cancelled_output(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.tui = n.TuiApp()
    runtime = n.TuiRuntime(command_loop)
    emitted = []

    def interrupt(_user_input):
        command_loop.model_stream_output("reasoning", "private reasoning")
        raise KeyboardInterrupt

    command_loop.agent.run = interrupt
    command_loop.emit = lambda text: emitted.append((text, command_loop.model_stream_fragments()))
    monkeypatch.setattr(n.CodeIndex, "update_pending_async", lambda _index: None)

    runtime.run_agent_turn("question")

    assert emitted[-1] == ("Cancelled", [])


def test_resumed_tui_auto_dispatches_persisted_queue_as_one_request(tmp_path, monkeypatch):
    saved = session(tmp_path)
    saved.enqueue_user_input("queued one")
    saved.enqueue_user_input("queued two")
    saved.save_snapshot()
    restored = n.Session.load_snapshot(saved.uid, config=saved.config)
    command_loop = n.CommandLoop(
        n.Agent(restored, output_fn=lambda _text: None),
        input_fn=lambda prompt="": "",
        output_fn=lambda _text: None,
    )
    requests = []

    class RecordingModel:
        def request(self, messages, tools=None):
            requests.append([message.get("content") for message in messages if message.get("role") == "user"])
            return {"role": "assistant", "content": "done"}, [], "done"

        def cancel(self):
            pass

    command_loop.agent.model = RecordingModel()
    monkeypatch.setattr(n.SessionSnapshotStore, "clean_expired", lambda _session: 0)
    monkeypatch.setattr(n.CodeIndex, "refresh_existing_async", lambda _index: False)
    monkeypatch.setattr(n.CodeIndex, "update_pending_async", lambda _index: None)
    monkeypatch.setattr(n.UpdateChecker, "start", lambda _checker: None)
    real_application = n.Application

    with create_pipe_input() as pipe_input:
        monkeypatch.setattr(n.tui, "Application", lambda **kwargs: real_application(input=pipe_input, **(kwargs | {"output": DummyOutput()})))

        def drive():
            wait_until(lambda: command_loop.tui is not None and command_loop.tui.app is not None and command_loop.tui.app.is_running)
            wait_until(lambda: len(requests) == 1)
            wait_until(lambda: command_loop.tui.input_mode == "chat")
            pipe_input.send_text("\x04")

        driver = threading.Thread(target=drive, daemon=True)
        driver.start()
        assert command_loop.run_tui() == 0
        driver.join(timeout=1)

    assert len(requests) == 1
    assert "queued one" in requests[0]
    marked_followup = n.Agent.LIVE_FOLLOWUP_PREFIX + "queued two"
    assert marked_followup in requests[0]
    assert requests[0].index("queued one") < requests[0].index(marked_followup)
    assert restored.pending_user_inputs == []


def test_processed_queued_message_does_not_return_to_input(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    first_request = threading.Event()
    release_first = threading.Event()
    requests = []

    class RecordingModel:
        def request(self, messages, tools=None):
            requests.append([message.get("content") for message in messages if message.get("role") == "user"])
            if len(requests) == 1:
                first_request.set()
                assert release_first.wait(timeout=1)
            return {"role": "assistant", "content": "done"}, [], "done"

        def cancel(self):
            pass

    command_loop.agent.model = RecordingModel()
    monkeypatch.setattr(n.SessionSnapshotStore, "clean_expired", lambda _session: 0)
    monkeypatch.setattr(n.CodeIndex, "refresh_existing_async", lambda _index: False)
    monkeypatch.setattr(n.CodeIndex, "update_pending_async", lambda _index: None)
    monkeypatch.setattr(n.UpdateChecker, "start", lambda _checker: None)
    real_application = n.Application

    with create_pipe_input() as pipe_input:
        monkeypatch.setattr(n.tui, "Application", lambda **kwargs: real_application(input=pipe_input, **(kwargs | {"output": DummyOutput()})))

        def drive():
            wait_until(lambda: command_loop.tui is not None and command_loop.tui.app is not None and command_loop.tui.app.is_running)
            pipe_input.send_text("first task\r")
            assert first_request.wait(timeout=1)
            pipe_input.send_text("queued task\r")
            wait_until(lambda: [item.text for item in command_loop.session.pending_user_inputs] == ["queued task"])
            release_first.set()
            wait_until(lambda: len(requests) == 2)
            wait_until(lambda: command_loop.tui.input_mode == "chat")
            assert command_loop.tui.input_buffer.text == ""
            pipe_input.send_text("\x04")

        driver = threading.Thread(target=drive, daemon=True)
        driver.start()
        assert command_loop.run_tui() == 0
        driver.join(timeout=1)

    assert not driver.is_alive()
    assert "queued task" in requests[1]


def test_interactive_tui_control_backslash_forces_exit(monkeypatch):
    forced = []
    app = None

    def force_exit():
        forced.append(True)
        app.app.exit()

    app = n.TuiApp(on_force_exit=force_exit)

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

    app = n.TuiApp(on_running_submit=submit, on_recall=recall)
    app.set_running("working")

    run_interactive_tui(monkeypatch, app, text="\x1b[A\r\x04")

    assert recalled == [True]
    assert received == ["edit queued message"]


@pytest.mark.parametrize("history_key", ["\x10", "\x1b[A"])
def test_interactive_tui_history_keys_recall_when_queue_is_empty(monkeypatch, tmp_path, history_key):
    received = []
    recalled = []
    app = n.TuiApp(
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
    app = n.TuiApp(completer=n.CommandCompleter())

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("/pro\t")
        wait_until(lambda: app.input_buffer.text == "/provider")
        assert app.input_buffer.complete_state is None
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert app.input_buffer.text == "/provider"


def test_interactive_tui_bracketed_paste_displays_all_lines(monkeypatch):
    app = n.TuiApp()
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
    app = n.TuiApp()
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
    app = n.TuiApp(activity_fragments_fn=lambda: [("", "working\n+ queued")])
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
    app = n.TuiApp()
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
    app = n.TuiApp(on_running_submit=received.append)
    app.set_running("working")
    app.input_buffer.insert_text("first\nsecond\nthird")

    app.input_buffer.validate_and_handle()

    assert received == ["first\nsecond\nthird"]
    assert app.input_buffer.text == ""


def test_tui_running_input_drops_whitespace_only_draft():
    received: list[str] = []
    app = n.TuiApp(on_running_submit=received.append)
    app.set_running("working")
    app.input_buffer.insert_text("  \n ")

    app.input_buffer.validate_and_handle()

    assert received == []
    assert app.input_buffer.text == ""


def test_tui_running_input_shows_contextual_placeholder():
    hint = {"text": "Enter queues follow-up"}
    placeholder = n.CallbackPlaceholder(lambda: hint["text"])

    def transform(text):
        document = n.Document(text)
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
    command_loop.tui = n.TuiApp()
    command_loop.tui.set_running("working")
    command_loop.session.enqueue_user_input("queued")

    assert command_loop.tui_input_hint() == "↑ recalls queued · Ctrl-C interrupts"


def test_tui_chat_input_shows_random_idle_placeholder(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = n.TuiApp()

    assert "Ctrl-X Ctrl-E opens $EDITOR" in n.CommandLoop.IDLE_HINTS
    hint = command_loop.tui_input_hint()
    assert hint in n.CommandLoop.IDLE_HINTS
    assert command_loop.tui_input_hint() == hint  # stable within a session (no flicker)


def test_tui_sigint_interrupts_dispatch_and_running_modes():
    interrupted = []
    app = n.TuiApp(on_interrupt=lambda: interrupted.append(True))
    bindings = app.make_bindings()
    handler = next(binding.handler for binding in bindings.bindings if binding.keys == (n.Keys.SIGINT,))
    event = type("Event", (), {})()

    app.set_dispatching()
    handler(event)
    app.set_running("working")
    handler(event)

    assert interrupted == [True, True]


def test_tui_ctrl_o_opens_latest_bash_output():
    expanded = []
    app = n.TuiApp(on_expand_output=lambda: expanded.append(True))
    binding = next(binding for binding in app.make_bindings().bindings if binding.keys == (n.Keys.ControlO,) and binding.filter())

    binding.handler(type("Event", (), {})())

    assert expanded == [True]


@pytest.mark.parametrize("mode", ["chat", "running"])
def test_tui_ctrl_d_deletes_at_cursor_when_input_is_nonempty(mode):
    app = n.TuiApp()
    app.input_buffer.reset(n.Document("abc", cursor_position=1))
    app.input_mode = mode
    binding = next(binding for binding in reversed(app.make_bindings().bindings) if binding.keys == (n.Keys.ControlD,) and binding.filter())
    event = type("Event", (), {"app": type("Application", (), {"exit": lambda self: None})()})()

    binding.handler(event)

    assert app.input_buffer.text == "ac"


def test_tui_ctrl_d_submits_multiline_approval_input():
    app = n.TuiApp()
    pending = threading.Event()
    app.input_mode = "approval"
    app._input_pending = pending
    app.input_buffer.reset(n.Document("first\nsecond"))
    binding = next(binding for binding in reversed(app.make_bindings().bindings) if binding.keys == (n.Keys.ControlD,) and binding.filter())
    event = type("Event", (), {"app": type("Application", (), {"exit": lambda self: None})()})()

    binding.handler(event)

    assert pending.is_set()
    assert app._input_result == "first\nsecond"


def test_resend_command_only_resends_while_running(tmp_path):
    command_loop = loop(tmp_path)
    retried = []
    command_loop.tui = n.TuiApp(on_retry=lambda: retried.append(True))

    # Reachable from the running follow-up input (queue region), not just the idle prompt.
    assert "/resend" in n.CommandLoop.QUEUE_RUN_COMMANDS

    # Idle chat: no-op with guidance.
    command_loop.tui.set_idle()
    command_loop.command("/resend")
    assert retried == []

    # Running but no model call in flight: still a no-op.
    command_loop.tui.set_running("working")
    command_loop.session.state.current_model_call_started_at = 0.0
    command_loop.command("/resend")
    assert retried == []

    # Running with a model call in flight: resends via on_retry.
    command_loop.session.state.current_model_call_started_at = 1.0
    command_loop.command("/resend")
    assert retried == [True]


def test_manual_resend_uses_transient_retry_status(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.tui = n.TuiApp()
    command_loop.tui.set_running("working")
    command_loop.session.state.current_model_call_started_at = 1.0
    runtime = n.TuiRuntime(command_loop)
    monkeypatch.setattr(runtime, "_interrupt_active", lambda _cancel: None)

    runtime._request_model_retry("working")

    assert command_loop.tui.status_label == "working"
    assert command_loop.session.state.manual_model_retry_requested is True
    assert command_loop.session.state.model_retry_count == 1


def test_retry_divider_keeps_pulse_and_elapsed_then_returns_to_working(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.tui = n.TuiApp()
    command_loop.tui.set_running("working")
    command_loop.status_bar.started_at = 90.0
    command_loop.session.state.current_model_call_started_at = 99.0
    now = [100.0]
    monkeypatch.setattr(n.time, "monotonic", lambda: now[0])

    command_loop.session.state.current_model_attempt = 2
    command_loop.session.state.model_retry_reason = "timeout"
    command_loop.session.state.model_retry_count += 1
    retrying = command_loop.queue_divider_fragments()
    retrying_text = "".join(text for _style, text in retrying)
    assert "retrying 2/6 · timeout (10s)" in retrying_text
    assert any(text == "● " for _style, text in retrying)
    assert ("retrying 2/6 · timeout", "warn") in command_loop.status_bar.entries(show_elapsed=True)

    now[0] = 102.1
    working = command_loop.queue_divider_fragments()
    working_text = "".join(text for _style, text in working)
    assert "working · attempt 2/6 (12s)" in working_text
    assert "retrying" not in working_text
    assert any(text == "● " for _style, text in working)
    assert ("attempt 2/6", "warn") in command_loop.status_bar.entries(show_elapsed=True)

    command_loop.session.state.current_model_call_started_at = 0.0
    assert all(text != "● " for _style, text in command_loop.queue_divider_fragments())


def test_tui_ctrl_g_and_ctrl_x_ctrl_e_open_editor():
    opened = []
    app = n.TuiApp()
    app.edit_input_in_editor = lambda: opened.append(True)
    bindings = app.make_bindings()
    event = type("Event", (), {})()

    # A fresh TuiApp is in chat mode, where the editor bindings are active.
    for keys in ((n.Keys.ControlG,), (n.Keys.ControlX, n.Keys.ControlE)):
        binding = next(binding for binding in bindings.bindings if binding.keys == keys)
        assert binding.filter()
        binding.handler(event)

    assert opened == [True, True]


def test_tui_activity_uses_transient_cancelling_status(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = n.TuiApp()
    command_loop.tui.set_running("cancelling")

    text = "".join(fragment for _style, fragment in command_loop.queue_divider_fragments())

    assert "cancelling" in text
    assert "working" not in text


@pytest.mark.parametrize("recall_key", [(n.Keys.Up,), (n.Keys.ControlP,)])
def test_tui_running_recall_removes_latest_pending_message(recall_key):
    pending = ["first", "second"]

    def recall():
        return pending.pop() if pending else ""

    app = n.TuiApp(on_recall=recall)
    app.set_running("working")
    bindings = app.make_bindings()
    event = type("Event", (), {"current_buffer": app.input_buffer})()
    handler = next(binding.handler for binding in reversed(bindings.bindings) if binding.keys == recall_key and binding.filter())

    handler(event)

    assert pending == ["first"]
    assert app.input_buffer.text == "second"


def test_interactive_tui_modal_uses_real_j_and_enter_keys(monkeypatch):
    app = n.TuiApp()
    selected = {"index": 0}
    result = []

    def key(key, _data):
        if key == "j":
            selected["index"] = 1
            return n.TUI_MODAL_PENDING
        if key == "enter":
            return selected["index"]
        return n.TUI_MODAL_PENDING

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
    app = n.TuiApp()
    output = ResizableOutput()
    result = []
    rendered = threading.Event()

    def fragments():
        return [("", "\n".join(f"choice {index}" for index in range(40)))]

    def key(key, _data):
        return None if key == "q" else n.TUI_MODAL_PENDING

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
    app = n.TuiApp(status_fragments_fn=lambda: [("", "status marker")])
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
            target=lambda: app.show_modal(lambda: [("", "modal marker")], lambda key, _data: None if key == "q" else n.TUI_MODAL_PENDING, exclusive=exclusive),
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
    marked_followup = n.Agent.LIVE_FOLLOWUP_PREFIX + "queued two"
    assert marked_followup in queued_request
    assert queued_request.index("queued one") < queued_request.index(marked_followup)
    assert outcome["draft_after_ctrl_c"] == [""]
    # The interrupted first turn produced no output, so it is retracted: "long request" leaves no
    # trace in the persisted conversation, while the queued follow-ups become the next turn.
    assert outcome["persisted_user_inputs"] == ["queued one", "queued two"]
    assert outcome["restored_queue"] == []


def test_tui_app_approval_mode_resolves_bridge_event():
    import threading as _threading

    app = n.TuiApp()
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
    app = n.TuiApp()
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

    app = n.TuiApp(on_chat_submit=submit)

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        waiter = threading.Thread(
            target=lambda: app.show_modal(lambda: [("", "selector")], lambda _key, _data: n.TUI_MODAL_PENDING),
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
    app = n.TuiApp()
    selected = []
    approved = []

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        selector = threading.Thread(
            target=lambda: selected.append(
                app.show_modal(
                    lambda: [("", "selector")],
                    lambda key, _data: "chosen" if key == "enter" else n.TUI_MODAL_PENDING,
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
    app = n.TuiApp()
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


def test_resume_history_prints_before_tui_starts(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.session.resumed = True
    command_loop.session.messages.extend(
        [
            {"role": "user", "content": "most recent question"},
            {"role": "assistant", "content": "most recent answer"},
        ]
    )
    command_loop.ui.color = True
    printed = []
    monkeypatch.setattr(n.render, "print_formatted_text", lambda value, *args, **kwargs: printed.append(fragment_list_to_text(to_formatted_text(value))))

    command_loop.render_resumed_session()

    text = "".join(printed)
    assert "most recent question" in text
    assert "most recent answer" in text


def test_desert_user_color_does_not_leak_into_default_ui_style(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    for mode, expected in (("dark", "#e0a96d"), ("light", "#9a5b2e")):
        monkeypatch.setattr(n.Theme, "_mode", mode)
        assert n.UiPrinter.user_log_style() == expected
        assert command_loop.style().get_attrs_for_style_str("").color == ""


def test_tui_commands_print_output_immediately(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.ui.color = True
    monkeypatch.setattr(command_loop, "status", lambda _args: "status marker")
    printed = []
    monkeypatch.setattr(n.render, "print_formatted_text", lambda value, *args, **kwargs: printed.append(fragment_list_to_text(to_formatted_text(value))))

    assert command_loop.command("/help") == (True, False)
    assert command_loop.command("/status") == (True, False)
    assert command_loop.command("/skills") == (True, False)

    assert len(printed) == 3
    text = "".join(printed)
    assert "/provider" in text
    assert "status marker" in text
    assert "No skills installed" in text


def test_background_output_is_closed_before_final_output(tmp_path):
    command_loop = loop(tmp_path)
    emitted = []
    command_loop.emit = emitted.append

    command_loop.close_background_output(lambda: emitted.append("final"))
    command_loop.emit_background("late worker output")

    assert emitted == ["final"]


def test_tool_labels_keep_legacy_green_style():
    assert n.UiPrinter.LOG_STYLES[n.LogRole.TOOL][0] == "ansigreen"


@pytest.mark.parametrize(("mode", "rgb"), [("dark", "224;169;109"), ("light", "154;91;46")])
def test_resumed_user_rendering_emits_desert_truecolor(mode, rgb, monkeypatch):
    monkeypatch.setattr(n.Theme, "_mode", mode)
    ui = n.UiPrinter(output_fn=lambda text: None)
    console = n.Console(force_terminal=True, color_system="truecolor", no_color=False, width=40)

    with console.capture() as capture:
        ui.render_message(console, "hello", "user", False, 0)

    assert f"\x1b[38;2;{rgb}m• hello\x1b[0m" in capture.get()


@pytest.mark.parametrize(
    ("configured", "colorfgbg", "expected"),
    [
        ("dark", "0;15", "dark"),
        ("light", "15;0", "light"),
        ("auto", "15;0", "dark"),
        ("auto", "0;7", "light"),
        ("auto", "7;8", "dark"),
        ("auto", "0;;15", "light"),
        ("auto", "invalid", "dark"),
    ],
)
def test_theme_resolution(configured, colorfgbg, expected, monkeypatch):
    monkeypatch.setenv("COLORFGBG", colorfgbg)
    assert n.Theme.resolve(configured) == expected


def test_tool_argument_rendering_tracks_theme_without_changing_text(monkeypatch):
    line = n.LogLine("Search", '"needle" path=src 0:20', n.LogRole.TOOL, syntax="tool-args")
    block = n.LogBlock([line])
    rendered = []

    for mode in ("dark", "light"):
        monkeypatch.setattr(n.Theme, "_mode", mode)
        segments = n.UiPrinter(output_fn=lambda text: None).log_segments(block)
        rendered.append(("".join(text for _style, text in segments), {style for style, text in segments if text.strip()}))

    assert rendered[0][0] == rendered[1][0] == '  Search  "needle" path=src 0:20\n'
    assert rendered[0][1] != rendered[1][1]


def test_interactive_renderer_keeps_theme_when_parent_exports_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(n.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(n.Theme, "_mode", "dark")
    emitted = []
    monkeypatch.setattr(n.render, "print_formatted_text", lambda value, **_kwargs: emitted.extend(to_formatted_text(value)))

    ui = n.UiPrinter()
    # Interactive TTY output stays colored regardless of NO_COLOR — minacode owns its theming and
    # renders through prompt_toolkit's ANSI path, so the parent env var is not honored.
    assert ui.color
    ui.emit_answer("sent message", role="user", rule=False)

    desert_text = "".join(text for style, text in emitted if style == "#e0a96d")
    assert "• sent message" in desert_text


def test_editor_and_queued_user_text_use_desert_style(tmp_path, monkeypatch):
    monkeypatch.setattr(n.Theme, "_mode", "dark")
    expected = n.UiPrinter.user_log_style()
    app = n.TuiApp()
    app.build_layout()
    assert app.input_window.style == expected

    command_loop = loop(tmp_path)
    command_loop.session.enqueue_user_input("queued message")
    assert any(style == expected and "queued message" in text for style, text in command_loop.queue_region_fragments())


@pytest.mark.parametrize("width", [20, 40, 80])
def test_styled_wrapping_respects_terminal_width_for_unicode(width):
    prefix = [("", "  Read  ")]
    continuation = [("", "        ")]
    content_text = "路径/非常长/🙂/é/模块/filename.py:123"
    rows = n.Text.wrap_styled(prefix, continuation, [("fg:default", content_text)], width)

    assert "".join(text for _style, text in rows[0]).startswith("  Read  ")
    assert all(sum(get_cwidth(text) for _style, text in row) <= width for row in rows)
    assert "".join(text for row in rows for _style, text in row).replace("  Read  ", "", 1).replace("        ", "") == content_text


class ModalHarness:
    def __init__(self, keys):
        self.keys = keys
        self.frames = []
        self.exclusive = []

    def show_modal(self, fragments_fn, key_fn, *, exclusive=False):
        self.exclusive.append(exclusive)
        self.frames.append(fragments_fn())
        result = n.TUI_MODAL_PENDING
        for key in self.keys:
            result = key_fn(key, key if len(key) == 1 else "")
            self.frames.append(fragments_fn())
            if result is not n.TUI_MODAL_PENDING:
                return result
        return None


def test_bash_output_viewer_browses_latest_ten_bounded_previews(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    for index in range(12):
        stdout = "\n".join(f"line {line}" for line in range(40)) if index == 10 else f"output {index}"
        stderr = "detail stderr" if index == 10 else ""
        command_loop.session.store_tool_result("Bash", [f"printf command-{index}"], n.Tool.process_result("BashToolResult", 0, stdout, stderr))
    command_loop.session.store_tool_result("Bash", ["true"], n.Tool.process_result("BashToolResult", 0, "", ""))
    modal = ModalHarness(["j", "enter", "escape", "G", "enter", "c-o"])
    command_loop.tui = modal

    # ``shutil`` is a shared module object also used by pytest's terminal reporter. Restore the
    # patch before pytest reports this test result, rather than waiting for fixture teardown.
    with monkeypatch.context() as patch:
        patch.setattr(n.shutil, "get_terminal_size", lambda fallback=(80, 24): n.os.terminal_size((50, 20)))
        command_loop.bash_output_viewer()

    listing = "".join(value for _style, value in modal.frames[0])
    assert listing.startswith("\n──── Bash outputs · latest 10 ")
    assert get_cwidth(listing.splitlines()[1]) == 48
    assert "command-11" in listing and "command-2" in listing
    assert "Bash printf command-1\n" not in listing and "Bash printf command-0\n" not in listing and "Bash true" not in listing
    second_detail = "".join(value for _style, value in modal.frames[2])
    assert second_detail.startswith("\n──── Bash output · tr.11 ")
    assert get_cwidth(second_detail.splitlines()[1]) == 48
    assert "command-10" in second_detail
    assert "line 0" in second_detail and "line 39" in second_detail
    assert "... 16 lines omitted ..." in second_detail
    assert "detail stderr" in second_detail
    assert "──── Bash outputs · latest 10 " in "".join(value for _style, value in modal.frames[3])
    oldest_detail = "".join(value for _style, value in modal.frames[5])
    assert "command-2" in oldest_detail and "output 2" in oldest_detail
    assert modal.exclusive == [False]


def test_bash_output_viewer_is_noop_without_stored_bash_output(tmp_path):
    command_loop = loop(tmp_path)
    modal = ModalHarness([])
    command_loop.tui = modal

    command_loop.bash_output_viewer()

    assert modal.frames == []


def test_bash_output_viewer_reads_resumed_history(tmp_path):
    saved = session(tmp_path)
    saved.store_tool_result("Bash", ["printf persisted"], n.Tool.process_result("BashToolResult", 0, "persisted output", ""))
    saved.save_snapshot()
    restored = n.Session.load_snapshot(saved.uid, config=saved.config)
    command_loop = n.CommandLoop(n.Agent(restored, output_fn=lambda _text: None), input_fn=lambda prompt="": "", output_fn=lambda _text: None)
    modal = ModalHarness(["enter", "q"])
    command_loop.tui = modal

    command_loop.bash_output_viewer()

    detail = "".join(value for _style, value in modal.frames[1])
    assert "Bash printf persisted" in detail
    assert "persisted output" in detail


def test_choice_navigation_uses_shared_modal_protocol(tmp_path):
    command_loop = loop(tmp_path)
    modal = ModalHarness(["j", "enter"])
    command_loop.tui = modal
    result = command_loop.choice_application("Pick", ("a", "b", "c"), {"a": "Alpha", "b": "Beta", "c": "Gamma"}, "", set())

    assert result == "b"
    assert "Beta" in "".join(text for frame in modal.frames for _style, text in frame)


def test_provider_selection_chains_provider_model_api_and_reasoning(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["other"] = n.ProviderConfig(model="model-b", available_models=("model-b",), reasoning="low")
    selected = iter(["other", "model-b", "responses", "high"])
    titles = []

    def select(title, *_args, **_kwargs):
        titles.append(title)
        return next(selected)

    command_loop.select_choice = select
    discovered = []
    command_loop.remote_models = lambda provider: discovered.append(provider.model) or ()

    result = command_loop.provider("")

    assert titles == ["Provider", "Model", "Request API", "Reasoning effort"]
    assert command_loop.session.config.active_provider == "other"
    assert command_loop.session.config.provider.model == "model-b"
    assert command_loop.session.config.provider.api == "responses"
    assert command_loop.session.config.provider.reasoning == "high"
    assert discovered == ["model-b"]
    assert "Set provider.model = model-b" in result
    assert "Set provider.api = responses (wire: responses)" in result


def test_provider_and_model_commands_validate_direct_arguments(tmp_path):
    command_loop = loop(tmp_path)

    assert command_loop.provider("one two") == "Usage: /provider [NAME]"
    assert command_loop.provider("missing") == "Unknown provider: missing"
    assert command_loop.model("one two") == "Usage: /model [MODEL]"


def test_reason_strict_and_set_commands_validate_values(tmp_path):
    from prompt_toolkit.document import Document

    command_loop = loop(tmp_path)

    assert command_loop.reason("invalid").startswith("Usage: /reason ")
    assert command_loop.strict("on") == "Usage: /strict"
    assert command_loop.set_value("") == "Usage: /set KEY VALUE"
    assert command_loop.set_value("unknown value") == "Unknown config key: unknown"
    assert command_loop.set_value("provider.timeout never") == "Invalid value for provider.timeout"
    assert command_loop.set_value("provider.temperature off") == "Set provider.temperature"
    assert command_loop.session.config.provider.temperature is None
    assert command_loop.set_value("provider.stream maybe") == "Invalid value for provider.stream"
    assert command_loop.set_value("provider.stream off") == "Set provider.stream"
    assert command_loop.session.config.provider.stream is False
    stream_values = [item.text for item in n.CommandCompleter().get_completions(Document("/set provider.stream "), None)]
    assert stream_values == ["on", "off"]
    assert command_loop.set_value("provider.image_input maybe") == "Invalid value for provider.image_input"
    assert command_loop.set_value("provider.image_input off") == "Set provider.image_input"
    assert command_loop.session.config.provider.image_input == "off"


def test_api_command_switches_the_request_wire_and_names_what_took_effect(tmp_path):
    # A model chosen with /model may not be served over the provider's configured protocol, so the
    # wire has to be switchable in-session rather than only in the config file.
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider
    provider.url = "https://example.com/compatible-mode/v1"
    provider.api = "responses"

    assert command_loop.api("grpc").startswith("Usage: /api ")
    assert provider.resolve().api == "responses"
    assert command_loop.api("chat") == "Set provider.api = chat (wire: chat)"
    assert provider.resolve().api == "chat"
    # "auto" reports the wire it inferred rather than echoing "auto" back.
    assert command_loop.api("auto") == "Set provider.api = auto (wire: chat)"

    provider.url = "https://example.com/v1/responses"
    assert command_loop.api("auto") == "Set provider.api = auto (wire: responses)"


def test_api_command_selection_offers_every_protocol_with_the_inferred_wire(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    provider = command_loop.session.config.provider
    provider.url = "https://example.com/v1/responses"
    provider.api = "chat"
    shown = {}

    def choose(title, choices, labels, current, _disabled):
        shown.update(title=title, choices=choices, labels=labels, current=current)
        return "auto"

    command_loop.choice_application = choose

    assert command_loop.api("") == "Set provider.api = auto (wire: responses)"
    assert shown["title"] == "Request API"
    assert shown["choices"] == n.PROVIDER_API_CHOICES
    assert shown["current"] == "chat"
    assert shown["labels"]["auto"] == "auto - infer from the endpoint URL and model (responses)"
    assert shown["labels"]["chat"] == "chat (current)"


def test_api_is_registered_like_reason_and_completes_its_choices(tmp_path):
    from prompt_toolkit.document import Document

    command_loop = loop(tmp_path)

    assert "/api" in n.CommandLoop.COMMANDS
    command_loop.command("/api anthropic")
    assert command_loop.session.config.provider.api == "anthropic"

    texts = [c.text for c in n.CommandCompleter().get_completions(Document("/api "), None)]
    assert set(texts) == set(n.PROVIDER_API_CHOICES)
    # The wire is a command, not a /set key, so it must not be reachable both ways.
    assert "provider.api" not in n.CommandCompleter.SET_KEYS
    assert command_loop.set_value("provider.api chat") == "Unknown config key: provider.api"


def test_model_chain_steps_back_from_the_wire_to_the_model_and_from_reasoning_to_the_wire(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    provider = command_loop.session.config.provider
    provider.available_models = ("model-a", "model-b")
    scripted = iter(
        [
            ("Model", "model-a"),
            ("Request API", n.SELECTION_BACK),  # back lands on the model picker again
            ("Model", "model-a"),
            ("Request API", "chat"),
            ("Reasoning effort", n.SELECTION_BACK),  # back lands on the wire, not the model
            ("Request API", "responses"),
            ("Reasoning effort", "high"),
        ]
    )
    titles = []

    def select(title, *_args, **_kwargs):
        expected_title, value = next(scripted)
        assert title == expected_title
        titles.append(title)
        return value

    command_loop.select_choice = select
    command_loop.remote_models = lambda _provider: ()

    result = command_loop.model("")

    assert titles == ["Model", "Request API", "Model", "Request API", "Reasoning effort", "Request API", "Reasoning effort"]
    assert provider.model == "model-a"
    assert provider.api == "responses"
    assert provider.reasoning == "high"
    assert "Set provider.api = responses (wire: responses)" in result


def test_model_chain_leaves_the_wire_alone_when_selection_is_unavailable(tmp_path):
    # Non-interactive input returns None from every picker; the model still applies, the wire is untouched.
    command_loop = loop(tmp_path)
    command_loop.interactive_input = False
    provider = command_loop.session.config.provider
    provider.api = "responses"
    provider.reasoning = "low"

    result = command_loop.set_model("model-a")

    assert result == "Set provider.model = model-a"
    assert provider.model == "model-a"
    assert provider.api == "responses"
    assert provider.reasoning == "low"


def test_remote_models_normalizes_sdk_results(monkeypatch, tmp_path):
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider
    provider.url = "https://example.com/v1"
    provider.key = "secret"
    calls = []

    class Models:
        def list(self):
            return SimpleNamespace(data=[{"id": "zeta"}, SimpleNamespace(id="alpha"), {"id": "zeta"}, {"missing": True}, None])

    def openai(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(models=Models())

    monkeypatch.setattr(n.loop, "OpenAI", openai)

    assert command_loop.remote_models(provider) == ("alpha", "zeta")
    assert calls[0]["api_key"] == "secret"
    assert calls[0]["max_retries"] == 0


def test_remote_models_is_optional_and_failure_safe(monkeypatch, tmp_path):
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider

    assert command_loop.remote_models(provider) == ()

    provider.url = "https://example.com/v1"
    provider.key = "secret"
    monkeypatch.setattr(n.loop, "OpenAI", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")))

    assert command_loop.remote_models(provider) == ()


def test_effort_is_an_alias_for_reason(tmp_path):
    command_loop = loop(tmp_path)

    # Registered as a command that dispatches to the same handler as /reason.
    assert "/effort" in n.CommandLoop.COMMANDS
    assert n.CommandLoop.COMMAND_HANDLERS["/effort"] == n.CommandLoop.COMMAND_HANDLERS["/reason"]

    # Dispatch sets reasoning effort exactly like /reason.
    command_loop.command("/effort high")
    assert command_loop.session.config.provider.reasoning == "high"

    # Tab completion offers the same reasoning choices.
    from prompt_toolkit.document import Document

    texts = [c.text for c in n.CommandCompleter().get_completions(Document("/effort "), None)]
    assert set(texts) == set(n.REASONING_CHOICES)


def test_model_selection_groups_configured_and_remote_choices_like_master(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    provider = command_loop.session.config.provider
    provider.model = "configured-model"
    provider.available_models = ("configured-model",)
    provider.url = "https://example.com/v1"
    provider.key = "key"
    shown = []

    def select(title, choices, **_kwargs):
        shown.append((title, choices))
        if title == "Reasoning effort":
            return "off"
        if title == "Request API":
            return "auto"
        return "remote-model"

    command_loop.select_choice = select
    command_loop.remote_models = lambda _provider: ("remote-model",)

    assert "Set provider.model = remote-model" in command_loop.model("")
    assert shown[0] == (
        "Model",
        (
            command_loop.MODEL_CONFIGURED_LABEL,
            "configured-model",
            command_loop.MODEL_DISCOVERED_LABEL,
            "remote-model",
        ),
    )


def test_model_discovery_shows_loading_state_for_selected_provider(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    provider = command_loop.session.config.provider
    provider.model = "configured-model"
    provider.available_models = ("configured-model",)
    provider.url = "https://example.com/v1"
    provider.key = "key"
    transitions = []
    command_loop.tui = n.TuiApp()
    command_loop.tui.set_dispatching = lambda prompt="": transitions.append(prompt)
    command_loop.remote_models = lambda selected: ("remote-model",)
    selected = iter(["remote-model", "auto", "off"])
    command_loop.select_choice = lambda *_args, **_kwargs: next(selected)

    assert "Set provider.model = remote-model" in command_loop.model("")
    assert transitions == ["Loading models...", ""]


def test_interactive_provider_chain_uses_one_inline_tui_and_real_navigation(monkeypatch, tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["zz-other"] = n.ProviderConfig(
        model="model-a",
        available_models=("model-a", "model-b"),
        reasoning="low",
    )
    app = n.TuiApp()
    command_loop.tui = app
    output = ResizableOutput(rows=20, columns=80)
    result = []
    application_ids = []

    def modal_title():
        modal = app.modal
        if modal is None:
            return ""
        return "".join(text for _style, text in modal.fragments_fn()).splitlines()[0]

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        application_ids.append(id(app.app))
        worker = threading.Thread(target=lambda: result.append(command_loop.provider("")), daemon=True)
        worker.start()
        for title in ("Provider", "Model", "Request API", "Reasoning effort"):
            wait_until(lambda title=title: modal_title().startswith(title))
            wait_until(lambda title=title: title in rendered_screen_text(app.app, output))
            application_ids.append(id(app.app))
            pipe_input.send_text("j\r")
        worker.join(timeout=1)
        assert not worker.is_alive()
        app.set_idle()
        wait_until(lambda: app.modal is None)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive, output=output)

    assert len(set(application_ids)) == 1
    assert command_loop.session.config.active_provider == "zz-other"
    assert command_loop.session.config.provider.model == "model-b"
    assert command_loop.session.config.provider.reasoning == "medium"
    assert "Set provider.model = model-b" in result[0]


def test_single_enabled_choice_is_selected_without_opening_modal(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.choice_application = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("modal should not open"))

    assert command_loop.select_choice("Provider", ("only",), current="only") == "only"
    assert command_loop.select_choice("Model", ("heading", "only"), disabled={"heading"}) == "only"


def test_provider_auto_selects_sole_provider_and_model(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    provider = command_loop.session.config.provider
    provider.available_models = ("only-model",)
    provider.model = "only-model"
    provider.url = ""
    provider.key = ""
    titles = []

    def choose(title, _choices, _labels, current, _disabled):
        titles.append(title)
        return current

    command_loop.choice_application = choose

    result = command_loop.provider("")

    assert titles == ["Request API", "Reasoning effort"]
    assert "Set provider.model = only-model" in result


def diff_loop(tmp_path):
    command_loop = loop(tmp_path)
    before = "".join(f"old {index}\n" for index in range(20))
    after = "".join(f"new {index}\n" for index in range(20))
    command_loop.session.store_turn_diff("tr.1", 1, "a.py", "unused", before=before, after=after, round=1)
    command_loop.session.store_turn_diff("tr.2", 2, "b.py", "unused", before="old\n", after="new\n", round=1)
    return command_loop


def test_diff_viewer_switches_tabs_and_opens_selected_file(tmp_path):
    command_loop = diff_loop(tmp_path)
    switched = ModalHarness(["l", "q"])
    command_loop.tui = switched
    command_loop.diff_viewer()
    opened = ModalHarness(["j", "enter", "q"])
    command_loop.tui = opened
    command_loop.diff_viewer()

    assert any(("class:tab.active", " Session ") in frame for frame in switched.frames)
    assert switched.exclusive == [True]
    assert opened.exclusive == [True]
    text = "".join(text for frame in opened.frames for _style, text in frame)
    assert "Edit · b.py" in text
    assert "[diff]" in text


def test_diff_viewer_ctrl_d_scrolls_file_preview(tmp_path):
    command_loop = diff_loop(tmp_path)
    initial = ModalHarness(["enter", "q"])
    command_loop.tui = initial
    command_loop.diff_viewer()
    scrolled = ModalHarness(["enter", "c-d", "c-d", "q"])
    command_loop.tui = scrolled
    command_loop.diff_viewer()

    initial_text = "".join(text for frame in initial.frames for _style, text in frame)
    scrolled_text = "".join(text for frame in scrolled.frames for _style, text in frame)
    assert initial_text != scrolled_text
    assert "[diff]" in scrolled_text


def test_empty_diff_viewer_reports_zero_position(tmp_path):
    command_loop = loop(tmp_path)
    modal = ModalHarness(["q"])
    command_loop.tui = modal
    command_loop.diff_viewer()
    text = "".join(text for frame in modal.frames for _style, text in frame)

    assert "No diffs" in text
    assert "[0/0]" in text


def test_diff_view_state_owns_navigation_transitions():
    state = n.DiffViewState(n.TabbedViewState(("Latest", "Session")))

    state.handle_key("down", 3, 10)
    assert state.file == 1
    state.handle_key("enter", 3, 10)
    assert state.mode is n.DiffViewState.Mode.FILE
    state.handle_key("c-d", 3, 10)
    assert state.view.scroll == 5
    assert state.handle_key("escape", 3, 10) is n.TUI_MODAL_PENDING
    assert state.mode is n.DiffViewState.Mode.LIST

    state.handle_key("right", 3, 10)
    assert state.view.tab == 1
    assert state.file == 0
    assert state.handle_key("r", 3, 10) is n.DiffViewState.REFRESH
    assert state.handle_key("q", 3, 10) is None


def test_diff_view_g_and_shift_g_jump_top_and_bottom():
    state = n.DiffViewState(n.TabbedViewState(("Latest", "Session")))

    # LIST mode: jump file selection to last / first.
    state.handle_key("G", 5, 10)
    assert state.file == 4
    state.handle_key("g", 5, 10)
    assert state.file == 0

    # FILE mode: jump scroll to bottom (clamped on render) / top.
    state.handle_key("enter", 5, 10)
    assert state.mode is n.DiffViewState.Mode.FILE
    state.handle_key("G", 5, 10)
    assert state.view.scroll > 0
    state.handle_key("g", 5, 10)
    assert state.view.scroll == 0


def test_choice_view_g_and_shift_g_jump_first_and_last():
    state = n.ChoiceViewState(choices=("one", "two", "three"), labels={}, disabled=set())

    state.handle_key("G")
    assert state.selected == 2
    state.handle_key("g")
    assert state.selected == 0

    # While searching, g/G are query text, not jumps.
    state.searching = True
    state.handle_key("g")
    assert state.query == "g"
    assert state.selected == 0


@pytest.mark.parametrize(("key", "expected_tab"), [("l", 1), ("tab", 1), ("h", 0)])
def test_diff_view_h_l_and_tab_switch_tabs_from_file_preview(key, expected_tab):
    state = n.DiffViewState(n.TabbedViewState(("Latest", "Session"), tab=0 if key != "h" else 1))
    state.open_file(3)

    state.handle_key(key, 3, 10)

    assert state.view.tab == expected_tab
    assert state.mode is n.DiffViewState.Mode.LIST
    assert state.file == 0


def test_bash_live_preview_clips_wide_output_to_terminal_width(monkeypatch):
    preview = n.BashLivePreview()
    preview.active = True
    preview.text = "界" * 20

    with monkeypatch.context() as patch:
        patch.setattr(n.shutil, "get_terminal_size", lambda fallback=(80, 24): n.os.terminal_size((20, 24)))
        assert all(get_cwidth(line) < 20 for line in preview.frame_lines())


def test_status_bar_clips_wide_model_name_by_display_width(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.config.provider.model = "模型" * 20

    with monkeypatch.context() as patch:
        patch.setattr(n.shutil, "get_terminal_size", lambda fallback=(80, 24): n.os.terminal_size((20, 24)))
        fragments = n.StatusBar(s).fragments(sweep=False, show_elapsed=False)

    assert get_cwidth("".join(text for _style, text in fragments)) < 20


def test_status_bar_does_not_treat_long_model_calls_as_pressure(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.config.provider.timeout = 120
    s.state.current_model_call_started_at = 1.0
    bar = n.StatusBar(s)
    now = [1.0]
    monkeypatch.setattr(n.time, "monotonic", lambda: now[0])

    initial = bar.sweep_fragments("status")
    now[0] = 121.0  # Same sweep phase after a full configured timeout.

    assert bar.sweep_fragments("status") == initial
    assert all("resend" not in text for text, _role in bar.entries(show_elapsed=True))


def test_bash_live_preview_rewrites_previous_frame_without_appending(tmp_path, monkeypatch, recording_output):
    now = [100.0]
    monkeypatch.setattr(n.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(n.render, "print_formatted_text", lambda *args, **kwargs: None)
    preview = n.BashLivePreview()
    preview.output = recording_output
    preview.active = True
    preview.started_at = 100.0

    preview.render()
    first_rows = preview.rendered_lines
    recording_output.events.clear()
    preview.text = "line one\nline two"
    preview.render()

    assert recording_output.events[0] == ("write", f"\x1b[{first_rows}A")
    assert sum(event == "erase" for event, _text in recording_output.events) == preview.rendered_lines
    assert recording_output.events[-1] == ("flush", "")


def test_status_clear_erases_rendered_line(tmp_path, recording_output):
    status = n.StatusBar(session(tmp_path))
    status.output = recording_output
    status.rendered = True

    status.clear()

    assert recording_output.events == [("write", "\r"), ("erase", ""), ("flush", "")]
    assert not status.rendered


def test_clip_width_returns_unchanged_text_when_within_width():
    assert n.Text.clip_width("hello", 10) == "hello"
    assert n.Text.clip_width("", 5) == ""
    assert n.Text.clip_width("hello", 5) == "hello"


def test_clip_width_clips_wide_text_with_ellipsis():
    assert n.Text.clip_width("hello world", 8) == "hello..."
    # When width is less than 3, the ellipsis shrinks to fit
    assert n.Text.clip_width("hello world", 1) == "."
    assert n.Text.clip_width("hello world", 2) == ".."
    assert n.Text.clip_width("hello world", 3) == "..."
    assert n.Text.clip_width("hello world", 4) == "h..."


def test_clip_width_clamps_negative_width_to_zero():
    assert n.Text.clip_width("hello", -1) == ""


def test_clip_width_handles_cjk_wide_characters():
    assert n.Text.clip_width("你好世界", 5) == "你..."
    assert n.Text.clip_width("a你好", 5) == "a你好"


def test_bash_live_preview_render_skips_identical_frames(monkeypatch, recording_output):
    now = [100.0]
    monkeypatch.setattr(n.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(n.render, "print_formatted_text", lambda *args, **kwargs: None)
    preview = n.BashLivePreview()
    preview.output = recording_output
    preview.active = True
    preview.started_at = 100.0

    preview.render()
    rows_before = preview.rendered_lines
    recording_output.events.clear()

    preview.render()
    assert len(recording_output.events) == 0
    assert preview.rendered_lines == rows_before

    preview.text = "new line"
    preview.render()
    assert len(recording_output.events) > 0


def test_choice_view_state_default_filtering():
    state = n.ChoiceViewState(
        choices=("alpha", "---", "beta", "---", "gamma"),
        labels={"alpha": "Alpha", "beta": "Beta", "gamma": "Gamma"},
        disabled={"---"},
    )
    assert state.visible() == ("alpha", "---", "beta", "---", "gamma")
    assert state.enabled() == ("alpha", "beta", "gamma")
    assert state.clamp() == ("alpha", "beta", "gamma")
    assert state.selected_choice() == "alpha"


def test_choice_view_state_search_filters_visible():
    state = n.ChoiceViewState(
        choices=("alpha", "---", "beta", "---", "gamma"),
        labels={"alpha": "Alpha", "beta": "Beta"},
        disabled={"---"},
    )
    state.set_query("beta")
    assert "beta" in state.visible()
    assert "alpha" not in state.visible()
    assert state.selected == 0


def test_choice_view_state_move_navigation():
    state = n.ChoiceViewState(
        choices=("a", "b", "c"),
        labels={},
        disabled=set(),
    )
    assert state.selected_choice() == "a"
    state.move(1)
    assert state.selected_choice() == "b"
    state.move(2)
    assert state.selected_choice() == "c"
    state.move(1)  # clamped at end
    assert state.selected_choice() == "c"
    state.move(-1)
    assert state.selected_choice() == "b"


def test_choice_view_state_no_enabled_choices_returns_none():
    state = n.ChoiceViewState(
        choices=("x",),
        labels={},
        disabled={"x"},
    )
    assert state.enabled() == ()
    assert state.selected_choice() is None


def test_choice_view_state_key_navigation_and_selection():
    state = n.ChoiceViewState(
        choices=("a", "---", "b", n.ChoiceViewState.FREE_TEXT),
        labels={n.ChoiceViewState.FREE_TEXT: "Type freely..."},
        disabled={"---"},
    )

    assert state.handle_key("j") is n.TUI_MODAL_PENDING
    assert state.selected_choice() == "b"
    assert state.handle_key("1") is n.TUI_MODAL_PENDING
    assert state.handle_key("enter") == "a"

    state.selected = 2
    assert state.handle_key("enter") is n.SELECTION_FREE_TEXT


def test_choice_view_state_search_and_escape_layers():
    state = n.ChoiceViewState(choices=("alpha", "beta"), labels={}, disabled=set())

    state.handle_key("/")
    state.handle_key("any", "b")
    assert state.searching
    assert state.query == "b"
    assert state.selected_choice() == "beta"
    assert state.handle_key("escape") is n.TUI_MODAL_PENDING
    assert not state.searching
    assert state.query == "b"
    assert state.handle_key("escape") is n.TUI_MODAL_PENDING
    assert state.query == ""
    assert state.handle_key("escape") is n.SELECTION_BACK


def test_choice_view_state_fragments_preserve_headers_and_preview():
    state = n.ChoiceViewState(
        choices=("--- Models ---", "alpha"),
        labels={"--- Models ---": "  ---- Models ----", "alpha": "Alpha"},
        disabled={"--- Models ---"},
    )

    fragments = state.fragments("Model", lambda _choice: "first\\nsecond")
    rendered = "".join(text for _style, text in fragments)

    assert "  ---- Models ----" in rendered
    assert ">  1. Alpha" in rendered
    assert "  │ first\n  │ second\n" in rendered


def test_start_session_discovers_mcp_off_the_main_thread(tmp_path, monkeypatch):
    """start_session must dispatch auto_connect MCP discovery in the background: an unreachable
    server otherwise blocks the prompt for the whole discovery timeout. Regression guard for the
    lifecycle refactor that had briefly made discover_auto a synchronous startup call."""
    config = n.Config.from_dict(
        {
            "provider": {"active": "d", "d": {"url": "u", "key": "k", "model": "m"}},
            "mcp": {"slow": {"url": "http://unreachable/mcp", "auto_connect": True}},
            "paths": {"data_dir": str(tmp_path / "data")},
        }
    )
    s = n.Session(cwd=str(tmp_path), config=config)
    command_loop = n.CommandLoop(
        n.Agent(s, output_fn=lambda text: None),
        input_fn=lambda prompt="": "",
        output_fn=lambda text: None,
    )

    monkeypatch.setattr(n.SessionSnapshotStore, "clean_expired", lambda _session: 0)
    monkeypatch.setattr(n.CodeIndex, "refresh_existing_async", lambda _index: False)
    monkeypatch.setattr(n.UpdateChecker, "start", lambda _checker: None)

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
