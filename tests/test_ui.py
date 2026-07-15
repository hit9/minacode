import multiprocessing
import threading
import time

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.history import FileHistory
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.utils import get_cwidth

import nanocode as n


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
    return "\n".join(
        "".join(screen.data_buffer[row][column].char for column in range(output.size.columns)).rstrip()
        for row in range(output.size.rows)
    )


def run_interactive_tui(monkeypatch, tui, *, text="", drive=None, output=None, after_render=None):
    real_application = n.Application
    output = output or DummyOutput()
    driver_errors = []
    with create_pipe_input() as pipe_input:
        def application(**kwargs):
            return real_application(input=pipe_input, after_render=after_render, **(kwargs | {"output": output}))

        monkeypatch.setattr(n, "Application", application)
        monkeypatch.setattr(tui, "dump_to_scrollback", lambda: None)
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
    preserved = []
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
    command_loop.discover_mcp = lambda: None
    n.SessionSnapshotStore.clean_expired = lambda _session: 0
    n.CodeIndex.refresh_existing_async = lambda _index: False
    n.CodeIndex.update_pending_async = lambda _index: None
    n.UpdateChecker.start = lambda _checker: None
    real_application = n.Application

    try:
        with create_pipe_input() as pipe_input:
            n.Application = lambda **kwargs: real_application(input=pipe_input, **(kwargs | {"output": DummyOutput()}))

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
                    wait_until(lambda: command_loop.tui.input_buffer.text == "unfinished draft")
                    preserved.append(command_loop.tui.input_buffer.text)
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
                "preserved": preserved,
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


def test_log_buffer_always_allocated():
    ui = n.UiPrinter(output_fn=lambda text: None)
    assert ui.log_buffer is not None
    assert ui.log_buffer.entries == []


def test_log_buffer_captures_emit_when_color_active(monkeypatch):
    monkeypatch.setattr(n.sys.stdout, "isatty", lambda: True)
    ui = n.UiPrinter()
    assert ui.log_buffer is not None
    # emit() mirrors its styled segments into the LogBuffer while also printing via
    # print_formatted_text (viewport gets the same content the scrollback would).
    monkeypatch.setattr(n, "print_formatted_text", lambda *a, **kw: None)
    ui.emit("hello")
    assert ui.log_buffer.entries
    fragments = ui.log_buffer.entries[-1].fragments
    assert any("hello" in text for _style, text in fragments)


def test_log_buffer_bounded_at_limit():
    buffer = n.LogBuffer()
    for i in range(n.LogBuffer.LIMIT + 50):
        buffer.append([("", str(i))])
    assert len(buffer.entries) == n.LogBuffer.LIMIT
    # Oldest entries drop off the front, tail preserved.
    tail = buffer.entries[-1].fragments[0][1]
    assert tail == str(n.LogBuffer.LIMIT + 49)


def test_log_buffer_notifies_observers():
    buffer = n.LogBuffer()
    calls = []
    buffer.observers.append(lambda: calls.append(True))
    buffer.append([("", "line")])
    assert calls == [True]


def test_tui_edit_preview_fills_width_and_reflows_on_resize(monkeypatch):
    preview = "--- foo.py\n+++ foo.py\n@@ -0,0 +1 @@\n+return 42"
    block = n.LogBlock.hierarchy(
        n.LogLine("Edit", "foo.py", n.LogRole.TOOL),
        [
            n.LogLine("preview", role=n.LogRole.META, edge=n.LogEdge.BRANCH),
            *(n.LogLine("", line, n.LogRole.DIFF, n.LogEdge.CONTINUE) for line in preview.splitlines()),
        ],
    )
    ui = n.UiPrinter(output_fn=lambda _text: None)
    ui.color = True
    ui.full_screen = True
    monkeypatch.setattr(n, "print_formatted_text", lambda *args, **kwargs: None)
    terminal_width = {"columns": 80}
    monkeypatch.setattr(
        n.shutil,
        "get_terminal_size",
        lambda *args: n.os.terminal_size((terminal_width["columns"], 24)),
    )

    ui.emit(block)
    entry = ui.log_buffer.entries[-1]

    normal_changed = next(line for line in ui.segment_lines(entry.fragments) if any("bg:" in style for style, _text in line))
    live_changed = next(line for line in ui.segment_lines(entry.render()) if any("bg:" in style for style, _text in line))
    assert sum(n.get_cwidth(text.rstrip("\n")) for _style, text in normal_changed) < 79
    assert sum(n.get_cwidth(text.rstrip("\n")) for _style, text in live_changed) == 79

    terminal_width["columns"] = 100
    resized_changed = next(line for line in ui.segment_lines(entry.render()) if any("bg:" in style for style, _text in line))
    assert sum(n.get_cwidth(text.rstrip("\n")) for _style, text in resized_changed) == 99


def test_tui_app_viewport_joins_log_entries_with_newlines():
    buffer = n.LogBuffer()
    buffer.append([("class:x", "first line")])
    buffer.append([("", "second"), ("class:y", " line")])
    app = n.TuiApp(buffer)
    fragments = app.viewport_fragments()
    text = "".join(t for _s, t in fragments)
    assert text == "first line\nsecond line"


def test_tui_viewport_initially_anchors_to_latest_history(monkeypatch):
    buffer = n.LogBuffer()
    for index in range(40):
        buffer.append([("", f"line {index}")])
    app = n.TuiApp(buffer)
    app.build_layout()
    monkeypatch.setattr(n.shutil, "get_terminal_size", lambda *args: n.os.terminal_size((80, 10)))

    scroll = app.vertical_scroll(app.viewport_window)
    content = app.viewport_window.content.create_content(width=80, height=8)

    assert content.cursor_position.y == content.line_count - 1
    assert scroll == app.viewport_line_count() - 8


def test_tui_app_build_layout_composes_viewport_input_and_status():
    buffer = n.LogBuffer()
    app = n.TuiApp(buffer)
    layout = app.build_layout()
    # focused element is the input window; the viewport lives above it in the HSplit.
    focused = layout.current_window
    assert focused is not None
    # Layout is composable and the focused element accepts typed input via app.input_buffer.
    app.input_buffer.insert_text("hi")
    assert app.input_buffer.text == "hi"


def test_tui_approval_prompt_keeps_connector_style_and_spinner(monkeypatch):
    app = n.TuiApp(n.LogBuffer())
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
    app = n.TuiApp(n.LogBuffer())
    app.set_dispatching("Loading models...")

    assert app.status_fragments() == [("ansibrightblack", "Loading models...")]


def test_tui_prompt_output_disables_cpr_probe(monkeypatch):
    output = type("Output", (), {"enable_cpr": True})()
    monkeypatch.setattr(n, "create_output", lambda: output)

    assert n.TuiApp.prompt_output() is output
    assert output.enable_cpr is False


def test_tui_app_accept_handler_fires_on_submit_and_clears_buffer():
    buffer = n.LogBuffer()
    received: list[str] = []
    app = n.TuiApp(buffer, on_chat_submit=received.append)
    app.input_buffer.insert_text("hello")
    app.input_buffer.validate_and_handle()
    assert received == ["hello"]
    assert app.input_buffer.text == ""


def test_interactive_tui_decodes_submit_and_eof(monkeypatch):
    received = []
    app = None

    def submit(text):
        received.append(text)
        app.set_idle()

    app = n.TuiApp(n.LogBuffer(), on_chat_submit=submit)

    run_interactive_tui(monkeypatch, app, text="hello from pipe\r\x04")

    assert received == ["hello from pipe"]
    assert app.app is None


@pytest.mark.parametrize("draft", ["", "unfinished draft"])
def test_interactive_tui_ctrl_c_cancels_idle_input_like_master(monkeypatch, draft):
    cancelled = []
    app = n.TuiApp(n.LogBuffer(), on_input_cancel=lambda: cancelled.append(True))

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text(draft + "\x03")
        wait_until(lambda: cancelled == [True])
        assert app.input_buffer.text == ""
        pipe_input.send_text("\x04")

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert cancelled == [True]


def test_full_tui_ctrl_d_emits_resume_command_before_exit(tmp_path, monkeypatch):
    scenario_session = session(tmp_path)
    scenario_session.messages.append({"role": "user", "content": "persist me"})
    output = []
    command_loop = n.CommandLoop(
        n.Agent(scenario_session, output_fn=output.append),
        input_fn=lambda prompt="": "",
        output_fn=output.append,
    )
    monkeypatch.setattr(command_loop, "discover_mcp", lambda: None)
    monkeypatch.setattr(n.SessionSnapshotStore, "clean_expired", lambda _session: 0)
    monkeypatch.setattr(n.CodeIndex, "refresh_existing_async", lambda _index: False)
    monkeypatch.setattr(n.UpdateChecker, "start", lambda _checker: None)
    real_application = n.Application
    dump_threads = []
    tui_daemon = []

    with create_pipe_input() as pipe_input:
        monkeypatch.setattr(n, "Application", lambda **kwargs: real_application(input=pipe_input, **(kwargs | {"output": DummyOutput()})))

        def drive():
            wait_until(lambda: command_loop.tui is not None and command_loop.tui.app is not None and command_loop.tui.app.is_running)
            tui_daemon.append(next(thread for thread in threading.enumerate() if thread.name == "tui").daemon)
            command_loop.tui.dump_to_scrollback = lambda: dump_threads.append(threading.current_thread())
            pipe_input.send_text("\x04")

        driver = threading.Thread(target=drive, daemon=True)
        driver.start()
        assert command_loop.run_tui() == 0
        driver.join(timeout=1)

    assert any(f"nanocode --resume {scenario_session.uid}" in line for line in output)
    assert tui_daemon == [False]
    assert dump_threads == [threading.main_thread()]


@pytest.mark.parametrize("entered", [" /help", "exit "])
def test_tui_runtime_strips_input_before_command_dispatch(tmp_path, entered):
    command_loop = loop(tmp_path)
    dispatched = []
    command_loop.command = lambda text: dispatched.append(text) or (True, False)
    command_loop.tui = n.TuiApp(command_loop.ui.log_buffer)
    runtime = n.TuiRuntime(command_loop)

    assert runtime.dispatch(entered)
    assert dispatched == [entered.strip()]


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
    monkeypatch.setattr(command_loop, "discover_mcp", lambda: None)
    monkeypatch.setattr(n.SessionSnapshotStore, "clean_expired", lambda _session: 0)
    monkeypatch.setattr(n.CodeIndex, "refresh_existing_async", lambda _index: False)
    monkeypatch.setattr(n.CodeIndex, "update_pending_async", lambda _index: None)
    monkeypatch.setattr(n.UpdateChecker, "start", lambda _checker: None)
    real_application = n.Application

    with create_pipe_input() as pipe_input:
        monkeypatch.setattr(n, "Application", lambda **kwargs: real_application(input=pipe_input, **(kwargs | {"output": DummyOutput()})))

        def drive():
            wait_until(lambda: command_loop.tui is not None and command_loop.tui.app is not None and command_loop.tui.app.is_running)
            command_loop.tui.dump_to_scrollback = lambda: None
            wait_until(lambda: len(requests) == 1)
            wait_until(lambda: command_loop.tui.input_mode == "chat")
            pipe_input.send_text("\x04")

        driver = threading.Thread(target=drive, daemon=True)
        driver.start()
        assert command_loop.run_tui() == 0
        driver.join(timeout=1)

    assert len(requests) == 1
    assert "queued one" in requests[0]
    assert "queued two" in requests[0]
    assert requests[0].index("queued one") < requests[0].index("queued two")
    assert restored.pending_user_inputs == []


def test_interactive_tui_control_backslash_forces_exit(monkeypatch):
    forced = []
    app = None

    def force_exit():
        forced.append(True)
        app.app.exit()

    app = n.TuiApp(n.LogBuffer(), on_force_exit=force_exit)

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

    app = n.TuiApp(n.LogBuffer(), on_running_submit=submit, on_recall=recall)
    app.set_running("working")

    run_interactive_tui(monkeypatch, app, text="\x1b[A\r\x04")

    assert recalled == [True]
    assert received == ["edit queued message"]


def test_interactive_tui_ctrl_p_recalls_submitted_queued_input(monkeypatch, tmp_path):
    received = []
    recalled = []
    app = n.TuiApp(
        n.LogBuffer(),
        on_running_submit=received.append,
        history=FileHistory(str(tmp_path / "history.txt")),
    )
    app.set_running("working")

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("queued message\r")
        wait_until(lambda: received == ["queued message"])
        pipe_input.send_text("\x10")
        wait_until(lambda: app.input_buffer.text == "queued message")
        recalled.append(app.input_buffer.text)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert recalled == ["queued message"]


def test_interactive_tui_tab_inserts_single_completion_without_menu(monkeypatch):
    app = n.TuiApp(n.LogBuffer(), completer=n.CommandCompleter())

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("/pro\t")
        wait_until(lambda: app.input_buffer.text == "/provider")
        assert app.input_buffer.complete_state is None
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert app.input_buffer.text == "/provider"


def test_interactive_tui_bracketed_paste_displays_all_lines(monkeypatch):
    app = n.TuiApp(n.LogBuffer())
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
    log = n.LogBuffer()
    log.append([("", "history")])
    app = n.TuiApp(log)
    frames = []
    rendered = threading.Event()

    def capture(application):
        screen = application.renderer.last_rendered_screen
        if screen is None:
            return
        positions = screen.visible_windows_to_write_positions
        if app.viewport_window in positions and app.input_window in positions and app.status_window in positions:
            frames.append((positions[app.viewport_window], positions[app.input_window], positions[app.status_window]))
        rendered.set()

    def drive(_pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        assert rendered.wait(timeout=1)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive, after_render=capture)

    assert frames
    viewport, prompt, status = frames[0]
    assert prompt.ypos == viewport.ypos + viewport.height + 1
    assert status.ypos == prompt.ypos + prompt.height + 1


def test_interactive_tui_keeps_padding_around_running_queue(monkeypatch):
    app = n.TuiApp(n.LogBuffer(), activity_fragments_fn=lambda: [("", "working\n+ queued")])
    app.set_running("working")
    frames = []
    rendered = threading.Event()

    def capture(application):
        screen = application.renderer.last_rendered_screen
        if screen is None:
            return
        positions = screen.visible_windows_to_write_positions
        windows = (app.viewport_window, app.activity_window, app.input_window, app.status_window)
        if all(window in positions for window in windows):
            frames.append(tuple(positions[window] for window in windows))
        rendered.set()

    def drive(_pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        assert rendered.wait(timeout=1)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive, after_render=capture)

    assert frames
    viewport, activity, prompt, status = frames[0]
    assert activity.ypos == viewport.ypos + viewport.height + 1
    assert prompt.ypos == activity.ypos + activity.height + 1
    assert status.ypos == prompt.ypos + prompt.height + 1


def test_tui_running_input_queues_one_multiline_message():
    received: list[str] = []
    app = n.TuiApp(n.LogBuffer(), on_running_submit=received.append)
    app.set_running("working")
    app.input_buffer.insert_text("first\nsecond\nthird")

    app.input_buffer.validate_and_handle()

    assert received == ["first\nsecond\nthird"]
    assert app.input_buffer.text == ""


def test_tui_running_input_drops_whitespace_only_draft():
    received: list[str] = []
    app = n.TuiApp(n.LogBuffer(), on_running_submit=received.append)
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


def test_tui_running_queue_hint_matches_legacy_send_now_behavior(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = n.TuiApp(command_loop.ui.log_buffer)
    command_loop.tui.set_running("working")
    command_loop.session.enqueue_user_input("queued")

    assert command_loop.tui_input_hint() == "↑ recalls queued · Ctrl-C sends now"


def test_tui_sigint_interrupts_dispatch_and_running_modes():
    interrupted = []
    app = n.TuiApp(n.LogBuffer(), on_interrupt=lambda: interrupted.append(True))
    bindings = app.make_bindings()
    handler = next(binding.handler for binding in bindings.bindings if binding.keys == (n.Keys.SIGINT,))
    event = type("Event", (), {})()

    app.set_dispatching()
    handler(event)
    app.set_running("working")
    handler(event)

    assert interrupted == [True, True]


@pytest.mark.parametrize("mode", ["chat", "running"])
def test_tui_ctrl_d_deletes_at_cursor_when_input_is_nonempty(mode):
    app = n.TuiApp(n.LogBuffer())
    app.input_buffer.reset(n.Document("abc", cursor_position=1))
    app.input_mode = mode
    binding = next(binding for binding in reversed(app.make_bindings().bindings) if binding.keys == (n.Keys.ControlD,) and binding.filter())
    event = type("Event", (), {"app": type("Application", (), {"exit": lambda self: None})()})()

    binding.handler(event)

    assert app.input_buffer.text == "ac"


def test_tui_ctrl_d_submits_multiline_approval_input():
    app = n.TuiApp(n.LogBuffer())
    pending = threading.Event()
    app.input_mode = "approval"
    app._input_pending = pending
    app.input_buffer.reset(n.Document("first\nsecond"))
    binding = next(binding for binding in reversed(app.make_bindings().bindings) if binding.keys == (n.Keys.ControlD,) and binding.filter())
    event = type("Event", (), {"app": type("Application", (), {"exit": lambda self: None})()})()

    binding.handler(event)

    assert pending.is_set()
    assert app._input_result == "first\nsecond"


def test_tui_ctrl_g_retries_only_while_running():
    retried = []
    app = n.TuiApp(n.LogBuffer(), on_retry=lambda: retried.append(True))
    bindings = app.make_bindings()
    event = type("Event", (), {})()

    app.set_running("working")
    binding = next(binding for binding in bindings.bindings if binding.keys == (n.Keys.ControlG,))
    assert binding.filter()
    binding.handler(event)
    app.set_idle()

    assert retried == [True]
    assert not binding.filter()


def test_tui_cancelling_is_transient_status_not_history():
    log = n.LogBuffer()
    app = n.TuiApp(log)
    app.set_running("working")

    app.set_cancelling()

    assert app.input_mode == "running"
    assert app.status_label == "cancelling"
    assert log.entries == []


def test_tui_activity_uses_transient_cancelling_status(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = n.TuiApp(command_loop.ui.log_buffer)
    command_loop.tui.set_cancelling()

    text = "".join(fragment for _style, fragment in command_loop.queue_divider_fragments())

    assert "cancelling" in text
    assert "working" not in text


def test_tui_running_recall_removes_latest_pending_message():
    pending = ["first", "second"]

    def recall():
        return pending.pop() if pending else ""

    app = n.TuiApp(n.LogBuffer(), on_recall=recall)
    app.set_running("working")
    bindings = app.make_bindings()
    event = type("Event", (), {"current_buffer": app.input_buffer})()
    handler = next(binding.handler for binding in reversed(bindings.bindings) if binding.keys == (n.Keys.Up,) and binding.filter())

    handler(event)

    assert pending == ["first"]
    assert app.input_buffer.text == "second"


def test_interactive_tui_modal_uses_real_j_and_enter_keys(monkeypatch):
    app = n.TuiApp(n.LogBuffer())
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
    app = n.TuiApp(n.LogBuffer())
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


@pytest.mark.parametrize(("exclusive", "history_visible"), [(False, True), (True, False)])
def test_interactive_tui_modal_presentation_matches_legacy_scope(monkeypatch, exclusive, history_visible):
    log = n.LogBuffer()
    log.append([("", "history marker")])
    app = n.TuiApp(log, status_fragments_fn=lambda: [("", "status marker")])
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

    assert bool("history marker" in frames[-1]) is history_visible
    assert "status marker" in frames[-1]
    if exclusive:
        assert frames[-1].splitlines()[-1] == "status marker"


def test_interactive_tui_survives_live_log_growth_during_resize(monkeypatch):
    log = n.LogBuffer()
    log.append([("", "initial")])
    app = n.TuiApp(log)
    output = ResizableOutput()
    rendered = threading.Event()

    def after_render(_application):
        rendered.set()

    def drive(_pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        for index, (rows, columns) in enumerate(((8, 30), (30, 100), (10, 45), (24, 80))):
            log.append([("", "\n".join(f"skill output {index}.{line}" for line in range(20)))])
            rendered.clear()
            output.size = Size(rows=rows, columns=columns)
            app.app.loop.call_soon_threadsafe(app.app._on_resize)
            assert rendered.wait(timeout=1)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive, output=output, after_render=after_render)


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
    assert "queued two" in queued_request
    assert queued_request.index("queued one") < queued_request.index("queued two")
    assert outcome["preserved"] == ["unfinished draft"]
    assert outcome["persisted_user_inputs"] == ["long request", "queued one", "queued two"]
    assert outcome["restored_queue"] == []


def test_tui_app_approval_mode_resolves_bridge_event():
    import threading as _threading

    buffer = n.LogBuffer()
    app = n.TuiApp(buffer)
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
    app = n.TuiApp(n.LogBuffer())
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

    app = n.TuiApp(n.LogBuffer(), on_chat_submit=submit)

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
    app = n.TuiApp(n.LogBuffer())
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
    app = n.TuiApp(command_loop.ui.log_buffer)
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


def test_resume_history_is_buffered_before_tui_starts(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.session.resumed = True
    command_loop.session.messages.extend(
        [
            {"role": "user", "content": "most recent question"},
            {"role": "assistant", "content": "most recent answer"},
        ]
    )
    command_loop.ui.color = True
    command_loop.ui.full_screen = True
    monkeypatch.setattr(n, "print_formatted_text", lambda *args, **kwargs: None)

    command_loop.render_resumed_session()

    text = "".join(fragment for entry in command_loop.ui.log_buffer.entries for _style, fragment in entry.fragments)
    assert "most recent question" in text
    assert "most recent answer" in text


def test_desert_user_color_does_not_leak_into_default_ui_style(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    for mode, expected in (("dark", "#e0a96d"), ("light", "#9a5b2e")):
        monkeypatch.setattr(n.Theme, "_mode", mode)
        assert n.UiPrinter.user_log_style() == expected
        assert command_loop.style().get_attrs_for_style_str("").color == ""


def test_full_tui_commands_append_output_immediately(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.ui.color = True
    command_loop.ui.full_screen = True
    monkeypatch.setattr(command_loop, "status", lambda _args: "status marker")

    before = len(command_loop.ui.log_buffer.entries)
    assert command_loop.command("/help") == (True, False)
    after_help = len(command_loop.ui.log_buffer.entries)
    assert command_loop.command("/status") == (True, False)
    after_status = len(command_loop.ui.log_buffer.entries)
    assert command_loop.command("/skills") == (True, False)
    after_skills = len(command_loop.ui.log_buffer.entries)

    assert before < after_help < after_status < after_skills
    text = "".join(fragment for entry in command_loop.ui.log_buffer.entries for _style, fragment in entry.fragments)
    assert "/provider" in text
    assert "status marker" in text
    assert "Skills ·" in text


def test_mcp_cancelled_error_notice_is_muted(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.session.mcp.server_errors.update({"openaiDeveloperDocs": "CancelledError", "broken": "connection failed"})

    notice = command_loop.mcp_error_notice()

    assert "openaiDeveloperDocs" not in notice
    assert "CancelledError" not in notice
    assert "mcp: broken: connection failed" in notice


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

    ui = n.UiPrinter()
    # Interactive TTY output stays colored regardless of NO_COLOR — nanocode owns its theming and
    # renders through prompt_toolkit's ANSI path, so the parent env var is not honored.
    assert ui.color


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


def test_choice_navigation_uses_shared_modal_protocol(tmp_path):
    command_loop = loop(tmp_path)
    modal = ModalHarness(["j", "enter"])
    command_loop.tui = modal
    result = command_loop.choice_application("Pick", ("a", "b", "c"), {"a": "Alpha", "b": "Beta", "c": "Gamma"}, "", set())

    assert result == "b"
    assert "Beta" in "".join(text for frame in modal.frames for _style, text in frame)


def test_provider_selection_chains_provider_model_and_reasoning(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["other"] = n.ProviderConfig(model="model-b", available_models=("model-b",), reasoning="low")
    selected = iter(["other", "model-b", "high"])
    titles = []

    def select(title, *_args, **_kwargs):
        titles.append(title)
        return next(selected)

    command_loop.select_choice = select
    discovered = []
    command_loop.remote_models = lambda provider: discovered.append(provider.model) or ()

    result = command_loop.provider("")

    assert titles == ["Provider", "Model", "Reasoning effort"]
    assert command_loop.session.config.active_provider == "other"
    assert command_loop.session.config.provider.model == "model-b"
    assert command_loop.session.config.provider.reasoning == "high"
    assert discovered == ["model-b"]
    assert "Set provider.model = model-b" in result


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
    command_loop.tui = n.TuiApp(command_loop.ui.log_buffer)
    command_loop.tui.set_dispatching = lambda prompt="": transitions.append(prompt)
    command_loop.remote_models = lambda selected: ("remote-model",)
    selected = iter(["remote-model", "off"])
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
    log = n.LogBuffer()
    log.append([("", "history marker")])
    app = n.TuiApp(log)
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
        for title in ("Provider", "Model", "Reasoning effort"):
            wait_until(lambda title=title: modal_title().startswith(title))
            wait_until(lambda title=title: title in rendered_screen_text(app.app, output))
            application_ids.append(id(app.app))
            screen = rendered_screen_text(app.app, output)
            assert "history marker" in screen
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

    assert titles == ["Reasoning effort"]
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


def test_bash_live_preview_clips_wide_output_to_terminal_width(monkeypatch):
    monkeypatch.setattr(n.shutil, "get_terminal_size", lambda *args: n.os.terminal_size((20, 24)))
    preview = n.BashLivePreview()
    preview.active = True
    preview.text = "界" * 20

    assert all(get_cwidth(line) < 20 for line in preview.frame_lines())


def test_status_bar_clips_wide_model_name_by_display_width(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.config.provider.model = "模型" * 20
    monkeypatch.setattr(n.shutil, "get_terminal_size", lambda *args: n.os.terminal_size((20, 24)))

    fragments = n.StatusBar(s).fragments(0.0, sweep=False, show_elapsed=False)

    assert get_cwidth("".join(text for _style, text in fragments)) < 20


def test_bash_live_preview_rewrites_previous_frame_without_appending(tmp_path, monkeypatch, recording_output):
    now = [100.0]
    monkeypatch.setattr(n.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(n, "print_formatted_text", lambda *args, **kwargs: None)
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
    monkeypatch.setattr(n, "print_formatted_text", lambda *args, **kwargs: None)
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
