import pytest
from prompt_toolkit.utils import get_cwidth

import nanocode as n


def session(tmp_path):
    config = n.Config()
    config.data_dir = str(tmp_path / "data")
    return n.Session(cwd=str(tmp_path), config=config)


def loop(tmp_path):
    return n.CommandLoop(n.Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt="": "", output_fn=lambda text: None)


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


def test_tui_app_accept_handler_fires_on_submit_and_clears_buffer():
    buffer = n.LogBuffer()
    received: list[str] = []
    app = n.TuiApp(buffer, on_chat_submit=received.append)
    app.input_buffer.insert_text("hello")
    # Simulate the accept handler pt would call on Enter in chat mode.
    app._accept(app.input_buffer)
    assert received == ["hello"]
    assert app.input_buffer.text == ""


def test_tui_running_input_queues_one_multiline_message():
    received: list[str] = []
    app = n.TuiApp(n.LogBuffer(), on_running_submit=received.append)
    app.set_running("working")
    app.input_buffer.insert_text("first\nsecond\nthird")

    app._accept(app.input_buffer)

    assert received == ["first\nsecond\nthird"]
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
    # Wait until the bg thread has switched us into approval mode.
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

    def show_modal(self, fragments_fn, key_fn):
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
    command_loop.remote_models = lambda _provider: (_ for _ in ()).throw(AssertionError("remote discovery must be lazy"))

    result = command_loop.provider("")

    assert titles == ["Provider", "Model", "Reasoning effort"]
    assert command_loop.session.config.active_provider == "other"
    assert command_loop.session.config.provider.model == "model-b"
    assert command_loop.session.config.provider.reasoning == "high"
    assert "Set provider.model = model-b" in result


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
