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


def test_no_color_disables_terminal_renderer(monkeypatch):
    monkeypatch.setattr(n.Theme, "NO_COLOR", True)
    monkeypatch.setattr(n.sys.stdout, "isatty", lambda: True)

    assert not n.UiPrinter().color


@pytest.mark.parametrize("width", [20, 40, 80])
def test_styled_wrapping_respects_terminal_width_for_unicode(width):
    prefix = [("", "  Read  ")]
    continuation = [("", "        ")]
    content_text = "路径/非常长/🙂/é/模块/filename.py:123"
    rows = n.Text.wrap_styled(prefix, continuation, [("fg:default", content_text)], width)

    assert "".join(text for _style, text in rows[0]).startswith("  Read  ")
    assert all(sum(get_cwidth(text) for _style, text in row) <= width for row in rows)
    assert "".join(text for row in rows for _style, text in row).replace("  Read  ", "", 1).replace("        ", "") == content_text


def test_choice_navigation_uses_real_key_bindings(tmp_path, ui_harness):
    command_loop = loop(tmp_path)
    run = ui_harness.run(
        command_loop,
        lambda: command_loop.choice_application("Pick", ("a", "b", "c"), {"a": "Alpha", "b": "Beta", "c": "Gamma"}, "", set()),
        "j\r",
    )

    assert run.result == "b"
    assert "Beta" in run.text


def diff_loop(tmp_path):
    command_loop = loop(tmp_path)
    before = "".join(f"old {index}\n" for index in range(20))
    after = "".join(f"new {index}\n" for index in range(20))
    command_loop.session.store_turn_diff("tr.1", 1, "a.py", "unused", before=before, after=after, round=1)
    command_loop.session.store_turn_diff("tr.2", 2, "b.py", "unused", before="old\n", after="new\n", round=1)
    return command_loop


def test_diff_viewer_switches_tabs_and_opens_selected_file(tmp_path, ui_harness):
    command_loop = diff_loop(tmp_path)
    switched = ui_harness.run(command_loop, command_loop.diff_viewer, "lq", size=(80, 12))
    opened = ui_harness.run(command_loop, command_loop.diff_viewer, "j\rq", size=(80, 12))

    assert ("class:tab.active", " Session ") in switched.fragments
    assert "Edit · b.py" in opened.text
    assert "[diff]" in opened.text


def test_diff_viewer_ctrl_d_scrolls_file_preview(tmp_path, ui_harness):
    command_loop = diff_loop(tmp_path)
    initial = ui_harness.run(command_loop, command_loop.diff_viewer, "\rq", size=(80, 12))
    scrolled = ui_harness.run(command_loop, command_loop.diff_viewer, "\r\x04\x04q", size=(80, 12))

    assert initial.text != scrolled.text
    assert "[diff]" in scrolled.text


def test_empty_diff_viewer_reports_zero_position(tmp_path, ui_harness):
    command_loop = loop(tmp_path)
    run = ui_harness.run(command_loop, command_loop.diff_viewer, "q")

    assert "No diffs" in run.text
    assert "[0/0]" in run.text


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


def test_status_bar_role_style_returns_theme_values():
    assert n.StatusBar.role_style("provider").startswith("#")
    assert n.StatusBar.role_style("unknown").startswith("#")
    assert n.StatusBar.role_style("unknown") == n.Theme.style("status.base")



def test_status_bar_entries_includes_expected_sections(tmp_path):
    s = n.Session(cwd=str(tmp_path))
    entries = n.StatusBar(s).entries(0.0, show_elapsed=True)
    texts = [text for text, _ in entries]
    joined = " | ".join(texts)
    # Always includes provider/model, context percent, step, tools
    assert any("ctx " in t for t in texts)
    assert any("step " in t for t in texts)
    assert any("tools " in t for t in texts)
    # Debug should not appear in status bar
    assert "debug" not in joined.lower()

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


def test_text_elapsed_since_zero_started_at(monkeypatch):
    monkeypatch.setattr(n.time, "monotonic", lambda: 100.0)
    assert n.Text.elapsed_since(0) == "0s"


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


def test_bash_live_preview_frame_lines_uses_log_block_format(monkeypatch):
    monkeypatch.setattr(n.shutil, "get_terminal_size", lambda *args: n.os.terminal_size((80, 24)))
    monkeypatch.setattr(n.time, "monotonic", lambda: 100.0)
    preview = n.BashLivePreview()
    preview.started_at = 90.0
    preview.text = "hello world"

    lines = preview.frame_lines()
    assert any("hello world" in line for line in lines)
    assert any("10s" in line for line in lines)


def test_prepared_request_dataclass():
    msg = {"role": "user", "content": "hi"}
    tools = [{"type": "function", "function": {"name": "test"}}]
    pending = [n.QueuedInput("extra")]
    req = n.PreparedRequest(messages=[msg], tools=tools, pending=pending)
    assert req.messages == [msg]
    assert req.tools == tools
    assert req.pending == pending
    assert req.messages is not None
