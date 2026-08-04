"""Rendering: themes, the external editor, the status bar, the Bash live preview, width
clipping, and choice-view state."""

import itertools
import os
import shutil
import sys
import time

import pytest
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.utils import get_cwidth
from rich.console import Console
from tui_harness import loop, session

import minacode.render as render_module
from minacode.base import (
    SELECTION_BACK,
    SELECTION_FREE_TEXT,
    LogBlock,
    LogLine,
    LogRole,
    Text,
    request_budget_for,
)
from minacode.render import BashLivePreview, StatusBar, Theme, UiPrinter
from minacode.tui import TUI_MODAL_PENDING, ChoiceViewState, TuiApp


def test_theme_palettes_have_identical_complete_keys():
    assert Theme.DARK.keys() == Theme.LIGHT.keys()
    assert all(Theme.DARK.values())
    assert all(Theme.LIGHT.values())


def test_status_roles_have_theme_entries():
    assert all(f"status.{role}" in Theme.DARK and f"status.{role}" in Theme.LIGHT for role in StatusBar.ROLE_KEYS)


def test_editor_command_prefers_visual_then_editor_then_vim(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    assert TuiApp.editor_command() == ["vim"]

    monkeypatch.setenv("EDITOR", "code --wait")
    assert TuiApp.editor_command() == ["code", "--wait"]

    monkeypatch.setenv("VISUAL", "nvim")
    assert TuiApp.editor_command() == ["nvim"]


def test_edit_text_in_editor_roundtrips_edited_content(tmp_path, monkeypatch):
    # A fake $EDITOR that appends a marker to whatever file it is given.
    editor = tmp_path / "fake_editor.sh"
    editor.write_text('#!/bin/sh\nprintf " EDITED" >> "$1"\n')
    editor.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(editor))
    monkeypatch.delenv("VISUAL", raising=False)

    assert TuiApp()._edit_text_in_editor("hello") == "hello EDITED"


def test_edit_text_in_editor_leaves_input_untouched_when_editor_missing(monkeypatch):
    monkeypatch.setenv("EDITOR", "definitely-not-an-editor-binary")
    monkeypatch.delenv("VISUAL", raising=False)

    assert TuiApp()._edit_text_in_editor("hello") is None


def test_edit_text_in_editor_leaves_input_untouched_on_nonzero_exit(monkeypatch):
    monkeypatch.setenv("EDITOR", "false")
    monkeypatch.delenv("VISUAL", raising=False)

    assert TuiApp()._edit_text_in_editor("hello") is None


def test_editor_text_compose_and_strip_roundtrip():
    # The editor receives the draft plus the agent's reply below a scissors line; stripping
    # drops the reference context and returns exactly the (possibly edited) draft.
    composed, marker = TuiApp._compose_editor_text("my draft", "reply line one\nline two")
    assert "my draft" in composed
    assert TuiApp.EDITOR_CONTEXT_MARKER in composed
    assert marker and marker in composed
    assert "reply line one" in composed
    assert TuiApp._strip_editor_context(composed, marker) == "my draft"
    # Editing above the scissors line survives; everything below it is dropped.
    assert TuiApp._strip_editor_context(composed.replace("my draft", "edited draft"), marker) == "edited draft"


def test_editor_text_compose_without_context_is_identity():
    assert TuiApp._compose_editor_text("draft", "") == ("draft", "")
    assert TuiApp._compose_editor_text("draft", "   ") == ("draft", "")
    assert TuiApp._strip_editor_context("plain text\n", "") == "plain text"


def test_editor_strip_preserves_a_scissors_line_the_user_typed():
    # Only the marker this composition added is stripped; a scissors line already in the draft
    # (pasted Markdown or code) survives, whether or not reference context was appended.
    draft = f"before\n{TuiApp.EDITOR_CONTEXT_MARKER}\nafter"
    assert TuiApp._strip_editor_context(draft, "") == draft
    composed, marker = TuiApp._compose_editor_text(draft, "reply")
    assert TuiApp._strip_editor_context(composed, marker) == draft


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


def test_desert_user_color_does_not_leak_into_default_ui_style(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    for mode, expected in (("dark", "#e0a96d"), ("light", "#9a5b2e")):
        monkeypatch.setattr(Theme, "_mode", mode)
        assert UiPrinter.user_log_style() == expected
        assert command_loop.style().get_attrs_for_style_str("").color == ""


def test_tool_labels_keep_legacy_green_style():
    assert UiPrinter.LOG_STYLES[LogRole.TOOL][0] == "ansigreen"


@pytest.mark.parametrize(("mode", "rgb"), [("dark", "224;169;109"), ("light", "154;91;46")])
def test_resumed_user_rendering_emits_desert_truecolor(mode, rgb, monkeypatch):
    monkeypatch.setattr(Theme, "_mode", mode)
    ui = UiPrinter(output_fn=lambda text: None)
    console = Console(force_terminal=True, color_system="truecolor", no_color=False, width=40)

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
    assert Theme.resolve(configured) == expected


def test_tool_argument_rendering_tracks_theme_without_changing_text(monkeypatch):
    line = LogLine("Search", '"needle" path=src 0:20', LogRole.TOOL, syntax="tool-args")
    block = LogBlock([line])
    rendered = []

    for mode in ("dark", "light"):
        monkeypatch.setattr(Theme, "_mode", mode)
        segments = UiPrinter(output_fn=lambda text: None).log_segments(block)
        rendered.append(("".join(text for _style, text in segments), {style for style, text in segments if text.strip()}))

    assert rendered[0][0] == rendered[1][0] == '  Search  "needle" path=src 0:20\n'
    assert rendered[0][1] != rendered[1][1]


def test_interactive_renderer_keeps_theme_when_parent_exports_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(Theme, "_mode", "dark")
    emitted = []
    monkeypatch.setattr(render_module, "print_formatted_text", lambda value, **_kwargs: emitted.extend(to_formatted_text(value)))

    ui = UiPrinter()
    # Interactive TTY output stays colored regardless of NO_COLOR — minacode owns its theming and
    # renders through prompt_toolkit's ANSI path, so the parent env var is not honored.
    assert ui.color
    ui.emit_answer("sent message", role="user", rule=False)

    desert_text = "".join(text for style, text in emitted if style == "#e0a96d")
    assert "• sent message" in desert_text


def test_emit_turn_end_non_color_uses_elapsed_since_format():
    emitted = []
    ui = UiPrinter(output_fn=emitted.append)
    assert not ui.color

    ui.emit_turn_end(time.monotonic() - 5)
    ui.emit_turn_end(time.monotonic() - 65)

    # The footer reuses the divider's `elapsed_since` format: no leading `0m`, seconds zero-padded.
    assert emitted == ["done in 5s", "done in 1m05s"]


def test_emit_turn_end_renders_a_left_aligned_rule_under_a_blank_line(monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    emitted = []
    monkeypatch.setattr(render_module, "print_formatted_text", lambda value, **_kwargs: emitted.extend(to_formatted_text(value)))

    ui = UiPrinter()
    assert ui.color
    ui.emit_turn_end(time.monotonic() - 65)

    text = "".join(fragment for _style, fragment in emitted)
    # A blank line lifts the rule off the answer; the label sits just past a short lead (left-biased,
    # not centered, not flush) with a long trail of dashes running to the full width.
    assert "\n── done in 1m05s " in text
    assert "──────" in text


def test_editor_and_queued_user_text_use_desert_style(tmp_path, monkeypatch):
    monkeypatch.setattr(Theme, "_mode", "dark")
    expected = UiPrinter.user_log_style()
    app = TuiApp()
    app.build_layout()
    assert app.input_window.style == expected

    command_loop = loop(tmp_path)
    command_loop.session.enqueue_user_input("queued message")
    sent, waiting = command_loop.followup_fragments()
    assert any(style == expected and "queued message" in text for style, text in [*sent, *waiting])


@pytest.mark.parametrize("width", [20, 40, 80])
def test_styled_wrapping_respects_terminal_width_for_unicode(width):
    prefix = [("", "  Read  ")]
    continuation = [("", "        ")]
    content_text = "路径/非常长/🙂/é/模块/filename.py:123"
    rows = Text.wrap_styled(prefix, continuation, [("fg:default", content_text)], width)

    assert "".join(text for _style, text in rows[0]).startswith("  Read  ")
    assert all(sum(get_cwidth(text) for _style, text in row) <= width for row in rows)
    assert "".join(text for row in rows for _style, text in row).replace("  Read  ", "", 1).replace("        ", "") == content_text


def test_bash_live_preview_clips_wide_output_to_terminal_width(monkeypatch):
    preview = BashLivePreview()
    preview.active = True
    preview.text = "界" * 20

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((20, 24)))
        assert all(get_cwidth(line) < 20 for line in preview.frame_lines())


def test_bash_live_preview_rewrites_previous_frame_without_appending(tmp_path, monkeypatch, recording_output):
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(render_module, "print_formatted_text", lambda *args, **kwargs: None)
    preview = BashLivePreview()
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


def test_bash_live_preview_render_skips_identical_frames(monkeypatch, recording_output):
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(render_module, "print_formatted_text", lambda *args, **kwargs: None)
    preview = BashLivePreview()
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


def test_status_bar_clips_wide_model_name_by_display_width(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.config.provider.model = "模型" * 20

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((20, 24)))
        fragments = StatusBar(s).fragments(sweep=False, show_elapsed=False)

    assert get_cwidth("".join(text for _style, text in fragments)) < 20


def test_status_bar_idle_clip_keeps_role_colors(tmp_path, monkeypatch):
    s = session(tmp_path)
    bar = StatusBar(s)

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((30, 24)))
        fragments = bar.fragments(sweep=False, show_elapsed=False)

    # A narrow idle bar clips but keeps its per-role colors instead of collapsing the whole line
    # to one status.base tone, which read as a colorless white bar in a tmux split.
    styles = {style for style, text in fragments if text.strip()}
    assert len(styles) > 1
    assert Theme.style("status.base") in styles
    assert Theme.style("status.reason") in styles
    assert get_cwidth("".join(text for _style, text in fragments)) < 30


def test_status_bar_clip_fragments_preserves_segment_styles():
    fragments = [("#aaaaaa", "alpha "), ("#bbbbbb", "beta "), ("#cccccc", "gamma")]

    clipped = StatusBar.clip_fragments(fragments, 12)

    # The clip cuts mid-second segment; each surviving segment keeps its own style and the
    # ellipsis inherits the style of the segment it interrupted.
    assert "".join(text for _style, text in clipped) == "alpha bet..."
    assert {style for style, _ in clipped} == {"#aaaaaa", "#bbbbbb"}


def test_status_bar_clip_fragments_mirrors_clip_width_ellipsis():
    fragments = [("#aaaaaa", "hello world")]

    assert StatusBar.clip_fragments(fragments, 0) == [("", "")]
    for width in (1, 2, 3, 4, 8):
        clipped = StatusBar.clip_fragments(fragments, width)
        assert "".join(text for _style, text in clipped) == Text.clip_width("hello world", width)
        assert get_cwidth("".join(text for _style, text in clipped)) <= width


def test_status_bar_sweep_shares_styles_between_neighbouring_cells(tmp_path, monkeypatch):
    s = session(tmp_path)
    bar = StatusBar(s)
    text = "dashscope/qwen3.7-plus | high | ctx 23% · cache 98% | index | step 160/200"
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    runs = []
    seen = set()
    for frame in range(120):  # four seconds of frames
        now[0] = 1000.0 + frame / 30
        styles = [style for style, _text in bar.sweep_fragments(text)]
        assert len(styles) == len(text)
        seen.update(styles)
        runs.append(1 + sum(1 for left, right in itertools.pairwise(styles) if left != right))

    # A colour per cell costs an escape sequence per column on every frame, and mints a style string
    # that prompt-toolkit's renderer caches for the life of the process. Quantized, neighbours share
    # a style, so the runs collapse and the set of strings stays bounded however long a turn runs.
    assert max(runs) < len(text) / 2
    assert len(seen) <= bar.SWEEP_BANDS * bar.SWEEP_LEVELS


def test_status_bar_sweep_crest_travels_and_stays_within_the_palette(tmp_path, monkeypatch):
    s = session(tmp_path)
    bar = StatusBar(s)
    text = "x" * 80
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    def crest_at(offset: float) -> int:
        now[0] = 1000.0 + offset
        styles = [style for style, _text in bar.sweep_fragments(text)]
        crest = Theme.style("status.sweep.crest")
        return min(range(len(styles)), key=lambda index: sum(abs(a - b) for a, b in zip(Theme.rgb(styles[index]), Theme.rgb(crest), strict=True)))

    # The crest crosses the line once per cycle and drifts by a cell or so per frame, which is what
    # keeps the band reading as a travelling light rather than a blink.
    positions = [crest_at(frame / 30) for frame in range(10)]
    assert positions == sorted(positions)
    assert 0 < positions[-1] - positions[0] <= 30

    quarter = crest_at(0.25 / bar.SWEEP_CYCLES_PER_SEC)
    assert abs(quarter - len(text) // 4) <= 2


def test_status_bar_does_not_treat_long_model_calls_as_pressure(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.config.provider.timeout = 120
    s.state.current_model_call_started_at = 1.0
    bar = StatusBar(s)
    now = [1.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    initial = bar.sweep_fragments("status")
    now[0] = 121.0  # Same sweep phase after a full configured timeout.

    assert bar.sweep_fragments("status") == initial
    assert all("resend" not in text for text, _role in bar.entries(show_elapsed=True))


def test_status_bar_shows_last_request_cache_hit_ratio(tmp_path):
    s = session(tmp_path)
    bar = StatusBar(s)

    def ctx_text() -> str:
        return next(text for text, role in bar.entries(show_elapsed=False) if role == "ctx")

    # No requests yet: the ctx segment carries no cache suffix.
    assert "cache" not in ctx_text()

    s.usage.last_prompt_tokens = 1000
    s.usage.last_cached_prompt_tokens = 870
    assert ctx_text().endswith("· cache 87%")
    # Rendering exercises the merged ctx/cache segment end-to-end.
    rendered = bar.fragments(sweep=False, show_elapsed=False)
    assert any("cache 87%" in text for _style, text in rendered)

    s.usage.last_cached_prompt_tokens = 0
    assert ctx_text().endswith("· cache 0%")


def test_status_bar_ctx_percent_uses_last_real_tokens_when_available(tmp_path):
    s = session(tmp_path)
    s.state.context_percent = 7  # estimate would claim 7%
    s.usage.last_prompt_tokens = 20_000  # provider reported 20K for the last request
    s.usage.last_prompt_budget = 80_000  # the budget that request was prepared against
    bar = StatusBar(s)

    ctx_text = next(text for text, role in bar.entries(show_elapsed=False) if role == "ctx")

    assert "ctx 25%" in ctx_text
    assert "ctx 7%" not in ctx_text


def test_status_bar_ctx_percent_keeps_the_request_time_budget(tmp_path):
    """Changing max_tokens or max_context_tokens after the request must not move the recorded fill:
    the denominator is the budget the last request was prepared against, not today's configuration."""
    s = session(tmp_path)
    s.settings.max_context_tokens = 100_000
    s.usage.last_prompt_tokens = 40_000
    s.usage.last_prompt_budget = request_budget_for(100_000, 10_000)  # 40K of an 85.9K budget
    bar = StatusBar(s)

    def ctx_percent() -> int:
        text = next(t for t, role in bar.entries(show_elapsed=False) if role == "ctx")
        return int(text.split("%")[0].split(" ")[1])

    recorded = 40_000 * 100 // request_budget_for(100_000, 10_000)
    assert ctx_percent() == recorded

    s.config.provider.max_tokens = 60_000  # today's budget would read as ~111% -> 100%
    assert ctx_percent() == recorded


def test_status_bar_ctx_percent_falls_back_to_estimate_without_requests(tmp_path):
    s = session(tmp_path)
    s.state.context_percent = 23
    bar = StatusBar(s)

    ctx_text = next(text for text, role in bar.entries(show_elapsed=False) if role == "ctx")

    assert f"ctx {s.state.context_percent}%" in ctx_text
    assert "cache" not in ctx_text


def test_status_bar_ctx_percent_falls_back_when_the_recorded_budget_is_missing(tmp_path):
    """A session resumed from a snapshot taken before last_prompt_budget existed has tokens but no
    budget; the estimate is the honest fallback rather than a division by zero."""
    s = session(tmp_path)
    s.state.context_percent = 31
    s.usage.last_prompt_tokens = 20_000
    s.usage.last_prompt_budget = 0
    bar = StatusBar(s)

    ctx_text = next(text for text, role in bar.entries(show_elapsed=False) if role == "ctx")

    assert f"ctx {s.state.context_percent}%" in ctx_text


def test_status_bar_shows_step_only_near_max_steps(tmp_path):
    s = session(tmp_path)
    bar = StatusBar(s)
    s.settings.max_steps = 200

    s.state.turn_step = 1
    assert all(not text.startswith("step ") for text, _role in bar.entries(show_elapsed=True))

    s.state.turn_step = 160
    assert ("step 160/200", "warn") in bar.entries(show_elapsed=True)


def test_status_clear_erases_rendered_line(tmp_path, recording_output):
    status = StatusBar(session(tmp_path))
    status.output = recording_output
    status.rendered = True

    status.clear()

    assert recording_output.events == [("write", "\r"), ("erase", ""), ("flush", "")]
    assert not status.rendered


def test_clip_width_returns_unchanged_text_when_within_width():
    assert Text.clip_width("hello", 10) == "hello"
    assert Text.clip_width("", 5) == ""
    assert Text.clip_width("hello", 5) == "hello"


def test_clip_width_clips_wide_text_with_ellipsis():
    assert Text.clip_width("hello world", 8) == "hello..."
    # When width is less than 3, the ellipsis shrinks to fit
    assert Text.clip_width("hello world", 1) == "."
    assert Text.clip_width("hello world", 2) == ".."
    assert Text.clip_width("hello world", 3) == "..."
    assert Text.clip_width("hello world", 4) == "h..."


def test_clip_width_clamps_negative_width_to_zero():
    assert Text.clip_width("hello", -1) == ""


def test_clip_width_handles_cjk_wide_characters():
    assert Text.clip_width("你好世界", 5) == "你..."
    assert Text.clip_width("a你好", 5) == "a你好"


def test_choice_view_g_and_shift_g_jump_first_and_last():
    state = ChoiceViewState(choices=("one", "two", "three"), labels={}, disabled=set())

    state.handle_key("G")
    assert state.selected == 2
    state.handle_key("g")
    assert state.selected == 0

    # While searching, g/G are query text, not jumps.
    state.searching = True
    state.handle_key("g")
    assert state.query == "g"
    assert state.selected == 0


def test_choice_view_state_default_filtering():
    state = ChoiceViewState(
        choices=("alpha", "---", "beta", "---", "gamma"),
        labels={"alpha": "Alpha", "beta": "Beta", "gamma": "Gamma"},
        disabled={"---"},
    )
    assert state.visible() == ("alpha", "---", "beta", "---", "gamma")
    assert state.enabled() == ("alpha", "beta", "gamma")
    assert state.clamp() == ("alpha", "beta", "gamma")
    assert state.selected_choice() == "alpha"


def test_choice_view_state_search_filters_visible():
    state = ChoiceViewState(
        choices=("alpha", "---", "beta", "---", "gamma"),
        labels={"alpha": "Alpha", "beta": "Beta"},
        disabled={"---"},
    )
    state.set_query("beta")
    assert "beta" in state.visible()
    assert "alpha" not in state.visible()
    assert state.selected == 0


def test_choice_view_state_move_navigation():
    state = ChoiceViewState(
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
    state = ChoiceViewState(
        choices=("x",),
        labels={},
        disabled={"x"},
    )
    assert state.enabled() == ()
    assert state.selected_choice() is None


def test_choice_view_state_key_navigation_and_selection():
    state = ChoiceViewState(
        choices=("a", "---", "b", ChoiceViewState.FREE_TEXT),
        labels={ChoiceViewState.FREE_TEXT: "Type freely..."},
        disabled={"---"},
    )

    assert state.handle_key("j") is TUI_MODAL_PENDING
    assert state.selected_choice() == "b"
    assert state.handle_key("1") is TUI_MODAL_PENDING
    assert state.handle_key("enter") == "a"

    state.selected = 2
    assert state.handle_key("enter") is SELECTION_FREE_TEXT


def test_choice_view_state_search_and_escape_layers():
    state = ChoiceViewState(choices=("alpha", "beta"), labels={}, disabled=set())

    state.handle_key("/")
    state.handle_key("any", "b")
    assert state.searching
    assert state.query == "b"
    assert state.selected_choice() == "beta"
    assert state.handle_key("escape") is TUI_MODAL_PENDING
    assert not state.searching
    assert state.query == "b"
    assert state.handle_key("escape") is TUI_MODAL_PENDING
    assert state.query == ""
    assert state.handle_key("escape") is SELECTION_BACK


def test_choice_view_state_fragments_preserve_headers_and_preview():
    state = ChoiceViewState(
        choices=("--- Models ---", "alpha"),
        labels={"--- Models ---": "  ---- Models ----", "alpha": "Alpha"},
        disabled={"--- Models ---"},
    )

    fragments = state.fragments("Model", lambda _choice: "first\\nsecond")
    rendered = "".join(text for _style, text in fragments)

    assert "  ---- Models ----" in rendered
    assert ">  1. Alpha" in rendered
    assert "  │ first\n  │ second\n" in rendered


def test_emit_answer_compact_drops_invisible_lines(monkeypatch):
    out = []
    monkeypatch.setattr(render_module, "print_formatted_text", lambda text, **kwargs: out.append(getattr(text, "value", str(text))))
    ui = UiPrinter(output_fn=lambda text: None)
    ui.color = True
    ui.emit_answer("### Parent\n| status | value |\n| --- | --- |\n| model | `x` |\n", rule=False, compact=True)
    rendered = out[0]
    visible = [line for line in rendered.split("\n") if UiPrinter.SGR_RE.sub("", line).strip()]
    assert rendered.split("\n") == visible + [""]  # no blank or box-padding lines survive
    assert "Parent" in rendered and "model" in rendered
