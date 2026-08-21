"""Tests for TUI modal state machines and live preview logic.

These tests exercise the stateful parts of the TUI without requiring a real terminal.
"""

import os
import shutil
import time
from types import SimpleNamespace

import minacode.render as render_module
from minacode.base import LogBlock, LogEdge, LogLine, LogRole, TurnBox
from minacode.cli import CommandLoop
from minacode.cli.runtime import RESUME_STATUS_LABEL
from minacode.cli.view import View
from minacode.config import (
    Config,
)
from minacode.engine import Agent
from minacode.render import BashLivePreview, LiveSpark, Theme
from minacode.session import Session
from minacode.tui import TUI_MODAL_PENDING, ChoiceViewState, DiffViewState, TabbedViewState


def test_diff_view_state_tab_switching():
    view = DiffViewState(view=TabbedViewState(titles=("latest", "net")))
    assert view.view.tab == 0
    view.switch_tab(1)
    assert view.view.tab == 1
    assert view.mode == view.Mode.LIST
    view.switch_tab(-1)
    assert view.view.tab == 0


def test_diff_view_state_file_navigation():
    view = DiffViewState(view=TabbedViewState(titles=("latest",)))
    view.move_file(1, 3)
    assert view.file == 1
    view.move_file(1, 3)
    assert view.file == 2
    view.move_file(1, 3)
    assert view.file == 0
    view.move_file(-1, 3)
    assert view.file == 2


def test_diff_view_state_open_and_close_file():
    view = DiffViewState(view=TabbedViewState(titles=("latest",)))
    assert view.mode == view.Mode.LIST
    view.open_file(2)
    assert view.mode == view.Mode.FILE
    assert view.view.scroll == 0
    view.close_file()
    assert view.mode == view.Mode.LIST


def test_diff_view_state_handle_key():
    view = DiffViewState(view=TabbedViewState(titles=("latest", "net")))
    # Down in list mode moves file
    result = view.handle_key("down", file_count=3, viewport=10)
    assert result == TUI_MODAL_PENDING
    assert view.file == 1
    # Enter opens file
    result = view.handle_key("enter", file_count=3, viewport=10)
    assert result == TUI_MODAL_PENDING
    assert view.mode == view.Mode.FILE
    # Page down in file mode scrolls
    result = view.handle_key("pagedown", file_count=3, viewport=10)
    assert result == TUI_MODAL_PENDING
    assert view.view.scroll == 10
    # Escape closes file
    result = view.handle_key("escape", file_count=3, viewport=10)
    assert result == TUI_MODAL_PENDING
    assert view.mode == view.Mode.LIST
    # q exits
    assert view.handle_key("q", file_count=3, viewport=10) is None


def test_choice_view_state_filtering():
    state = ChoiceViewState(
        choices=("a", "b", "c", "d"),
        labels={"a": "alpha", "b": "beta", "c": "gamma", "d": "delta"},
        disabled=set(),
    )
    assert state.visible() == ("a", "b", "c", "d")
    state.set_query("al")
    assert state.visible() == ("a",)
    state.set_query("be")
    assert state.visible() == ("b",)
    state.set_query("")
    assert state.visible() == ("a", "b", "c", "d")


def test_choice_view_state_disabled_headers():
    state = ChoiceViewState(
        choices=("header", "a", "b", "other", "c"),
        labels={},
        disabled={"header", "other"},
    )
    assert state.visible() == ("header", "a", "b", "other", "c")
    assert state.enabled() == ("a", "b", "c")
    state.set_query("a")
    assert state.visible() == ("header", "a")
    assert state.enabled() == ("a",)


def test_choice_view_state_movement():
    state = ChoiceViewState(
        choices=("a", "b", "c"),
        labels={},
        disabled=set(),
    )
    assert state.selected == 0
    state.move(1)
    assert state.selected == 1
    state.move(1)
    assert state.selected == 2
    state.move(1)  # clamp at end
    assert state.selected == 2
    state.move(-1)
    assert state.selected == 1
    state.move(-10)  # clamp at start
    assert state.selected == 0


def test_choice_view_state_selected_choice():
    state = ChoiceViewState(
        choices=("a", "b", "c"),
        labels={},
        disabled={"b"},
    )
    assert state.selected_choice() == "a"
    state.move(1)
    assert state.selected_choice() == "c"  # skips disabled b
    state.set_query("z")
    assert state.selected_choice() is None


def test_choice_view_state_window_uncapped_and_fitting():
    choices = tuple(f"r{index}" for index in range(50))
    state = ChoiceViewState(choices=choices, labels={}, disabled=set(), max_rows=0)  # no cap: the whole list
    visible = state.visible()
    assert state.window(visible, state.clamp()) == (0, 50)

    state.max_rows = 50  # exactly fits: still the whole list, no viewport
    assert state.window(visible, state.clamp()) == (0, 50)

    state.max_rows = 10
    assert state.window(visible, state.clamp()) == (0, 10)  # a long list caps at the viewport size


def test_choice_view_state_window_centers_the_selection_and_clamps_to_edges():
    choices = tuple(f"r{index}" for index in range(50))
    state = ChoiceViewState(choices=choices, labels={}, disabled=set(), max_rows=10)
    visible = state.visible()

    state.selected = 0  # head: anchored at the top
    assert state.window(visible, state.clamp()) == (0, 10)

    state.selected = 2  # near the head: centering is clamped by the top edge, not shifted down
    assert state.window(visible, state.clamp()) == (0, 10)

    state.selected = 24  # middle: centred, half the viewport above, half below
    assert state.window(visible, state.clamp()) == (19, 29)

    state.selected = 49  # tail: the window never runs past the last row
    assert state.window(visible, state.clamp()) == (40, 50)


def test_choice_view_state_window_counts_disabled_headers_in_the_row_range():
    # Filtering keeps section headers in `visible`; the window is over `visible`, so a header row
    # shifts the selection's row index but not the numbering (which counts enabled rows only).
    choices = ("topic", *tuple(f"r{index}" for index in range(30)), "tail", "extra")
    state = ChoiceViewState(choices=choices, labels={}, disabled={"topic", "tail"}, max_rows=10)
    state.set_query("r")  # drops "topic"; "extra" survives the filter
    visible = state.visible()
    options = state.clamp()
    assert len(options) == 31 and len(visible) == 33  # 2 headers + 31 enabled rows

    state.selected = 0  # "r0" sits one row below the first header
    assert state.window(visible, state.clamp()) == (0, 10)

    state.selected = 15  # centred over the row index, header included
    assert state.window(visible, state.clamp()) == (11, 21)

    state.selected = len(options) - 1  # "extra" is the last visible row
    assert state.window(visible, state.clamp()) == (23, 33)


def test_choice_view_state_window_counter_and_numbering_stay_stable():
    state = ChoiceViewState(choices=tuple(f"r{index}" for index in range(50)), labels={}, disabled=set(), max_rows=10)
    state.selected = 49
    parts = state.fragments("list")
    text = "".join(value for _style, value in parts)
    assert "showing 41-50 of 50" in text
    assert "41. r40" in text and "50. r49" in text

    state.selected = 0
    text = "".join(value for _style, value in state.fragments("list"))
    assert "showing 1-10 of 50" in text
    assert "1. r0" in text and "10. r9" in text


def test_bash_live_preview_frame_rows():
    preview = BashLivePreview()
    preview.active = True
    preview.text = "line1\nline2\n"
    preview.started_at = time.monotonic() - 1.5

    rows = preview.frame_rows()
    lines = ["".join(text for _style, text in row) for row in rows]
    assert any("line1" in line for line in lines)
    assert any("line2" in line for line in lines)
    assert any("output" in line.lower() or "running" in line.lower() for line in lines)
    # The spark caps the rail in place of the old BRANCH glyph, which hung off a root that the
    # frame never had: it is built as `hierarchy(None, ...)`. Same mark as the stream preview --
    # both say the same thing, that the region is live with nothing new to show yet.
    assert lines[0].startswith(LogBlock.margin(2) + LiveSpark.GLYPH)
    assert rows[0][0][0] == LiveSpark.style(preview.started_at)  # the one fragment that breathes,
    # anchored to this command rather than the wall clock, so it opens at its crest.
    preview.started_at = time.monotonic()
    assert preview.frame_rows()[0][0][0] == LiveSpark.ramp()[-1]
    assert not any(LogEdge.BRANCH.value in line for line in lines)


def test_bash_live_preview_text_accumulation():
    preview = BashLivePreview()
    preview.active = True
    preview.update("hello ")
    preview.update("world")
    assert preview.text == "hello world"
    preview.update("x" * preview.MAX_CHARS)
    assert len(preview.text) <= preview.MAX_CHARS


def test_bash_live_preview_finish():
    preview = BashLivePreview()
    preview.active = True
    preview.text = "output"
    preview.finish()
    assert not preview.active
    assert preview.text == ""


def test_live_spark_breathes_across_a_wide_range_of_the_divider_accent(monkeypatch):
    """A slow triangular breath, dark to bright and back, around the divider's own accent.

    The reach matters as much as the curve: a shallow fade reads as the terminal mis-drawing a
    cell. It spans at least as far as WAITING_PULSE_STYLES, the pulse it is a sibling of."""
    clock = [0.0]
    monkeypatch.setattr(render_module.time, "monotonic", lambda: clock[0])
    ramp = LiveSpark.ramp()

    # Measured from the region's own start, and opening at the crest: the frame that announces a
    # region has to be its loudest. Timed off the wall clock, a region appearing near the trough
    # would open near-black and stay unreadable for over a second.
    for born in (0.0, 0.7, 1.5, 2.6, 11.3):  # wherever the wall clock happens to be
        clock[0] = born
        assert LiveSpark.style(started_at=born) == ramp[-1]

    clock[0] = 100.0 + LiveSpark.PERIOD / 2
    assert LiveSpark.style(started_at=100.0) == ramp[0]  # trough at the half-way point

    clock[0] = 100.0 + LiveSpark.PERIOD  # and back up: the breath is a loop, not a sawtooth
    assert LiveSpark.style(started_at=100.0) == ramp[-1]

    clock[0] = 7.0  # no anchor: still breathing, just at whatever phase the clock is in
    assert LiveSpark.style() in ramp

    # The two stars swap at the darkest point of the breath (the half-period), so the change
    # reads as the color fading rather than a flicker; phase zero is GLYPH.
    clock[0] = 0.0
    assert LiveSpark.glyph(started_at=0.0) == LiveSpark.GLYPH
    clock[0] = 0.25 * LiveSpark.PERIOD
    assert LiveSpark.glyph(started_at=0.0) == LiveSpark.GLYPH  # the bright half keeps GLYPH
    clock[0] = 0.75 * LiveSpark.PERIOD
    assert LiveSpark.glyph(started_at=0.0) == LiveSpark.GLYPHS[1]  # swapped after the trough
    clock[0] = LiveSpark.PERIOD
    assert LiveSpark.glyph(started_at=0.0) == LiveSpark.GLYPH  # and back at the crest

    def luma(style):
        red, green, blue = Theme.rgb(style.split()[0])
        return 0.299 * red + 0.587 * green + 0.114 * blue

    accent = Theme.style(LiveSpark.ROLE)
    assert luma(ramp[0]) < luma(accent) < luma(ramp[-1])  # the breath brackets the accent
    assert luma(ramp[-1]) - luma(ramp[0]) >= luma(View.WAITING_PULSE_STYLES[-1]) - luma(View.WAITING_PULSE_STYLES[0])
    assert all(step.endswith(" bold") for step in ramp)  # the star is thin; bold carries its weight

    # Slower than the divider's in-flight heartbeat, which sits above a much quieter line.
    assert LiveSpark.PERIOD > View.WAITING_PULSE_PERIOD


def test_model_stream_preview_draws_the_same_tree_as_the_log(tmp_path):
    """The spark's row belongs to the spark: a gray phase word (`thinking`, then `responding`) sits
    beside it, and the streamed text starts on its own rail row below, so the first line never
    races for whatever room the spark leaves -- with or without the word, the layout is the same.

    The rows carry CONTINUE and nothing carries BRANCH -- `├` is a T-junction, and there is no
    line above the block for one to join. Nothing closes it either: the stream is still arriving,
    and a `└` would say it had finished."""
    config = Config()
    config.data_dir = str(tmp_path / "data")
    loop = CommandLoop(Agent(Session(cwd=str(tmp_path), config=config)), input_fn=lambda _prompt: "", output_fn=lambda _text: None)

    loop.model_stream_output("reasoning", "weighing the two paths\nthe second option is cleaner")
    lines = "".join(text for _style, text in loop.view.model_stream_fragments()).splitlines()

    rail = LogBlock.prefix(TurnBox.CONTENT_LEVEL + 1, LogEdge.CONTINUE)
    assert any(
        lines[0] == LogBlock.margin(TurnBox.CONTENT_LEVEL + 1) + glyph + "thinking" for glyph in LiveSpark.GLYPHS
    )  # either star may lead the preview: no anchor means the wall clock picks the phase
    assert lines[1] == ""  # the blank row lifts the spark off the rail below it
    assert lines[2] == rail + "weighing the two paths"
    assert lines[3] == rail + "the second option is cleaner"
    assert len(LiveSpark.GLYPH) == len(LogBlock.RAIL)  # so the spark sits in the rail's column
    assert all(len(glyph) == len(LogBlock.RAIL) for glyph in LiveSpark.GLYPHS)  # and its swapped partner does too
    assert not any(LogEdge.BRANCH.value in line or LogEdge.END.value in line for line in lines)
    # The column a tool's own output lines are drawn in: the two trees share a grid.
    tool = str(LogBlock.hierarchy(LogLine("Bash", "pytest -q", LogRole.TOOL), [LogLine("", "output line", LogRole.OUTPUT, LogEdge.CONTINUE)]))
    assert tool.splitlines()[1].index(LogEdge.CONTINUE.value) == lines[2].index(LogEdge.CONTINUE.value)


def test_model_stream_preview_switches_phase_and_clears(tmp_path):
    config = Config()
    config.data_dir = str(tmp_path / "data")
    loop = CommandLoop(Agent(Session(cwd=str(tmp_path), config=config)), input_fn=lambda _prompt: "", output_fn=lambda _text: None)

    # The phase word rides beside the spark and follows the stream: `thinking` while the model
    # reasons, `responding` once it answers; the preview carries only the text besides that.
    loop.model_stream_output("reasoning", "checking the request")
    reasoning = "".join(text for _style, text in loop.view.model_stream_fragments())
    assert "checking the request" in reasoning
    assert "thinking" in reasoning
    assert "thinking" in "".join(text for _style, text in loop.view.queue_divider_fragments())

    loop.model_stream_output("output", "answering now")
    output = "".join(text for _style, text in loop.view.model_stream_fragments())
    assert "answering now" in output
    assert "thinking" not in output  # the word follows the phase instead of staying stale
    assert "responding" in output
    assert "checking the request" not in output
    assert "responding" in "".join(text for _style, text in loop.view.queue_divider_fragments())

    loop.model_stream_output("correcting malformed tool call 1/5 · Bash", "")
    assert loop.view.model_stream_fragments() == []
    divider = "".join(text for _style, text in loop.view.queue_divider_fragments())
    assert "correcting malformed tool call 1/5 · Bash" in divider  # the phase stays whole

    loop.model_stream_output("", "")
    assert loop.view.model_stream_fragments() == []
    assert "working" in "".join(text for _style, text in loop.view.queue_divider_fragments())


def test_model_stream_preview_styles_inline_markdown(tmp_path, monkeypatch):
    """Closed inline markdown tokens in the live preview render with their own styles -- bold,
    code, italic -- marked in the preview's own gray tones (no color), while plain text stays
    gray and an unclosed marker stays literal, so a growing stream never toggles a token's
    style frame to frame."""
    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback: os.terminal_size((100, 20)))
    config = Config()
    config.data_dir = str(tmp_path / "data")
    loop = CommandLoop(Agent(Session(cwd=str(tmp_path), config=config)), input_fn=lambda _prompt: "", output_fn=lambda _text: None)

    loop.model_stream_output("reasoning", "**bold** `code` *italic* and **unclosed")

    styled = {(text, style) for style, text in loop.view.model_stream_fragments() if text}
    assert ("bold", "ansibrightblack bold") in styled
    assert ("code", "ansibrightblack underline") in styled
    assert ("italic", "ansibrightblack italic") in styled
    assert (" and **unclosed", "ansibrightblack") in styled  # no closing marker: the tail stays literal


def test_model_stream_preview_keeps_malformed_star_runs_literal(tmp_path, monkeypatch):
    """A lone star can never be borrowed from a double-star run: `**a*` and `*a**` are unclosed
    markers, not italic, and stay gray instead of mis-rendering as `*a*`."""
    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback: os.terminal_size((100, 20)))
    config = Config()
    config.data_dir = str(tmp_path / "data")
    loop = CommandLoop(Agent(Session(cwd=str(tmp_path), config=config)), input_fn=lambda _prompt: "", output_fn=lambda _text: None)

    loop.model_stream_output("reasoning", "**a* *a** **** **a**b**")

    styled = {(text, style) for style, text in loop.view.model_stream_fragments() if text}
    assert ("**a* *a** **** ", "ansibrightblack") in styled  # unclosed and empty star runs stay literal
    assert ("a", "ansibrightblack bold") in styled  # the closed bold inside the last token still renders
    assert ("b**", "ansibrightblack") in styled  # the trailing unclosed run stays literal
    assert not any(text == "a" and "italic" in style for text, style in styled)  # no italic borrowed from `**`


def test_sweep_divider_widens_for_long_labels_and_keeps_the_track(tmp_path, monkeypatch):
    """A label that would fill the default rule is accommodated by widening the rule instead of
    being clipped: both sides keep at least MIN_TRAIL dashes (up to the terminal width) and the
    label text stays whole, so the comet reads as motion rather than a frantic bounce."""
    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback: os.terminal_size((100, 20)))
    config = Config()
    config.data_dir = str(tmp_path / "data")
    loop = CommandLoop(Agent(Session(cwd=str(tmp_path), config=config)), input_fn=lambda _prompt: "", output_fn=lambda _text: None)
    view = loop.view

    long_label = "[worker] thinking (5m07s · ↓ 75 tok/s) [ 1 queued ]"
    fragments = view.sweep_divider_fragments(long_label)
    dashes = sum(1 for _style, text in fragments if text == "-")
    assert dashes >= 3 + 12  # lead + the minimum trail
    assert any(text == long_label for _style, text in fragments)  # never clipped

    short_label = "working"
    plain = view.sweep_divider_fragments(short_label)
    assert any(text == short_label for _style, text in plain)  # a short label is never clipped


def test_queue_divider_resuming_status_is_a_quiet_gray_line(tmp_path):
    """While a session is being restored the divider is one gray line: no sweep, no pulse, no
    elapsed time, because nothing is streaming and the replay that follows is the whole story."""
    config = Config()
    config.data_dir = str(tmp_path / "data")
    loop = CommandLoop(Agent(Session(cwd=str(tmp_path), config=config)), input_fn=lambda _prompt: "", output_fn=lambda _text: None)
    loop.tui = SimpleNamespace(status_label=RESUME_STATUS_LABEL)
    assert loop.view.queue_divider_fragments() == [("ansibrightblack", RESUME_STATUS_LABEL)]


def test_divider_shows_output_rate_while_a_response_streams(tmp_path):
    """The elapsed time says how long the wait has been; the rate says whether it is moving. Both
    live in the same parenthesis, and the rate leaves when the stream does."""
    config = Config()
    config.data_dir = str(tmp_path / "data")
    session = Session(cwd=str(tmp_path), config=config)
    loop = CommandLoop(Agent(session), input_fn=lambda _prompt: "", output_fn=lambda _text: None)

    loop.model_stream_output("output", "answering now")
    assert "tok/s" not in "".join(text for _style, text in loop.view.queue_divider_fragments())

    loop.status_bar.started_at = time.monotonic() - 4.0
    session.state.stream_started_at = time.monotonic() - 4.0
    session.state.stream_chars = 800
    assert "responding (4s · ↓ 50 tok/s)" in "".join(text for _style, text in loop.view.queue_divider_fragments())

    session.state.stream_started_at = 0.0
    label = "".join(text for _style, text in loop.view.queue_divider_fragments())
    assert "responding (" in label and "tok/s" not in label


def test_sent_followup_moves_above_activity_and_failed_request_requeues_it(tmp_path):
    config = Config()
    config.data_dir = str(tmp_path / "data")
    session = Session(cwd=str(tmp_path), config=config)
    loop = CommandLoop(Agent(session), input_fn=lambda _prompt: "", output_fn=lambda _text: None)
    session.enqueue_user_input("use black instead")
    claimed = session.claim_user_inputs()
    loop.model_stream_output("reasoning", "checking the formatter")

    activity = "".join(text for _style, text in loop.view.tui_activity_fragments())
    assert activity.count("use black instead") == 1
    # Either star may lead the preview row; the text after it is what the order checks.
    assert activity.index("• use black instead") < activity.index("checking the formatter") < activity.rindex("thinking")
    assert "+ use black instead" not in activity
    assert "queued" not in activity and "sent" not in activity

    session.release_user_inputs()
    requeued = "".join(text for _style, text in loop.view.tui_activity_fragments())
    assert "• use black instead" not in requeued
    assert "[ 1 queued ]" in requeued
    assert requeued.rindex("thinking") < requeued.index("+ use black instead")

    session.claim_user_inputs()
    session.acknowledge_user_inputs(claimed)
    committed = "".join(text for _style, text in loop.view.tui_activity_fragments())
    assert "use black instead" not in committed


def test_activity_echo_gets_a_blank_row_before_the_divider(tmp_path):
    """The echoed follow-up sits directly above the standing divider while no stream exists.
    Without a blank row it presses against the divider; the same gap the stream path already
    draws must separate it from the divider too."""
    config = Config()
    config.data_dir = str(tmp_path / "data")
    session = Session(cwd=str(tmp_path), config=config)
    loop = CommandLoop(Agent(session), input_fn=lambda _prompt: "", output_fn=lambda _text: None)
    session.enqueue_user_input("\u770b\u770b shot.png")
    session.claim_user_inputs()  # in-flight: the echo renders above the divider

    activity = "".join(text for _style, text in loop.view.tui_activity_fragments())
    lines = activity.splitlines()
    echo = next(index for index, line in enumerate(lines) if line.startswith("\u2022 \u770b\u770b shot.png"))
    divider = next(index for index, line in enumerate(lines) if "working" in line)
    assert divider == echo + 2  # one blank row between the echo and the divider


def test_model_stream_preview_keeps_only_the_latest_six_lines(tmp_path, monkeypatch):
    config = Config()
    config.data_dir = str(tmp_path / "data")
    loop = CommandLoop(Agent(Session(cwd=str(tmp_path), config=config)), input_fn=lambda _prompt: "", output_fn=lambda _text: None)
    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback: os.terminal_size((40, 20)))

    loop.model_stream_output("output", "\n".join(f"line {index} with a deliberately long suffix" for index in range(8)))

    preview = "".join(text for _style, text in loop.view.model_stream_fragments())
    assert "line 0" not in preview
    assert "line 1" not in preview
    assert "line 2" in preview
    assert "line 7" in preview
    assert all(len(line) <= 40 for line in preview.splitlines())
