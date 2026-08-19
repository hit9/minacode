"""Tests for TUI modal state machines and live preview logic.

These tests exercise the stateful parts of the TUI without requiring a real terminal.
"""

import os
import shutil
import time

from minacode.cli import CommandLoop
from minacode.config import (
    Config,
)
from minacode.engine import Agent
from minacode.render import BashLivePreview
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


def test_bash_live_preview_frame_lines():
    preview = BashLivePreview()
    preview.active = True
    preview.text = "line1\nline2\n"
    preview.started_at = time.monotonic() - 1.5

    lines = preview.frame_lines()
    assert any("line1" in line for line in lines)
    assert any("line2" in line for line in lines)
    assert any("output" in line.lower() or "running" in line.lower() for line in lines)


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


def test_model_stream_preview_switches_phase_and_clears(tmp_path):
    config = Config()
    config.data_dir = str(tmp_path / "data")
    loop = CommandLoop(Agent(Session(cwd=str(tmp_path), config=config)), input_fn=lambda _prompt: "", output_fn=lambda _text: None)

    loop.model_stream_output("reasoning", "checking the request")
    reasoning = "".join(text for _style, text in loop.view.model_stream_fragments())
    assert "thinking" in reasoning
    assert "checking the request" in reasoning
    assert "thinking" in "".join(text for _style, text in loop.view.queue_divider_fragments())

    loop.model_stream_output("output", "answering now")
    output = "".join(text for _style, text in loop.view.model_stream_fragments())
    assert "responding" in output
    assert "answering now" in output
    assert "checking the request" not in output
    assert "responding" in "".join(text for _style, text in loop.view.queue_divider_fragments())

    loop.model_stream_output("correcting malformed tool call 1/5 · Bash", "")
    assert loop.view.model_stream_fragments() == []
    assert "correcting malformed tool call 1/5 · Bash" in "".join(text for _style, text in loop.view.queue_divider_fragments())

    loop.model_stream_output("", "")
    assert loop.view.model_stream_fragments() == []
    assert "working" in "".join(text for _style, text in loop.view.queue_divider_fragments())


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
    assert "responding (4s · ~50 tok/s)" in "".join(text for _style, text in loop.view.queue_divider_fragments())

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
    assert activity.index("• use black instead") < activity.index("├─ thinking") < activity.rindex("thinking")
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
