"""Tests for TUI modal state machines and live preview logic.

These tests exercise the stateful parts of the TUI without requiring a real terminal.
"""

import time

import nanocode as n


def test_diff_view_state_tab_switching():
    view = n.DiffViewState(view=n.TabbedViewState(titles=("latest", "net")))
    assert view.view.tab == 0
    view.switch_tab(1)
    assert view.view.tab == 1
    assert view.mode == view.Mode.LIST
    view.switch_tab(-1)
    assert view.view.tab == 0


def test_diff_view_state_file_navigation():
    view = n.DiffViewState(view=n.TabbedViewState(titles=("latest",)))
    view.move_file(1, 3)
    assert view.file == 1
    view.move_file(1, 3)
    assert view.file == 2
    view.move_file(1, 3)
    assert view.file == 0
    view.move_file(-1, 3)
    assert view.file == 2


def test_diff_view_state_open_and_close_file():
    view = n.DiffViewState(view=n.TabbedViewState(titles=("latest",)))
    assert view.mode == view.Mode.LIST
    view.open_file(2)
    assert view.mode == view.Mode.FILE
    assert view.view.scroll == 0
    view.close_file()
    assert view.mode == view.Mode.LIST


def test_diff_view_state_handle_key():
    view = n.DiffViewState(view=n.TabbedViewState(titles=("latest", "net")))
    # Down in list mode moves file
    result = view.handle_key("down", file_count=3, viewport=10)
    assert result == n.TUI_MODAL_PENDING
    assert view.file == 1
    # Enter opens file
    result = view.handle_key("enter", file_count=3, viewport=10)
    assert result == n.TUI_MODAL_PENDING
    assert view.mode == view.Mode.FILE
    # Page down in file mode scrolls
    result = view.handle_key("pagedown", file_count=3, viewport=10)
    assert result == n.TUI_MODAL_PENDING
    assert view.view.scroll == 10
    # Escape closes file
    result = view.handle_key("escape", file_count=3, viewport=10)
    assert result == n.TUI_MODAL_PENDING
    assert view.mode == view.Mode.LIST
    # q exits
    assert view.handle_key("q", file_count=3, viewport=10) is None


def test_choice_view_state_filtering():
    state = n.ChoiceViewState(
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
    state = n.ChoiceViewState(
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
    state = n.ChoiceViewState(
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
    state = n.ChoiceViewState(
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
    preview = n.BashLivePreview()
    preview.active = True
    preview.text = "line1\nline2\n"
    preview.started_at = time.monotonic() - 1.5

    lines = preview.frame_lines()
    assert any("line1" in line for line in lines)
    assert any("line2" in line for line in lines)
    assert any("output" in line.lower() or "running" in line.lower() for line in lines)


def test_bash_live_preview_text_accumulation():
    preview = n.BashLivePreview()
    preview.active = True
    preview.update("hello ")
    preview.update("world")
    assert preview.text == "hello world"
    preview.update("x" * preview.MAX_CHARS)
    assert len(preview.text) <= preview.MAX_CHARS


def test_bash_live_preview_finish():
    preview = n.BashLivePreview()
    preview.active = True
    preview.text = "output"
    preview.finish()
    assert not preview.active
    assert preview.text == ""
