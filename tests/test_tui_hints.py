"""tui hints (split from tests/test_tui_app.py)."""
import asyncio
import multiprocessing
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
from tui_harness import ResizableOutput, loop, rendered_screen_text, run_interactive_tui, session, wait_until
import minacode.tui.app as tui_module
from minacode.base import (
    SESSION_EVENT_KEY,
    LogBlock,
    LogEdge,
)
from minacode.cli import CommandCompleter, CommandLoop, TuiRuntime, hints
from minacode.cli.commands import select_choice
from minacode.cli.hints import HintPicker
from minacode.cli.update import UpdateChecker
from minacode.config import (
    Config,
)
from minacode.engine import Agent
from minacode.mentions import FilePick, active_mention
from minacode.prompts import LIVE_FOLLOWUP_PREFIX
from minacode.session import Session, SessionSnapshotStore
from minacode.tools import CodeIndex
from minacode.tui import TUI_MODAL_PENDING, CallbackPlaceholder, TuiApp
from test_tui_app import _StubJob, quick_hint_app

def test_tui_chat_input_shows_random_idle_placeholder(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()

    hint = command_loop.view.tui_input_hint()
    assert hint in {entry.text for entry in hints.HINTS}
    assert command_loop.view.tui_input_hint() == hint  # stable within a situation (no flicker)

def test_tui_idle_hint_sessions_only_before_work(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    command_loop.view._hint_picker = HintPicker(choice=lambda pool: pool[-1])

    # Early session: the pool ends with the /sessions hint (the only early-only entry).
    assert command_loop.view.tui_input_hint() == "/sessions resumes a past session"

    # Once work exists the session is no longer early and /sessions leaves the pool.
    command_loop.session.store_tool_result("Bash", ["ls"], "ok")
    assert command_loop.view.tui_input_hint() == "Type / for commands"  # last technique hint

def test_tui_idle_hint_favors_diff_right_after_editing(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    command_loop.view._hint_picker = HintPicker(choice=lambda pool: pool[-1])
    command_loop.session.store_tool_result("Bash", ["ls"], "ok")  # mature phase
    command_loop.session.state.round_count = 1
    command_loop.session.store_turn_diff("tr.1", 1, "a.py", "diff", round=1)

    # The post-edit pool ends with the weighted /diff copies.
    assert command_loop.view.tui_input_hint() == "/diff reviews recent edits"

    # A later round without edits drops /diff back out of the pool.
    command_loop.session.state.round_count = 2
    assert command_loop.view.tui_input_hint() == "Type / for commands"

def test_tui_idle_hint_ps_while_jobs_running(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    command_loop.view._hint_picker = HintPicker(choice=lambda pool: pool[-1])
    command_loop.session.store_tool_result("Bash", ["ls"], "ok")  # mature

    assert command_loop.view.tui_input_hint() == "Type / for commands"  # no jobs yet

    command_loop.session.jobs["j1"] = _StubJob("running")
    assert command_loop.view.tui_input_hint() == "/ps lists background jobs"

    command_loop.session.jobs["j1"] = _StubJob("done")  # finished -> hint clears
    assert command_loop.view.tui_input_hint() == "Type / for commands"

def test_tui_hint_context_projects_availability(tmp_path):
    command_loop = loop(tmp_path)
    session = command_loop.session

    session.skills = None
    session.mcp = None
    session.jobs.clear()
    ctx = command_loop.view._hint_context()
    assert not ctx.skills_available and not ctx.mcp_connected and not ctx.jobs_running

    session.skills = SimpleNamespace(skills={"demo": object()})
    session.mcp = SimpleNamespace(tools={"srv": []})
    session.jobs["j1"] = _StubJob("running")
    ctx = command_loop.view._hint_context()
    assert ctx.skills_available and ctx.mcp_connected and ctx.jobs_running

def test_tui_idle_hint_rerolls_each_turn(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    picks = iter(["first", "second"])
    command_loop.view._hint_picker = HintPicker(choice=lambda pool: next(picks))
    command_loop.session.state.round_count = 1
    assert command_loop.view.tui_input_hint() == "first"
    assert command_loop.view.tui_input_hint() == "first"  # stable within the round
    command_loop.session.state.round_count = 2
    assert command_loop.view.tui_input_hint() == "second"  # a new turn re-rolls

def test_quick_hint_tab_cycles_focus_and_wraps():
    app, _ = quick_hint_app()
    assert app.quick_hint_focus == -1
    for expected in (0, 1, 2, -1):
        app.tab_or_complete(app.input_buffer, reverse=False)
        assert app.quick_hint_focus == expected

def test_quick_hint_shift_tab_cycles_focus_backwards_and_wraps():
    app, _ = quick_hint_app()
    assert app.quick_hint_focus == -1
    for expected in (2, 1, 0, -1):
        app.tab_or_complete(app.input_buffer, reverse=True)
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

def test_quick_hint_enter_picks_chip_and_returns_focus():
    app, submitted = quick_hint_app()
    app.quick_hint_focus = 1
    assert app._pick_quick_hint(app.input_buffer)
    assert submitted == []
    assert app.input_buffer.text == "show the diff"
    assert app.quick_hint_focus == -1  # Enter returns focus to the input line
    app._accept(app.input_buffer)  # a second Enter sends
    assert [str(value) for value in submitted] == ["show the diff"]

def test_quick_hint_pick_ignored_with_completion_menu_open():
    app, submitted = quick_hint_app()
    app.quick_hint_focus = 1
    app.input_buffer.complete_state = object()  # a completion menu is open
    assert app._pick_quick_hint(app.input_buffer) is False
    assert submitted == []
    assert app.input_buffer.text == ""

def test_quick_hint_pick_ignored_while_running():
    app, submitted = quick_hint_app()
    app.quick_hint_focus = 0
    app.set_running("working")
    assert app._pick_quick_hint(app.input_buffer) is False
    assert submitted == []

def test_quick_hint_enter_on_empty_unfocused_input_does_nothing():
    app, submitted = quick_hint_app()
    app._accept(app.input_buffer)
    assert submitted == []

def test_quick_hint_enter_picks_chips_one_per_tab_and_sends():
    """Enter picks the focused chip and returns to the prompt; Tab to the next chip and Enter
    again combines, and a final Enter sends the whole text."""
    app, submitted = quick_hint_app()
    app.quick_hint_focus = 0
    assert app._pick_quick_hint(app.input_buffer)  # Enter picks "run the tests"
    assert app.input_buffer.text == "run the tests"
    assert app.quick_hint_focus == -1  # focus is back on the input line
    app.tab_or_complete(app.input_buffer, reverse=False)  # resumes after picked chip 0 -> 1
    assert app._pick_quick_hint(app.input_buffer)  # Enter picks "show the diff"
    assert app.input_buffer.text == "run the tests\nshow the diff"
    assert app.quick_hint_picked == ["run the tests", "show the diff"]
    assert submitted == []
    assert app._accept(app.input_buffer)  # a final Enter sends
    assert [str(value) for value in submitted] == ["run the tests\nshow the diff"]

def test_quick_hint_enter_keys_pick_and_send_through_real_bindings(monkeypatch):
    """The wiring, not just the methods: Tab focuses, Enter picks and returns to the prompt, and
    only the Enter with no chip focused sends."""
    received = []
    app = None

    def submit(text):
        received.append(str(text))
        app.set_idle()

    app = TuiApp(on_chat_submit=submit, quick_hints_fn=lambda: ("run the tests", "show the diff", "commit"))
    app.set_idle()

    # Tab focuses chip 0, Enter picks it; one Tab reaches chip 1, Enter picks it; the final
    # Enter, with focus back on the input line, sends the combined text.
    run_interactive_tui(monkeypatch, app, text="\t\r\t\r\r\x04")

    assert received == ["run the tests\nshow the diff"]

def test_quick_hint_space_is_plain_through_real_bindings(monkeypatch):
    """Space no longer picks a focused chip: it reaches the buffer as an ordinary character."""
    received = []
    app = None

    def submit(text):
        received.append(str(text))
        app.set_idle()

    app = TuiApp(on_chat_submit=submit, quick_hints_fn=lambda: ("run the tests", "show the diff", "commit"))
    app.set_idle()

    # Tab focuses chip 0, but the space lands in the buffer instead of picking it.
    run_interactive_tui(monkeypatch, app, text="\t ok\r\x04")

    assert received == [" ok"]

def test_quick_hint_enter_again_unpicks_chip():
    app, _ = quick_hint_app()
    app.quick_hint_focus = 0
    app._pick_quick_hint(app.input_buffer)  # pick "run the tests", focus back on the input
    assert app.input_buffer.text == "run the tests"
    app.quick_hint_focus = 0  # cycling back to a picked chip makes Enter a toggle
    assert app._pick_quick_hint(app.input_buffer)  # Enter toggles it off
    assert app.quick_hint_picked == []
    assert app.input_buffer.text == ""

def test_quick_hint_pick_uses_one_hint_snapshot():
    states = iter((("old hint",), ()))
    app = TuiApp(quick_hints_fn=lambda: next(states))
    app.set_idle()
    app.quick_hint_focus = 0

    app._pick_quick_hint(app.input_buffer)
    assert app.input_buffer.text == "old hint"

def test_quick_hint_tab_cycles_while_picked():
    app, _ = quick_hint_app()
    app.quick_hint_focus = 0
    app._pick_quick_hint(app.input_buffer)  # pick -> focus back on the input
    for expected in (1, 2, -1, 0):
        app.tab_or_complete(app.input_buffer, reverse=False)
        assert app.quick_hint_focus == expected

def test_quick_hint_manual_edit_drops_picked_and_sends_edited_text():
    app, submitted = quick_hint_app()
    app.quick_hint_focus = 0
    app._pick_quick_hint(app.input_buffer)  # pick -> buffer = "run the tests"
    app.input_buffer.insert_text("!")
    assert app.quick_hint_picked == []
    assert app.quick_hint_focus == -1
    assert app.input_buffer.text == "run the tests!"
    app._accept(app.input_buffer)
    assert [str(value) for value in submitted] == ["run the tests!"]

def test_quick_hint_hints_change_resets_picked_keeps_text():
    hints = ["run the tests", "show the diff"]
    submitted = []
    app = TuiApp(on_chat_submit=submitted.append, quick_hints_fn=lambda: tuple(hints))
    app.set_idle()
    app.quick_hint_focus = 0
    app._pick_quick_hint(app.input_buffer)  # pick -> buffer = "run the tests"
    assert app.quick_hint_picked == ["run the tests"]
    hints[:] = ["commit", "push"]
    app.quick_hints()  # lazy comparison resets the picked state
    assert app.quick_hint_picked == []
    assert app.quick_hint_focus == -1
    assert app.input_buffer.text == "run the tests"

def test_quick_hint_tab_refreshes_hints_before_deciding_to_cycle():
    hints = ["old hint"]
    app, _ = quick_hint_app(tuple(hints))
    app.quick_hint_focus = 0
    app._pick_quick_hint(app.input_buffer)
    app.quick_hints_fn = lambda: tuple(hints)
    hints[:] = ["new hint"]

    app.tab_or_complete(app.input_buffer, reverse=False)

    assert app.quick_hint_picked == []
    assert app.quick_hint_focus == -1
    assert app.input_buffer.text == "old hint"

def test_quick_hint_external_edit_drops_picked_state(monkeypatch):
    app, _ = quick_hint_app()
    app.quick_hint_focus = 0
    app._pick_quick_hint(app.input_buffer)

    async def edit_in_terminal(callback, *, in_executor):
        del callback, in_executor
        return "edited text"

    monkeypatch.setattr("minacode.tui.app.run_in_terminal", edit_in_terminal)
    asyncio.run(app._run_input_editor())

    assert app.input_buffer.text == "edited text"
    assert app.quick_hint_picked == []
    assert app.quick_hint_focus == -1

def test_quick_hint_fragments_mark_picked_chips():
    app, _ = quick_hint_app(("a", "b"))
    app.quick_hint_focus = 0
    app._pick_quick_hint(app.input_buffer)  # pick "a"; the ✓ mark survives the pick
    fragments = app.quick_hint_fragments()
    assert ("class:quickhint", " \u2713 a ") in fragments
    assert ("class:quickhint", " b ") in fragments

def test_quick_hint_fragments_highlight_focused_chip():
    app, _ = quick_hint_app(("a", "b"))
    # With no terminal width (the app is not running) chips stay on one horizontal row,
    # separated by a bar, exactly as before the wrap-aware layout.
    assert app.quick_hint_fragments() == [("class:quickhint", " a "), ("class:quickhint.sep", " \u2502 "), ("class:quickhint", " b ")]
    app.quick_hint_focus = 0
    assert ("class:quickhint.focused", " a ") in app.quick_hint_fragments()

def test_quick_hints_all_visible_at_narrow_width(monkeypatch):
    """Chips flow left to right and wrap only between chips once the row is full, so all three
    hints stay fully visible — never truncated or split mid-text — at a narrow terminal.
    Exercises the real prompt_toolkit layout/render boundary, not just quick_hint_fragments()."""
    hints = ("run the tests and check the coverage", "构建文档并同步中文 locale 目录", "commit the work with a clear message")
    app = TuiApp(quick_hints_fn=lambda: hints)
    output = ResizableOutput(rows=14, columns=30)
    frames = []
    rendered = threading.Event()

    def after_render(application):
        frames.append(rendered_screen_text(application, output))
        rendered.set()

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        assert rendered.wait(timeout=1)
        pipe_input.send_text("\x04")
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive, output=output, after_render=after_render)

    assert frames, "the app rendered at least one frame"
    # A chip wider than the terminal wraps onto extra visual lines and eats the whitespace at
    # the break, so compare compact forms: one frame must show every hint's full text.
    compact_frames = ["".join(frame.split()) for frame in frames]
    assert any(all("".join(hint.split()) in screen for hint in hints) for screen in compact_frames), "no single rendered frame showed every hint"

def test_quick_hint_flow_lays_chips_horizontally_and_wraps_between_chips():
    """The wrap-aware layout keeps as many chips as fit on one row and only breaks between
    chips, never inside one; a narrow column pushes later chips to their own lines."""
    flow = TuiApp._flow_quick_hints

    # Wide enough: all three chips on one horizontal row, bar-separated, no newlines.
    wide = flow(("run", "show diff", "commit"), columns=100, focus=-1, picked=())
    assert wide == [
        ("class:quickhint", " run "),
        ("class:quickhint.sep", " \u2502 "),
        ("class:quickhint", " show diff "),
        ("class:quickhint.sep", " \u2502 "),
        ("class:quickhint", " commit "),
    ]

    # Unknown width (0) keeps the single horizontal row too, letting the window wrap as fallback.
    unknown = flow(("a", "b"), columns=0, focus=-1, picked=())
    assert "\n" not in [text for _style, text in unknown]

    # Narrow: " run the tests " (15) fits alone, but " b " would not fit the remaining row,
    # so it wraps to its own line; no chip is ever split and no bar appears mid-row.
    narrow = flow(("run the tests", "b"), columns=16, focus=-1, picked=())
    lines = "".join(text for _style, text in narrow)
    assert lines == " run the tests \n b "
    assert "\u2502" not in lines

    # A chip wider than the whole row still stays whole on its own line.
    lone = flow(("构建文档并同步中文 locale 目录", "x"), columns=12, focus=-1, picked=())
    lines = "".join(text for _style, text in lone)
    assert lines.startswith(" 构建文档并同步中文 locale 目录 ")
    assert " \u2502 " not in lines

    # Focus and picked marks keep working through the flow layout.
    picked = flow(("a", "b"), columns=100, focus=1, picked=("a",))
    assert ("class:quickhint", " \u2713 a ") in picked
    assert ("class:quickhint.focused", " b ") in picked

    # At most MAX_QUICK_HINTS_PER_ROW chips per row even when the terminal is wide: the fourth
    # short hint starts its own line instead of crowding one row.
    four = flow(("a", "b", "c", "d"), columns=100, focus=-1, picked=())
    lines = "".join(text for _style, text in four)
    assert lines == " a  \u2502  b  \u2502  c \n d "

    # Width and the per-row cap both end a row; the width check still wins on a narrow column.
    cap_and_width = flow(("a", "b", "c", "d"), columns=5, focus=-1, picked=())
    assert "".join(text for _style, text in cap_and_width) == " a \n b \n c \n d "

def test_quick_hint_pick_and_send_still_work_after_wrapping(monkeypatch):
    """Chips that wrap onto multiple visual lines keep Tab focus, Enter pick/unpick, and the
    final Enter submission."""
    received = []
    app = None

    def submit(text):
        received.append(str(text))
        app.set_idle()

    app = TuiApp(
        on_chat_submit=submit,
        quick_hints_fn=lambda: ("run the tests and check the coverage", "构建文档并同步中文 locale 目录", "commit the work"),
    )
    app.set_idle()

    # Tab focuses chip 0, Enter picks it; one Tab reaches chip 1, Enter picks it; the final
    # Enter, with focus back on the input line, sends the combined text.
    run_interactive_tui(monkeypatch, app, text="\t\r\t\r\r\x04")

    assert received == ["run the tests and check the coverage\n构建文档并同步中文 locale 目录"]

def test_quick_hint_placeholder_hints_keys_until_focused():
    app, _ = quick_hint_app()
    assert app.placeholder_text() == "Tab cycles suggestions \u00b7 Enter picks \u00b7 Enter sends"
    app.quick_hint_focus = 0
    assert app.placeholder_text() == ""

def test_quick_hint_pick_ignored_when_input_was_edited():
    app, _ = quick_hint_app()
    app.quick_hint_focus = 0
    app.input_buffer.insert_text("hello")
    assert app._pick_quick_hint(app.input_buffer) is False
    assert app.quick_hint_picked == []
    assert app.quick_hint_focus == 0  # unchanged, so Enter falls through to sending

def test_quick_hint_enter_without_focus_sends():
    """Enter with no focused chip sends; it never unpicks picked text."""
    app, submitted = quick_hint_app()
    app.quick_hint_focus = 0
    app._pick_quick_hint(app.input_buffer)  # pick -> buffer = "run the tests"
    assert app.quick_hint_focus == -1
    assert app._accept(app.input_buffer) is True  # Enter sends, it never unpicks
    assert [str(value) for value in submitted] == ["run the tests"]
    assert app.quick_hint_focus == -1  # sending clears the quick-hint state

def test_quick_hint_placeholder_falls_back_without_hints():
    app, _ = quick_hint_app(())
    assert app.placeholder_text() == app.input_hint_fn()

def test_quick_hint_mode_change_resets_focus_and_picked():
    app, _ = quick_hint_app()
    app.quick_hint_focus = 2
    app.quick_hint_picked = ["run the tests"]
    app.set_running("working")
    assert app.quick_hint_focus == -1
    assert app.quick_hint_picked == []
