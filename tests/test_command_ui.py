"""Interactive command surfaces: the provider/model/api/reason selection chains, the diff
viewer, and the stored Bash output viewer."""

import os
import shutil
import threading
from types import SimpleNamespace

import openai as openai_module
import pytest
from prompt_toolkit.utils import get_cwidth
from tui_harness import ResizableOutput, loop, rendered_screen_text, run_interactive_tui, session, wait_until

import minacode.cli.commands as commands_mod
import minacode.cli.modals as modals_mod
from minacode.base import (
    SELECTION_BACK,
    ModelError,
)
from minacode.cli import COMMANDS, CommandCompleter, CommandLoop
from minacode.cli import worker as worker_mod
from minacode.cli.commands import (
    SET_KEYS,
    api,
    config,
    language_command,
    model,
    provider,
    reason,
    remote_models,
    set_model,
    set_value,
    strict,
)
from minacode.cli.modals import choice_application, diff_viewer, select_choice, tool_output_viewer
from minacode.cli.worker import WorkerFlow, worker_command
from minacode.config import (
    PROVIDER_API_CHOICES,
    REASONING_CHOICES,
    Config,
    ProviderConfig,
)
from minacode.engine import Agent
from minacode.model import ModelClient
from minacode.runner import ToolRunner
from minacode.session import Session
from minacode.tools import Tool
from minacode.tui import TUI_MODAL_PENDING, DiffViewState, TabbedViewState, TuiApp


def diff_loop(tmp_path):
    command_loop = loop(tmp_path)
    before = "".join(f"old {index}\n" for index in range(20))
    after = "".join(f"new {index}\n" for index in range(20))
    command_loop.session.store_turn_diff("tr.1", 1, "a.py", "unused", before=before, after=after, round=1)
    command_loop.session.store_turn_diff("tr.2", 2, "b.py", "unused", before="old\n", after="new\n", round=1)
    return command_loop


# The registry is the single source of command metadata; HELP stays a hand-written literal with
# manual wrapping and non-command sections, so every registered name and alias must appear in it.
# `/worker` is a pre-existing gap: it is registered in master's COMMAND_HANDLERS but missing from
# master's HELP literal. It is listed here so the omission stays visible instead of silent; any
# new registered command missing from HELP fails this test unless explicitly added to the set.
HELP_OMISSIONS = frozenset({"/worker"})


def test_registry_names_and_aliases_appear_in_help():
    missing = {name for command in COMMANDS for name in (command.name, *command.aliases) if name not in CommandLoop.HELP}
    assert missing <= HELP_OMISSIONS, f"registered commands missing from HELP: {sorted(missing - HELP_OMISSIONS)}"


class ModalHarness:
    def __init__(self, keys, *, consumed=False):
        self.keys = list(keys)
        # consumed=True hands each key to the next modal in line instead of replaying the whole
        # sequence for every modal, which is how a multi-modal flow (list -> detail -> list) is
        # driven end to end.
        self.consumed = consumed
        self.pos = 0
        self.frames = []
        self.exclusive = []

    def show_modal(self, fragments_fn, key_fn, *, exclusive=False):
        self.exclusive.append(exclusive)
        self.frames.append(fragments_fn())
        result = TUI_MODAL_PENDING
        keys = self.keys[self.pos :] if self.consumed else self.keys
        for key in keys:
            if self.consumed:
                self.pos += 1
            result = key_fn(key, key if len(key) == 1 else "")
            self.frames.append(fragments_fn())
            if result is not TUI_MODAL_PENDING:
                return result
        return None


def test_tool_output_viewer_browses_recent_calls_through_a_viewport_and_opens_full_output(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    for index in range(12):
        stdout = "\n".join(f"line {line}" for line in range(40)) if index == 10 else f"output {index}"
        stderr = "detail stderr" if index == 10 else ""
        command_loop.session.store_tool_result("Bash", [f"printf command-{index}"], Tool.process_result("BashToolResult", 0, stdout, stderr))
    command_loop.session.store_tool_result("Bash", ["true"], Tool.process_result("BashToolResult", 0, "", ""))
    modal = ModalHarness(["j", "enter", "G"])  # second entry, then scroll the viewer to the bottom
    command_loop.tui = modal

    # ``shutil`` is a shared module object also used by pytest's terminal reporter. Restore the
    # patch before pytest reports this test result, rather than waiting for fixture teardown.
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((50, 20)))
        tool_output_viewer(command_loop)

    listing = "".join(value for _style, value in modal.frames[0])
    assert listing.startswith("\n──── Tool output · latest 12 ")
    assert get_cwidth(listing.splitlines()[1]) == 48
    assert "command-11" in listing and "command-2" in listing
    # A twenty-row terminal draws ten of the twelve: the rest are a scroll away, not dropped, and
    # the counter is what says so. `true` printed nothing and is not an entry at all.
    assert "Bash printf command-1\n" not in listing and "Bash printf command-0\n" not in listing and "Bash true" not in listing
    assert "showing 1-10 of 12" in listing
    # The second entry opens in the scrolling viewer: the command as its body, the streams below.
    frames = ["".join(value for _style, value in frame) for frame in modal.frames]
    viewer = [frame for frame in frames if "read-only" in frame]
    assert "Output · tr.11 · read-only" in viewer[0]
    assert "1  printf command-10" in viewer[0]
    assert "── result " in viewer[0]
    assert "stdout:" in viewer[0] and "line 0" in viewer[0]
    assert "stderr:" in viewer[-1] and "detail stderr" in viewer[-1]  # both streams, a scroll away
    assert modal.exclusive == [False, True]  # the list shares the screen; the viewer takes it


def test_tool_output_browser_marks_bash_results_ok_and_fail(tmp_path, monkeypatch):
    """A Bash row's first column carries its verdict: a green ✓ for exit 0, a red ✗ for any other
    exit. The list is mostly bash, so the failures should be scannable by color; entries with no
    exit code (a script, an order) keep the cell blank instead of guessing."""
    command_loop = loop(tmp_path)
    command_loop.session.store_tool_result("Bash", ["printf ok"], Tool.process_result("BashToolResult", 0, "ok output", ""))
    command_loop.session.store_tool_result("Bash", ["make check"], Tool.process_result("BashToolResult", 2, "", "target failed"))
    modal = ModalHarness(["j", "q"])
    command_loop.tui = modal

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((50, 20)))
        tool_output_viewer(command_loop)

    # The selected row is reversed as a whole, which hides its mark's own color, so the two frames
    # (cursor on the failure, then on the success) each expose one verdict column in its color.
    pairs = [(style, value) for frame in modal.frames for style, value in frame]
    assert ("class:choice.output.ok", "✓ ") in pairs
    assert ("class:choice.output.fail", "✗ ") in pairs


def test_tool_output_browser_lists_past_the_old_fifty_entry_cap(tmp_path, monkeypatch):
    """The browser's list reaches as far back as the session stores: 400 results, not a page of
    fifty. The viewport still shows one screenful with the counter saying how far there is to go."""
    command_loop = loop(tmp_path)
    for index in range(55):
        command_loop.session.store_tool_result("Bash", [f"printf {index}"], Tool.process_result("BashToolResult", 0, f"out {index}", ""))
    modal = ModalHarness(["q"])
    command_loop.tui = modal

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((50, 20)))
        tool_output_viewer(command_loop)

    listing = "".join(value for _style, value in modal.frames[0])
    assert listing.startswith("\n──── Tool output · latest 55 ")
    assert "showing 1-10 of 55" in listing
    assert "Bash printf 54" in listing  # the newest is in view


def test_tool_output_browser_keeps_every_stored_record_with_a_running_script(tmp_path, monkeypatch):
    """A running ToolScript's live entry does not push the oldest stored record out: with a full
    session the browser lists all 400 stored results plus the running one, and the viewport still
    bounds the screen."""
    command_loop = loop(tmp_path)
    for index in range(400):
        command_loop.session.store_tool_result(
            "Bash", [f"printf {index}"], Tool.process_result("BashToolResult", 0, f"out {index}", "")
        )
    command_loop.script_running_code = "print('hi')\n"
    modal = ModalHarness(["q"])
    command_loop.tui = modal

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((50, 20)))
        tool_output_viewer(command_loop)

    listing = "".join(value for _style, value in modal.frames[0])
    assert listing.startswith("\n──── Tool output · latest 401 ")
    assert "showing 1-10 of 401" in listing
    assert "ToolScript" in listing  # the running script's live entry is listed too


def test_tool_output_viewer_escape_returns_to_the_list_with_the_cursor_kept(tmp_path, monkeypatch):
    """Esc (or q) in a detail goes back to the list instead of closing the whole browser, and
    the reopened list still points at the entry the reader came from."""
    command_loop = loop(tmp_path)
    for index in range(5):
        command_loop.session.store_tool_result(
            "Bash",
            [f"printf command-{index}"],
            Tool.process_result("BashToolResult", 0, f"output {index}", ""),
        )
    # j moves to the second entry, enter opens it, escape returns to the list, enter opens the
    # same entry again, c-o closes the whole browser.
    modal = ModalHarness(["j", "enter", "escape", "enter", "c-o"], consumed=True)
    command_loop.tui = modal
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((50, 20)))
        tool_output_viewer(command_loop)

    frames = ["".join(value for _style, value in frame) for frame in modal.frames]
    listings = [frame for frame in frames if "Tool output" in frame]
    assert len(listings) == 5  # two list passes: three renders, then two on the reopened one
    assert modal.exclusive == [False, True, False, True]  # list, detail, list, detail

    def selected_line(listing: str) -> str:
        return next(row for row in listing.splitlines() if row.startswith("> "))

    assert "command-4" in selected_line(listings[0])  # first list starts at the newest entry
    assert "command-3" in selected_line(listings[1])  # j moved to the second entry
    # The reopened list still sits on the entry the escape came back from, not the top.
    assert "command-3" in selected_line(listings[3])
    # The same detail opened twice: once before the escape, once before the c-o, each rendering
    # its title row twice (initial frame plus the frame after its closing key).
    assert sum("read-only" in frame for frame in frames) == 4


def test_tool_output_viewer_q_in_a_detail_also_returns_to_the_list(tmp_path, monkeypatch):
    """q behaves exactly like Esc inside a detail: back to the list, not out of the browser."""
    command_loop = loop(tmp_path)
    for index in range(3):
        command_loop.session.store_tool_result(
            "Bash",
            [f"printf command-{index}"],
            Tool.process_result("BashToolResult", 0, f"output {index}", ""),
        )
    # enter opens the top entry, q returns to the list, enter opens it again, c-o closes.
    modal = ModalHarness(["enter", "q", "enter", "c-o"], consumed=True)
    command_loop.tui = modal
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((50, 20)))
        tool_output_viewer(command_loop)

    frames = ["".join(value for _style, value in frame) for frame in modal.frames]
    listings = [frame for frame in frames if "Tool output" in frame]
    assert modal.exclusive == [False, True, False, True]  # list, detail, list, detail
    assert len(listings) == 4  # q came back to the list: two renders per list pass
    assert sum("read-only" in frame for frame in frames) == 4  # the detail opened twice


def test_tool_output_viewer_ctrl_c_in_a_detail_also_returns_to_the_list(tmp_path, monkeypatch):
    """Ctrl-C behaves like Esc inside a detail: back to the list, not out of the browser."""
    command_loop = loop(tmp_path)
    for index in range(3):
        command_loop.session.store_tool_result(
            "Bash",
            [f"printf command-{index}"],
            Tool.process_result("BashToolResult", 0, f"output {index}", ""),
        )
    # enter opens the top entry, c-c returns to the list, enter opens it again, c-o closes.
    modal = ModalHarness(["enter", "c-c", "enter", "c-o"], consumed=True)
    command_loop.tui = modal
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((50, 20)))
        tool_output_viewer(command_loop)

    frames = ["".join(value for _style, value in frame) for frame in modal.frames]
    listings = [frame for frame in frames if "Tool output" in frame]
    assert modal.exclusive == [False, True, False, True]  # list, detail, list, detail
    assert len(listings) == 4  # c-c came back to the list: two renders per list pass
    assert sum("read-only" in frame for frame in frames) == 4  # the detail opened twice


def test_tool_output_viewer_ctrl_o_in_a_detail_closes_the_browser(tmp_path, monkeypatch):
    """Ctrl-O inside a detail still closes the whole browser: only Esc/q go back to the list."""
    command_loop = loop(tmp_path)
    for index in range(3):
        command_loop.session.store_tool_result(
            "Bash",
            [f"printf command-{index}"],
            Tool.process_result("BashToolResult", 0, f"output {index}", ""),
        )
    modal = ModalHarness(["j", "enter", "c-o"], consumed=True)
    command_loop.tui = modal
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((50, 20)))
        tool_output_viewer(command_loop)

    frames = ["".join(value for _style, value in frame) for frame in modal.frames]
    assert modal.exclusive == [False, True]  # list, detail -- and no reopened list
    assert sum("Tool output" in frame for frame in frames) == 3  # the list rendered its three frames once
    assert sum("read-only" in frame for frame in frames) == 2  # the detail closed the browser


def test_tool_output_viewer_keeps_the_search_filter_across_an_escape(tmp_path, monkeypatch):
    """A `/` filter survives Esc back to the list: the reopened list is still filtered."""
    command_loop = loop(tmp_path)
    for index in range(5):
        command_loop.session.store_tool_result(
            "Bash",
            [f"printf command-{index}"],
            Tool.process_result("BashToolResult", 0, f"output {index}", ""),
        )
    # Filter to the newest entry, open it, Esc back: the reopened list still shows just that
    # entry, then the second enter opens it again and c-o closes the browser.
    modal = ModalHarness(["/", *"command-4", "enter", "enter", "escape", "enter", "c-o"], consumed=True)
    command_loop.tui = modal
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((50, 20)))
        tool_output_viewer(command_loop)

    frames = ["".join(value for _style, value in frame) for frame in modal.frames]
    listings = [frame for frame in frames if "Tool output" in frame]
    assert modal.exclusive == [False, True, False, True]
    assert "command-4" in listings[0] and "command-3" in listings[0]  # the full list first
    # The reopened list is still filtered to the one matching entry, not the full list again.
    assert "command-4" in listings[-1]
    assert "command-3" not in listings[-1] and "command-0" not in listings[-1]
    assert sum("read-only" in frame for frame in frames) == 4


def test_tool_output_viewer_folds_a_multiline_command_into_one_row(tmp_path, monkeypatch):
    """A row is one row. `short_call` keeps a multi-line command whole for the transcript, and a
    `git commit -m` with a real message would otherwise spill its row over several lines, carrying
    the numbering and the selection bar with it."""
    command_loop = loop(tmp_path)
    command_loop.session.store_tool_result(
        "Bash",
        ['git commit -m "fix the parser\n\nIt dropped the last token."'],
        Tool.process_result("BashToolResult", 0, "1 file changed", ""),
    )
    modal = ModalHarness([])
    command_loop.tui = modal

    # ``shutil`` is a shared module object also used by pytest's terminal reporter. Restore the
    # patch before pytest reports this test result, rather than waiting for fixture teardown.
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((120, 20)))
        tool_output_viewer(command_loop)

    listing = "".join(value for _style, value in modal.frames[0])
    rows = [row for row in listing.splitlines() if "tr.1" in row]
    assert len(rows) == 1
    assert "fix the parser It dropped the last token." in rows[0]

    # The viewer still opens the command exactly as it was run, newlines and all.
    view = modals_mod.record_view(command_loop, command_loop.session.tool_records[0])
    assert view is not None and view.text.count("\n") == 2


def test_tool_output_viewer_reopens_a_delegate_order_with_the_worker_answer(tmp_path):
    """An order is written to be read twice: at the send prompt, and again when the worker's answer
    has to be judged against what was actually asked. The transcript keeps only the `Delegate send`
    line, so this browser is the second reading."""
    command_loop = loop(tmp_path)
    order = "Goal: rename the flag.\nFiles: minacode/config.py\nVerify: uv run pytest tests/test_cli.py"
    command_loop.session.store_tool_result(
        "Delegate",
        [{"action": "send", "order": order, "title": "rename the flag"}],
        '<Delegate action="send" files="minacode/config.py">\n<worker>renamed it at config.py:118</worker>\n</Delegate>',
    )
    command_loop.session.store_tool_result("Delegate", [{"action": "status"}], '<Delegate action="status" alive="true"/>')
    modal = ModalHarness(["enter"])
    command_loop.tui = modal

    tool_output_viewer(command_loop)

    listing = "".join(value for _style, value in modal.frames[0])
    assert "Tool output · latest 1" in listing  # a status carries no order and is not an entry
    viewer = next(frame for frame in ("".join(value for _style, value in f) for f in modal.frames) if "read-only" in frame)
    assert "Order · tr.1 · read-only" in viewer
    assert "rename the flag" in viewer and "Verify: uv run pytest" in viewer
    assert "renamed it at config.py:118" in viewer  # the answer, below the order it is judged against


def test_tool_output_viewer_shows_the_whole_output_not_the_transcript_preview(tmp_path):
    """The transcript keeps three lines. The point of opening an entry is the other thirty-seven."""
    command_loop = loop(tmp_path)
    stdout = "\n".join(f"line {line}" for line in range(40))
    command_loop.session.store_tool_result("Bash", ["seq 40"], Tool.process_result("BashToolResult", 0, stdout, ""))
    modal = ModalHarness(["enter", "G"])
    command_loop.tui = modal

    tool_output_viewer(command_loop)

    frames = ["".join(value for _style, value in frame) for frame in modal.frames]
    viewer = [frame for frame in frames if "read-only" in frame]
    assert "line 0" in viewer[0]
    assert "line 39" in viewer[-1]  # reachable by scrolling, not elided
    assert "lines omitted" not in "".join(viewer)


def test_tool_output_viewer_bounds_a_huge_result_and_says_so(tmp_path):
    """Stored output has no cap and the wrapper is quadratic in one line's length, so the viewer
    bounds what it renders -- and says how much it is showing, because a reader who cannot tell an
    elided result from a complete one has to distrust every result."""
    command_loop = loop(tmp_path)
    stdout = "\n".join(f"line {line}" for line in range(ToolRunner.VIEWER_LINES * 2))
    command_loop.session.store_tool_result("Bash", ["seq huge"], Tool.process_result("BashToolResult", 0, stdout, ""))
    command_loop.session.store_tool_result("Bash", ["one long line"], Tool.process_result("BashToolResult", 0, "x" * (ToolRunner.VIEWER_LINE_CHARS * 3), ""))
    modal = ModalHarness(["enter"])
    command_loop.tui = modal

    tool_output_viewer(command_loop)

    frames = ["".join(value for _style, value in frame) for frame in modal.frames]
    viewer = next(frame for frame in frames if "read-only" in frame)
    # The newest entry is the one long line; the header says the clip happened rather than
    # presenting a truncated line as the whole of it.
    assert f"long lines clipped at {ToolRunner.VIEWER_LINE_CHARS}" in viewer
    rendered = [row for row in viewer.splitlines() if row.strip().startswith("x")]
    assert rendered and all(len(row) <= ToolRunner.VIEWER_LINE_CHARS + 10 for row in rendered)


def test_tool_output_viewer_bounds_a_result_with_too_many_lines(tmp_path):
    """The line bound counts against the streams, not the stored envelope, so the note it prints
    is a fact about the output rather than about the tags wrapped around it."""
    command_loop = loop(tmp_path)
    stdout = "\n".join(f"line {line}" for line in range(ToolRunner.VIEWER_LINES * 2))
    command_loop.session.store_tool_result("Bash", ["seq huge"], Tool.process_result("BashToolResult", 0, stdout, ""))
    modal = ModalHarness(["enter"])
    command_loop.tui = modal

    tool_output_viewer(command_loop)

    viewer = next(frame for frame in ("".join(v for _s, v in f) for f in modal.frames) if "read-only" in frame)
    assert f"{ToolRunner.VIEWER_LINES} shown of {ToolRunner.VIEWER_LINES * 2}" in viewer
    # The elision is marked in the text too, where it happens -- the header note is derived from it.
    bounded, note = command_loop.agent.tools.bash_viewer_output(command_loop.session.tool_records[-1].output)
    assert "lines omitted" in bounded and note.startswith(f"{ToolRunner.VIEWER_LINES} shown of ")


def test_tool_output_viewer_is_noop_without_stored_bash_output(tmp_path):
    command_loop = loop(tmp_path)
    modal = ModalHarness([])
    command_loop.tui = modal

    tool_output_viewer(command_loop)

    assert modal.frames == []


def test_tool_output_viewer_offers_the_script_that_is_still_running(tmp_path):
    """A long batch is exactly when the reader wants to look; the record only arrives at the end."""
    command_loop = loop(tmp_path)
    command_loop.session.store_tool_result("Bash", ["printf done"], Tool.process_result("BashToolResult", 0, "done", ""))
    command_loop.toolscript_run_status(True, 'for key in KEYS:\n    call("server.tool", {"key": key})\n')
    modal = ModalHarness(["enter"])
    command_loop.tui = modal

    tool_output_viewer(command_loop)

    listing = "".join(value for _style, value in modal.frames[0])
    assert "running  ToolScript call 2 lines" in listing  # first row, above the stored Bash entry
    assert listing.index("running") < listing.index("Bash")
    frames = ["".join(value for _style, value in frame) for frame in modal.frames]
    viewer = [frame for frame in frames if "read-only" in frame]
    assert "Script · running · read-only" in viewer[0]
    assert 'call("server.tool"' in viewer[0]

    # It leaves with the script: once the batch returns, only the stored record remains.
    command_loop.tui = None
    command_loop.toolscript_run_status(False)
    modal = ModalHarness([])
    command_loop.tui = modal
    tool_output_viewer(command_loop)
    assert "running" not in "".join(value for _style, value in modal.frames[0])


def test_tool_output_list_rows_are_coloured_by_part(tmp_path):
    """All-grey rows read as a wall; the key, the tool name, and the arguments each get their own."""
    command_loop = loop(tmp_path)
    for index in range(2):
        command_loop.session.store_tool_result("Bash", [f"printf hi-{index}"], Tool.process_result("BashToolResult", 0, "hi", ""))
    modal = ModalHarness([])
    command_loop.tui = modal

    tool_output_viewer(command_loop)

    row = [(style, text) for style, text in modal.frames[0] if text.strip()]
    assert ("class:choice.meta", "tr.1  ") in row
    assert ("class:choice.tool", "Bash ") in row
    assert ("", "printf hi-0") in row
    # The selected row is left as one reverse bar rather than repainted part by part.
    assert not [style for style, _text in row if "choice.selected" in style and ("meta" in style or "tool" in style)]


def test_tool_output_viewer_reads_resumed_history(tmp_path):
    saved = session(tmp_path)
    saved.store_tool_result("Bash", ["printf persisted"], Tool.process_result("BashToolResult", 0, "persisted output", ""))
    saved.save_snapshot()
    restored = Session.load_snapshot(saved.uid, config=saved.config)
    command_loop = CommandLoop(Agent(restored, output_fn=lambda _text: None), input_fn=lambda prompt="": "", output_fn=lambda _text: None)
    modal = ModalHarness(["enter", "c-o"], consumed=True)
    command_loop.tui = modal

    tool_output_viewer(command_loop)

    detail = next(frame for frame in ("".join(value for _style, value in f) for f in modal.frames) if "read-only" in frame)
    assert "printf persisted" in detail
    assert "persisted output" in detail


def test_choice_navigation_uses_shared_modal_protocol(tmp_path):
    command_loop = loop(tmp_path)
    modal = ModalHarness(["j", "enter"])
    command_loop.tui = modal
    result = choice_application(command_loop, "Pick", ("a", "b", "c"), {"a": "Alpha", "b": "Beta", "c": "Gamma"}, "", set())

    assert result == "b"
    assert "Beta" in "".join(text for frame in modal.frames for _style, text in frame)


def test_provider_selection_chains_provider_model_api_and_reasoning(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["other"] = ProviderConfig(model="model-b", available_models=("model-b",), reasoning="low")
    selected = iter(["other", "model-b", "responses", "high"])
    titles = []

    def select(_loop, title, *_args, **_kwargs):
        titles.append(title)
        return next(selected)

    monkeypatch.setattr(commands_mod, "select_choice", select)
    discovered = []
    monkeypatch.setattr(commands_mod, "remote_models", lambda _loop, provider: discovered.append(provider.model) or ())

    result = provider(command_loop, "")

    assert titles == ["Provider", "Model", "Request API", "Reasoning effort"]
    assert command_loop.session.config.active_provider == "other"
    assert command_loop.session.config.provider.model == "model-b"
    assert command_loop.session.config.provider.api == "responses"
    assert command_loop.session.config.provider.reasoning == "high"
    assert discovered == ["model-b"]
    assert "Set provider.model = model-b" in result
    assert "Set provider.api = responses (wire: responses)" in result


def test_runtime_provider_switches_are_recorded_for_resume(tmp_path):
    """Every runtime switch is recorded per entry so a later --resume can restore it."""
    command_loop = loop(tmp_path)
    command_loop.interactive_input = False
    session = command_loop.session
    session.config.providers["other"] = ProviderConfig(model="m", api="chat", reasoning="low")

    assert provider(command_loop, "other") == "Set provider = other"
    assert session.provider_overrides == {"active_provider": "other"}

    assert set_model(command_loop, "model-b") == "Set provider.model = model-b"
    assert session.provider_overrides["providers"]["other"]["model"] == "model-b"

    assert reason(command_loop, "high") == "Set provider.reasoning = high"
    assert session.provider_overrides["providers"]["other"]["reasoning"] == "high"

    assert api(command_loop, "responses") == "Set provider.api = responses (wire: responses)"
    assert session.provider_overrides["providers"]["other"]["api"] == "responses"
    assert session.provider_overrides["active_provider"] == "other"


def test_model_override_binds_to_the_entry_it_was_set_on(tmp_path):
    """model/reasoning/api overrides key on the active entry at switch time, so a /provider after a
    /model restores each switch to the entry it was made on."""
    command_loop = loop(tmp_path)
    command_loop.interactive_input = False
    session = command_loop.session
    session.config.providers["a"] = ProviderConfig(model="ma", api="chat", reasoning="low")
    session.config.providers["b"] = ProviderConfig(model="mb", api="chat", reasoning="low")

    set_model(command_loop, "model-on-default")
    assert session.provider_overrides["providers"]["default"]["model"] == "model-on-default"

    provider(command_loop, "a")
    set_model(command_loop, "model-on-a")
    assert session.provider_overrides["providers"]["a"]["model"] == "model-on-a"
    assert session.provider_overrides["active_provider"] == "a"

    session.provider_overrides["active_provider"] = "default"
    session.apply_provider_overrides()
    assert session.config.active_provider == "default"
    assert session.config.providers["default"].model == "model-on-default"
    assert session.config.providers["a"].model == "model-on-a"


def test_provider_and_model_commands_validate_direct_arguments(tmp_path):
    command_loop = loop(tmp_path)

    assert provider(command_loop, "one two") == "Usage: /provider [NAME]"
    assert provider(command_loop, "missing") == "Unknown provider: missing"
    assert model(command_loop, "one two") == "Usage: /model [MODEL]"


def test_reason_strict_and_set_commands_validate_values(tmp_path):
    from prompt_toolkit.document import Document

    command_loop = loop(tmp_path)

    assert reason(command_loop, "invalid").startswith("Usage: /reason ")
    assert reason(command_loop, "max") == "Set provider.reasoning = max"
    assert command_loop.session.config.provider.reasoning == "max"
    assert strict(command_loop, "on") == "Usage: /strict"
    assert set_value(command_loop, "") == "Usage: /set KEY VALUE"
    assert set_value(command_loop, "unknown value") == "Unknown config key: unknown"
    assert set_value(command_loop, "provider.timeout never") == "Invalid value for provider.timeout"
    assert set_value(command_loop, "provider.response_timeout 900") == "Set provider.response_timeout"
    assert command_loop.session.config.provider.response_timeout == 900
    assert set_value(command_loop, "provider.temperature off") == "Set provider.temperature"
    assert command_loop.session.config.provider.temperature is None
    assert set_value(command_loop, "provider.stream maybe") == "Invalid value for provider.stream"
    assert set_value(command_loop, "provider.stream off") == "Set provider.stream"
    assert command_loop.session.config.provider.stream is False
    stream_values = [item.text for item in CommandCompleter().get_completions(Document("/set provider.stream "), None)]
    assert stream_values == ["on", "off"]
    assert set_value(command_loop, "provider.image_input maybe") == "Invalid value for provider.image_input"
    assert set_value(command_loop, "provider.image_input off") == "Set provider.image_input"
    assert command_loop.session.config.provider.image_input == "off"


def test_config_shows_the_reasoning_effort_resolved_for_the_active_model(tmp_path):
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider
    provider.url = "https://api.openai.com/v1"
    provider.model = "gpt-5.5"
    provider.reasoning = "max"

    assert "provider.resolved_reasoning_effort: xhigh" in config(command_loop, "")


def test_language_command_shows_sets_and_resets(tmp_path):
    command_loop = loop(tmp_path)

    assert language_command(command_loop, "") == "Reply language: auto (follows your messages)"

    assert language_command(command_loop, "Chinese") == "Reply language set: Chinese"
    assert language_command(command_loop, "") == "Reply language: Chinese"
    assert command_loop.session.settings.language == "Chinese"

    # the value is normalized (stripped), and free text like CJK names is allowed
    assert language_command(command_loop, "  简体中文  ") == "Reply language set: 简体中文"

    assert language_command(command_loop, "  AUTO  ") == "Reply language reset to auto"
    assert language_command(command_loop, "") == "Reply language: auto (follows your messages)"

    # invalid values return the validation message instead of raising
    assert language_command(command_loop, "Chinese\nJapanese").startswith("runtime.language")
    assert language_command(command_loop, "x" * 65).startswith("runtime.language")
    assert command_loop.session.settings.language == "auto"  # unchanged after the rejected set


def test_config_shows_runtime_language(tmp_path):
    command_loop = loop(tmp_path)
    assert "runtime.language: auto" in config(command_loop, "")

    command_loop.session.settings.language = "Chinese"
    assert "runtime.language: Chinese" in config(command_loop, "")


def test_api_command_switches_the_request_wire_and_names_what_took_effect(tmp_path):
    # A model chosen with /model may not be served over the provider's configured protocol, so the
    # wire has to be switchable in-session rather than only in the config file.
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider
    provider.url = "https://example.com/compatible-mode/v1"
    provider.api = "responses"

    assert api(command_loop, "grpc").startswith("Usage: /api ")
    assert provider.resolve().api == "responses"
    assert api(command_loop, "chat") == "Set provider.api = chat (wire: chat)"
    assert provider.resolve().api == "chat"
    # "auto" reports the wire it inferred rather than echoing "auto" back.
    assert api(command_loop, "auto") == "Set provider.api = auto (wire: chat)"

    provider.url = "https://example.com/v1/responses"
    assert api(command_loop, "auto") == "Set provider.api = auto (wire: responses)"


def test_api_command_selection_offers_every_protocol_with_the_inferred_wire(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    provider = command_loop.session.config.provider
    provider.url = "https://example.com/v1/responses"
    provider.api = "chat"
    shown = {}

    def choose(_loop, title, choices, labels, current, _disabled):
        shown.update(title=title, choices=choices, labels=labels, current=current)
        return "auto"

    monkeypatch.setattr(modals_mod, "choice_application", choose)

    assert api(command_loop, "") == "Set provider.api = auto (wire: responses)"
    assert shown["title"] == "Request API"
    assert shown["choices"] == PROVIDER_API_CHOICES
    assert shown["current"] == "chat"
    assert shown["labels"]["auto"] == "auto - infer from the endpoint URL and model (responses)"
    assert shown["labels"]["chat"] == "chat (current)"


def test_api_is_registered_like_reason_and_completes_its_choices(tmp_path):
    from prompt_toolkit.document import Document

    command_loop = loop(tmp_path)

    assert "/api" in CommandLoop.COMMANDS
    command_loop.command("/api anthropic")
    assert command_loop.session.config.provider.api == "anthropic"

    texts = [c.text for c in CommandCompleter().get_completions(Document("/api "), None)]
    assert set(texts) == set(PROVIDER_API_CHOICES)
    # The wire is a command, not a /set key, so it must not be reachable both ways.
    assert "provider.api" not in SET_KEYS
    assert set_value(command_loop, "provider.api chat") == "Unknown config key: provider.api"


def test_model_chain_steps_back_from_the_wire_to_the_model_and_from_reasoning_to_the_wire(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    provider = command_loop.session.config.provider
    provider.available_models = ("model-a", "model-b")
    scripted = iter(
        [
            ("Model", "model-a"),
            ("Request API", SELECTION_BACK),  # back lands on the model picker again
            ("Model", "model-a"),
            ("Request API", "chat"),
            ("Reasoning effort", SELECTION_BACK),  # back lands on the wire, not the model
            ("Request API", "responses"),
            ("Reasoning effort", "high"),
        ]
    )
    titles = []

    def select(_loop, title, *_args, **_kwargs):
        expected_title, value = next(scripted)
        assert title == expected_title
        titles.append(title)
        return value

    monkeypatch.setattr(commands_mod, "select_choice", select)
    monkeypatch.setattr(commands_mod, "remote_models", lambda _loop, _provider: ())

    result = model(command_loop, "")

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

    result = set_model(command_loop, "model-a")

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

    monkeypatch.setattr(openai_module, "OpenAI", openai)

    assert remote_models(command_loop, provider) == ("alpha", "zeta")
    assert calls[0]["api_key"] == "secret"
    assert calls[0]["max_retries"] == 0


def test_remote_models_is_optional_and_failure_safe(monkeypatch, tmp_path):
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider

    assert remote_models(command_loop, provider) == ()

    provider.url = "https://example.com/v1"
    provider.key = "secret"
    monkeypatch.setattr(openai_module, "OpenAI", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")))

    assert remote_models(command_loop, provider) == ()


def test_effort_is_an_alias_for_reason(tmp_path):
    command_loop = loop(tmp_path)

    # Registered as a command that dispatches to the same handler as /reason.
    assert "/effort" in CommandLoop.COMMANDS
    reason_command = next(command for command in COMMANDS if command.name == "/reason")
    assert "/effort" in reason_command.aliases

    # Dispatch sets reasoning effort exactly like /reason.
    command_loop.command("/effort high")
    assert command_loop.session.config.provider.reasoning == "high"

    # Tab completion offers the same reasoning choices.
    from prompt_toolkit.document import Document

    texts = [c.text for c in CommandCompleter().get_completions(Document("/effort "), None)]
    assert set(texts) == set(REASONING_CHOICES)


def test_model_selection_groups_configured_and_remote_choices_like_master(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    provider = command_loop.session.config.provider
    provider.model = "configured-model"
    provider.available_models = ("configured-model",)
    provider.url = "https://example.com/v1"
    provider.key = "key"
    shown = []

    def select(_loop, title, choices, **_kwargs):
        shown.append((title, choices))
        if title == "Reasoning effort":
            return "off"
        if title == "Request API":
            return "auto"
        return "remote-model"

    monkeypatch.setattr(commands_mod, "select_choice", select)
    monkeypatch.setattr(commands_mod, "remote_models", lambda _loop, _provider: ("remote-model",))

    assert "Set provider.model = remote-model" in model(command_loop, "")
    assert shown[0] == (
        "Model",
        (
            commands_mod.MODEL_CONFIGURED_LABEL,
            "configured-model",
            commands_mod.MODEL_DISCOVERED_LABEL,
            "remote-model",
        ),
    )


def test_model_discovery_shows_loading_state_for_selected_provider(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    provider = command_loop.session.config.provider
    provider.model = "configured-model"
    provider.available_models = ("configured-model",)
    provider.url = "https://example.com/v1"
    provider.key = "key"
    transitions = []
    command_loop.tui = TuiApp()
    command_loop.tui.set_dispatching = lambda prompt="": transitions.append(prompt)
    monkeypatch.setattr(commands_mod, "remote_models", lambda _loop, selected: ("remote-model",))
    selected = iter(["remote-model", "auto", "off"])
    monkeypatch.setattr(commands_mod, "select_choice", lambda *_args, **_kwargs: next(selected))

    assert "Set provider.model = remote-model" in model(command_loop, "")
    assert transitions == ["Loading models...", ""]


def test_interactive_provider_chain_uses_one_inline_tui_and_real_navigation(monkeypatch, tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["zz-other"] = ProviderConfig(
        model="model-a",
        available_models=("model-a", "model-b"),
        reasoning="low",
    )
    app = TuiApp()
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
        worker = threading.Thread(target=lambda: result.append(provider(command_loop, "")), daemon=True)
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


def test_single_enabled_choice_is_selected_without_opening_modal(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    monkeypatch.setattr(modals_mod, "choice_application", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("modal should not open")))

    assert select_choice(command_loop, "Provider", ("only",), current="only") == "only"
    assert select_choice(command_loop, "Model", ("heading", "only"), disabled={"heading"}) == "only"


def test_provider_auto_selects_sole_provider_and_model(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    provider_config = command_loop.session.config.provider
    provider_config.available_models = ("only-model",)
    provider_config.model = "only-model"
    provider_config.url = ""
    provider_config.key = ""
    titles = []

    def choose(_loop, title, _choices, _labels, current, _disabled):
        titles.append(title)
        return current

    monkeypatch.setattr(modals_mod, "choice_application", choose)

    result = provider(command_loop, "")

    assert titles == ["Request API", "Reasoning effort"]
    assert "Set provider.model = only-model" in result


def test_diff_viewer_switches_tabs_and_opens_selected_file(tmp_path):
    command_loop = diff_loop(tmp_path)
    switched = ModalHarness(["l", "q"])
    command_loop.tui = switched
    diff_viewer(command_loop)
    opened = ModalHarness(["j", "enter", "q"])
    command_loop.tui = opened
    diff_viewer(command_loop)

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
    diff_viewer(command_loop)
    scrolled = ModalHarness(["enter", "c-d", "c-d", "q"])
    command_loop.tui = scrolled
    diff_viewer(command_loop)

    initial_text = "".join(text for frame in initial.frames for _style, text in frame)
    scrolled_text = "".join(text for frame in scrolled.frames for _style, text in frame)
    assert initial_text != scrolled_text
    assert "[diff]" in scrolled_text


def test_empty_diff_viewer_reports_zero_position(tmp_path):
    command_loop = loop(tmp_path)
    modal = ModalHarness(["q"])
    command_loop.tui = modal
    diff_viewer(command_loop)
    text = "".join(text for frame in modal.frames for _style, text in frame)

    assert "No diffs" in text
    assert "[0/0]" in text


def test_diff_view_state_owns_navigation_transitions():
    state = DiffViewState(TabbedViewState(("Latest", "Session")))

    state.handle_key("down", 3, 10)
    assert state.file == 1
    state.handle_key("enter", 3, 10)
    assert state.mode is DiffViewState.Mode.FILE
    state.handle_key("c-d", 3, 10)
    assert state.view.scroll == 5
    assert state.handle_key("escape", 3, 10) is TUI_MODAL_PENDING
    assert state.mode is DiffViewState.Mode.LIST

    state.handle_key("right", 3, 10)
    assert state.view.tab == 1
    assert state.file == 0
    assert state.handle_key("r", 3, 10) is DiffViewState.REFRESH
    assert state.handle_key("q", 3, 10) is None


def test_diff_view_g_and_shift_g_jump_top_and_bottom():
    state = DiffViewState(TabbedViewState(("Latest", "Session")))

    # LIST mode: jump file selection to last / first.
    state.handle_key("G", 5, 10)
    assert state.file == 4
    state.handle_key("g", 5, 10)
    assert state.file == 0

    # FILE mode: jump scroll to bottom (clamped on render) / top.
    state.handle_key("enter", 5, 10)
    assert state.mode is DiffViewState.Mode.FILE
    state.handle_key("G", 5, 10)
    assert state.view.scroll > 0
    state.handle_key("g", 5, 10)
    assert state.view.scroll == 0


@pytest.mark.parametrize(("key", "expected_tab"), [("l", 1), ("tab", 1), ("h", 0)])
def test_diff_view_h_l_and_tab_switch_tabs_from_file_preview(key, expected_tab):
    state = DiffViewState(TabbedViewState(("Latest", "Session"), tab=0 if key != "h" else 1))
    state.open_file(3)

    state.handle_key(key, 3, 10)

    assert state.view.tab == expected_tab
    assert state.mode is DiffViewState.Mode.LIST
    assert state.file == 0


def test_api_command_reports_an_incompatible_builtin_tools_configuration_without_clearing_it(tmp_path):
    """Switching /api reports inactive builtin tools and never rewrites provider config."""
    command_loop = loop(tmp_path)
    provider_config = command_loop.session.config.provider
    provider_config.url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    provider_config.model = "qwen3.8-max-preview"
    provider_config.key = "sk-test"
    provider_config.api = "responses"
    provider_config.builtin_tools = ({"type": "web_search"}, {"type": "web_extractor"})

    assert api(command_loop, "chat") == "Set provider.api = chat (wire: chat); builtin_tools inactive on chat"
    # The requested API value is applied and the provider configuration is left intact.
    assert provider_config.api == "chat"
    assert provider_config.builtin_tools == ({"type": "web_search"}, {"type": "web_extractor"})

    # The next request projects no provider-native tools on the mismatched wire.
    assert ModelClient(command_loop.session).builtin_tools() == []

    # Switching back restores the working Responses configuration without erasing it.
    assert api(command_loop, "responses") == "Set provider.api = responses (wire: responses)"
    assert provider_config.builtin_tools == ({"type": "web_search"}, {"type": "web_extractor"})


def test_api_command_reports_when_no_wire_accepts_the_configured_builtin_tools(tmp_path):
    """DeepSeek has no provider-side tools channel, so the shared config stays inactive."""
    command_loop = loop(tmp_path)
    provider_config = command_loop.session.config.provider
    provider_config.url = "https://api.deepseek.com/v1"
    provider_config.model = "deepseek-chat"
    provider_config.key = "sk-test"
    provider_config.builtin_tools = ({"type": "web_search"},)

    assert api(command_loop, "chat") == "Set provider.api = chat (wire: chat); builtin_tools inactive on chat"
    assert provider_config.builtin_tools == ({"type": "web_search"},)


def test_config_distinguishes_configured_and_active_builtin_tools(tmp_path):
    command_loop = loop(tmp_path)
    provider_config = command_loop.session.config.provider
    provider_config.url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    provider_config.model = "qwen3.8-max-preview"
    provider_config.api = "chat"
    provider_config.builtin_tools = ({"type": "web_search"}, {"type": "web_extractor"})

    inactive = config(command_loop, "")
    assert "provider.builtin_tools: web_search, web_extractor" in inactive
    assert "provider.resolved_builtin_tools: inactive on chat: web_search, web_extractor" in inactive

    provider_config.api = "responses"
    active = config(command_loop, "")
    assert "provider.resolved_builtin_tools: active: web_search, web_extractor" in active


def test_api_command_uses_the_same_entry_policy_as_the_request_boundary(tmp_path):
    """A valid wire with an unsupported entry is reported immediately, not only on send."""
    command_loop = loop(tmp_path)
    provider_config = command_loop.session.config.provider
    provider_config.url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    provider_config.model = "qwen3.8-max-preview"
    provider_config.key = "sk-test"
    provider_config.builtin_tools = ({"type": "code_interpreter"},)

    assert api(command_loop, "responses") == "Set provider.api = responses (wire: responses); unsupported builtin_tools: code_interpreter"
    with pytest.raises(ModelError):
        ModelClient(command_loop.session).builtin_tools()


# --- /worker pickers and tab completion, mirroring the /provider /model /reason surfaces. ---


def test_worker_command_completion(tmp_path):
    from prompt_toolkit.document import Document

    command_loop = loop(tmp_path)
    command_loop.session.config.providers["alt"] = ProviderConfig(model="m", available_models=("w-a", "w-b"))
    command_loop.session.config.worker_provider = "alt"
    completer = CommandCompleter(
        providers=lambda: tuple(sorted(command_loop.session.config.providers)),
        worker_models=lambda: tuple(
            dict.fromkeys(
                (
                    *command_loop.session.config.providers[
                        command_loop.session.config.worker_provider or command_loop.session.config.active_provider
                    ].available_models,
                    "default",
                )
            )
        ),
    )

    sub_texts = [c.text for c in completer.get_completions(Document("/worker "), None)]
    assert set(sub_texts) == {"status", "reset", "on", "off", "provider", "model", "reason", "api"}

    provider_texts = [c.text for c in completer.get_completions(Document("/worker provider "), None)]
    assert set(provider_texts) == {"default", "alt", "off"}

    model_texts = [c.text for c in completer.get_completions(Document("/worker model "), None)]
    assert set(model_texts) == {"w-a", "w-b", "default"}

    reason_texts = [c.text for c in completer.get_completions(Document("/worker reason "), None)]
    assert set(reason_texts) == set(REASONING_CHOICES) | {"default"}

    api_texts = [c.text for c in completer.get_completions(Document("/worker api "), None)]
    assert set(api_texts) == set(PROVIDER_API_CHOICES) | {"default"}


# /worker api is the typed form of the [worker] api knob: it sets the override, "default" clears
# it back to inheriting the entry's own protocol, and an unknown value is rejected with usage.
def test_worker_api_subcommand_sets_clears_and_rejects(tmp_path):
    command_loop = loop(tmp_path)

    assert worker_command(command_loop, "api responses") == "Set worker.api = responses"
    assert command_loop.session.config.worker_api == "responses"

    assert worker_command(command_loop, "api default") == "worker api: (inherit)"
    assert command_loop.session.config.worker_api == ""

    assert worker_command(command_loop, "api oai") == "Usage: /worker api " + "|".join(PROVIDER_API_CHOICES)
    assert command_loop.session.config.worker_api == ""

    assert worker_command(command_loop, "api chat responses") == "Usage: /worker api [API]"


def test_worker_api_picker_sets_and_clears_like_the_typed_form(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    picks = iter(["chat", "default"])
    calls = []

    def select(_loop, title, choices, **kwargs):
        calls.append((title, choices, kwargs))
        return next(picks)

    monkeypatch.setattr(worker_mod, "select_choice", select)

    assert worker_command(command_loop, "api") == "Set worker.api = chat"
    assert command_loop.session.config.worker_api == "chat"
    assert calls[0][0] == "Worker api"
    assert set(calls[0][1]) == set(PROVIDER_API_CHOICES) | {"default"}
    assert calls[0][2]["labels"]["default"].startswith("default")

    assert worker_command(command_loop, "api") == "worker api: (inherit)"
    assert command_loop.session.config.worker_api == ""


def test_worker_status_line_reports_worker_config(tmp_path):
    command_loop = loop(tmp_path)

    assert "worker: no active session" in worker_command(command_loop, "")


# The confirm-time `c` loop reuses the shared choice selector: pick a knob, drive the matching
# /worker picker, and loop until done/Esc (or a non-interactive select yields nothing).
def test_run_worker_config_drives_pickers_until_done(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    picks = iter(["provider", "api", "done"])
    calls = []
    driven = []

    def select(_loop, title, choices, **kwargs):
        calls.append((title, choices, kwargs))
        return next(picks)

    monkeypatch.setattr(worker_mod, "select_choice", select)
    monkeypatch.setattr(WorkerFlow, "_worker_provider_picker", lambda self: driven.append("provider"))
    monkeypatch.setattr(WorkerFlow, "_worker_model_picker", lambda self: driven.append("model"))
    monkeypatch.setattr(WorkerFlow, "_worker_reason_picker", lambda self: driven.append("effort"))
    monkeypatch.setattr(WorkerFlow, "_worker_api_picker", lambda self: driven.append("api"))

    WorkerFlow(command_loop).run_worker_config()

    assert driven == ["provider", "api"]
    assert calls[0][0] == "Worker config"
    assert calls[0][1] == ("provider", "model", "effort", "api", "done")
    assert calls[0][2]["current"] == "done"  # Enter with nothing selected exits
    assert calls[0][2]["labels"]["provider"].startswith("provider:")
    assert calls[0][2]["labels"]["model"].startswith("model:")

    # Esc (SELECTION_BACK) and a non-interactive select (None) both exit without driving pickers.
    for value in (SELECTION_BACK, None):
        monkeypatch.setattr(worker_mod, "select_choice", lambda *a, value=value, **k: value)
        WorkerFlow(command_loop).run_worker_config()
    assert driven == ["provider", "api"]


# The no-arg pickers follow the /provider picker pattern: select_choice is stubbed, and the
# selection runs the exact same set path as the typed form (live-apply, frozen-gate note).
def test_worker_provider_picker_sets_and_clears_like_the_typed_form(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["alt"] = ProviderConfig(model="m")
    calls = []
    # First picker: provider "alt", then the cascade's model/reason pickers (default keeps the
    # entry's values), then a second /worker provider that clears with "off".
    picks = iter(["alt", "default", "default", "off"])

    def select(_loop, title, choices, **kwargs):
        calls.append((title, choices, kwargs))
        return next(picks)

    monkeypatch.setattr(worker_mod, "select_choice", select)

    first = worker_command(command_loop, "provider")
    assert calls[0][0] == "Worker provider"
    assert "off" in calls[0][1]
    assert calls[0][1][-1] == "off"  # the clear entry trails the provider names
    assert [call[0] for call in calls] == ["Worker provider", "Worker model", "Worker reasoning"]
    assert first.startswith("Set worker provider = alt")
    assert "worker model: (inherit)" in first
    assert "worker reasoning: (inherit)" in first
    assert command_loop.session.config.worker_provider == "alt"
    assert command_loop.session.config.worker_model == ""
    assert command_loop.session.config.worker_reasoning == ""

    cleared = worker_command(command_loop, "provider")
    assert calls[3][0] == "Worker provider"
    assert calls[3][2]["labels"] == {"alt": "alt (current)"}  # the live entry is marked
    assert cleared == "worker provider: off"  # picking "off" clears without cascading
    assert command_loop.session.config.worker_provider == ""


def test_worker_model_picker_sets_the_override_without_the_model_chain(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["alt"] = ProviderConfig(model="m-a", available_models=("m-a", "m-b"))
    command_loop.session.config.worker_provider = "alt"
    command_loop.session.config.worker_model = "m-c"
    titles = []
    discovered = []
    picks = iter(["m-b"])

    def select(_loop, title, choices, **kwargs):
        titles.append(title)
        assert "default" in choices and "m-c" in choices and "m-a" in choices
        return next(picks)

    monkeypatch.setattr(worker_mod, "select_choice", select)
    monkeypatch.setattr(commands_mod, "remote_models", lambda _loop, entry: discovered.append(entry.model) or ("m-remote",))

    result = worker_command(command_loop, "model")
    assert titles == ["Worker model"]
    assert discovered == ["m-a"]  # discovery ran against the worker's entry, not the parent's
    assert command_loop.session.config.worker_model == "m-b"
    assert result == "Set worker.model = m-b"


def test_worker_reason_picker_covers_efforts_and_default(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.worker_reasoning = "high"
    picks = iter(["low"])

    def select(_loop, title, choices, **kwargs):
        assert set(choices) == set(REASONING_CHOICES) | {"default"}
        assert kwargs["labels"] == {"default": "default - inherit the provider entry's reasoning", "high": "high (current)"}
        return next(picks)

    monkeypatch.setattr(worker_mod, "select_choice", select)
    result = worker_command(command_loop, "reason")
    assert command_loop.session.config.worker_reasoning == "low"
    assert result == "Set worker.reasoning = low"


def test_worker_pickers_return_no_change_on_back(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["alt"] = ProviderConfig(model="m", available_models=("m",))
    command_loop.session.config.worker_provider = "alt"
    command_loop.session.config.worker_model = "m-x"
    command_loop.session.config.worker_reasoning = "high"
    monkeypatch.setattr(worker_mod, "select_choice", lambda *_args, **_kwargs: SELECTION_BACK)
    assert worker_command(command_loop, "provider") == "No change"
    assert worker_command(command_loop, "model") == "No change"
    assert worker_command(command_loop, "reason") == "No change"
    assert command_loop.session.config.worker_provider == "alt"
    assert command_loop.session.config.worker_model == "m-x"
    assert command_loop.session.config.worker_reasoning == "high"


def test_worker_model_and_reason_pickers_clear_via_default(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.worker_model = "m-x"
    command_loop.session.config.worker_reasoning = "high"
    picks = iter(["default", "default"])
    monkeypatch.setattr(worker_mod, "select_choice", lambda *_args, **_kwargs: next(picks))
    assert worker_command(command_loop, "model") == "worker model: (inherit)"
    assert worker_command(command_loop, "reason") == "worker reasoning: (inherit)"
    assert command_loop.session.config.worker_model == ""
    assert command_loop.session.config.worker_reasoning == ""


# --- /worker provider cascade: the no-arg picker flows provider -> model -> reasoning. ---


def test_worker_provider_picker_cascades_into_model_and_reasoning(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["fast"] = ProviderConfig(model="fast-model", available_models=("fast-model", "fast-mini"))
    worker = SimpleNamespace(config=Config())  # a live worker session, like session.worker after a spawn
    command_loop.session.worker = worker
    titles = []
    discovered = []
    picks = iter(["fast", "fast-mini", "high"])

    def select(_loop, title, choices, **kwargs):
        titles.append(title)
        return next(picks)

    monkeypatch.setattr(worker_mod, "select_choice", select)
    monkeypatch.setattr(commands_mod, "remote_models", lambda _loop, entry: discovered.append(entry.model) or ("remote-mini",))

    result = worker_command(command_loop, "provider")

    assert titles == ["Worker provider", "Worker model", "Worker reasoning"]
    assert discovered == ["fast-model"]  # discovery ran against the newly selected entry
    assert command_loop.session.config.worker_provider == "fast"
    assert command_loop.session.config.worker_model == "fast-mini"
    assert command_loop.session.config.worker_reasoning == "high"
    assert "Set worker provider = fast" in result
    assert "Set worker.model = fast-mini" in result
    assert "Set worker.reasoning = high" in result
    # the live worker's detached entry reflects all three stages
    assert worker.config.active_provider == "fast"
    assert worker.config.providers["fast"].model == "fast-mini"
    assert worker.config.providers["fast"].reasoning == "high"


def test_worker_provider_cascade_aborts_at_model_stage_keeping_earlier_stages(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["fast"] = ProviderConfig(model="fast-model", available_models=("fast-model",))
    command_loop.session.config.worker_model = "m-x"
    command_loop.session.config.worker_reasoning = "high"
    picks = iter(["fast", SELECTION_BACK])
    monkeypatch.setattr(worker_mod, "select_choice", lambda *_args, **_kwargs: next(picks))
    monkeypatch.setattr(commands_mod, "remote_models", lambda _loop, _entry: ())

    result = worker_command(command_loop, "provider")

    assert command_loop.session.config.worker_provider == "fast"  # the provider stage landed
    assert command_loop.session.config.worker_model == "m-x"  # model/reasoning untouched
    assert command_loop.session.config.worker_reasoning == "high"
    assert "Set worker provider = fast" in result
    assert "worker model: unchanged" in result
    assert "worker reasoning" not in result


def test_worker_provider_cascade_aborts_at_reason_stage_keeping_model(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["fast"] = ProviderConfig(model="fast-model", available_models=("fast-model",))
    command_loop.session.config.worker_reasoning = "high"
    picks = iter(["fast", "fast-model", None])  # None = the picker was dismissed
    monkeypatch.setattr(worker_mod, "select_choice", lambda *_args, **_kwargs: next(picks))
    monkeypatch.setattr(commands_mod, "remote_models", lambda _loop, _entry: ())

    result = worker_command(command_loop, "provider")

    assert command_loop.session.config.worker_provider == "fast"
    assert command_loop.session.config.worker_model == "fast-model"
    assert command_loop.session.config.worker_reasoning == "high"  # untouched by the dismissal
    assert "Set worker.model = fast-model" in result
    assert "worker reasoning: unchanged" in result


def test_worker_provider_typed_form_does_not_cascade(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.session.config.providers["alt"] = ProviderConfig(model="m")
    monkeypatch.setattr(worker_mod, "select_choice", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("the typed form opens no picker")))

    result = worker_command(command_loop, "provider alt")

    assert result.startswith("Set worker provider = alt")
    assert command_loop.session.config.worker_provider == "alt"
    assert command_loop.session.config.worker_model == ""
    assert command_loop.session.config.worker_reasoning == ""


def test_tool_output_viewer_opens_a_stored_script_in_the_scrolling_viewer(tmp_path):
    """Ctrl-O is the only door to a script under yolo, where no prompt ever offered `v`: the entry
    hands the stored source to the same read-only viewer the confirm-time key opens."""
    command_loop = loop(tmp_path)
    code = "\n".join(f"x{index} = {index}" for index in range(30))
    envelope = "ToolScript ok\ncalls: 2 [tr.1-2]\nstdout:\ncounted 30 rows"
    command_loop.session.store_tool_result("ToolScript", [{"action": "call", "code": code}], envelope)
    modal = ModalHarness(["enter", "G"])  # open the entry, then scroll the viewer to the bottom
    command_loop.tui = modal

    tool_output_viewer(command_loop)

    listing = "".join(value for _style, value in modal.frames[0])
    assert "ToolScript call 30 lines" in listing
    frames = ["".join(value for _style, value in frame) for frame in modal.frames]
    viewer = [frame for frame in frames if "Script · tr.1 · read-only" in frame]
    assert viewer, "the entry hands off to the read-only script viewer"
    assert " 1  x0 = 0" in viewer[0]  # numbered, so a traceback's line N is findable
    assert "x29 = 29" in viewer[-1]  # the whole script is reachable, not just the excerpt
    assert "calls  2" in viewer[0]
    # A script is a question and its printed output is the answer, so the entry carries both.
    assert "── result " in viewer[-1]
    assert "counted 30 rows" in viewer[-1]


def test_tool_output_viewer_skips_a_describe_with_no_script(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.session.store_tool_result("ToolScript", [{"action": "describe", "tools": ["Read"]}], "Read\njson:    no")
    modal = ModalHarness([])
    command_loop.tui = modal

    tool_output_viewer(command_loop)

    assert modal.frames == []


def test_tool_output_viewer_shows_a_failed_script_with_its_traceback(tmp_path):
    """The log line clips a failure to one row. Here the whole traceback sits under the numbered
    source, so `File "<toolscript>", line N` resolves against the line it names."""
    command_loop = loop(tmp_path)
    code = "rows = []\nprint(rows[2])\n"
    envelope = 'ToolScript failed\ncalls: 0\nerror:\nTraceback (most recent call last):\n  File "<toolscript>", line 2, in <module>\nIndexError: list index out of range'
    command_loop.session.store_tool_result("ToolScript", [{"action": "call", "code": code}], envelope)
    modal = ModalHarness(["enter"])
    command_loop.tui = modal

    tool_output_viewer(command_loop)

    viewer = "".join(value for _style, value in modal.frames[-1])
    assert " 2  print(rows[2])" in viewer
    assert "IndexError: list index out of range" in viewer
    assert 'File "<toolscript>", line 2' in viewer


def test_tool_output_viewer_shows_the_whole_command_not_the_clipped_log_line(tmp_path):
    """The transcript row collapses and clips a command at 200 characters. A viewer opened to see
    what was run has to show what was run."""
    command_loop = loop(tmp_path)
    command = "rg --json " + " ".join(f"--glob '!vendor/{index}/**'" for index in range(30)) + " pattern"
    command_loop.session.store_tool_result("Bash", [command], Tool.process_result("BashToolResult", 0, "hit", ""))
    modal = ModalHarness(["enter"])
    command_loop.tui = modal

    tool_output_viewer(command_loop)

    viewer = next(frame for frame in ("".join(v for _s, v in f) for f in modal.frames) if "read-only" in frame)
    assert "vendor/29" in viewer  # the tail of the command survived
    assert "..." not in viewer.split("── result")[0]
