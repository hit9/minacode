"""tool output browser (split from tests/test_command_ui.py)."""

import os
import shutil
import time

from prompt_toolkit.utils import get_cwidth
from test_command_ui import ModalHarness
from tui_harness import loop, session

import minacode.cli.modals as modals_mod
from minacode.cli import CommandLoop
from minacode.cli.modals import job_view, tool_output_viewer
from minacode.engine import Agent
from minacode.session import Session
from minacode.session.jobs import BackgroundJob
from minacode.tools import BashTool, JobTool, Tool, tooloutput


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

    listing = "".join(value for _, value in modal.frames[0])
    assert listing.startswith("\n──── Tool output · latest 12 ")
    assert get_cwidth(listing.splitlines()[1]) == 48
    assert "command-11" in listing and "command-2" in listing
    # A twenty-row terminal draws ten of the twelve: the rest are a scroll away, not dropped, and
    # the counter is what says so. `true` printed nothing and is not an entry at all.
    assert "Bash printf command-1\n" not in listing and "Bash printf command-0\n" not in listing and "Bash true" not in listing
    assert "showing 1-10 of 12" in listing
    # The second entry opens in the scrolling viewer: the command as its body, the streams below.
    frames = ["".join(value for _, value in frame) for frame in modal.frames]
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

    listing = "".join(value for _, value in modal.frames[0])
    assert listing.startswith("\n──── Tool output · latest 55 ")
    assert "showing 1-10 of 55" in listing
    assert "Bash printf 54" in listing  # the newest is in view


def test_tool_output_browser_keeps_every_stored_record_with_a_running_script(tmp_path, monkeypatch):
    """A running ToolScript's live entry does not push the oldest stored record out: with a full
    session the browser lists all 400 stored results plus the running one, and the viewport still
    bounds the screen."""
    command_loop = loop(tmp_path)
    for index in range(400):
        command_loop.session.store_tool_result("Bash", [f"printf {index}"], Tool.process_result("BashToolResult", 0, f"out {index}", ""))
    command_loop.script_running_code = "print('hi')\n"
    modal = ModalHarness(["q"])
    command_loop.tui = modal

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((50, 20)))
        tool_output_viewer(command_loop)

    listing = "".join(value for _, value in modal.frames[0])
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

    frames = ["".join(value for _, value in frame) for frame in modal.frames]
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

    frames = ["".join(value for _, value in frame) for frame in modal.frames]
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

    frames = ["".join(value for _, value in frame) for frame in modal.frames]
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

    frames = ["".join(value for _, value in frame) for frame in modal.frames]
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

    frames = ["".join(value for _, value in frame) for frame in modal.frames]
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

    listing = "".join(value for _, value in modal.frames[0])
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

    listing = "".join(value for _, value in modal.frames[0])
    assert "Tool output · latest 1" in listing  # a status carries no order and is not an entry
    viewer = next(frame for frame in ("".join(value for _, value in f) for f in modal.frames) if "read-only" in frame)
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

    frames = ["".join(value for _, value in frame) for frame in modal.frames]
    viewer = [frame for frame in frames if "read-only" in frame]
    assert "line 0" in viewer[0]
    assert "line 39" in viewer[-1]  # reachable by scrolling, not elided
    assert "lines omitted" not in "".join(viewer)


def test_tool_output_viewer_bounds_a_huge_result_and_says_so(tmp_path):
    """Stored output has no cap and the wrapper is quadratic in one line's length, so the viewer
    bounds what it renders -- and says how much it is showing, because a reader who cannot tell an
    elided result from a complete one has to distrust every result."""
    command_loop = loop(tmp_path)
    stdout = "\n".join(f"line {line}" for line in range(tooloutput.VIEWER_LINES * 2))
    command_loop.session.store_tool_result("Bash", ["seq huge"], Tool.process_result("BashToolResult", 0, stdout, ""))
    command_loop.session.store_tool_result("Bash", ["one long line"], Tool.process_result("BashToolResult", 0, "x" * (tooloutput.VIEWER_LINE_CHARS * 3), ""))
    modal = ModalHarness(["enter"])
    command_loop.tui = modal

    tool_output_viewer(command_loop)

    frames = ["".join(value for _, value in frame) for frame in modal.frames]
    viewer = next(frame for frame in frames if "read-only" in frame)
    # The newest entry is the one long line; the header says the clip happened rather than
    # presenting a truncated line as the whole of it.
    assert f"long lines clipped at {tooloutput.VIEWER_LINE_CHARS}" in viewer
    rendered = [row for row in viewer.splitlines() if row.strip().startswith("x")]
    assert rendered and all(len(row) <= tooloutput.VIEWER_LINE_CHARS + 10 for row in rendered)


def test_tool_output_viewer_bounds_a_result_with_too_many_lines(tmp_path):
    """The line bound counts against the streams, not the stored envelope, so the note it prints
    is a fact about the output rather than about the tags wrapped around it."""
    command_loop = loop(tmp_path)
    stdout = "\n".join(f"line {line}" for line in range(tooloutput.VIEWER_LINES * 2))
    command_loop.session.store_tool_result("Bash", ["seq huge"], Tool.process_result("BashToolResult", 0, stdout, ""))
    modal = ModalHarness(["enter"])
    command_loop.tui = modal

    tool_output_viewer(command_loop)

    viewer = next(frame for frame in ("".join(v for _, v in f) for f in modal.frames) if "read-only" in frame)
    assert f"{tooloutput.VIEWER_LINES} shown of {tooloutput.VIEWER_LINES * 2}" in viewer
    # The elision is marked in the text too, where it happens -- the header note is derived from it.
    bounded, note = tooloutput.bash_viewer_output(command_loop.session.tool_records[-1].output)
    assert "lines omitted" in bounded and note.startswith(f"{tooloutput.VIEWER_LINES} shown of ")


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

    listing = "".join(value for _, value in modal.frames[0])
    assert "running  ToolScript call 2 lines" in listing  # first row, above the stored Bash entry
    assert listing.index("running") < listing.index("Bash")
    frames = ["".join(value for _, value in frame) for frame in modal.frames]
    viewer = [frame for frame in frames if "read-only" in frame]
    assert "Script · running · read-only" in viewer[0]
    assert 'call("server.tool"' in viewer[0]

    # It leaves with the script: once the batch returns, only the stored record remains.
    command_loop.tui = None
    command_loop.toolscript_run_status(False)
    modal = ModalHarness([])
    command_loop.tui = modal
    tool_output_viewer(command_loop)
    assert "running" not in "".join(value for _, value in modal.frames[0])


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
    assert not [style for style, _ in row if "choice.selected" in style and ("meta" in style or "tool" in style)]


def test_tool_output_viewer_reads_resumed_history(tmp_path):
    saved = session(tmp_path)
    saved.store_tool_result("Bash", ["printf persisted"], Tool.process_result("BashToolResult", 0, "persisted output", ""))
    saved.save_snapshot()
    restored = Session.load_snapshot(saved.uid, config=saved.config)
    command_loop = CommandLoop(Agent(restored, output_fn=lambda _text: None), input_fn=lambda prompt="": "", output_fn=lambda _text: None)
    modal = ModalHarness(["enter", "c-o"], consumed=True)
    command_loop.tui = modal

    tool_output_viewer(command_loop)

    detail = next(frame for frame in ("".join(value for _, value in f) for f in modal.frames) if "read-only" in frame)
    assert "printf persisted" in detail
    assert "persisted output" in detail


def _finished_job(tmp_path, job_id: str, command: str, log_text: str) -> "BackgroundJob":
    """A done BackgroundJob whose log file is on disk, the state a real ``Job(start)`` leaves
    behind after the process exits."""
    import subprocess

    log_path = tmp_path / f"{job_id}.log"
    log_path.write_text(log_text)
    proc = subprocess.Popen(["true"])
    proc.wait()
    return BackgroundJob(id=job_id, command=command, process=proc, log_path=str(log_path), started_at=0.0, status="done", exit_code=0)


def test_tool_output_browser_shows_the_job_log_for_a_known_job(tmp_path):
    """A Job status/wait record opens the job's full log while the job still exists in the
    session, so the browser is where a backgrounded build's real output is read."""
    command_loop = loop(tmp_path)
    command_loop.session.jobs["job.3"] = _finished_job(tmp_path, "job.3", "make build", "compiling...\nbuild ok\n")
    command_loop.session.store_tool_result("Job", [{"action": "wait", "job": "job.3"}], "Job: job.3\nStatus: done\n--- output ---\nbuild ok")

    view = job_view(command_loop, command_loop.session.tool_records[-1])

    assert view is not None
    assert view.label == "job · tr.1"
    assert "compiling..." in view.text and "build ok" in view.text
    assert ("job", "job.3") in view.rows
    assert ("status", "done") in view.rows
    assert ("exit", "0") in view.rows
    assert ("command", "make build") in view.rows
    assert "Status: done" in view.result


def test_tool_output_browser_job_view_falls_back_to_the_return_value_when_the_job_is_gone(tmp_path):
    """After a resume the session's job table is gone and a kill deletes the log file; those
    records show the tool's return value instead of pretending there is a log."""
    command_loop = loop(tmp_path)
    command_loop.session.store_tool_result("Job", [{"action": "wait", "job": "job.9"}], "Job: job.9\nStatus: done")

    view = job_view(command_loop, command_loop.session.tool_records[-1])

    assert view is not None
    assert view.text == "Job: job.9\nStatus: done"
    assert view.result == ""


def test_tool_output_browser_job_start_record_links_to_the_job_it_started(tmp_path):
    """A ``Job(start)`` call names the job only in its return value; the view finds the id there
    and shows the log anyway."""
    command_loop = loop(tmp_path)
    command_loop.session.jobs["job.1"] = _finished_job(tmp_path, "job.1", "long task", "started output\n")
    command_loop.session.store_tool_result("Job", [{"action": "start", "command": "long task"}], "Started job.1: long task")

    view = job_view(command_loop, command_loop.session.tool_records[-1])

    assert view is not None
    assert "started output" in view.text


def test_job_view_falls_back_when_a_known_jobs_log_was_removed(tmp_path):
    command_loop = loop(tmp_path)
    job = _finished_job(tmp_path, "job.1", "finished task", "finished output\n")
    command_loop.session.jobs[job.id] = job
    os.unlink(job.log_path)
    command_loop.session.store_tool_result("Job", [{"action": "kill", "job": job.id}], "Killed job.1 (status=done, exit_code=0)")

    view = job_view(command_loop, command_loop.session.tool_records[-1])

    assert view.text == "Killed job.1 (status=done, exit_code=0)"
    assert view.result == ""


def test_promoted_bash_job_view_includes_output_from_before_and_after_promotion(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.session.settings.bash_wait_timeout = 0.03

    promoted = BashTool(command_loop.session, ["printf early; sleep 0.08; printf late"]).call()
    job = command_loop.session.jobs["job.1"]
    status = JobTool(command_loop.session, [{"action": "wait", "job": job.id, "timeout": 2}]).call()
    # The process can exit just before its drainer consumes EOF. Wait for that observable state
    # instead of relying on one scheduler-sized sleep.
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        snapshot = job.log_snapshot(BackgroundJob.BUFFER_LIMIT)
        if snapshot is not None and "late" in snapshot[0]:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("background drainer did not capture the completed output")
    command_loop.session.store_tool_result("Job", [{"action": "wait", "job": job.id}], status)

    view = job_view(command_loop, command_loop.session.tool_records[-1])

    assert "early" in promoted
    assert "early" in view.text
    assert "late" in view.text


def test_disk_job_log_snapshot_reads_both_ends_with_a_fixed_bound(tmp_path):
    job = _finished_job(tmp_path, "job.1", "large task", "0123456789")

    snapshot = job.log_snapshot(6)

    assert snapshot is not None
    text, bounded = snapshot
    assert text.startswith("012") and text.endswith("789")
    assert "middle of job log omitted" in text
    assert bounded is True


def test_tool_output_browser_defers_job_log_read_until_the_row_opens(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    job = _finished_job(tmp_path, "job.1", "large task", "output")
    command_loop.session.jobs[job.id] = job
    command_loop.session.store_tool_result("Job", [{"action": "status", "job": job.id}], "Job: job.1\nStatus: done")
    reads = []
    monkeypatch.setattr(job, "log_snapshot", lambda _limit: reads.append(True) or ("output", False))
    command_loop.tui = ModalHarness(["q"])

    tool_output_viewer(command_loop)

    assert reads == []
