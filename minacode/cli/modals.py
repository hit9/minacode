"""Modal / question UI flows as free functions taking the CommandLoop.

These render blocking prompt_toolkit UIs (choice lists, free-text questions, the diff viewer,
the bash output viewer, and the MCP manager). They return the selected value, or None when
dismissed. Free functions so the command handlers in commands.py and the runtime can call them
without a CommandLoop instance.
"""

from __future__ import annotations

import contextlib
import re
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples, to_formatted_text
from prompt_toolkit.utils import get_cwidth
from rich.console import Console
from rich.markdown import Markdown

from minacode.base import DISMISSED, SELECTION_BACK, ApprovalView, Text, ToolCall, ToolError, TurnBox, oneline
from minacode.render import UiPrinter
from minacode.session import BackgroundJob, ToolResultRecord
from minacode.tools import AskSpec, BashTool, DelegateTool, ToolScript, tooloutput
from minacode.tui import (
    ASK_DONE,
    ASK_FREE_TEXT,
    TUI_MODAL_PENDING,
    AskViewState,
    ChoiceViewState,
    DiffViewState,
    SegmentLogViewState,
    TabbedViewState,
)

if TYPE_CHECKING:
    from minacode.cli import CommandLoop
    from minacode.session import HistorySegment

# A detail opened from the Ctrl-O browser can say ``Esc`` to go back to the list instead of
# closing the whole browser. ``show_modal`` closes on any non-pending return, so the viewer
# hands this sentinel back and ``tool_output_viewer`` reopens the list around it.

_TOOL_OUTPUT_BACK = object()


def wrapped_rows(text: str, width: int, margin: str = "  ", style: str = "") -> list[StyleAndTextTuples]:
    """Source text as display rows, one logical line at a time. Text.wrap_styled measures in
    terminal cells, so CJK text (two cells per character) wraps where it actually reaches the right
    edge instead of overflowing at twice the width, and each continuation row re-indents to its
    source line's own indent so indented code keeps its shape.

    This is the plain path on purpose: rendering conversation text as markdown would fold the
    structural newlines and drop anything that looks like an HTML tag, and a viewer whose whole
    job is showing what was evicted may not quietly edit it."""
    rows: list[StyleAndTextTuples] = []
    for raw in text.splitlines():
        if not raw.strip():
            rows.append([])
            continue
        indent = raw[: len(raw) - len(raw.lstrip())][: max(0, width // 4)]
        rows.extend(
            cast(
                list[StyleAndTextTuples],
                Text.wrap_styled([("", margin)], [("", margin + indent)], [(style, raw)], width),
            )
        )
    return rows


# The stored scope/trigger are internal words; these are what the reader sees. The list gets the
# short form, the opened segment gets the sentence — same fact, room for plain language.
SEGMENT_SCOPE_WORDS = {"history": ("earlier", "earlier conversation"), "turn": ("this turn", "the turn that was running")}
SEGMENT_TRIGGER_WORDS = {"auto": ("automatic", "Compacted automatically"), "manual": ("manual", "Compacted by /compact")}


def segment_columns(segment: HistorySegment) -> tuple[str, str, str]:
    """One segment as (when, kind, messages) display columns. Older snapshots predate the metadata,
    so every field falls back to a dash rather than inventing a value."""
    stamp = segment.created_at
    when = f"{stamp[5:10]} {stamp[11:16]}" if len(stamp) >= 16 else "—"
    trigger = SEGMENT_TRIGGER_WORDS.get(segment.trigger, (segment.trigger, ""))[0]
    scope = SEGMENT_SCOPE_WORDS.get(segment.scope, (segment.scope, ""))[0]
    kind = " · ".join(part for part in (trigger, scope) if part) or "—"
    if segment.fallback:
        kind += " · no summary"
    return when, kind, f"{segment.messages} msgs" if segment.messages else "—"


def missing_summary_note(segment: HistorySegment) -> str:
    """Stand-in for a segment with no summary, saying which of the two reasons it is."""
    return "(none recorded)" if segment.trigger else "(not recorded — this segment predates the log)"


def segment_story(segment: HistorySegment) -> tuple[str, str]:
    """(what this compaction was, what to know about it) as sentences for the opened segment.

    The reader is looking at conversation that is no longer in context and wants to know why it
    left and how much to trust what replaced it — not the internal words for the code path."""
    trigger = SEGMENT_TRIGGER_WORDS.get(segment.trigger, ("", ""))[1]
    scope = SEGMENT_SCOPE_WORDS.get(segment.scope, ("", ""))[1]
    if not trigger:
        # Recorded by a build that kept only the text: say that, so the missing detail below does
        # not read as something that went wrong here.
        return "Compacted before minacode kept these details", ""
    headline = f"{trigger} to free room in the context" if not scope else f"{trigger}, dropping {scope}"
    if segment.messages:
        headline += f" · {segment.messages} messages"
    if segment.model:
        headline += f" · model {segment.model}"
    if segment.fallback:
        return headline, "Summarizing failed, so this was trimmed without a summary — what it dropped survives only in the excerpt the agent can recall."
    return headline, ""


def mcp_manager(loop: CommandLoop) -> None:
    mcp = loop.session.mcp
    tui = loop.tui
    if mcp is None or tui is None:
        return
    configs = tuple(mcp.parse_configs())
    if not configs:
        loop.ui.emit_answer(mcp.render_server_status(), indent=TurnBox.CONTENT_LEVEL)
        return

    state = ChoiceViewState(tuple(config.name for config in configs), {}, set())
    transitions: dict[str, str] = {}
    errors: dict[str, str] = {}
    state_lock = threading.Lock()
    modal_open = threading.Event()
    modal_open.set()

    def server_labels() -> dict[str, str]:
        with state_lock:
            changing = dict(transitions)
            failed = dict(errors)
        server_rows = []
        for config in configs:
            if transition := changing.get(config.name):
                status = mcp.STATUS_MARKER + " " + transition
            elif config.name in failed:
                status = mcp.STATUS_MARKER + " error"
            elif issue := mcp.server_issue(config.name):
                status = mcp.STATUS_MARKER + " " + issue[0]
            elif mcp.connected(config.name):
                status = mcp.STATUS_MARKER + " connected"
            else:
                status = mcp.STATUS_MARKER + " disconnected"
            mode = "auto" if config.auto_connect else "manual"
            count = len(mcp.tools.get(config.name, []))
            server_rows.append((config.name, status, mode, count))
        name_width = max(len(name) for name, *_ in server_rows)
        status_width = max(len(mcp.STATUS_MARKER + " disconnecting"), *(len(status) for _, status, _, _ in server_rows))
        return {name: f"{name:<{name_width}}  {status:<{status_width}}  {mode:<6}  {count:>3} tools" for name, status, mode, count in server_rows}

    def preview(name: str) -> str:
        with state_lock:
            if message := errors.get(name):
                return message
        if issue := mcp.server_issue(name):
            return issue[1]
        return ""

    def fragments() -> StyleAndTextTuples:
        state.labels = server_labels()
        return state.fragments("MCP servers · Enter toggles connection", preview)

    def toggle(name: str, connect: bool) -> None:
        try:
            if connect:
                result = mcp.connect_server(name, interactive=True, notify=loop.emit)
            else:
                result = mcp.disconnect_server(name)
        except Exception as error:  # noqa: BLE001 - keep background MCP failures visible in the selector.
            result = f"MCP server error: {name}: {error}"

        succeeded = mcp.connected(name) == connect
        with state_lock:
            transitions.pop(name, None)
            if succeeded:
                errors.pop(name, None)
            else:
                errors[name] = result
        if modal_open.is_set():
            tui.invalidate()
        else:
            loop.emit_background(result)

    def handle_key(key: str, data: str = "") -> Any:
        result = state.handle_key(key, data)
        if not isinstance(result, str):
            return result
        with state_lock:
            if result in transitions:
                return TUI_MODAL_PENDING
            connect = not mcp.connected(result)
            errors.pop(result, None)
            transitions[result] = "connecting" if connect else "disconnecting"
        threading.Thread(target=toggle, args=(result, connect), name="mcp-toggle-" + result, daemon=True).start()
        return TUI_MODAL_PENDING

    try:
        tui.show_modal(fragments, handle_key)
    finally:
        modal_open.clear()


def select_choice(
    loop: CommandLoop,
    title: str,
    choices: tuple[str, ...],
    *,
    labels: dict[str, str] | None = None,
    current: str = "",
    disabled: set[str] | frozenset[str] = frozenset(),
    preview_fn: Callable[[str], StyleAndTextTuples | str] | None = None,
) -> str | object | None:
    labels = labels or {}
    if not choices or not loop.interactive_input:
        return None
    enabled = tuple(choice for choice in choices if choice not in disabled)
    if len(enabled) == 1:
        return enabled[0]
    try:
        return choice_application(loop, title, choices, labels, current, set(disabled), preview_fn=preview_fn)
    except (EOFError, KeyboardInterrupt):
        loop.emit_turn("Cancelled")
        return None


def choice_application(
    loop: CommandLoop,
    title: str,
    choices: tuple[str, ...],
    labels: dict[str, str],
    current: str,
    disabled: set[str],
    *,
    preview_fn: Callable[[str], StyleAndTextTuples | str] | None = None,
    label_fn: Callable[[str], StyleAndTextTuples] | None = None,
    exclusive: bool = False,
    max_rows: int = 0,
) -> str | object | None:
    state = ChoiceViewState(choices, labels, disabled, max_rows=max_rows)
    options = state.enabled()
    state.selected = options.index(current) if current in options else 0
    if loop.tui is None:
        return None
    result = loop.tui.show_modal(lambda: state.fragments(title, preview_fn, label_fn), state.handle_key, exclusive=exclusive)
    if isinstance(result, KeyboardInterrupt):
        raise result
    return result


def question_interaction(loop: CommandLoop, specs: list[AskSpec]) -> list[str]:
    """Entry point for Ask (the whole batch in one call). With a live TUI the batch runs in a
    single selector modal -- one page per question, options left, the selected option's rich
    markdown preview right (below the options on narrow terminals); a free-text page drops to
    the shared input row mid-flow and the modal reopens for the rest. Headless runs keep the
    plain per-question text prompts. The final tool log renders the returned answers."""
    if loop.tui is None or not loop.interactive_input:
        return [loop.read_input("\n" + spec.question) for spec in specs]
    state = AskViewState.build(specs)
    while True:
        size = shutil.get_terminal_size((120, 24))
        result = loop.tui.show_modal(
            lambda size=size: state.fragments(size.columns, max(1, size.lines - 6)),
            state.handle_key,
        )
        if result is ASK_DONE:
            break
        if isinstance(result, tuple) and len(result) == 2 and result[0] is ASK_FREE_TEXT:
            index = result[1]
            prompt = f"({index + 1}/{len(specs)}) {specs[index].question}" if len(specs) > 1 else specs[index].question
            answer = loop.tui.request_input("\n" + prompt)
            if answer is None:
                return [DISMISSED] * len(specs)  # Ctrl-C on a free-text page dismisses the batch
            state.picked[index] = answer
            if all(picked is not None for picked in state.picked):
                break  # a free-text answer to the last question submits without re-entering the modal
            state.active = state.picked.index(None)
            continue
        if result is SELECTION_BACK or result is None:
            return [DISMISSED] * len(specs)
        if isinstance(result, KeyboardInterrupt):
            raise result
        return [DISMISSED] * len(specs)
    answers: list[str] = []
    for index, spec in enumerate(specs):
        picked = state.picked[index]
        if picked is None:
            # Unanswered pages should never reach here (ASK_DONE requires the whole batch
            # answered); defensively cancel rather than leak the question text as an answer.
            return [DISMISSED] * len(specs)
        answer = picked
        if note := state.notes.get(index):
            answer += "\n\nUser notes: " + note
        answers.append(answer)
    return answers


def script_view(loop: CommandLoop, record: ToolResultRecord) -> ApprovalView | None:
    """The stored ToolScript call as a viewable script: the source it ran, plus what the envelope
    says came back. Rebuilt from the record rather than kept aside, so it is readable long after
    the call -- and under yolo, where no confirmation prompt ever offered `v`."""
    code = ToolScript(loop.session, record.args).script()
    if not code.strip():
        return None
    rows = [("key", record.key), ("lines", str(len(code.splitlines())))]
    fields = tooloutput.toolscript_result_fields(record.output)
    if fields is not None:
        rows.append(("calls", fields[0]))
    # The envelope rides along: a script is a question and its printed output is the answer, and
    # the transcript only kept the first lines of it. A failed script keeps its whole traceback
    # here too, which is the one place the clipped error line in the log can be resolved against
    # the numbered source right above it.
    result, note = tooloutput.viewer_text(record.output)
    if note:
        rows.append(("shown", note))
    return ApprovalView(f"script · {record.key}", code, "python", rows, result)


def bash_command(loop: CommandLoop, record: ToolResultRecord) -> str:
    """The command as it was run. Taken from the tool rather than from its log display, which is
    collapsed to one line and clipped at 200 characters -- fine for a transcript row, wrong for a
    viewer whose whole job is showing the thing in full."""
    try:
        return BashTool(loop.session, record.args).command()
    except ToolError:
        return tooloutput.short_call(loop.session, ToolCall("", "Bash", record.args)).removeprefix("Bash").strip()


def bash_view(loop: CommandLoop, record: ToolResultRecord) -> ApprovalView | None:
    """The stored Bash call as a viewable command: what was run, plus the streams it produced.

    Same two halves as a script -- what was asked, and what came back -- so both kinds of entry
    open the one viewer. The output is bounded (see ToolRunner.VIEWER_LINES): stored Bash output
    has no cap, and a viewer that hangs on the one command that printed a megabyte is worse than
    one that says how much it is showing."""
    command = bash_command(loop, record)
    streams, note = tooloutput.bash_viewer_output(record.output)
    if not streams:
        # A command that printed nothing has nothing here the transcript does not already show. A
        # script is different: its source is worth reading whether or not it printed anything.
        return None
    rows = [("key", record.key)]
    if code := tooloutput.bash_exit_code(record.output):
        rows.append(("exit", code))
    if note:
        rows.append(("shown", note))
    return ApprovalView(f"output · {record.key}", command, "bash", rows, streams)


@dataclass(frozen=True)
class OutputEntry:
    """One row of the Ctrl-O browser: how it reads in the list, and what it opens."""

    key: str
    name: str
    detail: str
    view: ApprovalView
    live: bool = False
    # The Bash result's verdict for the row's first column: "ok" (exit 0), "fail" (nonzero
    # exit), or "" when the entry is not a Bash result (a script or an order has no exit code
    # to promise). Computed once, at browse time, next to `record_view`.
    status: str = ""


def running_script_entry(loop: CommandLoop) -> OutputEntry | None:
    """The ToolScript running right now, if one is. It has no stored record yet -- that arrives
    only when the whole batch returns -- and a long batch is exactly when the reader wants to see
    what is running, so the browser offers it from the live source instead."""
    code = loop.script_running_code
    if not code.strip():
        return None
    lines = len(code.splitlines())
    detail = f"call {lines} line{'' if lines == 1 else 's'} ({len(code)} chars)"
    rows = [("status", "running"), ("lines", str(lines))]
    return OutputEntry("running", "ToolScript", detail, ApprovalView("script · running", code, "python", rows), live=True)


def tool_output_viewer(loop: CommandLoop) -> None:
    """Browse what recent calls produced without copying it into scrollback.

    Every entry -- a Bash command with its output, a ToolScript with its script and result, a
    Delegate order with the worker's answer -- opens the same read-only scrolling viewer.
    ToolScript is here because yolo has no other door to it; Bash is here because a bounded excerpt
    under the list was never the whole answer; Delegate is here because judging the answer means
    reading the order again, and the transcript kept only the `Delegate send` line.

    A detail's Esc (or q, or Ctrl-C) returns to the list with the cursor where it was; Ctrl-O
    closes the whole browser."""
    if loop.tui is None:
        return
    # Built once, on the way in: a record with nothing to show is also a record with no view, so
    # the same call decides whether to list it and what to open. Kept alongside its record rather
    # than rebuilt on selection -- the bounding work is proportional to the stored result, and
    # doing it twice for a multi-megabyte one would be paid on a keypress.
    entries: list[OutputEntry] = []
    if (running := running_script_entry(loop)) is not None:
        entries.append(running)
    for record in reversed(loop.session.tool_records):
        view = record_view(loop, record)
        if view is not None:
            status = ""
            if record.name == "Bash":
                code = tooloutput.bash_exit_code(record.output)
                status = "ok" if code == "0" else ("fail" if code else "")
            entries.append(
                OutputEntry(record.key, record.name, tooloutput.short_call(loop.session, ToolCall("", record.name, record.args)), view, status=status)
            )
    if not entries:
        return
    # One list state for the whole browser: reopening the list after a detail's Esc keeps the
    # cursor where it was instead of restarting at the top.
    state: ChoiceViewState | None = None
    while True:
        picked, state = _tool_output_list(loop, entries, state)
        if picked is None:
            return
        # Esc, q, or Ctrl-C in a detail goes back to the list; Ctrl-O closes the whole browser.
        if approval_text_viewer(loop, picked, back_on_escape=True) is not _TOOL_OUTPUT_BACK:
            return


def record_view(loop: CommandLoop, record: ToolResultRecord) -> ApprovalView | None:
    """The viewable form of a stored result, or None for a record this browser does not show."""
    if record.name == "Bash":
        return bash_view(loop, record)
    if record.name == "ToolScript":
        return script_view(loop, record)
    if record.name == "Delegate":
        return delegate_view(loop, record)
    if record.name == "Job":
        return job_view(loop, record)
    return None


def delegate_view(loop: CommandLoop, record: ToolResultRecord) -> ApprovalView | None:
    """The stored Delegate call as its order plus what the worker sent back.

    An order is the one text in a session written to be read twice: once at the send prompt, and
    again when the worker's answer has to be judged against what was actually asked. The transcript
    keeps neither -- just the `Delegate send` line -- so this is the second reading. Only a send has
    an order; status and reset return None from `approval_view` and are skipped like any other
    record this browser does not show."""
    view = DelegateTool(loop.session, record.args).approval_view()
    if view is None:
        return None
    result, note = tooloutput.viewer_text(record.output)
    rows = [("key", record.key), *view.rows]
    if note:
        rows.append(("shown", note))
    return ApprovalView(f"order · {record.key}", view.text, view.lexer, rows, result)


def job_view(loop: CommandLoop, record: ToolResultRecord) -> ApprovalView | None:
    """The stored Job call as the job's full log, when the job is still around; its return
    value otherwise.

    A Job start/status/wait writes the process output to a session log (a temp file for
    ``Job start``, an in-memory tail for a Bash call promoted to a job), and the record keeps
    only the tool's short return value. The browser shows the real log while the job still
    exists in ``session.jobs``; after a resume that table is gone, and a ``kill`` deletes the
    log file, so those records fall back to the return value like any other call."""

    payload = next((arg for arg in record.args if isinstance(arg, dict)), {})
    action = str(payload.get("action") or "")
    job_id = str(payload.get("job") or "").strip()
    if job_id and not job_id.startswith("job.") and job_id.isdigit():
        job_id = f"job.{job_id}"
    if not job_id and action == "start":
        match = re.search(r"\bjob\.\d+\b", record.output)
        if match is not None:
            job_id = match.group(0)
    job = loop.session.jobs.get(job_id) if job_id else None
    result, note = tooloutput.viewer_text(record.output)
    if job is None:
        rows = [("key", record.key)]
        if note:
            rows.append(("shown", note))
        return ApprovalView(f"job · {record.key}", result, "", rows)
    job.update_status()
    log = _job_log(job)
    bounded, log_note = tooloutput.viewer_text(log)
    rows = [("key", record.key), ("job", job.id), ("status", job.status)]
    if job.exit_code is not None:
        rows.append(("exit", str(job.exit_code)))
    if job.command:
        rows.append(("command", job.command))
    for extra in (log_note, note):
        if extra:
            rows.append(("shown", extra))
    return ApprovalView(f"job · {record.key}", bounded, "bash", rows, result)


def _job_log(job: BackgroundJob) -> str:
    """The job's merged stdout+stderr in full: the memory tail for a promoted Bash call, the
    log file for one started via ``Job(start)``."""

    if job.stream_buffer is not None:
        with job.stream_lock or contextlib.nullcontext():
            return "".join(job.stream_buffer)
    try:
        with open(job.log_path, "rb") as file:
            return file.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _tool_output_list(loop: CommandLoop, entries: list[OutputEntry], state: ChoiceViewState | None = None) -> tuple[ApprovalView | None, ChoiceViewState]:
    """The list modal itself: pick one, and it closes returning the view to open.

    `state` keeps the cursor (and any `/` filter) across reopenings, so a detail's Esc lands back
    on the entry the reader came from instead of the top of the list. The same state comes back in
    the return so the caller can pass it to the next opening.

    Rows are coloured the way the transcript colours the same call -- dim key, green tool name,
    plain arguments -- so a row is scannable by shape instead of read word by word. The first
    column is the Bash verdict where one exists: a green ✓ for exit 0, a red ✗ for any other
    exit, and a blank cell for entries that have no exit code (a script, an order, a running
    batch). The label is still the flat text, which is what `/` searches over."""
    assert loop.tui is not None
    width = max(20, shutil.get_terminal_size((120, 20)).columns - 12)
    parts: dict[str, StyleAndTextTuples] = {}
    labels: dict[str, str] = {}
    # Two cells wide so the verdict column lines up whether or not a row has one.
    status_marks = {
        "ok": ("class:choice.output.ok", "✓ "),
        "fail": ("class:choice.output.fail", "✗ "),
        "": ("", "  "),
    }
    for index, entry in enumerate(entries):
        mark = status_marks[entry.status]
        head = f"{entry.key}  "
        # Folded to one line before it is measured. `short_call` keeps a multi-line command whole,
        # which is right in the transcript and wrong here: a row is one row, and an embedded newline
        # spills it over several, taking the numbering and the selection bar with it. `git commit -m`
        # with a real message is the everyday case. The full command is a keypress away in the viewer.
        detail = oneline(entry.detail.removeprefix(entry.name).strip(), 400)
        detail = Text.clip_width(detail, max(8, width - get_cwidth(head + entry.name) - 1 - 2))
        labels[str(index)] = f"{head}{entry.name} {detail}".rstrip()
        parts[str(index)] = [
            mark,
            ("class:choice.live" if entry.live else "class:choice.meta", head),
            ("class:choice.tool", entry.name + " "),
            ("", detail),
        ]
    # Leave room for the rule, the help row, the counter, and the input region below.
    height = shutil.get_terminal_size((120, 24)).lines
    state = state or ChoiceViewState(tuple(labels), labels, set(), max_rows=max(5, min(20, height - 10)))

    def rule(label: str) -> StyleAndTextTuples:
        cols = shutil.get_terminal_size((80, 20)).columns
        rule_width = max(20, min(72, cols - 2))
        lead = "──── "
        trail = " " + "─" * max(3, rule_width - get_cwidth(lead + label) - 1)
        return [("", "\n"), ("class:choice.disabled", lead + label + trail + "\n")]

    def fragments() -> StyleAndTextTuples:
        list_fragments = state.fragments("", label_fn=lambda choice: parts.get(choice, []))
        return [*rule(f"Tool output · latest {len(entries)}"), *list_fragments[1:]]

    def handle_key(key: str, data: str) -> Any:
        if key in {"c-o", "q"}:
            return None
        result = state.handle_key(key, data)
        if result is SELECTION_BACK:
            return None
        return entries[int(result)].view if isinstance(result, str) else TUI_MODAL_PENDING

    picked = loop.tui.show_modal(fragments, handle_key)
    return (picked if isinstance(picked, ApprovalView) else None), state


def code_rows(text: str, lexer: str, width: int, margin: str = "  ") -> list[StyleAndTextTuples]:
    """Source text as display rows: a dim line-number gutter, then the line highlighted by the same
    whole-block lexer the transcript uses, so the viewer and the approval block agree on colors.

    Numbering matters more here than anywhere else: a failed script reports its traceback as
    `File "<toolscript>", line N`, and this is where the reader goes to find line N."""
    lines = text.splitlines() or [""]
    highlighted = UiPrinter.code_lines(text, lexer)
    number_width = len(str(len(lines)))
    rows: list[StyleAndTextTuples] = []
    for number, line in enumerate(lines, 1):
        rendered = highlighted[number - 1] if highlighted is not None and number - 1 <= len(highlighted) - 1 else [("fg:default", line)]
        prefix: list[tuple[str, str]] = [("", margin), ("ansibrightblack", f"{number:>{number_width}}  ")]
        continuation: list[tuple[str, str]] = [("", margin + " " * (number_width + 2))]
        rows.extend(cast(list[StyleAndTextTuples], Text.wrap_styled(prefix, continuation, rendered, width)))
    return rows


def approval_text_viewer(loop: CommandLoop, view: ApprovalView, *, back_on_escape: bool = False) -> object:
    """Read-only viewer for the text behind a confirmation: header rows plus the complete text,
    rendered as highlighted code when the view names a lexer and as markdown when it does not (an
    order is prose; a script is not). Esc/q closes back to the approval prompt; nothing here edits
    anything. The same viewer is what the Ctrl-O browser opens after the fact, which is how a
    script is read under yolo, where no prompt ever stops to offer `v`.

    With `back_on_escape`, Esc, q, or Ctrl-C returns `_TOOL_OUTPUT_BACK` instead of closing, so
    the caller can reopen the list; Ctrl-O still closes."""
    if loop.tui is None:
        return
    margin = "  "
    wrapped: dict[int, list[StyleAndTextTuples]] = {}
    header_rows = view.rows

    def markdown_rows(text: str, width: int) -> list[StyleAndTextTuples]:
        """Render `text` as markdown through the same Rich capture pipeline the scrollback
        renderer uses, then split the styled ANSI into display rows. The console width is the
        modal content width (terminal width minus the two-space margins); Rich measures wide
        characters itself, so CJK orders wrap at the real right edge.

        Every source line gets a hard line break first: an order's newlines are structural (file
        lists, steps, plain-text instructions), and Markdown otherwise folds in-paragraph newlines
        to spaces, running them together into one block the approver has to re-read."""
        hard_breaks = "\n".join(line.rstrip() + "  " for line in text.split("\n"))
        content_width = max(1, width - 4)
        console = Console(force_terminal=True, color_system="truecolor", no_color=False, width=content_width)
        with console.capture() as capture:
            console.print(Markdown(hard_breaks, hyperlinks=False))
        cleaned = UiPrinter.strip_unknown_escapes(UiPrinter.strip_trailing_pad(capture.get()))
        rows: list[StyleAndTextTuples] = [[]]
        for style, fragment in cast(list[tuple[str, str]], list(to_formatted_text(ANSI(cleaned)))):
            for index, part in enumerate(fragment.split("\n")):
                if index:
                    rows.append([])
                if part:
                    rows[-1].append((style, part))
        return [[("", margin), *row] for row in rows]

    def separator(width: int, label: str = "") -> StyleAndTextTuples:
        """The rule between the viewer's sections, optionally naming the one it opens. Labeled or
        not, it runs to the same right edge, so the sections read as one document."""
        if not label:
            return [("", margin), ("ansibrightblack", "─" * max(0, width - 4))]
        lead = f"── {label} "
        return [("", margin), ("ansibrightblack", lead + "─" * max(0, width - 4 - get_cwidth(lead)))]

    def layout(width: int) -> list[StyleAndTextTuples]:
        """Field header rows, a separator, the whole text, and -- when the call has already run --
        what it returned below a second rule. Cached per width: the wrap has to be redone when the
        terminal is resized, but not on every keypress."""
        if width in wrapped:
            return wrapped[width]
        lines: list[StyleAndTextTuples] = []
        label_width = max((get_cwidth(label) for label, _ in header_rows), default=0)
        for label, value in header_rows:
            padded = label + " " * max(0, label_width - get_cwidth(label))
            lines.extend(
                cast(
                    list[StyleAndTextTuples],
                    Text.wrap_styled(
                        [("", margin), ("ansicyan", padded), ("", "  ")],
                        [("", margin + " " * (label_width + 2))],
                        [("fg:default", value)],
                        width,
                    ),
                )
            )
        # A blank line sets the fields apart from the body below the rule, and the body from
        # the result: three sections instead of one wall of text.
        lines.append([])
        lines.append(separator(width))
        lines.extend(code_rows(view.text, view.lexer, width, margin) if view.lexer else markdown_rows(view.text, width))
        if view.result.strip():
            # Plain, unlexed, and whole: this is the result exactly as the model received it, and a
            # viewer opened to check what a script did may not quietly edit or clip it. Default
            # foreground, not dimmed, so the answer reads as plainly as the text above it.
            lines.extend([[], separator(width, "result")])
            lines.extend(wrapped_rows(view.result.rstrip(), width, margin))
        wrapped[width] = lines
        return lines

    scroll = 0

    def size() -> tuple[int, int]:
        columns, rows = shutil.get_terminal_size((120, 24))
        return max(20, columns), max(3, rows - 6)

    def viewport() -> int:
        return size()[1]

    def fragments() -> StyleAndTextTuples:
        nonlocal scroll
        width, height = size()
        lines = layout(width)
        scroll = min(scroll, max(0, len(lines) - height))
        # The full legend needs ~78 cells; drop to the key names alone rather than let it spill past
        # the right edge on a narrow terminal, where the modal window would just cut it off.
        legend = "  ↑/↓ scroll · Ctrl-U/D half-page · PgUp/Dn page · g/G top/bottom · Esc/q close"
        if back_on_escape:
            legend = "  ↑/↓ scroll · Ctrl-U/D half-page · PgUp/Dn page · g/G top/bottom · Esc/q back · c-o close"
        if get_cwidth(legend) > width:
            legend = "  ↑/↓ · Ctrl-U/D · g/G · Esc/q back · c-o close" if back_on_escape else "  ↑/↓ · Ctrl-U/D · g/G · Esc/q close"
        parts: StyleAndTextTuples = [("class:choice.disabled", f"  {view.label[:1].upper() + view.label[1:]} · read-only\n")]
        for line in lines[scroll : scroll + height]:
            parts.extend(line)
            parts.append(("", "\n"))
        parts.append(("class:choice.disabled", Text.clip_width(legend, width) + "\n"))
        return parts

    def handle_key(key: str, data: str) -> Any:
        nonlocal scroll
        if back_on_escape and key in {"q", "escape", "c-c"}:
            return _TOOL_OUTPUT_BACK
        if key in {"q", "c-o", "escape"}:
            return None
        height = viewport()
        if key in {"down", "j"}:
            scroll += 1
        elif key in {"up", "k"}:
            scroll -= 1
        elif key in {"pagedown", "c-d"}:
            scroll += height if key == "pagedown" else height // 2
        elif key in {"pageup", "c-u"}:
            scroll -= height if key == "pageup" else height // 2
        elif key in {"g", "G"}:
            scroll = 0 if key == "g" else 10**9
        scroll = max(0, scroll)
        return TUI_MODAL_PENDING

    return loop.tui.show_modal(fragments, handle_key, exclusive=True)


def diff_viewer(loop: CommandLoop) -> None:
    """Interactive diff viewer. First shows a file list; open a file to see its diff.

    List mode: ↑/↓ or j/k move, h/l or ←/→ switches tabs, Enter opens the selected file,
    r refreshes, q/Esc closes.
    Diff mode: ↑/↓ scroll one line, Ctrl-U/Ctrl-D half a page, PgUp/PgDn a page,
    Esc/← returns to list, r refreshes, q closes.
    """
    state = DiffViewState(TabbedViewState(("Latest", "Session")))

    def build_model() -> list[list[tuple[str, str, str]]]:
        latest = loop.agent.session.latest_round_diff_sections()
        return [latest[1] if latest is not None else [], loop.agent.session.session_diff_sections()]

    model = build_model()

    def viewport() -> int:
        return max(3, shutil.get_terminal_size().lines - 7)

    def active_sections() -> list[tuple[str, str, str]]:
        return model[state.view.tab]

    def list_fragments(parts: StyleAndTextTuples, sections: list[tuple[str, str, str]]) -> None:
        parts.append(("", "\n"))
        counts = [loop.diff_counts(diff) for _, _, diff in sections]
        added_width = max(len(str(added)) for added, _ in counts)
        removed_width = max(len(str(removed)) for _, removed in counts)
        for index, ((_, path, _), (added, removed)) in enumerate(zip(sections, counts)):
            selected = index == state.file
            marker = "> " if selected else "  "
            style = "ansicyan" if selected else "class:choice.disabled"
            parts.extend(
                [
                    (style, marker),
                    ("ansigreen", f"+{added:>{added_width}}"),
                    ("", " "),
                    ("ansired", f"-{removed:>{removed_width}}"),
                    (style, f" {path}\n"),
                ]
            )
        parts.append(("", "\n"))

    def file_fragments(parts: StyleAndTextTuples, sections: list[tuple[str, str, str]]) -> None:
        state.clamp_file(len(sections))
        status, path, diff = sections[state.file]
        parts.append(("", "\n"))
        parts.append(("ansicyan", f"  {status.title()} · {path}\n"))
        lines = loop.ui.segment_lines(loop.ui.diff_segments_live(diff))
        visible = state.view.visible(lines, viewport())
        for line in visible:
            parts.extend(line)
        if not visible or not visible[-1] or not visible[-1][-1][1].endswith("\n"):
            parts.append(("", "\n"))

    def fragments() -> StyleAndTextTuples:
        parts: StyleAndTextTuples = [("", "\n")]
        parts.extend(loop.ui.tab_segments(state.view.titles, state.view.tab))
        parts.append(("", "\n"))

        sections = active_sections()
        if not sections:
            parts.append(("class:choice.disabled", "  No diffs\n"))
        elif state.mode is DiffViewState.Mode.LIST:
            list_fragments(parts, sections)
        else:
            file_fragments(parts, sections)
        mode_hint = "list" if state.mode is DiffViewState.Mode.LIST else "diff"
        if state.mode is DiffViewState.Mode.LIST:
            hint = "↑/↓ or j/k move · ←/→ or h/l tab · Enter open · r refresh · Esc/q close"
        else:
            hint = "↑/↓ scroll · Ctrl-U/D half-page · PgUp/PgDn page · Esc/← back · r refresh · q close"
        position = f"{state.file + 1 if sections else 0}/{len(sections)}"
        parts.append(("class:choice.disabled", f"\n  [{mode_hint}] {hint} [{position}]\n"))
        return parts

    if loop.tui is None:
        return

    def modal_key(key: str, _data: str) -> Any:
        nonlocal model
        result = state.handle_key(key, len(active_sections()), viewport())
        if result is DiffViewState.REFRESH:
            model = build_model()
            return TUI_MODAL_PENDING
        return result

    loop.tui.show_modal(fragments, modal_key, exclusive=True)


def compaction_log_viewer(loop: CommandLoop) -> None:
    """Read-only viewer for `/compact log`: the stored compaction segments newest first, and the
    summary plus verbatim excerpt of the one opened with Enter. This is the user's half of what
    `RecallContext` gives the model — same segments, nothing here writes.

    List mode: ↑/↓ or j/k move, Enter/→ opens, g/G first/last, Esc/q closes.
    Detail mode: ↑/↓ scroll one line, Ctrl-U/D half a page, PgUp/PgDn a page, Esc/← back, q closes.
    """
    if loop.tui is None:
        return
    segments = list(reversed(loop.session.history))  # newest first, like RecallContext(list)
    state = SegmentLogViewState()
    detail: dict[tuple[str, int], list[StyleAndTextTuples]] = {}

    def size() -> tuple[int, int]:
        columns, rows = shutil.get_terminal_size((120, 24))
        return max(20, columns), max(3, rows - 7)

    def header() -> StyleAndTextTuples:
        """Segments are what compaction stored, `compaction_count` is what it ran: a pass with
        nothing evictable stores none, so report both rather than implying they always match."""
        count = loop.session.state.compaction_count
        stored = f"{len(segments)} stored segment{'' if len(segments) == 1 else 's'}"
        return [("class:choice.disabled", f"  Compaction log · {count} compaction{'' if count == 1 else 's'} · {stored}\n\n")]

    def list_rows(width: int) -> list[StyleAndTextTuples]:
        columns = [segment_columns(segment) for segment in segments]
        key_width = max((get_cwidth(segment.key) for segment in segments), default=0)
        when_width = max((get_cwidth(when) for when, _, _ in columns), default=0)
        kind_width = max((get_cwidth(kind) for _, kind, _ in columns), default=0)
        messages_width = max((get_cwidth(messages) for _, _, messages in columns), default=0)
        rows: list[StyleAndTextTuples] = []
        for index, (segment, (when, kind, messages)) in enumerate(zip(segments, columns)):
            selected = index == state.selected
            style = "ansicyan" if selected else "class:choice.disabled"
            lead = f"{'> ' if selected else '  '}{segment.key:<{key_width}}  {when:<{when_width}}  {kind:<{kind_width}}  "
            lead += f"{messages:>{messages_width}}  "
            title = Text.clip_width(segment.title, max(8, width - get_cwidth(lead)))
            rows.append([(style, lead), ("fg:default" if selected else "class:choice.disabled", title)])
        return rows

    def detail_rows(segment: HistorySegment, width: int) -> list[StyleAndTextTuples]:
        """What the compaction was, then what it kept. The stored excerpt stays in the segment for
        the model's RecallContext, but it is the raw conversation the summary already stands for —
        showing it here buried the one thing worth reading."""
        when, _, _ = segment_columns(segment)
        headline, caveat = segment_story(segment)
        rows: list[StyleAndTextTuples] = [
            [("ansicyan", f"  {segment.key}"), ("class:choice.disabled", f"  {when}")],
            *wrapped_rows(segment.title, width),
            [],
            [("", "  "), ("class:choice.disabled", Text.clip_width(headline, width - 2))],
        ]
        if caveat:
            rows.extend(wrapped_rows(caveat, width))
        rows.append([])
        rows.extend(wrapped_rows(segment.summary or missing_summary_note(segment), width))
        return rows

    def body(width: int, height: int) -> list[StyleAndTextTuples]:
        if state.mode is SegmentLogViewState.Mode.LIST:
            return list_rows(width)
        segment = segments[state.selected]
        cached = detail.get((segment.key, width))
        if cached is None:
            cached = detail_rows(segment, width)
            detail[(segment.key, width)] = cached
        return state.visible(cached, height)

    def fragments() -> StyleAndTextTuples:
        width, height = size()
        parts: StyleAndTextTuples = [("", "\n")]
        parts.extend(header())
        if not segments:
            parts.append(("class:choice.disabled", "  No compaction has stored a segment yet\n"))
        else:
            state.selected = state.selected % len(segments)
            for row in body(width, height):
                parts.extend(row)
                parts.append(("", "\n"))
        if state.mode is SegmentLogViewState.Mode.LIST:
            hint = "  [list] ↑/↓ or j/k move · Enter open · g/G first/last · Esc/q close"
        else:
            hint = "  [detail] ↑/↓ scroll · Ctrl-U/D half-page · PgUp/PgDn page · Esc/← back · q close"
        position = f" [{state.selected + 1 if segments else 0}/{len(segments)}]"
        parts.append(("class:choice.disabled", "\n" + Text.clip_width(hint + position, width) + "\n"))
        return parts

    def modal_key(key: str, _data: str) -> Any:
        return state.handle_key(key, len(segments), size()[1])

    loop.tui.show_modal(fragments, modal_key, exclusive=True)
