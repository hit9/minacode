"""Log-block assembly for tool calls: what an approval brief, a rejection, and a finished call
look like as a LogBlock tree, independent of execution state.

The sibling of `tooloutput`, which bounds and parses tool *result text*; this module builds the
*tree structure* the renderer paints. Neither reaches into ToolRunner: everything here is derived
from the call, the tool, and the session's configuration, so a resumed transcript and a live turn
can render the same call the same way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from prompt_toolkit.utils import get_cwidth

from minacode.base import ApprovalView, LogBlock, LogEdge, LogLine, LogRole, ToolCall, oneline
from minacode.session import Session
from minacode.tools import tooloutput
from minacode.tools.base import Tool
from minacode.tools.delegate import DelegateTool
from minacode.tools.editplan import EditBatchPlan
from minacode.tools.files import EditTool

# How much of an approval view's text the block itself shows. The rest is one keypress away in
# the viewer (`v`, or Ctrl-O afterwards), and a script long enough to overflow this is exactly
# the one whose full body in the transcript would bury everything it then goes on to do.
VIEW_EXCERPT_LINES = 10
# Slack before clipping is worth it: the `… +N more lines` line costs a row of its own, so
# hiding one or two lines buys nothing and merely sends the reader to the viewer for them.
VIEW_EXCERPT_SLACK = 2
# The typed protocol, spelled out for runs with no action row to show it: headless, piped stdin.
# Keyed by the action's answer so the legend can be built from the offered actions and cannot
# advertise one the call has no use for; ordered here rather than by the row, which leads with
# what is reached most often while a legend reads best with the two answers first.
APPROVAL_LEGEND_SEGMENTS: tuple[tuple[str, str], ...] = (
    ("", "Y/Enter approve"),
    ("n", "n refuse"),
    ("c", "c worker config"),
    ("v", "v view {label}"),
)


@dataclass
class ToolDisplay:
    """How one tool call renders: the batch-counter suffix, the short call line, whether it prints
    as a nested tree, and whether it was auto/user approved. Threaded from run_one into finish/reject."""

    batch_suffix: str = ""
    display: str | None = None
    nested_display: bool = False
    approved: bool = False
    auto: bool = False
    # Non-empty when a ViewImage call bridged to the [vision] entry; finish_display draws it as a
    # child of the call line so the bridge trace can never precede the call's own root.
    vision_entry: str = ""


def log_root(display: str, role: LogRole = LogRole.TOOL, batch_suffix: str = "", call: ToolCall | None = None) -> LogLine:
    from minacode.tools import TOOL_REGISTRY  # local import: the registry is built on top of every tool

    name, _, args = display.partition(" ")
    tool_class = TOOL_REGISTRY.get(name)
    syntax = ""
    if tool_class is not None:
        syntax = tool_class.log_lexer(call.args) if call is not None else tool_class.LOG_LEXER
    if role is LogRole.MUTED:
        syntax = ""
    # The batch counter goes into `meta` (rendered gray) instead of `args` (syntax-highlighted),
    # so it reads as a subdued tag on the same line rather than another highlighted token.
    meta = ("  " + batch_suffix) if batch_suffix else ""
    return LogLine(name, args, role, meta=meta, syntax=syntax)


def field_pairs(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Pad each label to the widest label in the block, CJK-safe (visible width via
    get_cwidth), so the values start on one column."""
    width = max(get_cwidth(label) for label, _ in rows) if rows else 0
    return [(label + " " * max(0, width - get_cwidth(label)), value) for label, value in rows]


def approval_actions(tool: Tool, always_option: bool) -> list[tuple[str, str]]:
    """What this prompt can do, as (label, answer) pairs with the default first.

    The one answer to the question, so the action row and the typed legend cannot disagree about
    it. Ordered by how often they are wanted, because reaching the nth costs n-1 Tabs. Refusing
    sits last despite being common: Escape already refuses in one key. Every action's answer is
    the whole line the user could have typed, so the typed protocol underneath is untouched and
    a headless run loses the row, not the actions."""
    actions = [("Approve", "")]
    view = tool.approval_view()
    if view is not None:
        actions.append((f"View {view.label}", "v"))  # a tool with nothing to view returns None, so it is never offered
    if always_option:
        actions.append(("Worker config", "c"))
    actions.append(("Refuse", "n"))
    return actions


def approval_prompt(always_option: bool, form: list[tuple[str, str]]) -> str:
    """The one-line prompt after the brief. The form renders its own labels above the input row,
    so it only needs the field name; without one the prompt carries the typed protocol, which is
    all a headless run has."""
    if form:
        return "reason › "
    return "Approve delegation? [Y/n/c] " if always_option else "Approve? [Y/n or reason] "


def approval_legend(actions: list[tuple[str, str]], view_label: str = "") -> str:
    offered = {answer for _, answer in actions}
    segments = [text.format(label=view_label) for answer, text in APPROVAL_LEGEND_SEGMENTS if answer in offered]
    return " · ".join(segments) + " · else reason"


def view_excerpt_children(view: ApprovalView, status: str, form: list[tuple[str, str]], actions: list[tuple[str, str]]) -> list[LogLine]:
    """The opening lines of an approval view, syntax-highlighted, under a header naming what is
    clipped. CODE-role lines are lexed as one block by the renderer, so a construct spanning
    lines (a triple-quoted string) still highlights correctly inside the excerpt."""
    lines = view.text.rstrip().splitlines()
    if not lines:
        return []
    kept = len(lines) if len(lines) <= VIEW_EXCERPT_LINES + VIEW_EXCERPT_SLACK else VIEW_EXCERPT_LINES
    shown, hidden = lines[:kept], len(lines) - kept
    children = [
        LogLine(view.label, "", LogRole.META, LogEdge.BRANCH),
        *(LogLine("", line, LogRole.CODE, LogEdge.CONTINUE, syntax=view.lexer) for line in shown),
    ]
    # The tail line is where the rest of the text is advertised, so it is also the legend's home
    # when there is no action row to carry the keys (headless, piped stdin). Under yolo nothing
    # stops to ask, so it points at the one door that is still open afterwards: the Ctrl-O
    # browser, which is the only way to read a script that was never confirmed.
    tail = f"… +{hidden} more line{'' if hidden == 1 else 's'}" if hidden else ""
    if status != "confirm":
        tail = (tail + " · " if tail else "") + "Ctrl-O for more"
    elif not form:
        tail = (tail + " · " if tail else "") + approval_legend(actions, view.label)
    if tail:
        # CONTINUE, not END: the call is about to run, and what it does next -- the prompt, the
        # calls the script makes, the result line -- hangs off this same gutter. Closing the
        # tree here and reopening it below would break the bracket in the middle.
        children.append(LogLine("", tail, LogRole.META, LogEdge.CONTINUE))
    return children


def delegate_approval_children(tool: DelegateTool, form: list[tuple[str, str]] | None = None, actions: list[tuple[str, str]] | None = None) -> list[LogLine]:
    """Approval brief for a Delegate send: title, a one-line order excerpt, explicit send
    parameters, and the worker configuration the send will run under. FIELD-role rows render
    cyan left-aligned labels (padded to one column, CJK-safe) with default-foreground values;
    everything is derived from the call and the session config, never from mutable worker state.

    The key legend closes the brief only when there is no action row: with one, the same
    choices sit live above the input line, and a copy frozen into the transcript would go stale
    the moment the worker config is edited."""
    order = tool.payload_dict().get("order")
    order_row = None
    if isinstance(order, str) and order.strip():
        lines = order.strip().splitlines()
        text = oneline(lines[0].strip(), 100)
        if len(lines) > 1:
            text += f"  (… {len(lines) - 1} more lines)"
        order_row = ("order", text)
    rows: list[tuple[str, str, LogRole]] = [(label, value, LogRole.FIELD) for label, value in field_pairs(tool.header_rows(order_row))]
    if not form:
        rows.append(("", approval_legend(actions if actions is not None else approval_actions(tool, True), "order"), LogRole.META))
    last = len(rows) - 1
    return [
        LogLine(label, value, role, LogEdge.END if index == last else LogEdge.BRANCH if index == 0 else LogEdge.CONTINUE)
        for index, (label, value, role) in enumerate(rows)
    ]


def approval_display(
    session: Session,
    call: ToolCall,
    tool: Tool,
    status: str,
    batch_suffix: str = "",
    planned_edit: EditBatchPlan.PlannedEdit | None = None,
    form: list[tuple[str, str]] | None = None,
    actions: list[tuple[str, str]] | None = None,
) -> LogBlock:
    role = LogRole.TOOL if status == "confirm" else LogRole.AUTO
    root = log_root(tooloutput.short_call(session, call), role, batch_suffix, call)
    children = []
    if isinstance(tool, DelegateTool) and tool.always_confirms():
        children.extend(delegate_approval_children(tool, form or [], actions))
    elif tool.NAME == "Edit":
        preview = planned_edit.preview(tool) if planned_edit and isinstance(tool, EditTool) else tool.preview()
        preview_lines = preview.rstrip().splitlines()
        if preview_lines:
            children.append(LogLine("preview", role=LogRole.META, edge=LogEdge.BRANCH))
            children.extend(LogLine("", line, LogRole.DIFF, LogEdge.CONTINUE) for line in preview_lines)
    elif (view := tool.approval_view()) is not None:
        children.extend(view_excerpt_children(view, status, form or [], actions or approval_actions(tool, False)))
    return LogBlock.hierarchy(root, children)


def reject_display(session: Session, call: ToolCall, output: str, *, d: ToolDisplay) -> LogBlock:
    # Argument/usage rejections are usually self-corrected on retry, so show a quiet one-liner
    # (rendered dim by UiPrinter) instead of the full red failed block. The model still receives
    # the complete error so it can correct the call.
    #
    # One line has to be enforced here, not assumed: a display is whatever the tool's short_args
    # produced, and Note's is the whole rendered note so that a successful call can print it.
    # Left alone, a rejected Note dims its entire body and hides the reason at the end of the
    # last line -- the reason for the rejection is the only part of a rejection worth reading.
    reason = oneline(output.removeprefix("ToolError:").strip(), 60)
    display = oneline(d.display or tooloutput.short_call(session, call), 120)
    return LogBlock.hierarchy(log_root(display + " · rejected: " + reason, LogRole.MUTED, d.batch_suffix, call), [])


def finish_display(
    session: Session,
    call: ToolCall,
    key: str,
    output: str,
    *,
    failed: bool,
    elapsed: float | None = None,
    d: ToolDisplay | None = None,
    worker_rule: Callable[[str], None] | None = None,
) -> str | LogBlock:
    """The block a finished call prints.

    `worker_rule` is the one output sink this reaches for rather than returns: a finished Delegate
    send closes its bracket with a full-width rule that has to be drawn as a sibling of the block,
    not inside it. Without one wired (headless, or a runner outside CommandLoop) the same detail
    falls back into the block's own child lines."""
    d = d or ToolDisplay()
    if call.name == "Note" and not failed and d.display:
        return tooloutput.with_batch_suffix(d.display.removeprefix("Note ").strip(), d.batch_suffix)
    tag = " [refused]" if failed and "user refused" in output else " [failed]" if failed else " [approved]" if d.approved else " [auto]" if d.auto else ""
    tree = d.nested_display or call.name in ("Bash", "Delegate") or bool(d.vision_entry)
    # A failed call explains itself in the error child below, so its root only has to identify
    # the call -- collapsed to one line, or a multi-line display (Note keeps the whole rendered
    # note there) paints its entire body red under the tag.
    label = d.display or tooloutput.short_call(session, call)
    root = log_root(oneline(label, 120) if failed else label, LogRole.ERROR if failed else LogRole.TOOL, d.batch_suffix, call)
    is_reset = call.name == "Delegate" and not failed and 'action="reset"' in output
    if call.name == "Delegate" and not failed and not is_reset:
        # The delegation bracket: the start marker opens with the yellow full-width rule; the
        # finish closes it with the sibling rule carrying the done summary. Without a wired
        # worker_rule the finish block falls back to the "[worker] ◀" root line with the detail
        # in the child lines below. Reset is a one-shot tool call, not a bracket: it keeps its
        # ordinary tool root and does not print a full-width rule.
        if worker_rule is not None:
            root = None
        else:
            root = log_root("[worker] ◀", LogRole.WORKER, d.batch_suffix, call)
    children = []
    if failed:
        label = "refused" if "user refused" in output else "error"
        children.append(LogLine(label, oneline(output, 220), LogRole.ERROR, LogEdge.END))
    elif call.name == "MCP":
        summary = tooloutput.mcp_result_summary(call, output, elapsed)
        if summary:
            children.append(LogLine("", summary, LogRole.META, LogEdge.END))
    elif call.name == "Bash":
        preview = tooloutput.bash_result_preview(output, tooloutput.BASH_TRANSCRIPT_PREVIEW_LINES)
        if preview:
            duration = f" · {elapsed:.1f}s" if elapsed is not None else ""
            children.append(LogLine("output" + duration, "Ctrl-O for more", LogRole.META, LogEdge.BRANCH))
            children.extend(LogLine("", line, LogRole.OUTPUT, LogEdge.CONTINUE) for line in preview.splitlines())
    elif call.name == "ToolScript":
        # Closes the bracket the nested calls were indented under: how many of them there were,
        # how long the script took, and the first lines of what it printed -- the printed output
        # being the whole point of a script, since only that comes back to the model. The script
        # body itself stays one keypress away rather than repeated here.
        fields = tooloutput.toolscript_result_fields(output)
        if fields is not None:  # a describe returns tool shapes, not a script envelope
            counted, stdout, error = fields
            duration = f" · {elapsed:.1f}s" if elapsed is not None else ""
            # A script that raised returns its envelope normally, so the call itself did not
            # fail and nothing above this line says otherwise. Said here, or a script that died
            # on call 2 of 40 reads exactly like one that finished -- and the error, unlike the
            # printed output, is the part the reader needs.
            head = ("failed · " if error else "") + f"calls {counted}" + duration
            children.append(LogLine(head, "Ctrl-O for more", LogRole.ERROR if error else LogRole.META, LogEdge.BRANCH))
            body = error or stdout
            children.extend(
                LogLine("", line, LogRole.ERROR if error else LogRole.OUTPUT, LogEdge.CONTINUE)
                for line in tooloutput.preview_lines(body, tooloutput.BASH_TRANSCRIPT_PREVIEW_LINES)
            )
    elif call.name == "Ask":
        children.append(LogLine("answer", oneline(output, 220), LogRole.META, LogEdge.END))
    elif call.name == "Delegate":
        if 'action="reset"' in output:
            # Reset is a one-shot tool call, not a delegation bracket: it keeps its ordinary
            # tool root (above) and only adds a plain done child stating what it cleared and
            # what survives. No full-width `worker reset` rule runs.
            children.append(LogLine("done", "worker context cleared; file changes and merged diffs kept", LogRole.META, LogEdge.BRANCH))
        else:
            summary = tooloutput.delegate_result_summary(output)
            if summary:
                if worker_rule is not None:
                    fields = tooloutput.delegate_result_fields(output)
                    # `summary` only renders when the envelope parsed, so fields is never None
                    # here; the guard exists for the type checker.
                    if fields is not None:
                        title = ""
                        if call.args and isinstance(call.args[0], dict):
                            raw_title = call.args[0].get("title")
                            title = raw_title.strip() if isinstance(raw_title, str) else ""
                        parts = ([title] if title else []) + [f"steps {fields.steps}", fields.elapsed]
                        if fields.in_tokens:
                            parts.append(f"{fields.in_tokens} in / {fields.out_tokens} out")
                        if fields.files != "(none)":
                            parts.append(fields.files if len(fields.files) <= 48 else fields.files[:47].rstrip() + "…")
                        worker_rule("worker done · " + " · ".join(parts))
                else:
                    children.append(LogLine("done", summary, LogRole.META, LogEdge.BRANCH))
                preview = tooloutput.delegate_answer_preview(output)
                if preview:
                    children.extend(LogLine("", line, LogRole.OUTPUT, LogEdge.CONTINUE) for line in preview.splitlines())
    elif call.name == "ViewImage" and d.vision_entry:
        # The vision-provider observation is a child of the call line, drawn before the stored row, so
        # the trace can never appear above its own call (the attachment path's standalone
        # line is the engine's, not a tool's). TOOL, not sibling children's META: the stored
        # row is bookkeeping, this is a real request on another paid entry.
        children.append(LogLine("described by", d.vision_entry, LogRole.TOOL, LogEdge.BRANCH))
    if tree and not failed:
        children.append(LogLine("stored" if key else "done", key + tag if key else tag.strip(), LogRole.META, LogEdge.END))
    elif not tree and root is not None:
        # root can be None only on the Delegate worker_rule path, where tree is always True.
        tail = ((" → " + key) if key else "") + tag
        root = LogLine(root.label, root.text, root.role, meta=root.meta + tail, syntax=root.syntax)
    return LogBlock.hierarchy(None if d.nested_display else root, children)


def full_text_block(view: ApprovalView) -> LogBlock:
    """Headless fallback for the `v` action: header rows then the whole text, one line each, so
    nothing behind the confirmation is dropped. Code keeps its lexer here too -- the fallback is
    what a piped-stdin run reads instead of the viewer, not a lesser copy of it."""
    role = LogRole.CODE if view.lexer else LogRole.OUTPUT
    return LogBlock(
        [
            LogLine(view.label, "", LogRole.FIELD, LogEdge.BRANCH),
            *(LogLine(label, value, LogRole.FIELD, LogEdge.CONTINUE) for label, value in field_pairs(view.rows)),
            *(LogLine("", line, role, LogEdge.CONTINUE, syntax=view.lexer) for line in view.text.splitlines()),
        ]
    )


def worker_config_rows(session: Session) -> list[tuple[str, str]]:
    """The effective worker provider/model/effort/api as (label, value) rows. Owned by
    DelegateTool, which also builds the send brief's header rows from it, so the `c` cycle and
    the brief can never disagree about what the worker is configured as."""
    return DelegateTool.worker_config_rows(session.config)


def worker_config_block(session: Session) -> LogBlock:
    """The current effective worker config as a log block: one row per knob, inherited values
    marked `(inherit)`, matching the approval brief's four worker rows (same cyan aligned
    labels)."""
    return LogBlock(
        [
            LogLine("worker config", "", LogRole.FIELD, LogEdge.BRANCH),
            *(LogLine(label, value, LogRole.FIELD, LogEdge.CONTINUE) for label, value in field_pairs(worker_config_rows(session))),
        ]
    )
