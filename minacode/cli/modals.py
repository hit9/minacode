"""Modal / question UI flows as free functions taking the CommandLoop.

These render blocking prompt_toolkit UIs (choice lists, free-text questions, the diff viewer,
the bash output viewer, and the MCP manager). They return the selected value, or None when
dismissed. Free functions so the command handlers in commands.py and the runtime can call them
without a CommandLoop instance.
"""

from __future__ import annotations

import shutil
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples, to_formatted_text
from prompt_toolkit.utils import get_cwidth
from rich.console import Console
from rich.markdown import Markdown

from minacode.base import DISMISSED, SELECTION_BACK, Text, ToolCall
from minacode.render import UiPrinter
from minacode.tools import AskSpec
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


def wrapped_rows(text: str, width: int, margin: str = "  ") -> list[StyleAndTextTuples]:
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
                Text.wrap_styled([("", margin)], [("", margin + indent)], [("", raw)], width),
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
        loop.ui.emit_answer(mcp.render_server_status())
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
        name_width = max(len(name) for name, *_rest in server_rows)
        status_width = max(len(mcp.STATUS_MARKER + " disconnecting"), *(len(status) for _name, status, _mode, _count in server_rows))
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
) -> str | object | None:
    labels = labels or {}
    if not choices or not loop.interactive_input:
        return None
    enabled = tuple(choice for choice in choices if choice not in disabled)
    if len(enabled) == 1:
        return enabled[0]
    try:
        return choice_application(loop, title, choices, labels, current, set(disabled))
    except (EOFError, KeyboardInterrupt):
        loop.emit("Cancelled")
        return None


def choice_application(
    loop: CommandLoop,
    title: str,
    choices: tuple[str, ...],
    labels: dict[str, str],
    current: str,
    disabled: set[str],
    *,
    preview_fn: Callable[[str], str] | None = None,
) -> str | object | None:
    state = ChoiceViewState(choices, labels, disabled)
    options = state.enabled()
    state.selected = options.index(current) if current in options else 0
    if loop.tui is None:
        return None
    result = loop.tui.show_modal(lambda: state.fragments(title, preview_fn), state.handle_key)
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


def bash_output_viewer(loop: CommandLoop) -> None:
    """Browse recent completed Bash previews without copying them into scrollback."""
    if loop.tui is None:
        return
    records = []
    for record in reversed(loop.session.tool_records):
        if record.name != "Bash":
            continue
        preview = loop.agent.tools.bash_result_preview(record.output)
        if preview:
            records.append((record, preview))
        if len(records) == 10:
            break
    if not records:
        return
    width = max(20, shutil.get_terminal_size((120, 20)).columns - 12)
    labels = {}
    calls = {}
    for index, (record, _preview) in enumerate(records):
        call = loop.agent.tools.short_call(ToolCall("", "Bash", record.args))
        choice = str(index)
        calls[choice] = call
        labels[choice] = Text.clip_width(f"{record.key}  {call}", width)
    choices = tuple(labels)
    state = ChoiceViewState(choices, labels, set())
    opened: str | None = None

    def rule(label: str) -> StyleAndTextTuples:
        cols = shutil.get_terminal_size((80, 20)).columns
        rule_width = max(20, min(72, cols - 2))
        lead = "──── "
        trail = " " + "─" * max(3, rule_width - get_cwidth(lead + label) - 1)
        return [("", "\n"), ("class:choice.disabled", lead + label + trail + "\n")]

    def fragments() -> StyleAndTextTuples:
        if opened is None:
            list_fragments = state.fragments("")
            return [*rule(f"Bash outputs · latest {len(records)}"), *list_fragments[1:]]
        record, preview = records[int(opened)]
        detail_width = max(20, shutil.get_terminal_size((120, 20)).columns - 6)
        parts: StyleAndTextTuples = [*rule(f"Bash output · {record.key}"), ("ansibrightblack", f"  {Text.clip_width(calls[opened], detail_width)}\n\n")]
        parts.extend(("ansibrightblack", f"  {Text.clip_width(line, detail_width)}\n") for line in preview.splitlines())
        parts.append(("class:choice.disabled", "\n  Esc / ← back · Ctrl-O / q closes\n"))
        return parts

    def handle_key(key: str, data: str) -> Any:
        nonlocal opened
        if key in {"c-o", "q"}:
            return None
        if opened is not None:
            if key in {"escape", "left", "h"}:
                opened = None
            return TUI_MODAL_PENDING
        result = state.handle_key(key, data)
        if result is SELECTION_BACK:
            return None
        if isinstance(result, str):
            opened = result
        return TUI_MODAL_PENDING

    loop.tui.show_modal(fragments, handle_key)


def delegate_order_viewer(loop: CommandLoop, order: str, header_rows: list[tuple[str, str]]) -> None:
    """Read-only viewer for the Delegate `v` key: header rows plus the complete order text
    rendered as markdown. Esc/q closes back to the approval prompt; nothing here edits
    anything."""
    if loop.tui is None:
        return
    margin = "  "
    wrapped: dict[int, list[StyleAndTextTuples]] = {}

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

    def layout(width: int) -> list[StyleAndTextTuples]:
        """Field header rows, a separator, then the whole order, wrapped for `width`. Cached per
        width: the wrap has to be redone when the terminal is resized, but not on every keypress."""
        if width in wrapped:
            return wrapped[width]
        lines: list[StyleAndTextTuples] = []
        label_width = max((get_cwidth(label) for label, _value in header_rows), default=0)
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
        lines.append([("", margin), ("ansibrightblack", "─" * max(0, width - 4))])
        lines.extend(markdown_rows(order, width))
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
        if get_cwidth(legend) > width:
            legend = "  ↑/↓ · Ctrl-U/D · g/G · Esc/q close"
        parts: StyleAndTextTuples = [("class:choice.disabled", "  Delegate order · read-only\n")]
        for line in lines[scroll : scroll + height]:
            parts.extend(line)
            parts.append(("", "\n"))
        parts.append(("class:choice.disabled", Text.clip_width(legend, width) + "\n"))
        return parts

    def handle_key(key: str, data: str) -> Any:
        nonlocal scroll
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

    loop.tui.show_modal(fragments, handle_key, exclusive=True)


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
        counts = [loop.diff_counts(diff) for _status, _path, diff in sections]
        added_width = max(len(str(added)) for added, _removed in counts)
        removed_width = max(len(str(removed)) for _added, removed in counts)
        for index, ((_status, path, _diff), (added, removed)) in enumerate(zip(sections, counts)):
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
        when_width = max((get_cwidth(when) for when, _kind, _messages in columns), default=0)
        kind_width = max((get_cwidth(kind) for _when, kind, _messages in columns), default=0)
        messages_width = max((get_cwidth(messages) for _when, _kind, messages in columns), default=0)
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
        when, _kind, _messages = segment_columns(segment)
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
