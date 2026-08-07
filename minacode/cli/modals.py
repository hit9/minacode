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
from typing import TYPE_CHECKING, Any

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.utils import get_cwidth

from minacode.base import DISMISSED, SELECTION_BACK, Text, ToolCall
from minacode.tools import AskSpec
from minacode.tui import (
    ASK_DONE,
    ASK_FREE_TEXT,
    TUI_MODAL_PENDING,
    AskViewState,
    ChoiceViewState,
    DiffViewState,
    TabbedViewState,
)

if TYPE_CHECKING:
    from minacode.cli import CommandLoop


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

