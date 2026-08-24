"""Tool result formatting: pure text assembly for tool calls, independent of execution state."""

from __future__ import annotations

import json
import re
from typing import NamedTuple

from minacode.base import Text, ToolCall, oneline
from minacode.tools import TOOL_REGISTRY, Tool

BASH_TRANSCRIPT_PREVIEW_LINES = 3
BASH_PREVIEW_LINE_LIMIT = 220
EDIT_PATH_RE = re.compile(r'<Edit\s+path=(".*?")')
MCP_CALL_RE = re.compile(r"(?s)<MCPCall\b[^>]*>\n?(.*?)\n?</MCPCall>\s*$")
# The envelope DelegateTool._send returns for a finished delegation: attributes in fixed order,
# the worker's answer wrapped in <worker> tags. Parsed with a couple of string scans — the
# format is ours, so no XML parser is needed.
DELEGATE_META_RE = re.compile(
    r'<Delegate action="send" steps="(\d+)" elapsed="([^"]+)" files="([^"]*)" stopped_at_max_steps="(true|false)"(?: tokens="([^"]*)")?(?: rounds="(\d+)")?(?: context_percent="(\d+)")?>'
)
# What a scrolling viewer renders: generous next to the three-line transcript preview, but not
# unbounded. Stored output has no cap of its own, and the text wrapper costs time quadratic in
# the length of a single line, so one minified-JSON line would freeze the modal until it gave
# up. Whatever these drop is still whole under the result's own tr.N key.
VIEWER_LINES = 2000
VIEWER_LINE_CHARS = 1000
# The marker preview_lines writes where it elided; read back to describe the bound, so the note
# in the viewer's header cannot drift from the text under it.
OMITTED_RE = re.compile(r"\.\.\. (\d+) lines? omitted \.\.\.")
# `calls: 5 [tr.95-99]`, `calls: 0`, or the bounded `calls: ... +120 keys` form, all of which
# lead with the count -- the keys themselves are already in the log, one per nested call line.
TOOLSCRIPT_CALLS_RE = re.compile(r"^calls: (?:\.\.\. \+)?(\d+)", re.MULTILINE)


class DelegateFields(NamedTuple):
    """The display fields of a finished Delegate send envelope. Named rather than a plain tuple
    because both readers want a different subset, and the envelope keeps gaining attributes."""

    steps: str
    elapsed: str
    files: str
    in_tokens: str
    out_tokens: str
    stopped: bool
    rounds: str
    context_percent: str


def bash_result_preview(output: str, line_limit: int, char_limit: int | None = None) -> str:
    sections = []
    for name in ("stdout", "stderr"):
        text = tagged_output(output, name).strip()
        if text:
            sections.extend([name + ":", *("  " + line for line in preview_lines(text, line_limit, char_limit))])
    return "\n".join(sections)


def viewer_text(text: str) -> tuple[str, str]:
    """Arbitrary result text bounded for a scrolling viewer, with a note saying what the bound
    dropped. The same head/tail elision the transcript preview uses, at a size meant to be read
    rather than glanced at."""
    bounded = "\n".join(preview_lines(text, VIEWER_LINES, VIEWER_LINE_CHARS))
    return bounded, viewer_note(text, bounded)


def bash_viewer_output(output: str) -> tuple[str, str]:
    """A Bash result's streams, labeled and bounded for a scrolling viewer, with the same note.

    Measured against the streams rather than the stored envelope: the envelope's tag lines are
    not output, and a note that counts them reads as nonsense on a one-line result."""
    bounded = bash_result_preview(output, VIEWER_LINES, VIEWER_LINE_CHARS)
    streams = [text for name in ("stdout", "stderr") if (text := tagged_output(output, name).strip())]
    return bounded, viewer_note("\n".join(streams), bounded)


def viewer_note(source: str, bounded: str) -> str:
    """What the viewer's bound dropped, as a header phrase -- empty when it dropped nothing.

    Both facts come from the source, not from sniffing the rendered text: a clipped line whose
    tail was whitespace comes back shorter than the limit, so a length test on the output would
    stay silent about exactly the clip it was meant to report. Silence is the dangerous half --
    a reader who cannot tell an elided result from a complete one has to distrust every one."""
    lines = source.splitlines()
    omitted = sum(int(match.group(1)) for match in OMITTED_RE.finditer(bounded))
    parts = []
    if omitted:
        parts.append(f"{len(lines) - omitted} shown of {len(lines)}")
    if any(len(line.rstrip()) > VIEWER_LINE_CHARS for line in lines):
        parts.append(f"long lines clipped at {VIEWER_LINE_CHARS}")
    return " · ".join(parts)


def bash_exit_code(output: str) -> str:
    """The exit code the envelope recorded, or "" when the output is not one."""
    for line in output.splitlines():
        if line.startswith("* exit_code: "):
            return line.removeprefix("* exit_code: ").strip()
    return ""


def tagged_output(output: str, name: str) -> str:
    start_tag = f"<{name}>"
    end_tag = f"</{name}>"
    start = output.find(start_tag)
    if start < 0:
        return ""
    start += len(start_tag)
    if output.startswith("\n", start):
        start += 1
    next_section = output.find("\n<stderr>\n", start) if name == "stdout" else output.find("\n</BashToolResult>", start)
    end = output.rfind(end_tag, start, next_section if next_section >= 0 else len(output))
    if end < 0:
        return ""
    text = output[start:end]
    return text.removesuffix("\n")


def toolscript_result_fields(output: str) -> tuple[str, str, str] | None:
    """(nested call count, printed stdout, error) from a ToolScript envelope, or None when the
    output is not one -- a `describe` returns tool shapes, and has no script to summarize."""
    if not output.startswith(("ToolScript ok", "ToolScript failed")):
        return None
    match = TOOLSCRIPT_CALLS_RE.search(output)
    sections: dict[str, list[str]] = {"stdout:": [], "error:": []}
    section = ""
    for line in output.splitlines():
        if line in ("stdout:", "stderr:", "error:"):
            section = line
        elif section in sections:
            sections[section].append(line)
    # The traceback's last line is the one that names what went wrong; the frames above it are
    # in the viewer, against the numbered source.
    error = "\n".join(sections["error:"]).strip()
    return match.group(1) if match else "0", "\n".join(sections["stdout:"]), error.splitlines()[-1] if error else ""


def preview_lines(text: str, line_limit: int, char_limit: int | None = None) -> list[str]:
    lines = [clip_preview_line(line, char_limit) for line in text.splitlines()]
    if len(lines) <= line_limit:
        return lines
    head = line_limit // 2
    tail = line_limit - head
    omitted = len(lines) - line_limit
    noun = "line" if omitted == 1 else "lines"
    return [*lines[:head], f"... {omitted} {noun} omitted ...", *lines[-tail:]]


def clip_preview_line(line: str, char_limit: int | None = None) -> str:
    limit = BASH_PREVIEW_LINE_LIMIT if char_limit is None else char_limit
    line = line.rstrip()
    return line if len(line) <= limit else line[: limit - 3].rstrip() + "..."


def mcp_result_summary(call: ToolCall, output: str, elapsed: float | None) -> str:
    if str((call.args[0] if call.args and isinstance(call.args[0], dict) else {}).get("action")) != "call":
        return ""
    inner = output
    match = MCP_CALL_RE.match(output)
    if match:
        inner = match.group(1).strip()
    if not inner:
        shape = "empty"
    else:
        try:
            data = json.loads(inner)
        except (json.JSONDecodeError, ValueError):
            data = None
        if isinstance(data, list):
            shape = f"{len(data)} items"
        elif isinstance(data, dict):
            shape = f"{len(data)} fields"
        else:
            shape = f"{inner.count(chr(10)) + 1} lines"
    parts = [f"{shape}, {human_size(len(inner))}"]
    if elapsed is not None:
        parts.append(f"{elapsed:.1f}s")
    return "→ " + " · ".join(parts)


def delegate_result_fields(output: str) -> DelegateFields | None:
    """Parse a finished Delegate send envelope into its display fields, or None when the
    envelope is missing. rounds/context_percent are "" when the envelope was written before
    they existed. Shared by delegate_result_summary (the fallback child line) and the finish
    rule label, so both show the same numbers.
    """
    match = DELEGATE_META_RE.search(output)
    if not match:
        return None
    steps, elapsed, files, stopped, tokens, rounds, context_percent = match.groups()
    if tokens is not None:
        in_tokens, out_tokens = tokens.split("/", 1)
        in_tokens = Text.abbreviate_count(int(in_tokens))
        out_tokens = Text.abbreviate_count(int(out_tokens))
    else:
        in_tokens = out_tokens = ""
    return DelegateFields(steps, elapsed, files, in_tokens, out_tokens, stopped == "true", rounds or "", context_percent or "")


def delegate_result_summary(output: str) -> str:
    """The one-line summary of a finished Delegate send, from its envelope attributes."""
    fields = delegate_result_fields(output)
    if fields is None:
        return ""
    parts = [f"steps {fields.steps}", fields.elapsed, fields.files]
    if fields.in_tokens:
        parts.append(f"{fields.in_tokens} in / {fields.out_tokens} out")
    if fields.rounds:
        parts.append(f"round {fields.rounds}")
    if fields.context_percent:
        parts.append(f"ctx {fields.context_percent}%")
    if fields.stopped:
        parts.append("stopped at max steps")
    return " · ".join(parts)


def delegate_answer_preview(output: str) -> str:
    """The worker's answer (the text between <worker> and </worker>), bounded like the Bash
    transcript preview: clipped per line and capped at BASH_TRANSCRIPT_PREVIEW_LINES."""
    start = output.find("<worker>")
    end = output.find("</worker>")
    if start < 0 or end <= start:
        return ""
    answer = output[start + len("<worker>") : end].strip()
    if not answer:
        return ""
    return "\n".join(preview_lines(answer, BASH_TRANSCRIPT_PREVIEW_LINES))


def human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes}B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f}KB"
    return f"{num_bytes / (1024 * 1024):.1f}MB"


def with_batch_suffix(text: str, suffix: str) -> str:
    return text + (("  " + suffix) if suffix else "")


def short_call(session, call: ToolCall, args: list[str] | None = None) -> str:
    tool_class = TOOL_REGISTRY.get(call.name)
    if args is None:
        try:
            args = tool_class(session, call.args).short_args() if tool_class is not None else [Tool.compact(arg) for arg in call.args]
        except Exception:  # noqa: BLE001 - display formatting must fall back for malformed tool arguments.
            args = [Tool.compact(arg) for arg in call.args]
    text = " ".join([call.name, *args]).strip()
    return text if "\n" in text else oneline(text, 200)
