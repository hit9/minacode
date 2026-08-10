"""File tools: reading, image viewing, and anchored editing."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

from minacode.base import Json, ModelError, Text, ToolArgs, ToolError
from minacode.image import ImageRef
from minacode.session import Session, TurnDiff
from minacode.tools.base import Tool


class ReadTool(Tool):
    NAME = "Read"
    MAX_ANCHOR_DRIFT: ClassVar[int] = 50
    _ANCHOR_RE: ClassVar[re.Pattern] = re.compile(r"(\d+):([0-9a-z]{5}|[0-9a-f]{8})")
    DESCRIPTION = (
        "Read UTF-8 file line ranges; returns file stat, total lines, and anchor=line:hash(line_content) text. Large outputs are bounded in conversation; "
        "use Recall(tr.N) for full stored output."
    )
    EXAMPLE = (
        'Read ranges. Example: {"path":"src/app.py","ranges":[[1,80],[120,180]]}',
        'Read several files. Example: {"files":[{"path":"src/app.py","ranges":[[1,80]]},{"path":"README.md","ranges":[[1,40]]}]}',
    )

    @classmethod
    def arg_schema(cls) -> Json:
        # fmt: off
        return cls.object_schema({
            "path": {"type": "string", "description": "File path to read"},
            "ranges": {"type": "array", "minItems": 1, "items": cls.RANGE_SCHEMA, "description": "Line ranges [[start,end],...], 1-based and inclusive of both ends; end 0 reads from start to the end of the file; omit to read the whole file"},
        }, ["path"])
        # fmt: on

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        return cls.object_schema({
            "path": {"type": "string", "description": "File path to read (single-file form)"},
            "ranges": {"type": "array", "items": cls.RANGE_SCHEMA, "minItems": 1, "description": "Line ranges [[start,end],...], 1-based and inclusive of both ends; end 0 reads from start to the end of the file; omit to read the whole file"},
            "files": {"type": "array", "items": cls.arg_schema(), "minItems": 1, "description": "Batch form: list of {path, ranges} to read several files in one call"},
        })
        # fmt: on

    @classmethod
    def payload_args(cls, payload: Json) -> ToolArgs:
        return (
            payload["files"]
            if isinstance(payload.get("files"), list)
            else [{"path": payload.get("path", ""), "ranges": cls.ranges_arg(payload.get("ranges") or [[1, 0]])}]
        )

    @classmethod
    def ranges_arg(cls, value: object) -> object:
        return [value] if isinstance(value, list) and len(value) == 2 and all(isinstance(item, int) and not isinstance(item, bool) for item in value) else value

    @staticmethod
    def line_hash(line: str) -> str:
        # Hash the visible content only. The trailing newline is stripped so the anchor matches the
        # line the model sees (anchor_line displays the stripped line), stays stable when only the
        # final newline changes, and is consistent with indexed_line_hash.
        return Text.base36(int(hashlib.sha1(line.rstrip("\n").encode("utf-8")).hexdigest()[:6], 16)).rjust(5, "0")

    @staticmethod
    def split_lines(text: str) -> list[str]:
        # Canonical line model shared by Read and Edit: split on "\n" only, keeping the newline
        # (like file.readlines()). str.splitlines(True) also breaks on \r, \v, \f, \x1c-\x1e, \x85,
        # \u2028, \u2029, which would number lines differently than Read and desync anchors.
        parts = text.split("\n")
        lines = [part + "\n" for part in parts[:-1]]
        if parts[-1]:
            lines.append(parts[-1])
        return lines

    @classmethod
    def anchor(cls, index: int, line: str) -> str:
        # Line numbers are 0-based everywhere inside the tools (they index straight into the line
        # list) and 1-based only in what the model reads, so they agree with the numbering it sees
        # from grep, tracebacks, diffs, and InspectCode. This and parse_anchor are the pair that
        # crosses that boundary; every caller on either side keeps working in 0-based indices.
        return f"{index + 1}:{cls.line_hash(line)}"

    @classmethod
    def anchor_line(cls, index: int, line: str) -> str:
        return f"anchor={cls.anchor(index, line)} | {line.rstrip(chr(10))}"

    @staticmethod
    def indexed_line_hash(line: str) -> str:
        return hashlib.sha256(line.rstrip("\n").encode("utf-8")).hexdigest()[:8]

    @staticmethod
    def parse_anchor(value: str) -> tuple[int, str] | None:
        """Return the anchor's 0-based line index and hash. The counterpart to `anchor`: the model
        writes a 1-based number, callers get an index they can use against a line list directly. An
        anchor carried over from a session written before line numbers became 1-based decodes one
        line early, which the hash check and `relocated_anchor` correct, so the index can be -1."""
        text = value.split("|", 1)[0].strip()
        if text.startswith("anchor="):
            text = text.removeprefix("anchor=").strip()
        match = ReadTool._ANCHOR_RE.fullmatch(text)
        return (int(match.group(1)) - 1, match.group(2).lower()) if match else None

    @staticmethod
    def require_anchor(anchor: str) -> tuple[int, str]:
        """Parse an anchor or raise the standard ToolError guiding the model to a real one."""
        parsed = ReadTool.parse_anchor(anchor)
        if parsed is None:
            raise ToolError('invalid anchor; use the "anchor=line:hash" value from Read, Search, or InspectCode')
        return parsed

    @classmethod
    def anchor_matches(cls, line: str, expected: str) -> bool:
        return expected == cls.line_hash(line) or expected == cls.indexed_line_hash(line)

    @classmethod
    def relocated_anchor(cls, lines: list[str], index: int, expected: str) -> int | None:
        matches = [current for current, line in enumerate(lines) if cls.anchor_matches(line, expected)]
        if len(matches) != 1 or abs(matches[0] - index) > cls.MAX_ANCHOR_DRIFT:
            return None
        return matches[0]

    def needs_confirmation(self) -> bool:
        return any(not self.session.in_cwd(path) for path, _ in self.targets())

    def call(self) -> str:
        return "\n\n".join(self.read_one(path, ranges) for path, ranges in self.targets())

    def short_args(self) -> list[str]:
        # This echoes the call back to the model, not just the terminal, so it has to read as a
        # range the model could have written. `end` 0 is the "to the end of the file" sentinel:
        # printed raw it says 1:0, which under 1-based inclusive bounds reads as an empty range.
        return [(self.session.relpath(path) + " " + ",".join(filter(None, map(self.range_label, ranges)))).rstrip() for path, ranges in self.targets()]

    @staticmethod
    def range_label(bounds: tuple[int, int]) -> str:
        start, end = bounds
        if end == 0:
            return "" if start <= 1 else f"{start}:"  # whole file / from start to the end
        return f"{start}:{end}"

    def targets(self) -> list[tuple[str, list[tuple[int, int]]]]:
        if not self.args:
            raise ToolError("Read requires at least one {path,ranges} object")
        targets = []
        for index, spec in enumerate(self.args):
            if not isinstance(spec, dict):
                raise ToolError("Read args must be {path,ranges} objects")
            if unexpected := sorted(set(spec) - {"path", "ranges"}):
                raise ToolError("Read unexpected field: " + ", ".join(unexpected))
            path = str(spec.get("path") or "").strip()
            raw_ranges = self.ranges_arg(spec.get("ranges") if "ranges" in spec else [[1, 0]])
            if not path:
                raise ToolError("Read requires non-empty path")
            if not isinstance(raw_ranges, list) or not raw_ranges:
                raise ToolError("Read requires non-empty ranges")
            ranges = [self.line_range(value, f"args[{index}].ranges") for value in raw_ranges]
            targets.append((self.session.resolve_path(path), ranges))
        return targets

    def read_one(self, path: str, ranges: list[tuple[int, int]]) -> str:
        with open(path, encoding="utf-8") as file:
            lines = file.readlines()
        out = [f"<Read path={json.dumps(self.session.relpath(path))}>", self.file_stat(path), f"<total_lines>{len(lines)}</total_lines>"]
        for requested_start, requested_end in ranges:
            # Requested bounds are 1-based and inclusive; `end` 0 still means "to the end of the
            # file". A start of 0 is not a valid line number but unambiguously means the top, so it
            # reads as line 1 rather than failing.
            start = min(max(requested_start, 1) - 1, len(lines))
            end = max(start, len(lines) if requested_end == 0 else min(len(lines), requested_end))
            out.append(f"<range>{start + 1}:{end}</range>")
            out.append("<content hashline-numbered>")
            out.extend(self.anchor_line(i, lines[i]) for i in range(start, end))
            out.append("</content>")
        out.append("</Read>")
        return "\n".join(out)


class ViewImageTool(Tool):
    NAME = "ViewImage"
    DESCRIPTION = (
        "View one local image as visual model input. Supports PNG, JPEG, WebP, and single-frame GIF; paths outside the workspace require confirmation."
    )
    PRODUCES_MODEL_OBSERVATION = True

    def __init__(self, session: Session, args: ToolArgs):
        super().__init__(session, args)
        self.image: ImageRef | None = None

    @classmethod
    def params_schema(cls) -> Json:
        return cls.object_schema({"path": {"type": "string", "minLength": 1, "description": "Local image path to view"}}, ["path"])

    @classmethod
    def payload_args(cls, payload: Json) -> ToolArgs:
        return [payload.get("path")]

    def path(self) -> str:
        path = self.strings(min_count=1, max_count=1)[0].strip()
        if not path:
            raise ToolError("ViewImage path must be non-empty")
        return self.session.resolve_path(path)

    def needs_confirmation(self) -> bool:
        return not self.session.in_cwd(self.path())

    def short_args(self) -> list[str]:
        return [self.session.relpath(self.path())]

    def call(self) -> str:
        path = self.path()
        try:
            self.image = self.session.images.load(path, source_text=self.session.relpath(path))
        except ModelError as error:
            raise ToolError(str(error)) from error
        return (
            f"<ViewImage path={json.dumps(self.session.relpath(path))} "
            f"media_type={json.dumps(self.image.media_type)} width={self.image.width} "
            f"height={self.image.height} bytes={self.image.size}/>"
        )

    def model_observation(self) -> Json | None:
        return self.session.images.tool_observation((self.image,)) if self.image is not None else None


@dataclass
class Edit:
    op: str
    start: str = ""
    end: str = ""
    content: str = ""
    old: str = ""
    new: str = ""


@dataclass
class EditApplyResult:
    content: str
    changes: list[tuple[int, int, int, int]]
    replacements: list[tuple[int, int, list[str]]]
    replace_all: bool = False


@dataclass
class EditWarning:
    """A post-edit observation rendered inside the Edit result. Never affects control flow:
    warnings are advisory only, and anchors always name lines of the edited file."""

    code: str  # rule name, e.g. "duplicate-lines"
    message: str
    anchors: list[tuple[int, str]] = field(default_factory=list)  # (line_index, line_text) in the edited file


def _duplicate_lines(before: str, after: str) -> EditWarning | None:
    """Warn when the edit introduces adjacent identical non-blank lines that were not adjacent
    in the original file. Pairs are compared by line content only: a pair that already existed
    before the edit is not reported, blank lines are never reported."""
    after_lines = ReadTool.split_lines(after)
    before_lines = ReadTool.split_lines(before)

    def pairs(lines: list[str]) -> set[tuple[str, str]]:
        return {(lines[i - 1], lines[i]) for i in range(1, len(lines)) if lines[i] == lines[i - 1] and lines[i].strip() != ""}

    new_pairs = pairs(after_lines) - pairs(before_lines)
    if not new_pairs:
        return None
    anchors: list[tuple[int, str]] = []
    for i in range(1, len(after_lines)):
        if (after_lines[i - 1], after_lines[i]) in new_pairs:
            anchors.extend(((i - 1, after_lines[i - 1]), (i, after_lines[i])))
    return EditWarning(
        "duplicate-lines",
        "adjacent identical lines after this edit; confirm intended",
        sorted(set(anchors)),
    )


# Rule tuple: pure (before: str, after: str) -> EditWarning | None functions. Adding a rule
# here is enough — call() only invokes warnings_block() and the envelope does not change.
EDIT_WARNINGS: tuple[Callable[[str, str], EditWarning | None], ...] = (_duplicate_lines,)


class EditTool(Tool):
    """Create or patch one file through content-verified anchors rather than line numbers.

    An anchor pairs a line index with a hash of that line's content, so an edit carries proof of what
    the model believed it was editing. A bare line number silently targets whatever moved into that
    position; here a mismatch is refused and reported with the line actually found. That check is what
    makes edits safe to plan and confirm before writing.

    Strict but not brittle: when the expected content still exists exactly once and has drifted only a
    short distance, the anchor relocates to it, since earlier edits shifting lines is ordinary rather
    than a conflict. Ambiguity or distance is refused instead of guessed, because a wrong resolution
    corrupts a file silently.

    A call that would change nothing is an error rather than a silent success — the model needs to
    learn that its anchor or search text was wrong.
    """

    NAME = "Edit"
    DESCRIPTION = (
        "Create or patch one UTF-8 file. op=create writes a new file; replace/delete cover the inclusive "
        "start..end range (the line at end is itself replaced or deleted); insert_before/insert_after keep "
        "the anchor line and only add content beside it; never restate lines that already exist in the file."
    )
    EXAMPLE = (
        'create file. Example: {"path":"src/app.py","edits":[{"op":"create","content":"print(1)\\n"}]}',
        'replace range. Example: {"path":"src/app.py","edits":[{"op":"replace","start":"10:1ab2c","end":"12:3de4f","content":"new_value = 1\\n"}]}',
        'replace_all exact text; do not mix with anchored ops. Example: {"path":"src/app.py","edits":[{"op":"replace_all","old":"OldName","new":"NewName"}]}',
    )
    MUTATES = True

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        edit = cls.object_schema({
            "op": {"type": "string", "description": "create|replace|delete|insert_before|insert_after|replace_all"},
            "start": {
                "type": "string",
                "description": (
                    "Exact current line:hash anchor copied verbatim from Read, Search, or InspectCode; "
                    "never invent or calculate it; re-read after any file change or stale-anchor error"
                ),
            },
            "end": {
                "type": "string",
                "description": (
                    "Exact current line:hash anchor copied verbatim from Read, Search, or InspectCode; "
                    "never invent or calculate it; re-read after any file change or stale-anchor error; "
                    "inclusive — the line at end is itself replaced or deleted"
                ),
            },
            "content": {
                "type": "string",
                "description": (
                    "New text for create/replace/insert. For replace: the complete new text of the inclusive range; "
                    "for insert_before/insert_after: only the new lines — the anchor line is kept and must not be "
                    "restated; for create: the whole file. The first and last content lines must correspond exactly "
                    "to the start/end anchor lines."
                ),
            },
            "old": {"type": "string", "description": "Text to find for replace_all"},
            "new": {"type": "string", "description": "Replacement text for replace_all"},
        }, ["op"])
        return cls.object_schema({
            "path": {"type": "string", "description": "File to create or patch"},
            "edits": {"type": "array", "items": edit, "minItems": 1, "description": "Ordered edit operations to apply"},
        }, ["path", "edits"])
        # fmt: on

    @classmethod
    def payload_args(cls, payload: Json) -> ToolArgs:
        path = payload.get("path", "")
        raw_edits = payload.get("edits", [])
        if not isinstance(raw_edits, list):
            return [path, raw_edits]

        # Some models repeat the top-level path inside an edit operation. It is safe to discard
        # only an exact duplicate; a different nested path remains invalid and is rejected later.
        edits = []
        for item in raw_edits:
            if isinstance(item, dict) and item.get("path") == path:
                item = {key: value for key, value in item.items() if key != "path"}
            edits.append(item)
        return [path, edits]

    def call(self) -> str:
        path, original, created, result = self.build()
        if result.content == original and not created:
            raise ToolError(self.no_changes_error(original, result))
        if created:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(result.content)
        self.last_path = self.session.relpath(path)
        self.last_diff = self.diff(path, original, result.content)
        self.last_before = original
        self.last_after = result.content
        parts = [
            f"<Edit path={json.dumps(self.last_path)}>",
            self.file_stat(path),
            self.last_diff.rstrip(),
        ]
        block = self.warnings_block(original, result.content)
        if block:
            parts.append(block)
        parts.append(self.edit_context(result.content, result.changes))
        parts.append("</Edit>")
        return "\n".join(parts)

    def warnings_block(self, before: str, after: str) -> str:
        """Render post-edit warnings as a `<warnings>` block, or "" when nothing fired. Warnings
        are advisory only and never change the edit; anchors always name lines of the edited
        file. Output is bounded: at most 3 warnings and 12 anchor lines, truncated with "..."
        (the format_current_ranges convention)."""
        collected = []
        for rule in EDIT_WARNINGS:
            warning = rule(before, after)
            if warning is not None:
                collected.append(warning)
        if not collected:
            return ""
        out = ["<warnings>"]
        shown = 0
        truncated = False
        for warning in collected[:3]:
            out.append(f"{warning.code}: {warning.message}")
            for index, line in warning.anchors:
                if shown >= 12:
                    truncated = True
                    break
                out.append(ReadTool.anchor_line(index, line))
                shown += 1
            if truncated:
                break
        if truncated or len(collected) > 3:
            out.append("...")
        out.append("</warnings>")
        return "\n".join(out)

    def turn_diff(self) -> TurnDiff | None:
        path, diff = getattr(self, "last_path", ""), getattr(self, "last_diff", "")
        if not (path and diff):
            return None
        return TurnDiff(key="", turn=0, path=path, diff=diff, before=getattr(self, "last_before", ""), after=getattr(self, "last_after", ""))

    def preview(self) -> str:
        path, original, _, result = self.build()
        if result.content == original and os.path.exists(path):
            raise ToolError(self.no_changes_error(original, result))
        return self.diff(path, original, result.content) or f"Edit({path})"

    def short_args(self) -> list[str]:
        path = self.parse()[0]
        return [self.session.relpath(path)]

    def diff(self, path: str, original: str, new_content: str) -> str:
        relpath = self.session.relpath(path)
        return "".join(
            difflib.unified_diff(
                ReadTool.split_lines(original),
                ReadTool.split_lines(new_content),
                fromfile="/dev/null" if not original and not os.path.exists(path) else relpath,
                tofile=relpath,
            )
        )

    def parse(self) -> tuple[str, list[Edit]]:
        if len(self.args) != 2:
            raise ToolError("Edit requires path and edits")
        if not isinstance(self.args[0], str):
            raise ToolError("Edit path must be a string")
        path = self.session.resolve_path(str(self.args[0]))
        raw_edits = self.args[1]
        if not isinstance(raw_edits, list) or not raw_edits:
            raise ToolError("Edit edits must be a non-empty array")
        edits = []
        for item in raw_edits:
            if not isinstance(item, dict):
                raise ToolError("each edit must be an object")
            if unexpected := sorted(set(item) - {"op", "start", "end", "content", "old", "new"}):
                raise ToolError("Edit unexpected field: " + ", ".join(unexpected))
            op = str(item.get("op") or "")
            if op not in {"create", "replace", "delete", "insert_before", "insert_after", "replace_all"}:
                raise ToolError("unknown edit op")
            if op == "create" and len(raw_edits) != 1:
                raise ToolError("create cannot be mixed with other edits")
            if op in {"replace", "delete"} and (not item.get("start") or not item.get("end")):
                raise ToolError(f"{op} requires start and end anchors")
            if op in {"insert_before", "insert_after"} and not item.get("start"):
                raise ToolError(f"{op} requires start anchor")
            edits.append(
                Edit(
                    op=op,
                    start=str(item.get("start") or ""),
                    end=str(item.get("end") or ""),
                    content=self.content_text(str(item.get("content") or "")),
                    old=self.normalize_text(str(item.get("old") or "")),
                    new=self.content_text(str(item.get("new") or "")),
                )
            )
        return path, edits

    def _validate_target(self, path: str, creating: bool) -> bool:
        """Validate an edit/create target and return whether its current contents should be read."""

        if os.path.exists(path):
            if creating:
                raise ToolError("file already exists")
            if os.path.isdir(path):
                raise ToolError("path is a directory")
            return True
        if creating:
            parent = os.path.dirname(path) or "."
            if os.path.isdir(parent):
                return False
            if os.path.exists(parent):
                raise ToolError("parent path is not a directory")
            if not self.session.in_cwd(parent):
                raise ToolError("parent directory outside workspace does not exist; create it with an approved Bash mkdir, then retry Edit")
            return False
        raise ToolError("file does not exist; use op=create to create it")

    def build(self) -> tuple[str, str, bool, EditApplyResult]:
        path, edits = self.parse()
        creating = edits[0].op == "create"
        if self._validate_target(path, creating):
            with open(path, encoding="utf-8") as file:
                original = file.read()
            created = False
        else:
            original, created = "", True
        result = self.apply(original, edits)
        return path, original, created, result

    def apply(self, original: str, edits: list[Edit], anchor_resolver: Callable[[str], int] | None = None) -> EditApplyResult:
        """Apply one call's edits to a file's lines, returning the new content and its change spans.

        Anchors are resolved against the original lines before anything is spliced, and the splices are
        then applied in reverse index order so each one leaves the earlier indices untouched. Resolving
        as you go, or splicing forward, shifts every later anchor by the size of the previous edit.
        Overlaps are rejected once the anchors are known and before any splice runs: two edits
        claiming the same range cannot both be honored, and the model needs to be told rather than
        silently obeyed.

        `changes` is rebuilt afterwards in forward order with a running delta, because it describes
        positions in the file that now exists rather than the one the anchors named.
        """
        if edits[0].op == "create":
            lines = self.content_lines(edits[0].content, False)
            return EditApplyResult("".join(lines), [(0, 0, 0, len(lines))], [])
        if any(edit.op == "replace_all" for edit in edits):
            if any(edit.op != "replace_all" for edit in edits):
                raise ToolError("replace_all cannot be mixed with anchored edits")
            content = original
            for edit in edits:
                if not edit.old and content:
                    raise ToolError("replace_all requires old")
                if edit.old and edit.old not in content:
                    raise ToolError("replace_all old text not found")
                content = content.replace(edit.old, edit.new)
            return EditApplyResult(content, [(0, 0, 0, len(ReadTool.split_lines(content)))], [], True)
        lines = ReadTool.split_lines(original)
        resolve_anchor = anchor_resolver or (lambda anchor: self.resolve_anchor(lines, anchor))
        replacements = []
        for edit in edits:
            start = resolve_anchor(edit.start)
            if edit.op in {"replace", "delete"}:
                end = resolve_anchor(edit.end)
                if end < start:
                    raise ToolError("end anchor is before start anchor")
                replacement = [] if edit.op == "delete" else self.content_lines(edit.content, end + 1 < len(lines))
                replacements.append((start, end + 1, replacement))
            elif edit.op in {"insert_before", "insert_after"}:
                index = start if edit.op == "insert_before" else start + 1
                replacements.append((index, index, self.content_lines(edit.content, index < len(lines))))
            else:
                raise ToolError("unknown edit op")
        previous = None
        for start, end, _ in sorted(replacements):
            if previous and (start < previous[1] or (start == previous[0] and end == previous[1])):
                raise ToolError(f"edits overlap or share an insertion point: {previous[0]}:{previous[1]} and {start}:{end}")
            previous = (start, end)
        new_lines = list(lines)
        for start, end, replacement in sorted(replacements, reverse=True):
            new_lines[start:end] = replacement
        changes = []
        delta = 0
        for start, end, replacement in sorted(replacements):
            new_start = start + delta
            new_end = new_start + len(replacement)
            clear_end = 0 if len(replacement) != end - start else new_start + (end - start)
            changes.append((new_start, clear_end, new_start, new_end))
            delta += len(replacement) - (end - start)
        return EditApplyResult("".join(new_lines), changes, replacements)

    def no_changes_error(self, original: str, result: EditApplyResult) -> str:
        return self.no_changes_error_from_lines(ReadTool.split_lines(original), result.replacements, result.replace_all)

    @classmethod
    def no_changes_error_from_lines(cls, lines: list[str], replacements: list[tuple[int, int, list[str]]], replace_all: bool) -> str:
        prefix = "edit produced no changes"
        if replace_all:
            return prefix + "; replace_all result is identical to current file"
        if not replacements:
            return prefix
        matching = [(start, end) for start, end, replacement in replacements if lines[start:end] == replacement]
        if len(matching) != len(replacements):
            return prefix + "; edits cancel out; check requested content"
        return prefix + "; requested content already matches target range\n" + cls.format_current_ranges(lines, matching)

    @classmethod
    def format_current_ranges(cls, lines: list[str], ranges: list[tuple[int, int]]) -> str:
        out = ["<current-target-ranges hashline-numbered>"]
        shown_lines = 0
        range_index = -1
        for range_index, (start, end) in enumerate(ranges[:3]):
            out.append(f"<target start={start + 1} end={end}>")
            if start == end:
                out.append("(empty range)")
            else:
                for index in range(start, end):
                    if shown_lines >= 12:
                        out.append("...")
                        break
                    line = lines[index]
                    out.append(ReadTool.anchor_line(index, line))
                    shown_lines += 1
            out.append("</target>")
            if shown_lines >= 12:
                break
        if len(ranges) > range_index + 1:
            out.append("...")
        out.append("</current-target-ranges>")
        return "\n".join(out)

    def edit_context(self, content: str, changes: list[tuple[int, int, int, int]]) -> str:
        lines = ReadTool.split_lines(content)
        out = []
        for clear_start, clear_end, start, end in changes:
            if start == 0 and end == len(lines):
                # create and replace_all cover the whole file: the refund table would be the entire
                # file's anchor list, ballooning one small edit into a full-context dump. Skip it;
                # the model re-Reads for anchors when it keeps editing such files.
                continue
            # Reported like every other range the model sees: 1-based, both ends inclusive.
            out.append(f"<invalidate>{clear_start + 1}:{clear_end}</invalidate>")
            shown = lines[start:end]
            if shown:
                out.append("<content hashline-numbered>")
                out.extend(ReadTool.anchor_line(start + index, line) for index, line in enumerate(shown))
                out.append("</content>")
        return "\n".join(out)

    def content_lines(self, content: str, followed_by_more: bool) -> list[str]:
        content = self.normalize_text(content)
        if content == "":
            return []
        lines = ReadTool.split_lines(content)
        if followed_by_more and lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        return lines

    @staticmethod
    def normalize_text(value: str) -> str:
        return value.replace("\r\n", "\n").replace("\r", "\n")

    @classmethod
    def content_text(cls, value: str) -> str:
        value = cls.normalize_text(value)
        return value.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t") if "\n" not in value and "\\n" in value else value

    def resolve_anchor(self, lines: list[str], anchor: str) -> int:
        index, expected = ReadTool.require_anchor(anchor)
        if 0 <= index < len(lines) and ReadTool.anchor_matches(lines[index], expected):
            return index
        relocated = ReadTool.relocated_anchor(lines, index, expected)
        if relocated is not None:
            return relocated
        if not 0 <= index < len(lines):
            raise ToolError("anchor line out of range")
        current = ReadTool.anchor_line(index, lines[index])
        raise ToolError(f"stale anchor {anchor}; current is {current}")
