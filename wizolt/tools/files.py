"""File tools: reading, image viewing, and source-view editing."""

from __future__ import annotations

import difflib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field

from wizolt.base import Json, ModelError, ToolArgs, ToolError, split_lines
from wizolt.image import ImageRef
from wizolt.session import Session, TurnDiff
from wizolt.source import (
    EDIT,
    READ,
    SOURCE_MISSING,
    SOURCE_PATH_MISMATCH,
    SOURCE_TARGET_CHANGED,
    SourceBlock,
    SourceSpan,
    SourceView,
    SourceViewDraft,
    TextBlock,
    ToolOutput,
    context_matches,
    relocate_target,
    same_position,
    source_error,
)
from wizolt.tools.base import Tool


class ReadTool(Tool):
    NAME = "Read"
    DESCRIPTION = (
        "Read UTF-8 file line ranges; returns a source view (source=view.N) with ordinary 1-based line numbers, and total lines. "
        "Large outputs are bounded in conversation; use Recall(tr.N) for full stored output. Edit existing text only through a source id and "
        "visible ordinary line numbers."
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

    def needs_confirmation(self) -> bool:
        return any(not (self.session.in_cwd(path) or self.session.owns_asset(path)) for path, _ in self.targets())

    def call(self) -> ToolOutput:
        parts: list[str | TextBlock | SourceBlock] = []
        per_path: dict[str, list[tuple[int, int]]] = {}
        order: list[str] = []
        for path, ranges in self.targets():
            if path not in per_path:
                per_path[path] = []
                order.append(path)
            per_path[path].extend(ranges)
        for path in order:
            # One block and one view per file: ranges across separate request items for the same
            # path are unioned, normalized, and merged, matching the model-facing lines label.
            parts.extend(self.read_one(path, per_path[path]).parts)
        return ToolOutput.rendered(parts)

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

    def read_one(self, path: str, ranges: list[tuple[int, int]]) -> ToolOutput:
        with open(path, encoding="utf-8") as file:
            lines = file.readlines()
        total = len(lines)
        resolved = []
        for requested_start, requested_end in ranges:
            # Requested bounds are 1-based and inclusive; `end` 0 still means "to the end of the
            # file". A start of 0 is not a valid line number but unambiguously means the top.
            start = min(max(requested_start, 1) - 1, total)
            end = max(start, total if requested_end == 0 else min(total, requested_end))
            if end > start:
                resolved.append((start + 1, end))
        block = SourceBlock.plain(SourceViewDraft(path, self.session.relpath(path), total, SourceSpan.build(lines, resolved), READ))
        return ToolOutput(block.render(), (block,))


class ViewImageTool(Tool):
    NAME = "ViewImage"
    DESCRIPTION = "View a local PNG, JPEG, WebP, or single-frame GIF. Uses the active model when possible and the configured vision provider as fallback. Use an attachment's session-owned path when provided; outside-workspace paths require confirmation."
    PRODUCES_MODEL_OBSERVATION = True

    def __init__(self, session: Session, args: ToolArgs):
        super().__init__(session, args)
        self.image: ImageRef | None = None
        # Injected by ToolRunner.call_tool. The tool owns validation and result shape; orchestration
        # owns the model-client lifecycle so Ctrl-C can reach every provider request.
        self.vision_observe: Callable[[tuple[ImageRef, ...], str], str] | None = None
        self._uses_vision_provider = False
        self._observation_text = ""
        # Set when the explicit call uses [vision], so the runner can render that paid request.
        self.vision_entry_label = ""

    @classmethod
    def params_schema(cls) -> Json:
        return cls.object_schema(
            {
                "path": {"type": "string", "minLength": 1, "description": "Local image path to view"},
                "question": {"type": "string", "description": "Optional question to answer about the image"},
            },
            ["path"],
        )

    @classmethod
    def payload_args(cls, payload: Json) -> ToolArgs:
        return [payload.get("path"), payload.get("question")]

    def path(self) -> str:
        raw = self.args[0] if self.args else None
        if not isinstance(raw, str):
            raise ToolError("ViewImage requires a path string")
        path = raw.strip()
        if not path:
            raise ToolError("ViewImage path must be non-empty")
        return self.session.resolve_path(path)

    def question(self) -> str:
        if len(self.args) < 2 or self.args[1] is None:
            return ""
        if not isinstance(self.args[1], str):
            raise ToolError("ViewImage question must be a string")
        return self.args[1].strip()

    def needs_confirmation(self) -> bool:
        path = self.path()
        return not (self.session.in_cwd(path) or self.session.owns_asset(path))

    def short_args(self) -> list[str]:
        return [self.session.relpath(self.path())]

    def _vision_label(self) -> str:
        provider = self.session.config.providers[self.session.config.vision_provider]
        return f"{self.session.config.vision_provider}/{provider.model}"

    def _vision_observe(self, question: str) -> str:
        assert self.image is not None  # call() loaded it before bridging
        if self.vision_observe is None:
            raise ToolError("ViewImage with a configured vision provider requires ToolRunner")
        return self.vision_observe((self.image,), question)

    def call(self) -> str:
        path = self.path()
        try:
            self.image = self.session.images.load(path, source_text=self.session.relpath(path))
        except ModelError as error:
            raise ToolError(str(error)) from error
        header = (
            f"<ViewImage path={json.dumps(self.session.relpath(path))} "
            f"media_type={json.dumps(self.image.media_type)} width={self.image.width} "
            f"height={self.image.height} bytes={self.image.size}"
        )
        # Main-first: only a text-only route (static catalog or session-learned 400) with a
        # configured [vision] entry bridges here; an unknown route keeps the raw observation so
        # the active multimodal model receives the original pixels even when [vision] exists.
        if self.session.image_route.delivery() != "vision":
            return header + "/>"
        self._uses_vision_provider = True
        self.vision_entry_label = self._vision_label()
        try:
            observation = self._vision_observe(self.question())
        except ModelError as error:
            raise ToolError(f"Vision observation failed: {error}") from error
        self._observation_text = observation
        return header + f" vision={json.dumps(self.vision_entry_label)}/>" + "\n" + observation

    def model_observation(self) -> Json | None:
        if self.image is None:
            return None
        if self._uses_vision_provider:
            # Durable text observation for a text-only route: no raw block ever reaches the main
            # route, and the refs stay only for asset ownership across resume.
            return self.session.images.text_observation((self.image,), self._observation_text, self.question())
        return self.session.images.tool_observation((self.image,), self.question())


@dataclass
class Edit:
    op: str
    start: int = 0  # 1-based inclusive first line of a replace/delete range
    end: int = 0  # 1-based inclusive last line of a replace/delete range
    content: str = ""


@dataclass
class EditApplyResult:
    content: str
    changes: list[tuple[int, int, int, int]]
    replacements: list[tuple[int, int, list[str]]]
    relocations: list[str] = field(default_factory=list)  # "relocated ... -> ..." reports


# Characters written in one call past which the call is worth splitting. About 1.5k tokens: large
# enough that ordinary edits never see it, small enough to stay well inside any output budget.
LARGE_EDIT_CHARS = 6000


def _large_edit(edits: list[Edit]) -> str | None:
    """The rendered `<warnings>` line when one call wrote enough text that it should have been several.

    Deliberately measured on what the model typed (`content`), not on the file's before and after:
    the subject is the assistant message the call arrived in, not the change it made. That message
    is generated in one stretch, and a response timeout or an output cap partway through discards
    all of it -- this edit, the reasoning that reached it, and every other call batched beside it.
    Smaller edits land as they go, and cost nothing extra when nothing goes wrong."""
    written = sum(len(edit.content) for edit in edits)
    if written < LARGE_EDIT_CHARS:
        return None
    return (
        "large-edit: "
        f"this call wrote {written} characters in one assistant message; a change this size is safer as several "
        "Edit calls, since a timeout mid-message loses the whole batch"
    )


class EditTool(Tool):
    """Create or patch one file through source views rather than bare line numbers.

    An edit names a source view (view.N) plus ordinary 1-based line numbers. The view is
    evidence, not authority: the complete target is extracted from the view and validated against
    the current file before anything is written. A mismatch relocates only when the exact target
    still exists exactly once nearby; ambiguity or distance is refused instead of guessed,
    because a wrong resolution corrupts a file silently.

    All operations in one call are resolved against the same view before any splice runs, and
    splices apply in reverse index order so each one leaves the earlier indices untouched.
    Every failure returns a small fresh view of the current file so the model can retry without
    a separate Read. A call that would change nothing is an error rather than a silent success.
    """

    NAME = "Edit"
    DESCRIPTION = (
        "Create or patch one UTF-8 file. op=create writes a new file and is the only operation in its call; "
        "replace/delete cover the inclusive 1-based start..end range inside the named source view (source=view.N from "
        "Read, Search, or InspectCode). Content is the complete final text of the named range; lines outside the range "
        "are preserved automatically and must not be copied in as context. An insertion is a replace over a single line: "
        "to add a line after line N, replace N:N with line N's text followed by the new line; to add one before it, put "
        "the new line first. To append to the end, replace the last line N:N with its text plus the new lines. To write "
        "into an existing empty file, use create. "
        "Work in small steps: one call per cohesive change, and split a large rewrite across several "
        "calls, because everything one call writes is generated inside a single assistant message "
        "and a timeout partway through loses all of it. Bash output is not a source: read the file "
        "through Read, Search, or InspectCode before editing."
    )
    EXAMPLE = (
        'create file. Example: {"path":"src/app.py","edits":[{"op":"create","content":"print(1)\\n"}]}',
        'replace range. Example: {"path":"src/app.py","source":"view.12","edits":[{"op":"replace","start":10,"end":12,"content":"new_value = 1\\n"}]}',
        'add a line after line 10. Example: {"path":"src/app.py","source":"view.12","edits":[{"op":"replace","start":10,"end":10,"content":"line 10 text\\nnew_value = 1\\n"}]}',
        'delete range. Example: {"path":"src/app.py","source":"view.12","edits":[{"op":"delete","start":10,"end":12}]}',
    )
    MUTATES = True
    # A recovery view answers the edit that failed, so it spans the requested lines plus context
    # rather than a fixed window. Past this many lines the request is better served by a Read: the
    # point is to save one round trip, not to page a file back through an error message.
    RECOVERY_CONTEXT_LINES = 3
    RECOVERY_MAX_LINES = 60

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        edit = cls.object_schema({
            "op": {"type": "string", "description": "create|replace|delete"},
            "start": {"type": "integer", "minimum": 1, "description": "First line of an inclusive 1-based replace/delete range; must be visible in the named source view"},
            "end": {"type": "integer", "minimum": 1, "description": "Last line of an inclusive 1-based replace/delete range; the line at end is itself replaced or deleted"},
            "content": {
                "type": "string",
                "description": (
                    "New text for create/replace. For replace: the complete final text of the inclusive start..end range; "
                    "lines before start and after end are preserved automatically and must not be copied into content merely as context. "
                    "An insertion is a replace over a single line whose content is that line's final text plus what is added around it; "
                    "an insertion point has two neighbours and either may serve as the range, so prefer the shorter one. "
                    "An explicit empty string deletes the matched range (replace). For create: the whole file."
                ),
            },
        }, ["op"])
        return cls.object_schema({
            "path": {"type": "string", "description": "File to create or patch"},
            "source": {"type": "string", "description": "The view.N id returned by Read, Search, or InspectCode for this path. Required for every existing-file call; forbidden for create"},
            "edits": {"type": "array", "items": edit, "minItems": 1, "description": "Ordered edit operations to apply"},
        }, ["path", "edits"])
        # fmt: on

    @classmethod
    def payload_args(cls, payload: Json) -> ToolArgs:
        path = payload.get("path", "")
        source = payload.get("source") or ""
        raw_edits = payload.get("edits", [])
        if not isinstance(raw_edits, list):
            return [path, source, raw_edits]
        # Some models repeat the top-level path inside an edit operation. It is safe to discard
        # only an exact duplicate; a different nested path remains invalid and is rejected later.
        edits = []
        for item in raw_edits:
            if isinstance(item, dict) and item.get("path") == path:
                item = {key: value for key, value in item.items() if key != "path"}
            edits.append(item)
        return [path, source, edits]

    def call(self) -> ToolOutput:
        path, source_name, edits = self.parse()
        creating = edits[0].op == "create"
        view = self.resolve_view(path, source_name, creating, edits)
        if self._validate_target(path, creating):
            with open(path, encoding="utf-8") as file:
                original = file.read()
            created = False
        else:
            original, created = "", True
        result = self.apply(original, edits, view)
        if result.content == original and not created:
            raise ToolError(self.no_changes_error(original, result), recovery=self.no_op_recovery(path, view, original, result.replacements))
        if created:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        return self.write_result(path, original, result.content, result.changes, self.warnings_block(edits), result.relocations)

    def write_result(
        self,
        path: str,
        before: str,
        after: str,
        changes: list[tuple[int, int, int, int]],
        warnings: str,
        relocations: list[str],
    ) -> ToolOutput:
        """Write `after` and render the Edit envelope: diff, warnings, relocations, fresh view.

        The batch plan writes through here too, so a single Edit and a planned one report a change
        the same way and the fresh view is minted in exactly one place.
        """
        with open(path, "w", encoding="utf-8") as file:
            file.write(after)
        self.last_path = self.session.relpath(path)
        self.last_diff = self.diff(path, before, after)
        self.last_before = before
        self.last_after = after
        parts: list[str | SourceBlock] = [f"<Edit path={json.dumps(self.last_path)}>", self.last_diff.rstrip()]
        if warnings:
            parts.append(warnings)
        if relocations:
            parts.append("\n".join(relocations))
        parts.append(self.fresh_block(path, split_lines(after), changes))
        parts.append("</Edit>")
        return ToolOutput.rendered(parts)

    def warnings_block(self, edits: list[Edit]) -> str:
        """Render the call's warnings as a `<warnings>` block, or "" when nothing fired.

        The only warning left is the large-edit advisory, which is about the assistant message that
        carried the call, not about the change it made: with insertions gone there is no content
        heuristic left to police, so the file's before and after are not inspected at all.
        """
        if (large := _large_edit(edits)) is None:
            return ""
        return f"<warnings>\n{large}\n</warnings>"

    def turn_diff(self) -> TurnDiff | None:
        path, diff = getattr(self, "last_path", ""), getattr(self, "last_diff", "")
        if not (path and diff):
            return None
        return TurnDiff(key="", turn=0, path=path, diff=diff, before=getattr(self, "last_before", ""), after=getattr(self, "last_after", ""))

    def preview(self) -> str:
        path, source_name, edits = self.parse()
        creating = edits[0].op == "create"
        view = self.resolve_view(path, source_name, creating, edits)
        if self._validate_target(path, creating):
            with open(path, encoding="utf-8") as file:
                original = file.read()
        else:
            original = ""
        result = self.apply(original, edits, view)
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
                split_lines(original),
                split_lines(new_content),
                fromfile="/dev/null" if not original and not os.path.exists(path) else relpath,
                tofile=relpath,
            )
        )

    @staticmethod
    def _int_arg(item: Json, key: str, op: str) -> int:
        value = item.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolError(f"{op} requires integer {key}")
        return int(value)

    def parse(self) -> tuple[str, str, list[Edit]]:
        if len(self.args) != 3:
            raise ToolError("Edit requires path, source, and edits")
        if not isinstance(self.args[0], str):
            raise ToolError("Edit path must be a string")
        if not isinstance(self.args[1], str):
            raise ToolError("Edit source must be a string")
        path = self.session.resolve_path(str(self.args[0]))
        source_name = str(self.args[1]).strip()
        raw_edits = self.args[2]
        if not isinstance(raw_edits, list) or not raw_edits:
            raise ToolError("Edit edits must be a non-empty array")
        edits = []
        for item in raw_edits:
            if not isinstance(item, dict):
                raise ToolError("each edit must be an object")
            if unexpected := sorted(set(item) - {"op", "start", "end", "content"}):
                raise ToolError("Edit unexpected field: " + ", ".join(unexpected))
            op = str(item.get("op") or "")
            if op not in {"create", "replace", "delete"}:
                raise ToolError("unknown edit op")
            if op == "create" and len(raw_edits) != 1:
                raise ToolError("create cannot be mixed with other edits")
            if op == "create" and source_name:
                raise ToolError("source is forbidden for create")
            if op != "create" and not source_name:
                raise ToolError(f"{op} requires source=view.N for an existing file; Read, Search, or InspectCode first")
            if op in {"replace", "delete"}:
                start = self._int_arg(item, "start", op)
                end = self._int_arg(item, "end", op)
                if start < 1 or end < start:
                    raise ToolError(f"{op} requires 1 <= start <= end")
                edits.append(Edit(op=op, start=start, end=end, content=self.normalize_text(str(item.get("content") or ""))))
            else:
                if "content" not in item or item["content"] is None:
                    raise ToolError("create requires content; use an explicit empty string to create an empty file")
                edits.append(Edit(op=op, content=self.normalize_text(str(item.get("content") or ""))))
        return path, source_name, edits

    def resolve_view(self, path: str, source_name: str, creating: bool, edits: list[Edit]) -> SourceView | None:
        """Load and validate the named view for an existing-file call, or None for create.

        An expired id is the one failure the model cannot fix by thinking harder: the view left
        active context, and nothing about `view.12` says what it held. So the refusal carries the
        requested lines as they are now, read fresh from the path the call named. That is not the
        old view reconstructed -- it cannot be -- it is current evidence for the same intent, and
        it costs the model one retry instead of a Read plus a retry.
        """
        if creating:
            if source_name:
                raise ToolError("source is forbidden for create; create writes a new file")
            return None
        if not source_name:
            raise ToolError("Edit requires source=view.N for an existing file; Read, Search, or InspectCode first")
        view = self.session.get_source_view(source_name)
        if view is None:
            recovery = self.current_view_recovery(path, edits)
            hint = "use the fresh view below" if recovery else "Read or Search again to obtain a current view"
            raise source_error(SOURCE_MISSING, f"{source_name} is unknown or expired; {hint}", recovery=recovery)
        if view.path != path:
            # Deliberately no fresh view: the model named two different files in one call, and
            # answering with the content of either one would encourage it to pick by content.
            raise source_error(SOURCE_PATH_MISMATCH, f"Edit path and {source_name} path differ; use the view returned for this path")
        return view

    def current_view_recovery(self, path: str, edits: list[Edit]) -> ToolOutput | None:
        """The lines this call asked to change, as they are in the file right now.

        Returns None when the path cannot be shown without approval or cannot be read: an expired
        id is not a reason to project a file the user has not agreed to open. A request too large
        to answer this way falls back to a bounded window, which tells the model the file is there
        and that the range it wants needs a Read of its own.
        """
        if not (self.session.in_cwd(path) or self.session.owns_asset(path)) or os.path.isdir(path):
            return None
        try:
            with open(path, encoding="utf-8") as file:
                lines = file.readlines()
        except (OSError, UnicodeDecodeError):
            return None
        ranges = [self.recovery_range(edit, len(lines)) for edit in edits]
        spans = SourceSpan.build(lines, ranges)
        if sum(len(span.lines) for span in spans) > self.RECOVERY_MAX_LINES:
            spans = SourceViewDraft.around(path, self.session.relpath(path), lines, ranges[0][0] - 1).spans
        block = SourceBlock.plain(SourceViewDraft(path, self.session.relpath(path), len(lines), spans, EDIT))
        return ToolOutput(block.render(), (block,))

    @classmethod
    def recovery_range(cls, edit: Edit, total: int) -> tuple[int, int]:
        """The 1-based inclusive range a failed edit needs to see, with context on both sides."""
        return max(1, edit.start - cls.RECOVERY_CONTEXT_LINES), min(total, edit.end + cls.RECOVERY_CONTEXT_LINES)

    def _validate_target(self, path: str, creating: bool) -> bool:
        """Validate an edit/create target and return whether its current contents should be read."""

        if os.path.exists(path):
            if creating:
                # An existing zero-byte file has nothing to preserve, so create may overwrite it;
                # anything non-empty still fails, since overwriting would destroy its content.
                if os.path.isfile(path) and os.path.getsize(path) == 0:
                    return False
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

    def apply(
        self,
        original: str,
        edits: list[Edit],
        view: SourceView | None,
        locate: Callable[[Edit, int, int], int | None] | None = None,
        recover: Callable[[Edit], ToolOutput] | None = None,
    ) -> EditApplyResult:
        """Resolve one call's edits against a file's lines and splice them.

        Every operation is resolved against `view` (its target is extracted from the view) and
        validated against the lines it is being applied to, then relocated
        exactly within MAX_VIEW_DRIFT when the exact text merely moved. All resolutions happen
        before any splice runs; splices then apply in reverse index order.

        This is the only edit resolution loop. The batch plan applies the same call to its planned
        in-memory file rather than to disk, and injects the two things that differ there:
        `locate`, which maps a range of view lines to the index those lines now sit at after
        earlier edits in the batch (None when the batch never tracked them, and an error when one
        of them was consumed), and `recover`, which builds the fresh view a failure returns -- the
        plan owes the model a view of the file on disk, not of a planned state that has not been
        written.
        """
        if edits[0].op == "create":
            lines = self.content_lines(edits[0].content, False)
            return EditApplyResult("".join(lines), [(0, 0, 0, len(lines))], [], relocations=[])
        assert view is not None
        lines = split_lines(original)
        replacements: list[tuple[int, int, list[str]]] = []
        relocations: list[str] = []
        for edit in edits:
            try:
                target = view.range_lines(edit.start, edit.end)
                planned = locate(edit, edit.start, edit.end) if locate else None
                found, report = self.locate_range(lines, view, edit, planned, target)
                if report:
                    relocations.append(report)
                replacement = [] if edit.op == "delete" else self.content_lines(edit.content, found + len(target) < len(lines))
                replacements.append((found, found + len(target), replacement))
            except ToolError as error:
                raise ToolError(str(error), recovery=recover(edit) if recover else self.fresh_recovery(view, lines, edit)) from error
        return self.splice_lines(lines, replacements, relocations)

    @staticmethod
    def locate_range(lines: list[str], view: SourceView, edit: Edit, planned: int | None, target: tuple[str, ...]) -> tuple[int, str]:
        """Resolve a replace/delete target: exact match in place, else exact bounded relocation.

        Returns the 0-based start and a relocation report ("" when the target did not move). Both
        the single-call and the batch path go through here, so a target that merely drifted is
        relocated under one set of rules and never resolved by position alone.

        `planned` is where a batch moved these lines after editing this file, or None when the
        view's own coordinate is the only place to start. The difference is what the position is
        worth as evidence. A planned index describes a state that plan owns and re-verifies against
        disk before writing, so matching text there settles it -- and the neighbours could not
        vouch for it anyway, since the batch may have rewritten them itself. A view coordinate is
        an assumption: among repeated lines, matching text at an assumed index can be a coincidence
        rather than the occurrence the model saw, so it holds only while the lines the view showed
        beside it are still there. When they are not, the position is re-derived through relocation,
        which either finds the one occurrence the neighbours single out or refuses; a target that is
        unique in the window resolves to that same index anyway, so this costs a scan, never a
        working edit.
        """
        index = edit.start - 1 if planned is None else planned
        before, after = view.neighbors(edit.start, edit.end)
        if same_position(lines, index, target) and (planned is not None or context_matches(lines, index, target, before, after)):
            return index, ""
        relocated = relocate_target(lines, index, target, before=before, after=after)
        if relocated is None:
            widen = "widen the range to include its neighbors, " if edit.start == edit.end else ""
            raise source_error(
                SOURCE_TARGET_CHANGED,
                f"exact target for {view.key} lines {edit.start}:{edit.end} differs and cannot relocate; {widen}use the fresh view below, or Read again",
            )
        if relocated == index:
            return relocated, ""
        return relocated, f"relocated {view.key} lines {edit.start}:{edit.end} -> current lines {relocated + 1}:{relocated + len(target)}"

    @staticmethod
    def fresh_recovery(view: SourceView, lines: list[str], edit: Edit) -> ToolOutput:
        """A fresh bounded view of `lines` around the coordinates the failed edit requested."""
        return ToolOutput("", (SourceBlock.around(view.path, view.display_path, lines, edit.start - 1),))

    @classmethod
    def splice_lines(cls, lines: list[str], replacements: list[tuple[int, int, list[str]]], relocations: list[str] | None = None) -> EditApplyResult:
        """Overlap-check and splice resolved replacements in reverse index order.

        `changes` is rebuilt afterwards in forward order with a running delta, because it
        describes positions in the file that now exists rather than the one the edits named.
        """
        previous = None
        for start, end, _ in sorted(replacements):
            if previous and (start < previous[1] or (start == previous[0] and end == previous[1])):
                raise ToolError(f"edits overlap or are identical ranges: {previous[0]}:{previous[1]} and {start}:{end}")
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
        return EditApplyResult("".join(new_lines), changes, replacements, relocations=relocations or [])

    def no_changes_error(self, original: str, result: EditApplyResult) -> str:
        return self.no_changes_error_from_lines(split_lines(original), result.replacements)

    @classmethod
    def no_changes_error_from_lines(cls, lines: list[str], replacements: list[tuple[int, int, list[str]]]) -> str:
        prefix = "edit produced no changes"
        if not replacements:
            return prefix
        matching = [(start, end) for start, end, replacement in replacements if lines[start:end] == replacement]
        if len(matching) != len(replacements):
            return prefix + "; edits cancel out; check requested content"
        return prefix + "; requested content already matches target range"

    def no_op_recovery(self, path: str, view: SourceView | None, original: str, replacements: list[tuple[int, int, list[str]]]) -> ToolOutput:
        """The no-op failure's fresh view: what the requested targets currently hold.

        Every target is shown, not only the ones whose content already matched: the model asked to
        change these lines and nothing happened, so these lines are exactly what it needs to see.
        """
        lines = split_lines(original)
        ranges = [(start + 1, end) if end > start else (max(1, start), min(len(lines), start + 1)) for start, end, _ in replacements]
        display = view.display_path if view else self.session.relpath(path)
        draft = SourceViewDraft(view.path if view else path, display, len(lines), SourceSpan.build(lines, ranges), EDIT)
        block = SourceBlock.plain(draft)
        return ToolOutput(block.render(), (block,))

    def fresh_block(self, path: str, lines: list[str], changes: list[tuple[int, int, int, int]]) -> SourceBlock:
        """The fresh view after a successful edit: every changed hunk plus up to three unchanged
        context lines on either side, as one new view the model can continue editing from."""
        ranges = []
        for clear_start, clear_end, start, end in changes:
            # A deletion has no changed line left to show, so the view covers the seam it left
            # behind: without it the block would be empty and the model would have to Read again
            # just to keep editing the file it only just changed.
            ranges.append((max(1, start - 2), min(len(lines), max(end, start) + 3)))
        return SourceBlock.plain(SourceViewDraft(path, self.session.relpath(path), len(lines), SourceSpan.build(lines, ranges), EDIT))

    def content_lines(self, content: str, followed_by_more: bool) -> list[str]:
        content = self.normalize_text(content)
        if content == "":
            return []
        lines = split_lines(content)
        if followed_by_more and lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        return lines

    @staticmethod
    def normalize_text(value: str) -> str:
        return value.replace("\r\n", "\n").replace("\r", "\n")
