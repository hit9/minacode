"""Source-view value types, ids, error categories, and view construction from lines."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from wizolt.base import ToolError

# How far a target may drift from its original coordinates before relocation refuses.
MAX_VIEW_DRIFT = 50

# Stable error categories, first line of a view-related ToolError.
SOURCE_MISSING = "source missing"
SOURCE_PATH_MISMATCH = "source path mismatch"
SOURCE_RANGE_UNSEEN = "source range unseen"
SOURCE_TARGET_CHANGED = "source target changed"
SOURCE_TARGET_AMBIGUOUS = "source target ambiguous"
SOURCE_TARGET_CONSUMED = "source target consumed"
PLANNED_EDIT_STALE = "planned edit stale"

# Producers that may mint source views.
READ = "Read"
SEARCH = "Search"
INSPECT = "InspectCode"
EDIT = "Edit"

_VIEW_KEY_RE = re.compile(r"^view\.(\d+)$")


def view_key(number: int) -> str:
    """The public id for view number `number`."""
    return f"view.{number}"


def parse_view_key(key: str) -> int | None:
    """The numeric part of a public view id, or None for a malformed key."""
    match = _VIEW_KEY_RE.fullmatch(key)
    return int(match.group(1)) if match else None


def source_error(category: str, detail: str) -> ToolError:
    """A ToolError whose first line is the stable error category."""
    return ToolError(f"{category} {detail}".rstrip())


@dataclass(frozen=True)
class SourceSpan:
    """One contiguous, 1-based inclusive line range inside a source view."""

    start: int  # 1-based inclusive
    lines: tuple[str, ...]  # exact normalized line strings, newline included

    @property
    def end(self) -> int:
        """1-based inclusive end line."""
        return self.start + len(self.lines) - 1


@dataclass(frozen=True)
class SourceViewDraft:
    """Immutable source evidence produced by a read-only tool before a public id is assigned."""

    path: str  # canonical resolved path
    display_path: str  # stable model-facing path
    total_lines: int
    spans: tuple[SourceSpan, ...]
    producer: str  # Read, Search, InspectCode, or Edit


@dataclass(frozen=True)
class SourceView:
    """A draft committed by the runner with a public `view.N` key and its round/step origin."""

    key: str  # view.N
    path: str
    display_path: str
    total_lines: int
    spans: tuple[SourceSpan, ...]
    producer: str
    round: int
    step: int

    def draft(self) -> SourceViewDraft:
        return SourceViewDraft(self.path, self.display_path, self.total_lines, self.spans, self.producer)


@dataclass(frozen=True)
class SourceBlock:
    """A draft plus per-line markers (e.g. Search's `>` match / ` ` context prefix).

    `bounded` records a projection clip: the visible spans are head spans followed by tail spans,
    and `split_span` is the index of the first tail span. The omitted middle is not part of any
    registered span and cannot be targeted.
    """

    draft: SourceViewDraft
    markers: tuple[str, ...]  # one marker per line across all spans, in order
    bounded: bool = False
    estimated_tokens: int = 0
    omitted_tokens: int = 0
    budget_tokens: int = 0  # the projection budget the block was clipped to
    split_span: int = 0  # index of the first tail span when bounded
    note_recall: str = ""  # tr.N key the full retained output lives under, filled by the runner
    note_file: str = ""  # materialized asset path for the full retained output, filled by the runner
    note_hint: str = ""  # what the model should do with the omitted middle


@dataclass(frozen=True)
class ToolOutput:
    """Structured tool output: the full plain retained text plus model-facing parts.

    A part is either literal text or a source block. Source blocks are rendered with their
    assigned `view.N` key on the main thread; the retained text carries ordinary line numbers and
    no `source=` attribute.
    """

    retained_text: str
    parts: tuple[str | SourceBlock, ...]

    @property
    def has_source(self) -> bool:
        return any(isinstance(part, SourceBlock) for part in self.parts)

    @property
    def drafts(self) -> tuple[SourceViewDraft, ...]:
        return tuple(part.draft for part in self.parts if isinstance(part, SourceBlock))


def as_tool_output(value: str | ToolOutput) -> ToolOutput:
    """Normalize an ordinary tool result to the same structured abstraction."""
    return value if isinstance(value, ToolOutput) else ToolOutput(value, (value,))


def merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Normalize 1-based inclusive ranges: sort, then merge ranges that overlap or touch.

    A range whose end is 0 means "to the end of the file" and is resolved by the caller, which
    knows the file's line count. This function only merges explicit bounds.
    """
    ordered = sorted((start, end) for start, end in ranges if start <= end)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        elif end > merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
    return merged


def spans_from_lines(lines: Sequence[str], ranges: Sequence[tuple[int, int]]) -> tuple[SourceSpan, ...]:
    """Build non-empty spans from 1-based inclusive ranges over `lines` (0-based indexed).

    Ranges are clamped to the file and normalized: sorted, merged when overlapping or touching.
    """
    total = len(lines)
    spans: list[SourceSpan] = []
    for start, end in merge_ranges(ranges):
        low = min(max(start, 1) - 1, total)
        high = max(low, min(end, total))
        if high > low:
            spans.append(SourceSpan(low + 1, tuple(lines[low:high])))
    return tuple(spans)


def _ranges_label(spans: Sequence[SourceSpan]) -> str:
    return ",".join(f"{span.start}:{span.end}" for span in spans)


def fresh_context_draft(path: str, display_path: str, lines: Sequence[str], center: int, producer: str = EDIT) -> SourceViewDraft:
    """A bounded fresh view of at most seven current lines around 0-based `center`."""
    if not lines:
        return SourceViewDraft(path, display_path, 0, (), producer)
    start = max(0, center - 3)
    end = min(len(lines), center + 4)
    span = SourceSpan(start + 1, tuple(lines[start:end]))
    return SourceViewDraft(path, display_path, len(lines), (span,), producer)


def fresh_context_block(path: str, display_path: str, lines: Sequence[str], center: int, producer: str = EDIT) -> SourceBlock:
    """A SourceBlock wrapping a fresh context draft, with plain markers."""
    draft = fresh_context_draft(path, display_path, lines, center, producer)
    markers = ("",) * sum(len(span.lines) for span in draft.spans)
    return SourceBlock(draft, markers)


def view_line(view: SourceView, line: int) -> str | None:
    """The exact view line at 1-based `line`, or None when not visible."""
    for span in view.spans:
        if span.start <= line <= span.end:
            return span.lines[line - span.start]
    return None
