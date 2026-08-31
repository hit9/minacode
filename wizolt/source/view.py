"""Source views: the immutable record of the lines a tool actually showed the model.

A view is evidence, not authority. Everything here is behavior of that evidence -- what it
contains, which line ranges it can vouch for, and what it says when asked for a range it never
projected -- so it lives on the values themselves rather than beside them. Nothing in this module
imports a runner, tool, context, or Session: views are built on worker threads and committed on
the main one, so they must be safe to construct anywhere.
"""

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

    def contains(self, start: int, end: int) -> bool:
        """True when the whole 1-based inclusive range lies inside this span."""
        return self.start <= start and end <= self.end

    def slice(self, start: int, end: int) -> tuple[str, ...]:
        """The exact lines for a 1-based inclusive range this span contains."""
        return self.lines[start - self.start : end - self.start + 1]

    @classmethod
    def build(cls, lines: Sequence[str], ranges: Sequence[tuple[int, int]]) -> tuple[SourceSpan, ...]:
        """Non-empty spans for 1-based inclusive `ranges` over `lines`, clamped and normalized.

        Ranges are sorted and merged when they overlap or touch, so the spans a view carries are
        exactly the spans its rendered `lines` label announces.
        """
        total = len(lines)
        spans: list[SourceSpan] = []
        for start, end in cls.merge(ranges):
            low = min(max(start, 1) - 1, total)
            high = max(low, min(end, total))
            if high > low:
                spans.append(cls(low + 1, tuple(lines[low:high])))
        return tuple(spans)

    @staticmethod
    def merge(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
        """Sort 1-based inclusive ranges and merge those that overlap or touch."""
        ordered = sorted((start, end) for start, end in ranges if start <= end)
        merged: list[tuple[int, int]] = []
        for start, end in ordered:
            if not merged or start > merged[-1][1] + 1:
                merged.append((start, end))
            elif end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        return merged


@dataclass(frozen=True)
class SourceViewDraft:
    """Immutable source evidence produced by a read-only tool before a public id is assigned."""

    path: str  # canonical resolved path
    display_path: str  # stable model-facing path
    total_lines: int
    spans: tuple[SourceSpan, ...]
    producer: str  # Read, Search, InspectCode, or Edit

    @property
    def line_count(self) -> int:
        """How many lines this draft actually projects."""
        return sum(len(span.lines) for span in self.spans)

    def ranges_label(self) -> str:
        """The `lines="..."` value naming every visible span."""
        return ",".join(f"{span.start}:{span.end}" for span in self.spans)

    @classmethod
    def around(cls, path: str, display_path: str, lines: Sequence[str], center: int, producer: str = EDIT) -> SourceViewDraft:
        """A bounded fresh draft of at most seven current lines around 0-based `center`.

        This is what a failed Edit hands back, so the model can retry against current text without
        spending a whole Read on it.
        """
        if not lines:
            return cls(path, display_path, 0, (), producer)
        start = max(0, center - 3)
        end = min(len(lines), center + 4)
        return cls(path, display_path, len(lines), (SourceSpan(start + 1, tuple(lines[start:end])),), producer)


@dataclass(frozen=True)
class SourceView:
    """A draft committed by the runner under a public `view.N` key.

    The key names exactly one immutable view for the life of a Session. `total_lines` is the
    captured file's line count, not the current one: a view describes what was shown, and the
    difference between that and the file now is precisely what Edit has to validate.
    """

    key: str  # view.N
    path: str
    display_path: str
    total_lines: int
    spans: tuple[SourceSpan, ...]
    producer: str
    round: int
    step: int

    @staticmethod
    def make_key(number: int) -> str:
        """The public id for view number `number`."""
        return f"view.{number}"

    @staticmethod
    def parse_key(key: str) -> int | None:
        """The numeric part of a public view id, or None for a malformed key."""
        match = _VIEW_KEY_RE.fullmatch(key)
        return int(match.group(1)) if match else None

    def draft(self) -> SourceViewDraft:
        return SourceViewDraft(self.path, self.display_path, self.total_lines, self.spans, self.producer)

    def ranges_label(self) -> str:
        return self.draft().ranges_label()

    def line(self, line: int) -> str | None:
        """The exact line this view shows at 1-based `line`, or None when it showed none."""
        for span in self.spans:
            if span.start <= line <= span.end:
                return span.lines[line - span.start]
        return None

    def range_lines(self, start: int, end: int) -> tuple[str, ...]:
        """The complete old line sequence for a replace/delete target.

        Raises `source range unseen` unless the whole 1-based inclusive range sits inside one
        contiguous visible span: a range that crosses a gap was never shown, and a target the
        model did not see is a target it cannot prove.
        """
        if start < 1 or end < start or end > self.total_lines:
            raise source_error(SOURCE_RANGE_UNSEEN, f"lines {start}:{end} are outside view {self.key}")
        for span in self.spans:
            if span.contains(start, end):
                return span.slice(start, end)
        raise source_error(
            SOURCE_RANGE_UNSEEN, f"lines {start}:{end} were not projected in {self.key}; only {self.ranges_label() or '(empty file)'} is visible"
        )

    def witness(self, line: int, after: bool) -> tuple[tuple[str, ...], int, int]:
        """The exact boundary witness for an insertion, as (lines, boundary_index, view_index).

        An insertion has no content of its own to validate, so it is proved by the lines around
        it: the anchor boundary plus up to two visible lines on each side. `boundary_index` counts
        how many witness lines precede the insertion point, and `view_index` is the 0-based
        insertion index in the view's own coordinates.
        """
        if self.total_lines == 0:
            if not after or line != 0:
                raise source_error(SOURCE_RANGE_UNSEEN, f"{self.key} represents an empty file; only insert_after with line 0 is valid")
            return (), 0, 0
        if line < 1 or line > self.total_lines:
            raise source_error(SOURCE_RANGE_UNSEEN, f"line {line} is outside view {self.key}")
        span = next((span for span in self.spans if span.start <= line <= span.end), None)
        if span is None:
            raise source_error(SOURCE_RANGE_UNSEEN, f"line {line} was not projected in {self.key}; only {self.ranges_label()} is visible")
        low = max(span.start, line - 2)
        high = min(span.end, line + 2)
        lines = span.slice(low, high)
        if after:
            return lines, line - low + 1, line  # after the anchor line itself
        return lines, line - low, line - 1  # before the anchor line itself
