"""Edit target resolution: extract claimed ranges and relocate them exactly in the current file."""

from __future__ import annotations

from collections.abc import Sequence

from wizolt.source.values import MAX_VIEW_DRIFT, SOURCE_RANGE_UNSEEN, SourceView, _ranges_label, source_error


def range_lines(view: SourceView, start: int, end: int) -> tuple[str, ...]:
    """The complete old line sequence for a replace/delete target.

    Raises `source range unseen` when the requested 1-based inclusive range is not wholly inside
    one contiguous visible span of the view.
    """
    if start < 1 or end < start or end > view.total_lines:
        raise source_error(SOURCE_RANGE_UNSEEN, f"lines {start}:{end} are outside view {view.key}")
    for span in view.spans:
        if span.start <= start and end <= span.end:
            return span.lines[start - span.start : end - span.start + 1]
    raise source_error(
        SOURCE_RANGE_UNSEEN, f"lines {start}:{end} were not projected in {view.key}; only {_ranges_label(view.spans) or '(empty file)'} is visible"
    )


def insertion_witness(view: SourceView, line: int, after: bool) -> tuple[tuple[str, ...], int, int]:
    """The exact boundary witness for an insertion, as (lines, boundary_index, original_index).

    `line` is the 1-based anchor line; `after` selects insert_after vs insert_before. The witness
    is the anchor boundary plus the immediately adjacent visible lines on both sides, up to two
    per side. `boundary_index` is how many witness lines precede the insertion point, and
    `original_index` is the 0-based insertion index in the current file.
    """
    if view.total_lines == 0:
        if not after or line != 0:
            raise source_error(SOURCE_RANGE_UNSEEN, f"{view.key} represents an empty file; only insert_after with line 0 is valid")
        return (), 0, 0
    if line < 1 or line > view.total_lines:
        raise source_error(SOURCE_RANGE_UNSEEN, f"line {line} is outside view {view.key}")
    span = next((span for span in view.spans if span.start <= line <= span.end), None)
    if span is None:
        raise source_error(SOURCE_RANGE_UNSEEN, f"line {line} was not projected in {view.key}; only {_ranges_label(view.spans)} is visible")
    low = max(span.start, line - 2)
    high = min(span.end, line + 2)
    lines = span.lines[low - span.start : high - span.start + 1]
    if after:
        boundary = line - low + 1  # after the anchor line itself
        insertion_index = line  # 0-based: insert after index line-1
    else:
        boundary = line - low  # before the anchor line itself
        insertion_index = line - 1
    return lines, boundary, insertion_index


def same_position(lines: Sequence[str], index: int, target: Sequence[str]) -> bool:
    """True when the current `lines` equal `target` starting at 0-based `index`."""
    return list(lines[index : index + len(target)]) == list(target)


def relocate_target(lines: Sequence[str], original_index: int, target: Sequence[str], max_drift: int = MAX_VIEW_DRIFT) -> int | None:
    """Unique exact relocation of `target` within `max_drift` lines of its original start.

    Returns the 0-based position of the match, or None when there are zero or multiple candidates
    in the window. Never similarity, syntax, or partial hashes: only exact line sequences.
    """
    if not target:
        return None
    lo = max(0, original_index - max_drift)
    hi = min(len(lines) - len(target) + 1, original_index + max_drift + 1)
    candidates = [index for index in range(lo, hi) if list(lines[index : index + len(target)]) == list(target)]
    return candidates[0] if len(candidates) == 1 else None


def relocate_witness(lines: Sequence[str], original_boundary: int, witness: Sequence[str], boundary: int, max_drift: int = MAX_VIEW_DRIFT) -> int | None:
    """Unique exact relocation of an insertion boundary.

    Searches the whole file for the exact witness sequence and requires exactly one candidate
    whose boundary (witness start + `boundary`) is within `max_drift` lines of the original
    boundary. Returns the relocated 0-based insertion index, or None.
    """
    if not witness:
        return None
    candidates = [index + boundary for index in range(len(lines) - len(witness) + 1) if list(lines[index : index + len(witness)]) == list(witness)]
    nearby = [index for index in candidates if abs(index - original_boundary) <= max_drift]
    return nearby[0] if len(nearby) == 1 else None
