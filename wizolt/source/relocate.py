"""Exact matching of a view's target against the lines a file currently has.

These are the only functions here that look at current file content rather than at a view, which
is why they are functions over `list[str]` and not methods on anything: the file is not a value
this package owns. Matching is exact and bounded by construction -- never similarity, syntax,
embeddings, or partial hashes -- because a plausible target is not proof, and the cost of being
wrong is a silently corrupted file.
"""

from __future__ import annotations

from collections.abc import Sequence

from wizolt.source.view import MAX_VIEW_DRIFT


def same_position(lines: Sequence[str], index: int, target: Sequence[str]) -> bool:
    """True when the current `lines` equal `target` starting at 0-based `index`."""
    return list(lines[index : index + len(target)]) == list(target)


def context_matches(lines: Sequence[str], index: int, target: Sequence[str], before: Sequence[str], after: Sequence[str]) -> bool:
    """True when the lines a view showed beside `target` still sit beside it at 0-based `index`.

    `before`/`after` are the up-to-2 lines on each side of the target in the view. An empty side
    matches vacuously, so a target at the edge of its span needs no special case; the
    `index >= len(before)` guard is what keeps a negative slice from silently matching.
    """
    return index >= len(before) and same_position(lines, index - len(before), before) and same_position(lines, index + len(target), after)


def relocate_target(
    lines: Sequence[str],
    original_index: int,
    target: Sequence[str],
    max_drift: int = MAX_VIEW_DRIFT,
    before: Sequence[str] = (),
    after: Sequence[str] = (),
) -> int | None:
    """Unique exact relocation of `target` within `max_drift` lines of its original start.

    Returns the 0-based position of the match, or None when the window holds zero candidates
    (changed or removed) or several (ambiguous). Both are refused rather than guessed.

    `before`/`after` are a tiebreaker, never a requirement: consulted only when the window already
    holds several candidates, to narrow them to those whose neighbours still match. A single
    candidate is resolved even when its neighbours have since changed, so the surrounding lines can
    only shrink a candidate set, never disqualify the one match there is.
    """
    if not target:
        return None
    low = max(0, original_index - max_drift)
    high = min(len(lines) - len(target) + 1, original_index + max_drift + 1)
    candidates = [index for index in range(low, high) if same_position(lines, index, target)]
    if len(candidates) > 1 and (before or after):
        candidates = [index for index in candidates if context_matches(lines, index, target, before, after)]
    return candidates[0] if len(candidates) == 1 else None
