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


def relocate_target(lines: Sequence[str], original_index: int, target: Sequence[str], max_drift: int = MAX_VIEW_DRIFT) -> int | None:
    """Unique exact relocation of `target` within `max_drift` lines of its original start.

    Returns the 0-based position of the match, or None when the window holds zero candidates
    (changed or removed) or several (ambiguous). Both are refused rather than guessed.
    """
    if not target:
        return None
    low = max(0, original_index - max_drift)
    high = min(len(lines) - len(target) + 1, original_index + max_drift + 1)
    candidates = [index for index in range(low, high) if same_position(lines, index, target)]
    return candidates[0] if len(candidates) == 1 else None


def relocate_witness(lines: Sequence[str], original_boundary: int, witness: Sequence[str], boundary: int, max_drift: int = MAX_VIEW_DRIFT) -> int | None:
    """Unique exact relocation of an insertion boundary.

    Searches for the exact witness sequence and requires exactly one candidate whose boundary
    (witness start + `boundary`) lands within `max_drift` lines of the original.
    """
    if not witness:
        return None
    candidates = [index + boundary for index in range(len(lines) - len(witness) + 1) if same_position(lines, index, witness)]
    nearby = [index for index in candidates if abs(index - original_boundary) <= max_drift]
    return nearby[0] if len(nearby) == 1 else None
