"""Net diffs: what a turn or a session changed, reconstructed from recorded edits.

A pure algorithm over `TurnDiff` records and the files on disk -- unified-diff generation,
forward/reverse hunk application, legacy reconstruction, and rename detection. It reads `cwd` as
an argument and nothing else from the session, which is why it lives beside `Session` rather than
on it: the session owns the records, this owns what they mean.

`Session.latest_round_diff_sections` and `Session.session_diff_sections` are the two entry points
the UI uses; everything else here is how they are computed.
"""

from __future__ import annotations

import difflib
import os
import re
from typing import TYPE_CHECKING

from wizolt.base import split_lines

if TYPE_CHECKING:
    from wizolt.session import TurnDiff


def net_diff_for_path(status: str, path: str, before: str, after: str) -> tuple[str, str, str] | None:
    if before == after:
        return None
    text = "".join(difflib.unified_diff(split_lines(before), split_lines(after), fromfile="/dev/null" if not before else path, tofile=path))
    return (status, path, text) if text else None


def net_diff_sections(diffs: list[TurnDiff], status: str, *, cwd: str = "") -> list[tuple[str, str, str]]:
    states: dict[str, tuple[str, str]] = {}
    legacy: dict[str, list[str]] = {}
    # Whether the most recent edit to each path carried snapshots. A path can hold both kinds
    # when a file grows past the snapshot size limit partway through a session, and the two
    # descriptions overlap — emitting both would repeat the file's changes.
    snapshot_tail: dict[str, bool] = {}
    paths: list[str] = []
    for diff in diffs:
        if diff.path not in paths:
            paths.append(diff.path)
        snapshot_tail[diff.path] = bool(diff.before or diff.after)
        if not diff.before and not diff.after:
            legacy.setdefault(diff.path, []).append(diff.diff)
            continue
        before, _ = states.get(diff.path, (diff.before, diff.after))
        states[diff.path] = (before, diff.after)

    # Bash can move a file between Edit calls. When one path's `.after` matches another path's
    # `.before` uniquely on both sides, that's the boundary of a move: merge into the target so
    # the logical history follows the file to its final path.
    while (move := _find_unambiguous_move(states, legacy)) is not None:
        source, target = move
        states[target] = (states[source][0], states[target][1])
        del states[source]

    sections = []
    for path in paths:
        chunk = net_diff_chunk(path, status, states, legacy, snapshot_tail, cwd)
        if chunk:
            sections.append((status, path, chunk.rstrip("\n") + "\n"))
    return sections


def net_diff_chunk(
    path: str,
    status: str,
    states: dict[str, tuple[str, str]],
    legacy: dict[str, list[str]],
    snapshot_tail: dict[str, bool],
    cwd: str,
) -> str:
    """One diff per path, from exactly one description of its history."""
    if path in states and snapshot_tail.get(path):
        # The last edit carried snapshots, so the recorded `after` is the file's final content.
        before, after = states[path]
        if legacy_chunks := legacy.get(path, []):
            # Snapshots cover only a suffix: snapshot-less edits ran before the first snapshot
            # (the file shrank past the limit mid-session), and their starting content isn't in
            # `states`. Walk their hunks back from the first snapshot's `before` to recover it so
            # the net diff spans the whole path. If they don't apply cleanly — they were
            # interleaved between snapshots, so the snapshot span already reflects them, or the
            # file was mutated outside Edit — the snapshot span stands as-is.
            original = _reverse_apply(before, legacy_chunks)
            if original is not None:
                before = original
        section = net_diff_for_path(status, path, before, after)
        return section[2] if section else ""
    if path in states and not snapshot_tail.get(path):
        # Snapshots stop partway through the path's history (the file grew past the limit); the
        # starting content is still known exactly. The end state is the file's current on-disk
        # content; if the file is gone, forward-apply the trailing snapshot-less hunks onto the
        # last snapshot's `after` to recover it, so the exactly-known snapshot history isn't
        # discarded. If neither is available, fall through to the raw-hunks fallback below.
        final = _current_content(cwd, path)
        if final is None:
            final = _forward_apply(states[path][1], legacy.get(path, []))
        if final is not None:
            section = net_diff_for_path(status, path, states[path][0], final)
            return section[2] if section else ""
    legacy_chunks = legacy.get(path, [])
    if not legacy_chunks:
        return ""
    # No usable snapshots for this file. Best effort: reconstruct the pre-edit content by
    # reverse-applying the recorded per-Edit hunks to the file's current on-disk state, then emit
    # one clean synthesized diff. Falls back to the raw per-Edit hunks concatenated when
    # reconstruction can't uniquely locate a hunk (e.g. the file was mutated outside Edit).
    reconstructed = _reconstruct_legacy_diff(cwd, path, legacy_chunks, status) if cwd else None
    if reconstructed is not None:
        return reconstructed
    return "\n".join(chunk.rstrip("\n") for chunk in legacy_chunks)


def _current_content(cwd: str, path: str) -> str | None:
    if not cwd:
        return None
    abspath = path if os.path.isabs(path) else os.path.join(cwd, path)
    try:
        with open(abspath, encoding="utf-8") as file:
            return file.read()
    except (OSError, UnicodeDecodeError):
        return None


_HUNK_RE: re.Pattern[str] = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")


def _reverse_apply(current: str, chunks: list[str]) -> str | None:
    """Walk `current` back to the state before the given per-Edit hunks by reverse-applying them
    in reverse chronological order. Each hunk's after-text must occur uniquely in the buffer; if
    not (external mutation, ambiguous context, or hunks that don't belong to this buffer's
    history), return None so the caller can fall back."""
    hunk_pairs: list[tuple[str, str]] = []
    for chunk in chunks:
        pairs = _split_hunks(chunk)
        if pairs is None:
            return None
        hunk_pairs.extend(pairs)
    for after_text, before_text in reversed(hunk_pairs):
        if not after_text or not before_text:
            return None
        if current.count(after_text) != 1:
            return None
        current = current.replace(after_text, before_text, 1)
    return current


def _forward_apply(current: str, chunks: list[str]) -> str | None:
    """Apply the given per-Edit hunks forward to `current` in chronological order, deriving the
    content they produce. Each hunk's before-text must occur uniquely in the buffer; if not
    (external mutation or ambiguous context), return None so the caller can fall back. The mirror
    of `_reverse_apply`: used to recover a file's final content from its last snapshot when the
    file is no longer on disk."""
    hunk_pairs: list[tuple[str, str]] = []
    for chunk in chunks:
        pairs = _split_hunks(chunk)
        if pairs is None:
            return None
        hunk_pairs.extend(pairs)
    for after_text, before_text in hunk_pairs:
        if not after_text or not before_text:
            return None
        if current.count(before_text) != 1:
            return None
        current = current.replace(before_text, after_text, 1)
    return current


def _reconstruct_legacy_diff(cwd: str, path: str, chunks: list[str], status: str) -> str | None:
    final = _current_content(cwd, path)
    if final is None:
        return None
    original = _reverse_apply(final, chunks)
    if original is None:
        return None
    section = net_diff_for_path(status, path, original, final)
    return section[2] if section else ""


def _split_hunks(chunk: str) -> list[tuple[str, str]] | None:
    pairs: list[tuple[str, str]] = []
    before_lines: list[str] | None = None
    after_lines: list[str] | None = None
    for line in chunk.splitlines():
        if line.startswith(("--- ", "+++ ")):
            continue
        if _HUNK_RE.match(line):
            if before_lines is not None and after_lines is not None:
                pairs.append(("\n".join(after_lines), "\n".join(before_lines)))
            before_lines, after_lines = [], []
            continue
        if before_lines is None or after_lines is None:
            return None
        if line.startswith("+"):
            after_lines.append(line[1:])
        elif line.startswith("-"):
            before_lines.append(line[1:])
        elif line.startswith(" "):
            before_lines.append(line[1:])
            after_lines.append(line[1:])
        elif line == "\\ No newline at end of file":
            continue
        else:
            return None
    if before_lines is not None and after_lines is not None:
        pairs.append(("\n".join(after_lines), "\n".join(before_lines)))
    return pairs


def _find_unambiguous_move(states: dict[str, tuple[str, str]], legacy: dict[str, list[str]]) -> tuple[str, str] | None:
    sources_by_after: dict[str, list[str]] = {}
    targets_by_before: dict[str, list[str]] = {}
    for path, (before, after) in states.items():
        if path in legacy:
            continue
        if after:
            sources_by_after.setdefault(after, []).append(path)
        if before:
            targets_by_before.setdefault(before, []).append(path)
    for content, sources in sources_by_after.items():
        targets = targets_by_before.get(content, [])
        if len(sources) == 1 and len(targets) == 1 and sources[0] != targets[0]:
            return sources[0], targets[0]
    return None
