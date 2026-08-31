"""Pure source-view relocation contracts, pinned at the function boundary.

Snapshot-backed editing only ever applies a target that matches exactly once inside the drift
window. Those functions are exercised through Edit indirectly in test_edit_source_views.py;
this file pins the contract -- empty targets, out-of-file originals, the drift edge, and
witness boundary semantics -- that the integration tests cannot reach.
"""

from wizolt.source import MAX_VIEW_DRIFT, relocate_target, relocate_witness


def test_relocate_target_is_unique_exact_and_drift_bounded():
    """A target relocates only when its complete text has exactly one candidate inside the drift
    window. Missing targets, duplicates, and out-of-window matches are all None, never a guess;
    the window edge itself is inclusive."""
    lines = [f"l{index}\n" for index in range(1, 200)]
    lines[50:52] = ["a\n", "b\n"]

    # Unique candidate at offset 49 (within the 50-line window) resolves.
    assert relocate_target(lines, 1, ["a\n", "b\n"]) == 50
    # A candidate at offset 51 (one past the window edge) is refused.
    beyond = [f"l{index}\n" for index in range(1, 200)]
    beyond[52:54] = ["a\n", "b\n"]
    assert relocate_target(beyond, 1, ["a\n", "b\n"]) is None

    # The original position itself still resolves when nothing moved.
    assert relocate_target(lines, 50, ["a\n", "b\n"]) == 50

    # Empty targets and originals beyond the file never match.
    assert relocate_target(lines, 0, []) is None
    assert relocate_target(lines, len(lines) + 50, ["a\n"]) is None

    # Duplicates inside the window are refused rather than disambiguated by order.
    dup = ["l0\n"] + ["same\n"] * 4 + ["l9\n"]
    assert relocate_target(dup, 2, ["same\n"]) is None


def test_relocate_witness_applies_the_window_to_the_boundary():
    """An insertion is proved by the lines around its boundary. The witness relocates only when
    exactly one candidate boundary lands inside the drift window; the window applies to the
    boundary, not to the witness start, and a repeated witness far away cannot steal it."""
    lines = ["def a():\n", "    pass\n", "\n", "def b():\n", "    pass\n"]

    # `pass` repeats, but the witness carries its neighbour, so only the intended boundary matches.
    assert relocate_witness(lines, 1, ["def a():\n", "    pass\n"], 1) == 1
    # A prepended line shifts the same witness down one; the boundary follows it.
    assert relocate_witness(["\n"] + lines, 0, ["def a():\n", "    pass\n"], 1) == 2

    # A repeated witness whose boundary is not nearby is refused.
    assert relocate_witness(["pass\n", "pass\n", "pass\n"], 1, ["pass\n"], 0) is None
    far = ["x\n"] * (MAX_VIEW_DRIFT + 2) + ["w0\n", "w1\n"]
    assert relocate_witness(far, 0, ["w0\n", "w1\n"], 1) is None

    # An empty witness never matches.
    assert relocate_witness(lines, 1, [], 0) is None
