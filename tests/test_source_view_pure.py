"""Pure source-view relocation contracts, pinned at the function boundary.

Snapshot-backed editing only ever applies a target that matches exactly once inside the drift
window. Those functions are exercised through Edit indirectly in test_edit_source_views.py;
this file pins the contract -- empty targets, out-of-file originals, the drift edge, and the
view-side context accessor -- that the integration tests cannot reach.
"""

from wizolt.source import EDIT, SourceSpan, SourceView, context_matches, relocate_target


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


def test_relocate_target_context_resolves_an_ambiguous_candidate():
    """When the window holds several exact matches, the lines beside the target narrow the set: a
    candidate whose neighbours match both sides is the one that resolves."""
    lines = ["w\n", "x\n", "z\n", "x\n", "q\n"]

    assert relocate_target(lines, 1, ["x\n"], before=["w\n"], after=["z\n"]) == 1
    assert relocate_target(lines, 3, ["x\n"], before=["z\n"], after=["q\n"]) == 3


def test_relocate_target_context_matching_several_candidates_still_refuses():
    """Context narrows but never fabricates: when the whole neighbourhood repeats, two candidates
    survive the filter and the relocation is still refused rather than guessed."""
    lines = ["p\n", "x\n", "q\n", "p\n", "x\n", "q\n"]

    assert relocate_target(lines, 2, ["x\n"], before=["p\n"], after=["q\n"]) is None


def test_relocate_target_context_is_a_tiebreaker_not_a_requirement():
    """A single candidate resolves even when its neighbours have since changed: context is only
    consulted to narrow a set of several, so it can never reject the one match there is."""
    lines = ["w\n", "x\n", "Q\n"]

    assert relocate_target(lines, 1, ["x\n"], before=["w\n"], after=["z\n"]) == 1


def test_relocate_target_context_at_a_span_edge_uses_the_remaining_side():
    """A target at the edge of its span has one empty context side; the surviving side alone
    still narrows, and the empty side matches vacuously."""
    lines = ["x\n", "z\n", "x\n", "q\n"]

    assert relocate_target(lines, 0, ["x\n"], after=["z\n"]) == 0
    assert relocate_target(lines, 2, ["x\n"], before=["z\n"]) == 2


def test_context_matches_is_exact_and_guards_the_leading_edge():
    """The predicate behind both the in-place check and the candidate filter: each side must sit
    exactly where the view showed it, an empty side is vacuously satisfied, and a target too close
    to the file's start for its `before` lines fails instead of matching a negative slice."""
    lines = ["a\n", "x\n", "b\n"]

    assert context_matches(lines, 1, ["x\n"], ["a\n"], ["b\n"])
    assert not context_matches(lines, 1, ["x\n"], ["A\n"], ["b\n"])  # before differs
    assert not context_matches(lines, 1, ["x\n"], ["a\n"], ["B\n"])  # after differs
    assert context_matches(lines, 1, ["x\n"], (), ())  # no context: vacuous
    assert not context_matches(lines, 0, ["a\n"], ["z\n"], ["x\n"])  # no room for `before`
    assert not context_matches(lines, 2, ["b\n"], ["x\n"], ["tail\n"])  # `after` past the end


def view_of(*ranges):
    """A view over `lines` showing only `ranges` (1-based inclusive), as spans of their own text."""
    lines = [f"l{index}\n" for index in range(1, 11)]
    return SourceView("view.1", "f.py", "f.py", len(lines), SourceSpan.build(lines, ranges), EDIT, 0, 0)


def test_neighbors_are_the_shown_lines_beside_a_range_clipped_to_its_span():
    """The context a relocation gets: up to two shown lines on each side, never reaching past the
    span's own edge, and empty on a side where the range touches that edge."""
    view = view_of((3, 8))

    assert view.neighbors(5, 6) == (("l3\n", "l4\n"), ("l7\n", "l8\n"))
    assert view.neighbors(4, 4) == (("l3\n",), ("l5\n", "l6\n"))  # only one line above inside the span
    assert view.neighbors(3, 3) == ((), ("l4\n", "l5\n"))  # at the span's first line
    assert view.neighbors(8, 8) == (("l6\n", "l7\n"), ())  # at the span's last line
    assert view.neighbors(3, 8) == ((), ())  # the whole span is the range


def test_neighbors_of_a_range_no_span_contains_is_empty_rather_than_an_error():
    """`range_lines` has already refused such a range before this is reached, so the accessor stays
    total and simply offers no context instead of raising a second, later error."""
    view = view_of((1, 2), (9, 10))

    assert view.neighbors(4, 5) == ((), ())  # inside the gap
    assert view.neighbors(2, 9) == ((), ())  # crosses the gap
