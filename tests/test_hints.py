"""Unit tests for the idle-input hint mechanism (minacode/hints.py)."""

from minacode.hints import HINTS, Context, HintPicker

DIFF = "/diff reviews recent edits"
SESSIONS = "/sessions resumes a past session"


def _pool_for(ctx: Context) -> list[str]:
    """Capture the weighted pool the picker draws from for a given context."""
    captured: dict[str, list[str]] = {}

    def capture(pool):
        captured["pool"] = list(pool)
        return pool[0]

    HintPicker(choice=capture).pick(ctx)
    return captured["pool"]


def test_pick_returns_an_applicable_hint():
    picker = HintPicker()
    ctx = Context(early=False, edited_round=None)
    assert picker.pick(ctx) in {hint.text for hint in HINTS}


def test_pick_is_stable_for_the_same_context():
    picker = HintPicker()
    ctx = Context(early=False, edited_round=None)
    assert picker.pick(ctx) == picker.pick(ctx)


def test_pick_rerolls_when_context_changes():
    picks = iter(["first", "second"])
    picker = HintPicker(choice=lambda pool: next(picks))
    ctx1 = Context(early=False, edited_round=1)
    assert picker.pick(ctx1) == "first"
    assert picker.pick(ctx1) == "first"  # cached within a context
    ctx2 = Context(early=False, edited_round=2)
    assert picker.pick(ctx2) == "second"  # a new editing round re-rolls


def test_sessions_hint_only_applies_when_early():
    sessions = next(hint for hint in HINTS if hint.text == SESSIONS)
    assert sessions.when is not None
    assert sessions.when(Context(early=True, edited_round=None))
    assert not sessions.when(Context(early=False, edited_round=None))


def test_mature_pool_excludes_navigation_and_diff():
    pool = _pool_for(Context(early=False, edited_round=None))
    assert SESSIONS not in pool
    assert DIFF not in pool


def test_early_pool_includes_sessions_but_not_diff():
    pool = _pool_for(Context(early=True, edited_round=None))
    assert SESSIONS in pool
    assert DIFF not in pool


def test_editing_pool_weights_diff_high():
    diff_hint = next(hint for hint in HINTS if hint.text == DIFF)
    assert diff_hint.weight > 1
    pool = _pool_for(Context(early=False, edited_round=3))
    assert pool.count(DIFF) == diff_hint.weight
    assert SESSIONS not in pool  # a mature session
