"""Catalog-wide invariants that do not duplicate its provider/model fact table."""

import re

from wizolt.providers.catalog import decode_bundled

EFFORTS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")


def test_anything_that_narrows_a_menu_cites_a_page_for_it():
    """A shortened `/reason` menu must account for itself with checkable evidence."""
    snapshot = decode_bundled()

    narrowing = [rule for rule in snapshot.model_rules if "reasoning.levels" in rule.set or rule.set.get("reasoning.mandatory")]
    for provider in snapshot.providers:
        narrowing.extend(rule for rule in provider.model_rules if "reasoning.levels" in rule.set or rule.set.get("reasoning.mandatory"))

    assert narrowing
    for entry in narrowing:
        selector = "/".join(entry.selector.prefixes) or entry.id
        why = entry.why
        assert why, selector
        assert any(str(evidence).startswith("https://") for evidence in entry.evidence), selector
        assert "\n" not in why and len(why) <= 80, selector
        # A restated level list is a second fact copy beside `reasoning.levels` and can drift.
        level = "|".join(EFFORTS)
        assert not re.search(rf"\b(?:{level})/(?:{level})\b", why), selector
