"""per-model declarations in `[provider.X.models]`: the effort scale a config states for a model."""
from dataclasses import replace

import pytest

from minacode.base import ConfigError
from minacode.config import ProviderConfig


def entry(**overrides) -> ProviderConfig:
    data = {
        "url": "https://gw.example/v1",
        "key": "k",
        "model": "gpt-5.6-fast",
        "models": {"gpt-5.6*": {"reasoning": ["low", "medium", "high", "ultra"]}},
    }
    return ProviderConfig.from_dict({**data, **overrides})


@pytest.mark.parametrize(
    ("chosen", "sent"),
    (("minimal", "low"), ("low", "low"), ("medium", "medium"), ("high", "high"), ("xhigh", "ultra"), ("max", "ultra")),
)
def test_a_declared_scale_replaces_the_catalog_guess_for_matching_models(chosen, sent):
    """Declaration order places a level minacode cannot name: `ultra` sits above `high`, so the two
    efforts above `high` land on it rather than folding back down to a level minacode knows."""
    assert replace(entry(), reasoning=chosen).resolve().reasoning_effort == sent

def test_a_declared_level_is_sent_as_written():
    assert entry(reasoning="ultra").resolve().reasoning_effort == "ultra"
    assert entry(reasoning="ultra").reasoning_effort() == "ultra"

def test_declarations_apply_only_to_models_their_glob_matches():
    """An entry switched to another model with /model keeps the catalog's own answer."""
    provider = replace(entry(reasoning="max"), model="gpt-4o")
    assert provider.declared_levels() == ()
    assert provider.resolve().reasoning_effort == "max"

def test_the_first_matching_glob_wins_like_a_catalog_rule():
    provider = ProviderConfig.from_dict(
        {
            "url": "https://gw.example/v1",
            "model": "gpt-5.6-fast",
            "models": {"gpt-5.6-fast": {"reasoning": ["low"]}, "gpt-5.6*": {"reasoning": ["high"]}},
        }
    )
    assert provider.declared_levels() == ("low",)

def test_a_scale_of_names_minacode_knows_none_of_still_orders_by_declaration():
    provider = ProviderConfig.from_dict({"url": "https://gw.example/v1", "model": "m", "models": {"m": {"reasoning": ["cheap", "normal", "deep"]}}})

    assert replace(provider, reasoning="minimal").resolve().reasoning_effort == "cheap"
    assert replace(provider, reasoning="medium").resolve().reasoning_effort == "normal"
    assert replace(provider, reasoning="max").resolve().reasoning_effort == "deep"

def test_only_a_declared_level_widens_what_reasoning_accepts():
    """`ultra` is a valid effort because a model declares it; an undeclared word stays a typo."""
    with pytest.raises(ConfigError):
        entry(reasoning="ultraa")
    with pytest.raises(ConfigError):
        ProviderConfig.from_dict({"url": "https://gw.example/v1", "model": "m", "reasoning": "ultra"})

def test_a_malformed_declaration_is_a_config_error():
    for models in ({"m": ["low"]}, {"m": {"reasoning": [""]}}, {"m": {"reasoning": 3}}):
        with pytest.raises(ConfigError):
            ProviderConfig.from_dict({"url": "https://gw.example/v1", "model": "m", "models": models})

    assert ProviderConfig.from_dict({}).model_overrides == ()

def test_declarations_follow_the_entry_into_its_copies():
    """Worker and compaction entries are dataclasses.replace copies, so a declaration reaches the
    requests they make without being configured a second time."""
    worker = replace(entry(reasoning="ultra"), model="gpt-5.6-mini")
    assert worker.resolve().reasoning_effort == "ultra"
