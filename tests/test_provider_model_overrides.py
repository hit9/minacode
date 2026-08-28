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


def test_a_declared_scale_is_what_the_model_offers():
    """The declaration replaces the catalog's answer for matching models, and it is a menu rather
    than a fold target: `/reason` offers exactly these, so each one reaches the wire as written."""
    assert entry().reasoning_choices() == ("off", "low", "medium", "high", "ultra")

    for level in ("low", "medium", "high", "ultra"):
        assert replace(entry(), reasoning=level).resolve().reasoning_effort == level

def test_a_declared_level_is_sent_as_written():
    assert entry(reasoning="ultra").resolve().reasoning_effort == "ultra"
    assert entry(reasoning="ultra").reasoning_effort() == "ultra"

@pytest.mark.parametrize(("stored", "aligned"), (("minimal", "low"), ("xhigh", "high"), ("max", "high")))
def test_an_effort_off_the_declared_scale_is_moved_onto_it(stored, aligned):
    """Only reachable by carrying an effort over from another model — the picker cannot produce
    one. It lands on the nearest level the declaration names that minacode also knows."""
    assert replace(entry(), reasoning=stored).normalized_reasoning() == aligned

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

def test_a_scale_of_names_minacode_knows_none_of_is_offered_as_written():
    """Nothing has to rank `cheap` against `deep`: they are the menu, and one of them is chosen."""
    provider = ProviderConfig.from_dict({"url": "https://gw.example/v1", "model": "m", "models": {"m": {"reasoning": ["cheap", "normal", "deep"]}}})

    assert provider.reasoning_choices() == ("off", "cheap", "normal", "deep")
    assert replace(provider, reasoning="deep").resolve().reasoning_effort == "deep"
    # An effort carried over from another model has no comparable rank on this scale, so it lands
    # in the middle rather than on a guessed one.
    assert replace(provider, reasoning="max").normalized_reasoning() == "normal"

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

def test_a_catalogued_model_offers_the_levels_it_documents():
    """The scale minacode ships is the endpoint's own: DeepSeek documents low/high/max, and the
    compatibility spellings it also accepts (medium, xhigh) both resolve to high server-side, so
    offering them would be offering two choices that do the same thing."""
    deepseek = ProviderConfig(url="https://api.deepseek.com", key="k", model="deepseek-v4-flash")

    assert deepseek.reasoning_choices() == ("off", "low", "high", "max")

def test_a_model_the_catalog_says_nothing_about_keeps_the_full_scale():
    """Unknown means unconstrained: an endpoint minacode has no evidence for must not have its
    choices narrowed on a guess."""
    unknown = ProviderConfig(url="https://gw.example/v1", key="k", model="custom-model")

    assert unknown.reasoning_choices() == ("off", "minimal", "low", "medium", "high", "xhigh", "max")

def test_switching_to_a_model_without_the_stored_level_moves_it_onto_that_model_s_scale():
    stored = ProviderConfig(url="https://api.openai.com/v1", key="k", model="gpt-5", reasoning="minimal")
    assert stored.reasoning_choices() == ("off", "minimal", "low", "medium", "high")
    assert stored.resolve().reasoning_effort == "minimal"

    on_deepseek = replace(stored, url="https://api.deepseek.com", model="deepseek-v4-flash")
    assert on_deepseek.normalized_reasoning() == "low"

def test_a_model_that_always_reasons_does_not_offer_off():
    """Grok, Kimi K3 and GLM-5.3 document that reasoning cannot be turned off. Offering `off`
    anyway would be the menu promising something the endpoint does not do."""
    for url, model in (
        ("https://api.x.ai/v1", "grok-4.6"),
        ("https://api.moonshot.ai/v1", "kimi-k3"),
        ("https://api.z.ai/api/paas/v4", "glm-5.3"),
        ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.5-flash"),
    ):
        assert "off" not in ProviderConfig(url=url, key="k", model=model).reasoning_choices(), model

    # Gemini 2.5 documents `none`, so it keeps the choice.
    gemini_25 = ProviderConfig(url="https://generativelanguage.googleapis.com/v1beta/openai", key="k", model="gemini-2.5-flash")
    assert gemini_25.reasoning_choices()[0] == "off"

def test_off_stored_against_an_always_reasoning_model_moves_to_its_weakest_level():
    """The request reasons either way, so leaving the setting on `off` would show an effort the
    model is not spending."""
    grok = ProviderConfig(url="https://api.x.ai/v1", key="k", model="grok-4.6", reasoning="off")

    assert grok.normalized_reasoning() == "low"

def test_kimi_k3_still_sends_its_closest_to_off_spelling_when_a_config_names_off():
    """`off` is not offered, but a config can still name it, and the open platform documents the
    weakest level as the closest thing it has."""
    kimi = ProviderConfig(url="https://api.moonshot.ai/v1", key="k", model="kimi-k3", reasoning="off")

    assert kimi.resolve().reasoning_effort == "low"
