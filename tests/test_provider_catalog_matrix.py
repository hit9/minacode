"""What every catalogued model resolves to, host by host and effort by effort.

The catalog's model knowledge is spread across prefix rules, version patterns, and per-host
overlays, and `resolve()` folds all of it into a handful of values. Reorganizing that data is
supposed to move where a fact is written, not what it resolves to — but nothing else in the suite
would notice a rule that quietly stopped matching. This matrix is the net: one row per
host/model pair, every effort spelled out, so a reorganization has to state which rows it changes.
"""

import re
from dataclasses import replace

import pytest

from minacode.config import ProviderConfig
from minacode.providers.catalog import decode_bundled
from minacode.providers.compat import bundled_policy

EFFORTS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")

# (url, model, api, chat_reasoning, chat_reasoning_history, {effort: sent})
MATRIX = (
    ("https://api.openai.com/v1", "gpt-5.6", "chat", "reasoning_effort", "all", ("none", "low", "low", "medium", "high", "xhigh", "max")),
    ("https://api.openai.com/v1", "gpt-5.5", "chat", "reasoning_effort", "all", ("none", "low", "low", "medium", "high", "xhigh", "xhigh")),
    ("https://api.openai.com/v1", "gpt-5.4-pro", "chat", "reasoning_effort", "all", ("medium", "medium", "medium", "medium", "high", "xhigh", "xhigh")),
    ("https://api.openai.com/v1", "gpt-5.3-codex", "chat", "reasoning_effort", "all", ("low", "low", "low", "medium", "high", "xhigh", "xhigh")),
    ("https://api.openai.com/v1", "gpt-5.1", "chat", "reasoning_effort", "all", ("none", "low", "low", "medium", "high", "high", "high")),
    ("https://api.openai.com/v1", "gpt-5", "chat", "reasoning_effort", "all", (None, "minimal", "low", "medium", "high", "high", "high")),
    ("https://api.openai.com/v1", "gpt-5-pro", "chat", "reasoning_effort", "all", ("high", "high", "high", "high", "high", "high", "high")),
    ("https://api.openai.com/v1", "o3", "chat", "reasoning_effort", "all", (None, "low", "low", "medium", "high", "high", "high")),
    ("https://api.openai.com/v1", "gpt-4o", "chat", "off", "all", (None, "minimal", "low", "medium", "high", "xhigh", "max")),
    # OpenRouter normalizes every upstream behind its own reasoning object, so a model's native
    # spelling must not reach it. It is the one host that proves model traits cannot simply
    # outrank an endpoint's own declaration.
    ("https://openrouter.ai/api/v1", "deepseek-v4-flash", "chat", "reasoning", "all", (None, "minimal", "low", "medium", "high", "xhigh", "max")),
    ("https://openrouter.ai/api/v1", "gpt-5.5", "chat", "reasoning", "all", (None, "minimal", "low", "medium", "high", "xhigh", "max")),
    ("https://openrouter.ai/api/v1", "glm-5.2", "chat", "reasoning", "all", (None, "minimal", "low", "medium", "high", "xhigh", "max")),
    ("https://openrouter.ai/api/v1", "kimi-k3", "chat", "reasoning", "all", (None, "minimal", "low", "medium", "high", "xhigh", "max")),
    # `chat_reasoning` is now stated by the gpt-5 trait rather than by the openai host, so it is
    # filled in here too. OpenCode routes gpt-* to the Responses wire, which builds its own
    # reasoning parameters, so the value is recorded but never sent.
    ("https://opencode.ai/zen/v1", "gpt-5.5", "responses", "reasoning_effort", "all", ("none", "low", "low", "medium", "high", "xhigh", "xhigh")),
    ("https://opencode.ai/zen/v1", "deepseek-v4-flash", "chat", "thinking", "tool_calls", (None, "low", "low", "high", "high", "max", "max")),
    ("https://opencode.ai/zen/v1", "glm-5.2", "chat", "thinking_effort", "all", (None, "high", "high", "high", "high", "max", "max")),
    ("https://opencode.ai/zen/v1", "glm-5", "chat", "thinking_toggle", "all", (None, "minimal", "low", "medium", "high", "xhigh", "max")),
    ("https://opencode.ai/zen/v1", "kimi-k3", "chat", "reasoning_effort", "all", ("low", "low", "low", "high", "high", "max", "max")),
    ("https://opencode.ai/zen/v1", "claude-sonnet-4-6", "anthropic", "off", "all", (None, "low", "low", "medium", "high", "xhigh", "max")),
    ("https://api.deepseek.com", "deepseek-v4-flash", "chat", "thinking", "tool_calls", (None, "low", "low", "high", "high", "max", "max")),
    ("https://api.deepseek.com", "deepseek-chat", "chat", "thinking", "tool_calls", (None, "low", "low", "high", "high", "xhigh", "max")),
    (
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen3.8-max",
        "chat",
        "reasoning_effort",
        "current_turn",
        ("none", "low", "low", "medium", "xhigh", "xhigh", "xhigh"),
    ),
    # Two endpoints that re-host other vendors' models and give them a scale of their own, so the
    # host's rule outranks the model's trait — what that precedence exists for. Volcengine hands
    # every model the same four levels; Alibaba narrows DeepSeek and GLM to high/max.
    (
        "https://ark.cn-beijing.volces.com/api/v3",
        "deepseek-v4-flash-ga-260731",
        "chat",
        "thinking",
        "tool_calls",
        (None, "minimal", "low", "medium", "high", "high", "high"),
    ),
    (
        "https://ark.cn-beijing.volces.com/api/v3",
        "glm-5-2-260617",
        "chat",
        "thinking_toggle",
        "all",
        (None, "minimal", "low", "medium", "high", "high", "high"),
    ),
    (
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "deepseek-v4-flash",
        "chat",
        "thinking",
        "tool_calls",
        (None, "high", "high", "high", "high", "max", "max"),
    ),
    (
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen3-max",
        "chat",
        "off",
        "current_turn",
        (None, "minimal", "low", "medium", "high", "xhigh", "max"),
    ),
    ("https://api.moonshot.ai/v1", "kimi-k3", "chat", "reasoning_effort", "all", ("low", "low", "low", "high", "high", "max", "max")),
    ("https://api.moonshot.ai/v1", "kimi-k2.5", "chat", "thinking_toggle", "current_turn", (None, "low", "low", "high", "high", "max", "max")),
    ("https://api.moonshot.ai/v1", "kimi-k2.6", "chat", "thinking_toggle", "current_turn", (None, "low", "low", "high", "high", "max", "max")),
    ("https://api.moonshot.ai/v1", "kimi-k2.7-code", "chat", "mandatory_thinking", "all", (None, "low", "low", "high", "high", "max", "max")),
    ("https://api.moonshot.ai/v1", "moonshot-v1-8k", "chat", "off", "current_turn", (None, "low", "low", "high", "high", "max", "max")),
    ("https://api.kimi.com/v1", "k3", "chat", "reasoning_effort", "all", ("none", "low", "low", "high", "high", "max", "max")),
    ("https://api.kimi.com/v1", "kimi-for-coding", "chat", "mandatory_thinking", "all", (None, "low", "low", "high", "high", "max", "max")),
    ("https://api.z.ai/api/paas/v4", "glm-5.2", "chat", "thinking_effort", "current_turn", (None, "high", "high", "high", "high", "max", "max")),
    ("https://api.z.ai/api/paas/v4", "glm-5", "chat", "thinking_toggle", "current_turn", (None, "minimal", "low", "medium", "high", "xhigh", "max")),
    ("https://api.z.ai/api/paas/v4", "glm-4.6", "chat", "thinking_toggle", "current_turn", (None, "minimal", "low", "medium", "high", "xhigh", "max")),
    ("https://api.z.ai/api/paas/v4", "glm-4.5", "chat", "thinking_toggle", "current_turn", (None, "minimal", "low", "medium", "high", "xhigh", "max")),
    ("https://open.bigmodel.cn/api/paas/v4", "glm-5.2", "chat", "thinking_effort", "current_turn", (None, "high", "high", "high", "high", "max", "max")),
    ("https://open.bigmodel.cn/api/paas/v4", "glm-5", "chat", "thinking_toggle", "current_turn", (None, "minimal", "low", "medium", "high", "xhigh", "max")),
    # Claude 4.6 folds through resolve() like every model: its catalog rule narrows the scale to
    # low..max (minimal folds to low), and the adaptive-xhigh-as-max recipe maps xhigh to max on
    # the wire, so the sent effort matches what the Messages wire used to derive itself.
    (
        "https://api.anthropic.com/v1/messages",
        "claude-sonnet-4-6",
        "anthropic",
        "off",
        "all",
        (None, "low", "low", "medium", "high", "xhigh", "max"),
    ),
    # Families the catalog knows without knowing their endpoints: neither api.x.ai nor Google's
    # OpenAI-compatible host has a provider entry, and the models resolve anyway.
    ("https://api.x.ai/v1", "grok-4.6", "chat", "reasoning_effort", "all", (None, "low", "low", "medium", "high", "xhigh", "xhigh")),
    ("https://api.x.ai/v1", "grok-4.5", "chat", "reasoning_effort", "all", (None, "low", "low", "medium", "high", "high", "high")),
    (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-3.5-flash",
        "chat",
        "reasoning_effort",
        "all",
        (None, "minimal", "low", "medium", "high", "high", "high"),
    ),
    (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-3.1-pro",
        "chat",
        "reasoning_effort",
        "all",
        (None, "low", "low", "medium", "high", "high", "high"),
    ),
    # The one Gemini family that can stop reasoning, and the only row here with an `off` spelling.
    (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-2.5-flash",
        "chat",
        "reasoning_effort",
        "all",
        ("none", "low", "low", "medium", "high", "high", "high"),
    ),
    # GLM-5.3 is forced thinking, so it must not reach the `glm-5` family rule and be offered a
    # disable it does not honour.
    ("https://api.z.ai/api/paas/v4", "glm-5.3", "chat", "thinking_effort", "current_turn", (None, "low", "low", "high", "high", "max", "max")),
    # An uncatalogued gateway serving a catalogued model gets the model's own knowledge: the same
    # thinking format, replay rule, and effort scale it would get from the model's own vendor. Only
    # endpoint policy is missing, because only the endpoint is unknown.
    ("https://gw.example/v1", "deepseek-v4-flash", "chat", "thinking", "tool_calls", (None, "low", "low", "high", "high", "max", "max")),
    ("https://gw.example/v1", "gpt-5.5", "chat", "reasoning_effort", "all", ("none", "low", "low", "medium", "high", "xhigh", "xhigh")),
    ("https://gw.example/v1", "custom-model", "chat", "off", "all", (None, "minimal", "low", "medium", "high", "xhigh", "max")),
)


@pytest.mark.parametrize("case", MATRIX, ids=lambda case: f"{case[0].split('/')[2]}-{case[1]}")
def test_catalogued_models_resolve_to_their_recorded_wire_settings(case):
    url, model, api, chat_reasoning, history, sent = case
    provider = ProviderConfig(url=url, key="k", model=model)

    policy = bundled_policy()
    resolved = policy.resolve(provider)
    assert (resolved.api, resolved.chat_reasoning, resolved.chat_reasoning_history) == (api, chat_reasoning, history)
    assert tuple(policy.resolve(replace(provider, reasoning=effort)).reasoning_effort for effort in EFFORTS) == sent

def test_anything_that_narrows_a_menu_cites_a_page_for_it():
    """`/reason` shows this text under a shortened list, so it is the user's only account of why
    their model has three levels instead of six. A rule that narrows without one leaves that
    question unanswered on screen — and an entry nobody can source is one nobody can check."""
    snapshot = decode_bundled()

    narrowing = [
        rule
        for rule in snapshot.model_rules
        if "reasoning.levels" in rule.set or rule.set.get("reasoning.mandatory")
    ]
    for provider in snapshot.providers:
        narrowing.extend(
            rule
            for rule in provider.model_rules
            if "reasoning.levels" in rule.set or rule.set.get("reasoning.mandatory")
        )

    assert narrowing
    for entry in narrowing:
        selector = "/".join(entry.selector.prefixes) or entry.id
        why = entry.why
        assert why, selector
        assert any(str(evidence).startswith("https://") for evidence in entry.evidence), selector
        # One line, and short enough to sit under a modal list on an ordinary terminal.
        assert "\n" not in why and len(why) <= 80, selector
        # It says what the endpoint does, not which levels survived: a restated list is a second
        # copy of the field beside it and drifts from it. Only the enumerated form is caught here
        # — naming levels to say how one is served ("serves anything but high as max") is the
        # point of the line, and no check can tell that from a list except by its shape.
        level = "|".join(EFFORTS)
        assert not re.search(rf"\b(?:{level})/(?:{level})\b", why), selector

def test_every_catalogued_host_appears_in_the_matrix():
    """A new host must state what its models resolve to, or the net has a hole the size of it."""
    covered = {url.split("/")[2] for url, *_ in MATRIX}

    for provider in decode_bundled().providers:
        assert any(host in domain or domain.endswith(host) for host in provider.hosts for domain in covered), provider.hosts
