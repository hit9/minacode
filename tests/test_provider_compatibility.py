"""provider compatibility (split from tests/test_core_logic.py)."""
import pytest
from test_core_logic import session

from minacode.config import (
    ProviderConfig,
)
from minacode.model import ModelClient


@pytest.mark.parametrize("model", ("o3", "o4-mini", "gpt-5.6"))
def test_openai_compatibility_recognizes_reasoning_model_families(model):
    provider = ProviderConfig(url="https://api.openai.com/v1", model=model)
    assert provider.resolve().chat_reasoning == "reasoning_effort"

def test_openai_compatibility_leaves_non_reasoning_chat_models_off():
    provider = ProviderConfig(url="https://api.openai.com/v1", model="gpt-4o")
    assert provider.resolve().chat_reasoning == "off"

def test_openai_compatibility_limits_responses_reasoning_to_reasoning_models():
    reasoning = ProviderConfig(url="https://api.openai.com/v1", model="gpt-5", api="responses")
    non_reasoning = ProviderConfig(url="https://api.openai.com/v1", model="gpt-4.1", api="responses")

    assert reasoning.resolve().responses_reasoning is True
    assert non_reasoning.resolve().responses_reasoning is False

@pytest.mark.parametrize(
    ("url", "model", "reasoning", "expected"),
    (
        ("https://api.openai.com/v1", "gpt-5.6-sol", "max", "max"),
        ("https://api.openai.com/v1", "gpt-5.6-terra", "minimal", "low"),
        ("https://api.openai.com/v1", "gpt-5.5", "minimal", "low"),
        ("https://api.openai.com/v1", "gpt-5.5", "max", "xhigh"),
        ("https://api.openai.com/v1", "gpt-5.5-pro", "low", "medium"),
        ("https://api.openai.com/v1", "gpt-5.4-pro", "max", "xhigh"),
        ("https://api.openai.com/v1", "gpt-5.2-pro", "minimal", "medium"),
        ("https://api.openai.com/v1", "gpt-5.3-codex", "max", "xhigh"),
        ("https://api.openai.com/v1", "gpt-5.2-codex", "minimal", "low"),
        ("https://api.openai.com/v1", "gpt-5.4-mini", "max", "xhigh"),
        ("https://api.openai.com/v1", "gpt-5.2", "minimal", "low"),
        ("https://api.openai.com/v1", "gpt-5.1", "xhigh", "high"),
        ("https://api.openai.com/v1", "gpt-5-pro", "minimal", "high"),
        ("https://api.openai.com/v1", "gpt-5", "max", "high"),
        ("https://api.openai.com/v1", "o3", "minimal", "low"),
        ("https://api.openai.com/v1", "o4-mini", "max", "high"),
        ("https://opencode.ai/zen/v1", "gpt-5.5", "max", "xhigh"),
        # Unknown future models stay on the generic pass-through path.
        ("https://api.openai.com/v1", "gpt-5.7", "max", "max"),
    ),
)
def test_openai_effort_uses_each_models_nearest_supported_level(url, model, reasoning, expected):
    provider = ProviderConfig(url=url, model=model, api="responses", reasoning=reasoning)

    assert provider.resolve().reasoning_effort == expected

@pytest.mark.parametrize(
    ("model", "expected"),
    (("gpt-5.6-sol", "none"), ("gpt-5.5-pro", "medium"), ("gpt-5.3-codex", "low")),
)
def test_openai_reasoning_off_uses_the_models_lowest_supported_level(model, expected):
    provider = ProviderConfig(url="https://api.openai.com/v1", model=model, api="responses", reasoning="off")

    assert provider.resolve().reasoning_effort == expected

def test_opencode_routes_grok_through_responses_and_uses_its_documented_levels():
    """Routing is OpenCode's; the effort scale is Grok's, and it reaches OpenCode without OpenCode
    having to say anything about Grok."""
    provider = ProviderConfig(url="https://opencode.ai/zen/v1", model="grok-4.5", reasoning="high")

    resolved = provider.resolve()

    assert resolved.api == "responses"
    assert resolved.reasoning_effort == "high"
    assert provider.reasoning_choices() == ("low", "medium", "high")

@pytest.mark.parametrize(
    ("model", "reasoning", "chat_reasoning", "effort"),
    (
        ("deepseek-v4-flash", "medium", "thinking", "high"),
        ("glm-5.2", "xhigh", "thinking_effort", "max"),
        ("kimi-k3", "medium", "reasoning_effort", "high"),
    ),
)
def test_opencode_reuses_model_family_reasoning_capabilities(model, reasoning, chat_reasoning, effort):
    provider = ProviderConfig(url="https://opencode.ai/zen/v1", model=model, reasoning=reasoning)

    resolved = provider.resolve()

    assert resolved.chat_reasoning == chat_reasoning
    assert resolved.reasoning_effort == effort

@pytest.mark.parametrize("url", ("https://api.deepseek.com", "https://opencode.ai/zen/v1"))
@pytest.mark.parametrize(("reasoning", "expected"), (("minimal", "low"), ("medium", "high"), ("high", "high"), ("max", "max")))
def test_deepseek_effort_is_resolved_before_request_construction(url, reasoning, expected, tmp_path):
    provider = ProviderConfig(url=url, model="deepseek-v4-flash", reasoning=reasoning)

    assert provider.resolve().reasoning_effort == expected

    params = {}
    ModelClient(session(tmp_path)).apply_provider_params(params, provider)
    assert params == {"reasoning_effort": expected, "extra_body": {"thinking": {"type": "enabled"}}}

def test_unknown_provider_can_explicitly_pass_through_max_effort(tmp_path):
    client = ModelClient(session(tmp_path))
    provider = ProviderConfig(url="https://models.example/v1", model="future-reasoner", reasoning="max", chat_reasoning="reasoning_effort")
    params = {}

    client.apply_provider_params(params, provider)

    assert params == {"reasoning_effort": "max"}

def test_qwen_token_plan_compatibility_uses_reasoning_effort(tmp_path):
    client = ModelClient(session(tmp_path))
    provider = ProviderConfig.from_dict(
        {
            "url": "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            "model": "qwen3.8-max-preview",
            "reasoning": "medium",
        }
    )
    assert provider.resolve().chat_reasoning == "reasoning_effort"
    assert provider.resolve().chat_reasoning_history == "current_turn"

    # Qwen3.8-Max documents low/medium/xhigh; the two levels it has no spelling for fold to the
    # nearest one it does, which for both is xhigh.
    for reasoning, expected in (
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "xhigh"),
        ("xhigh", "xhigh"),
        ("max", "xhigh"),
    ):
        provider.reasoning = reasoning
        params = {}
        client.apply_provider_params(params, provider)
        assert params == {"reasoning_effort": expected}

    provider.reasoning = "off"
    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"reasoning_effort": "none"}

    provider.url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert provider.resolve().chat_reasoning == "reasoning_effort"

    # A lookalike domain is not the endpoint — but the model is still the model, and how it takes
    # reasoning is a fact about it, so the format follows the model name rather than the domain.
    provider.url = "https://notaliyuncs.com/compatible-mode/v1"
    assert provider.resolve().chat_reasoning == "reasoning_effort"
    assert provider.resolve().chat_reasoning_history == "all"

    provider.model = "other-model"
    assert provider.resolve().chat_reasoning == "off"

def test_kimi_compatibility_uses_model_native_reasoning_controls(tmp_path):
    client = ModelClient(session(tmp_path))
    provider = ProviderConfig(url="https://api.moonshot.ai/v1", model="kimi-k3", reasoning="medium", temperature=0.2)
    resolved = provider.resolve()
    assert resolved.chat_reasoning == "reasoning_effort"
    assert resolved.prompt_cache_key is True
    assert resolved.chat_reasoning_history == "all"
    assert client.prompt_cache_key(provider, None).startswith("minacode-")

    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"reasoning_effort": "high"}

    provider.reasoning = "max"
    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"reasoning_effort": "max"}

    provider.reasoning = "off"
    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"reasoning_effort": "low"}

    provider.model = "kimi-k2.6"
    assert provider.resolve().chat_reasoning_history == "current_turn"
    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"extra_body": {"thinking": {"type": "disabled"}}}

    provider.reasoning = "low"
    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"extra_body": {"thinking": {"type": "enabled"}}}

    provider.model = "kimi-k2.7-code-highspeed"
    params = {}
    client.apply_provider_params(params, provider)
    assert params == {}

    provider.url = "https://api.moonshot.cn/v1"
    assert provider.resolve().chat_reasoning == "mandatory_thinking"

def test_kimi_code_compatibility_is_distinct_from_open_platform(tmp_path):
    client = ModelClient(session(tmp_path))
    provider = ProviderConfig(url="https://api.kimi.com/coding/v1", model="k3", reasoning="medium", temperature=0.2)
    resolved = provider.resolve()
    assert resolved.chat_reasoning == "reasoning_effort"
    assert resolved.prompt_cache_key is True
    assert resolved.chat_reasoning_history == "all"
    assert client.prompt_cache_key(provider, None).startswith("minacode-")

    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"temperature": 0.2, "reasoning_effort": "high"}

    provider.reasoning = "off"
    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"temperature": 0.2, "reasoning_effort": "none"}

    provider.model = "kimi-for-coding-highspeed"
    provider.reasoning = "high"
    params = {}
    client.apply_provider_params(params, provider)
    assert provider.resolve().chat_reasoning == "mandatory_thinking"
    assert params == {"temperature": 0.2}

@pytest.mark.parametrize("url", ("https://api.z.ai/api/paas/v4", "https://open.bigmodel.cn/api/paas/v4"))
# GLM-5.2 documents high and max only, and resolves anything that is not "high" to max. Sending an
# unfolded level would buy the most expensive setting for a request that asked for the cheapest, so
# everything at or below high folds to high — its low end — and the rest folds up to max.
@pytest.mark.parametrize(
    ("reasoning", "expected"),
    (("minimal", "high"), ("low", "high"), ("medium", "high"), ("high", "high"), ("xhigh", "max"), ("max", "max")),
)
def test_zai_regional_endpoints_share_documented_reasoning_effort(url, reasoning, expected, tmp_path):
    client = ModelClient(session(tmp_path))
    provider = ProviderConfig(url=url, model="glm-5.2", reasoning=reasoning, temperature=0.6)
    resolved = provider.resolve()
    assert resolved.chat_reasoning == "thinking_effort"
    assert resolved.prompt_cache_key is False
    assert resolved.chat_reasoning_history == "current_turn"
    assert client.prompt_cache_key(provider, None) == ""

    params = {}
    client.apply_provider_params(params, provider)
    assert params == {
        "temperature": 0.6,
        "reasoning_effort": expected,
        "extra_body": {"thinking": {"type": "enabled"}},
    }

    provider.reasoning = "off"
    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"temperature": 0.6, "extra_body": {"thinking": {"type": "disabled"}}}

@pytest.mark.parametrize("url", ("https://api.z.ai/api/paas/v4", "https://open.bigmodel.cn/api/paas/v4"))
def test_zai_older_reasoning_families_use_only_thinking_toggle(url, tmp_path):
    client = ModelClient(session(tmp_path))
    provider = ProviderConfig(url=url, model="glm-5.1", reasoning="high", temperature=0.6)
    assert provider.resolve().chat_reasoning == "thinking_toggle"

    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"temperature": 0.6, "extra_body": {"thinking": {"type": "enabled"}}}

@pytest.mark.parametrize(
    ("url", "model"),
    (
        ("https://api.moonshot.ai.evil.test/v1", "kimi-k3"),
        ("https://notmoonshot.cn/v1", "kimi-k3"),
        ("https://api.kimi.com.evil.test/coding/v1", "k3"),
        ("https://notz.ai/api/paas/v4", "glm-5.2"),
        ("https://notbigmodel.cn/api/paas/v4", "glm-5.2"),
    ),
)
def test_provider_compatibility_requires_a_real_domain_boundary(url, model):
    """A lookalike domain adopts none of the real endpoint's policy.

    What it does keep is the model's own reasoning format, which is deliberate: that is a fact
    about the model and applies wherever the model is served. The boundary exists to stop a
    domain from claiming an endpoint's caching, strictness, and tool contracts — the settings
    that describe the service — not to make a known model unrecognizable.
    """
    provider = ProviderConfig(url=url, model=model, reasoning="high", temperature=0.4)
    resolved = provider.resolve()

    assert resolved.prompt_cache_key is True
    assert resolved.strict_tools_active is False
    assert resolved.builtin_tools_by_wire is None
    assert ProviderConfig(url=url, model="other-model").resolve().chat_reasoning == "off"

def test_unknown_provider_resolution_stays_generic_and_explicit_values_win():
    provider = ProviderConfig(
        url="https://gateway.example/v1/responses",
        model="custom-model",
        api="chat",
        chat_reasoning="enable_thinking",
        reasoning="low",
        temperature=0.4,
        strict_tools=True,
    )

    resolved = provider.resolve()

    assert resolved.api == "chat"
    assert resolved.chat_reasoning == "enable_thinking"
    assert resolved.reasoning_effort == "low"
    assert resolved.suppress_temperature is True
    assert resolved.prompt_cache_key is True
    assert resolved.strict_tools_active is False
