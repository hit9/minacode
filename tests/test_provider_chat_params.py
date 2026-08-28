"""provider chat params (split from tests/test_core_logic.py)."""
import pytest
from catalog_harness import resolve
from test_core_logic import session

from minacode.base import (
    RESPONSES_OUTPUT_KEY,
    ConfigError,
)
from minacode.config import (
    CHAT_REASONING_CHOICES,
    Config,
    ProviderConfig,
    RuntimeSettings,
)
from minacode.context import ContextManager
from minacode.model import ModelClient
from minacode.providers.catalog import decode_bundled


def test_runtime_settings_reads_theme_from_config():
    settings = RuntimeSettings.from_dict(
        {"runtime": {"theme": "light"}},
    )
    assert settings.theme == "light"

    # default when not set
    settings = RuntimeSettings.from_dict({})
    assert settings.theme == "auto"

    # override via keyword
    settings = RuntimeSettings.from_dict({"runtime": {"theme": "light"}}, theme="dark")
    assert settings.theme == "dark"

    # keyword override even when config is absent
    settings = RuntimeSettings.from_dict({}, theme="light")
    assert settings.theme == "light"

def test_config_validates_provider_selection_and_provider_fields():
    config = Config.from_dict(
        {
            "provider": {
                "active": "main",
                "main": {"url": "https://example.test/v1", "key": "k", "model": "m", "available_models": "a,b", "temperature": "off"},
            },
            "paths": {"data_dir": ".data"},
        }
    )
    assert config.active_provider == "main"
    assert config.provider.available_models == ("a", "b")
    assert config.provider.temperature is None
    assert config.data_dir == ".data"

    with pytest.raises(ConfigError):
        Config.from_dict({"provider": {"active": "missing", "main": {}}})
    with pytest.raises(ConfigError):
        ProviderConfig.from_dict({"api": "bad"})
    with pytest.raises(ConfigError):
        ProviderConfig.from_dict({"reasoning": "bad"})
    with pytest.raises(ConfigError):
        ProviderConfig.from_dict({"chat_reasoning": "bad"})
    with pytest.raises(ConfigError):
        ProviderConfig.from_dict({"prompt_cache_key": "not stable"})

    assert ProviderConfig.from_dict({"reasoning": "max"}).reasoning == "max"

def test_chat_provider_params_cover_reasoning_variants(tmp_path):
    client = ModelClient(session(tmp_path))

    params = {}
    client.apply_provider_params(params, ProviderConfig(url="https://openrouter.ai/api/v1", model="x", reasoning="max"))
    assert params["extra_body"] == {"reasoning": {"effort": "max"}}

    params = {}
    client.apply_provider_params(params, ProviderConfig(url="https://api.openai.com/v1", model="gpt-5-mini", reasoning="low"))
    assert params["reasoning_effort"] == "low"

    params = {}
    client.apply_provider_params(params, ProviderConfig(url="https://api.deepseek.com/v1", model="deepseek-chat", reasoning="off"))
    assert params["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in params

def test_every_resolvable_chat_reasoning_mode_is_configurable_by_hand():
    """`chat_reasoning` is the escape hatch when auto guesses wrong for a gateway or an
    unrecognized model name, so every dialect the catalog can resolve to must also be
    accepted from config."""
    snapshot = decode_bundled()

    def dialects(setmaps) -> set[str]:
        return {str(mapping["reasoning.dialect"]) for mapping in setmaps if "reasoning.dialect" in mapping}

    resolvable = dialects(rule.set for rule in snapshot.model_rules)
    for provider in snapshot.providers:
        resolvable |= dialects([provider.defaults])
        resolvable |= dialects(rule.set for rule in provider.model_rules)

    assert resolvable <= set(CHAT_REASONING_CHOICES), sorted(resolvable - set(CHAT_REASONING_CHOICES))
    for mode in resolvable:
        assert ProviderConfig.from_dict({"chat_reasoning": mode}).chat_reasoning == mode

def test_openai_suppresses_temperature_only_for_reasoning_families(tmp_path):
    """Reasoning models reject temperature outright, while sibling chat models still take it."""
    client = ModelClient(session(tmp_path))
    reasoning = ProviderConfig(url="https://api.openai.com/v1", model="gpt-5", reasoning="medium", temperature=0.7)
    assert resolve(reasoning).suppress_temperature is True
    params = {}
    client.apply_provider_params(params, reasoning)
    assert params == {"reasoning_effort": "medium"}

    chat = ProviderConfig(url="https://api.openai.com/v1", model="gpt-4o", temperature=0.7)
    assert resolve(chat).suppress_temperature is False
    params = {}
    client.apply_provider_params(params, chat)
    assert params == {"temperature": 0.7}

def test_opencode_routes_each_model_family_to_its_documented_protocol():
    """One base URL multiplexes three wire protocols by model, so api=auto cannot read the URL."""

    def api(model):
        return resolve(ProviderConfig(url="https://opencode.ai/zen/v1", model=model)).api

    assert api("claude-sonnet-5") == "anthropic"
    assert api("qwen3-coder") == "anthropic"
    assert api("gpt-5.6") == "responses"
    assert api("deepseek-v4") == "chat"

def test_anthropic_omits_temperature_while_thinking_is_enabled(tmp_path):
    """Thinking pins sampling to the default; any other temperature is rejected."""
    client = ModelClient(session(tmp_path))
    provider = client.session.config.provider
    provider.url, provider.model, provider.api = "https://api.anthropic.com", "claude-sonnet-4-5", "anthropic"
    provider.temperature, provider.reasoning = 0.3, "medium"

    params = client.wire(client.session.config.provider).params([{"role": "user", "content": "hi"}], None)
    assert params["thinking"]["type"] == "enabled"
    assert "temperature" not in params

    provider.reasoning = "off"
    params = client.wire(client.session.config.provider).params([{"role": "user", "content": "hi"}], None)
    assert "thinking" not in params
    assert "temperature" not in params
    assert params["extra_body"]["temperature"] == 0.3

    provider.model = "claude-fable-5"
    params = client.wire(client.session.config.provider).params([{"role": "user", "content": "hi"}], None)
    assert "thinking" not in params
    assert "temperature" not in params

    provider.model, provider.reasoning = "claude-sonnet", "medium"
    params = client.wire(client.session.config.provider).params([{"role": "user", "content": "hi"}], None)
    assert "thinking" not in params
    assert "temperature" not in params
    assert params["extra_body"]["temperature"] == 0.3

@pytest.mark.parametrize(
    ("model", "expected"),
    (
        # Extended thinking is the only mode at 4.5 and earlier. The high budget (8,192) no longer
        # fits under the conservative unset cap (8,192), so it is clamped to cap - 1,024 (7,168); a
        # configured max_tokens still scales it (see the clamp test below).
        ("claude-sonnet-4-5", {"thinking": {"type": "enabled", "budget_tokens": 7168}}),
        (
            "claude-opus-4-5-20251101",
            {"thinking": {"type": "enabled", "budget_tokens": 7168}, "output_config": {"effort": "high"}},
        ),
        ("anthropic.claude-haiku-4-5-20251001-v1:0", {"thinking": {"type": "enabled", "budget_tokens": 7168}}),
        ("claude-3-7-sonnet-20250219", {"thinking": {"type": "enabled", "budget_tokens": 7168}}),
        # The 4.6 generation accepts both; adaptive is the documented recommendation.
        ("claude-sonnet-4-6", {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}),
        # 4.7 and later reject "enabled" outright.
        ("claude-opus-4-7", {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}),
        ("claude-sonnet-5", {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}),
        ("claude-fable-5", {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}),
        # An alias with no generation stays generic: forcing either adaptive or manual thinking
        # can make a gateway reject an otherwise valid model name.
        ("claude-sonnet", {}),
    ),
)
def test_anthropic_thinking_matches_the_generation_of_the_model(tmp_path, model, expected):
    client = ModelClient(session(tmp_path))
    provider = client.session.config.provider
    provider.url, provider.api, provider.reasoning = "https://api.anthropic.com", "anthropic", "high"
    provider.model = model

    params = client.wire(client.session.config.provider).params([{"role": "user", "content": "hi"}], None)

    assert {key: params[key] for key in ("thinking", "output_config") if key in params} == expected

def test_anthropic_reasoning_off_respects_models_that_cannot_stop_thinking(tmp_path):
    """Adaptive models think by default, so "off" has to say so — except on the always-thinking
    families, which reject `disabled` with a 400 and have to be left unconfigured."""
    client = ModelClient(session(tmp_path))
    provider = client.session.config.provider
    provider.url, provider.api, provider.reasoning = "https://api.anthropic.com", "anthropic", "off"

    def thinking(model):
        provider.model = model
        params = client.wire(client.session.config.provider).params([{"role": "user", "content": "hi"}], None)
        return params.get("thinking")

    assert thinking("claude-sonnet-5") == {"type": "disabled"}
    assert thinking("claude-opus-4-7") == {"type": "disabled"}
    assert thinking("claude-fable-5") is None
    assert thinking("claude-mythos-5") is None
    # Extended-thinking models think only when asked, so the parameter is simply absent.
    assert thinking("claude-sonnet-4-5") is None

def test_anthropic_effort_uses_the_highest_level_each_generation_accepts(tmp_path):
    """xhigh arrived after the 4.6 generation, which tops out at max."""
    client = ModelClient(session(tmp_path))
    provider = client.session.config.provider
    provider.url, provider.api, provider.reasoning = "https://api.anthropic.com", "anthropic", "xhigh"

    def effort(model):
        provider.model = model
        return client.wire(client.session.config.provider).params([{"role": "user", "content": "hi"}], None)["output_config"]["effort"]

    assert effort("claude-sonnet-4-6") == "max"
    assert effort("claude-opus-4-7") == "xhigh"
    assert effort("claude-opus-5") == "xhigh"

    provider.reasoning = "max"
    assert effort("claude-sonnet-4-6") == "max"
    assert effort("claude-opus-4-7") == "max"
    assert effort("claude-opus-5") == "max"

    provider.reasoning = "minimal"
    assert effort("claude-opus-5") == "low"

    # Opus 4.5 is the one manual-thinking generation that also accepts output_config.effort.
    provider.reasoning = "medium"
    provider.model = "claude-opus-4-5"
    params = client.wire(client.session.config.provider).params([{"role": "user", "content": "hi"}], None)
    assert params["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert params["output_config"] == {"effort": "medium"}

def test_anthropic_thinking_budget_stays_under_the_requested_output_budget(tmp_path):
    """The API rejects a budget that is not strictly below max_tokens, so max_tokens has to lower it.

    The conservative unset cap (8,192) no longer clears the `high` budget, and a configured smaller
    cap or a larger effort still collides, which is what the cases below cover."""
    client = ModelClient(session(tmp_path))
    provider = client.session.config.provider
    provider.url, provider.api, provider.model = "https://api.anthropic.com", "anthropic", "claude-3-7-sonnet-20250219"

    for max_tokens, reasoning in ((8_192, "high"), (4_096, "max"), (2_048, "xhigh"), (0, "max")):
        provider.max_tokens, provider.reasoning = max_tokens, reasoning
        params = client.wire(client.session.config.provider).params([{"role": "user", "content": "hi"}], None)
        budget = params["thinking"]["budget_tokens"]
        assert 1_024 <= budget < params["max_tokens"], (max_tokens, reasoning, budget)

    # A budget that already fits is left alone.
    provider.max_tokens, provider.reasoning = 32_000, "medium"
    assert client.wire(client.session.config.provider).params([{"role": "user", "content": "hi"}], None)["thinking"]["budget_tokens"] == 4_096

def test_openrouter_reasoning_object_recipe_sends_the_resolved_effort(tmp_path):
    """OpenRouter normalizes reasoning behind its own object, so the recipe reads the resolved
    effort rather than the configured value directly (equal here, because the host's ignore-mode
    leaves the model's scale unconstrained)."""
    client = ModelClient(session(tmp_path))
    params: dict = {}

    client.apply_provider_params(params, ProviderConfig(url="https://openrouter.ai/api/v1", model="kimi-k3", reasoning="medium"))

    assert params["extra_body"] == {"reasoning": {"effort": "medium"}}

def test_anthropic_assistant_turns_are_echoed_back_verbatim(tmp_path):
    """The API verifies that thinking blocks return exactly as it produced them, signature
    included, so a rebuilt assistant turn breaks any tool loop that thought."""
    client = ModelClient(session(tmp_path))
    blocks = [
        {"type": "thinking", "thinking": "", "signature": "sig-abc"},
        {"type": "text", "text": "checking"},
        {"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"command": "ls"}},
    ]
    assistant, calls, _ = client.wire(ProviderConfig(api="anthropic", model="claude")).result({"content": blocks})
    assert [call.name for call in calls] == ["Bash"]

    params = client.wire(ProviderConfig(api="anthropic", model="claude")).params(
        [{"role": "user", "content": "go"}, assistant, {"role": "tool", "tool_call_id": "tu_1", "content": "out"}],
        None,
    )

    assert params["messages"][1]["content"] == blocks

@pytest.mark.parametrize(
    ("model", "keeps_prior"),
    [
        ("claude-sonnet-4-5", False),
        ("claude-haiku-4-5", False),
        ("claude-opus-4-5", True),
        ("claude-sonnet-4-6", True),
        ("claude-custom-alias", True),
    ],
)
def test_anthropic_replays_thinking_according_to_model_generation(tmp_path, model, keeps_prior):
    s = session(tmp_path)
    s.config.provider.api = "anthropic"
    s.config.provider.model = model
    client = ModelClient(s)
    prior = {
        "role": "assistant",
        "content": "checking",
        "_anthropic_content": [
            {"type": "thinking", "thinking": "R" * 800, "signature": "signature"},
            {"type": "text", "text": "checking"},
            {"type": "tool_use", "id": "tu", "name": "Read", "input": {"path": "a"}},
        ],
    }
    final = {
        "role": "assistant",
        "content": "done",
        "_anthropic_content": [{"type": "thinking", "thinking": "recent", "signature": "recent-signature"}, {"type": "text", "text": "done"}],
    }
    history = [
        {"role": "user", "content": "first"},
        prior,
        {"role": "tool", "tool_call_id": "tu", "content": "done"},
        final,
        {"role": "user", "content": "second"},
    ]

    blocks = client.wire(client.session.config.provider).messages(history)[1]["content"]
    tokens = client.estimated_request_tokens(history)
    without_old_thinking = [
        history[0],
        {**prior, "_anthropic_content": [block for block in prior["_anthropic_content"] if block["type"] != "thinking"]},
        *history[2:],
    ]

    # Always return complete blocks on the wire; older models filter all but the latest turn
    # server-side, which the context estimate mirrors without mutating the request.
    assert {"type": "thinking", "thinking": "R" * 800, "signature": "signature"} in blocks
    assert {"type": "tool_use", "id": "tu", "name": "Read", "input": {"path": "a"}} in blocks
    assert (tokens > client.estimated_request_tokens(without_old_thinking) + 150) is keeps_prior

def test_anthropic_always_replays_current_tool_loop_thinking(tmp_path):
    s = session(tmp_path)
    s.config.provider.api = "anthropic"
    s.config.provider.model = "claude-sonnet-4-5"
    blocks = [{"type": "thinking", "thinking": "reasoning", "signature": "signature"}, {"type": "tool_use", "id": "tu", "name": "Read", "input": {}}]
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": None, "_anthropic_content": blocks},
        {"role": "tool", "tool_call_id": "tu", "content": "done"},
    ]

    assert ModelClient(s).wire(ProviderConfig(api="anthropic", model="claude")).messages(history)[1]["content"] == blocks

def test_context_estimate_ignores_opaque_echo_bytes_but_counts_readable_reasoning(tmp_path):
    """Serialized ciphertext/signatures are not prompt text, but readable reasoning replayed by
    a protocol still occupies context and must not disappear from the estimate."""
    context = ContextManager(session(tmp_path))
    plain = {"role": "assistant", "content": "hello world"}
    carrying = {
        **plain,
        RESPONSES_OUTPUT_KEY: [
            {"id": "rs_1", "type": "reasoning", "encrypted_content": "E" * 8000, "summary": []},
            {"id": "msg_1", "type": "message", "content": [{"type": "output_text", "text": "hello world"}]},
        ],
    }

    assert context.estimated_tokens([carrying]) == context.estimated_tokens([plain])

    carrying[RESPONSES_OUTPUT_KEY][0]["summary"] = [{"type": "summary_text", "text": "R" * 800}]
    assert context.estimated_tokens([carrying]) > context.estimated_tokens([plain]) + 150

    anthropic = {
        **plain,
        "_anthropic_content": [
            {"type": "thinking", "thinking": "T" * 800, "signature": "S" * 8000},
            {"type": "text", "text": "hello world"},
        ],
    }
    assert context.estimated_tokens([anthropic]) > context.estimated_tokens([plain]) + 150

def test_context_gate_estimates_the_actual_chat_reasoning_history(tmp_path):
    s = session(tmp_path)
    s.config.provider.url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    s.config.provider.model = "qwen3.8-max-preview"
    s.config.provider.max_tokens = 1000
    s.settings.max_context_tokens = 6000
    model = ModelClient(s)
    context = ContextManager(s, model)
    reasoning = "R" * 20_000
    plain = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]
    final_reasoning = [plain[0], {**plain[1], "reasoning_content": reasoning}]
    tool_plain = [
        plain[0],
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "Read", "arguments": "{}"}}]},
    ]
    tool_reasoning = [tool_plain[0], {**tool_plain[1], "reasoning_content": reasoning}]

    # Qwen's default does not put final-answer reasoning on the next wire request, so it cannot
    # trigger compaction. Reasoning attached to a tool call is replayed and still counts.
    assert context.request_tokens(final_reasoning) == context.request_tokens(plain)
    assert context.request_tokens(final_reasoning) < context.request_token_budget()
    assert context.request_tokens(tool_reasoning) > context.request_tokens(tool_plain) + 4_000
    assert context.request_tokens(tool_reasoning) >= context.request_token_budget()

def test_context_estimate_uses_each_protocols_replayed_reasoning_shape(tmp_path):
    plain = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]

    responses = session(tmp_path / "responses")
    responses.config.provider.api = "responses"
    responses_model = ModelClient(responses)
    response_history = [
        plain[0],
        {
            **plain[1],
            RESPONSES_OUTPUT_KEY: [
                {"id": "rs", "type": "reasoning", "encrypted_content": "E" * 20_000, "summary": [{"type": "summary_text", "text": "S" * 800}]},
                {"id": "msg", "type": "message", "content": [{"type": "output_text", "text": "answer"}]},
            ],
        },
    ]
    response_tokens = responses_model.estimated_request_tokens(response_history)
    response_without_ciphertext = responses_model.estimated_request_tokens(
        [
            plain[0],
            {
                **response_history[1],
                RESPONSES_OUTPUT_KEY: [{**response_history[1][RESPONSES_OUTPUT_KEY][0], "encrypted_content": ""}, response_history[1][RESPONSES_OUTPUT_KEY][1]],
            },
        ]
    )
    assert response_tokens == response_without_ciphertext
    assert response_tokens > responses_model.estimated_request_tokens(plain) + 150

    anthropic = session(tmp_path / "anthropic")
    anthropic.config.provider.api = "anthropic"
    anthropic_model = ModelClient(anthropic)
    anthropic_history = [
        plain[0],
        {**plain[1], "_anthropic_content": [{"type": "thinking", "thinking": "T" * 800, "signature": "X" * 20_000}, {"type": "text", "text": "answer"}]},
    ]
    assert anthropic_model.estimated_request_tokens(anthropic_history) > anthropic_model.estimated_request_tokens(plain) + 150
    without_signature = [
        plain[0],
        {**plain[1], "_anthropic_content": [{"type": "thinking", "thinking": "T" * 800, "signature": ""}, {"type": "text", "text": "answer"}]},
    ]
    assert anthropic_model.estimated_request_tokens(anthropic_history) == anthropic_model.estimated_request_tokens(without_signature)
