import json
import threading
import time
from types import SimpleNamespace

import code_symbol_index as csi
import pytest

import minacode.__main__ as cli
import minacode.update as update_module
from minacode.__main__ import main
from minacode.base import (
    CHAT_REASONING_CHOICES,
    HTTP_USER_AGENT,
    RESPONSES_OUTPUT_KEY,
    Config,
    ConfigError,
    ConfigFile,
    ModelError,
    ModelUsage,
    ProviderConfig,
    RuntimeSettings,
    ToolCall,
    UpdateStatus,
    __version__,
)
from minacode.context import ContextManager
from minacode.engine import Agent
from minacode.loop import CommandLoop
from minacode.model import ModelClient
from minacode.render import StatusBar
from minacode.runner import ToolRunner
from minacode.session import Session, SessionSnapshotStore
from minacode.tools import TOOL_REGISTRY, CodeIndex, Tool
from minacode.update import UpdateChecker


def session(tmp_path):
    return Session(cwd=str(tmp_path))


def data_session(tmp_path):
    return Session(cwd=str(tmp_path), config=Config(data_dir=str(tmp_path / ".data")))


@pytest.mark.parametrize("flag", ["-c", "--last", "--latest"])
def test_continue_flags_resume_latest_session_in_current_project(tmp_path, monkeypatch, flag):
    config = Config(data_dir=str(tmp_path / "data"))
    settings = RuntimeSettings()
    resumed = SimpleNamespace(settings=settings, mcp=None)
    selected = []

    monkeypatch.setattr(ConfigFile, "load", lambda _path: {})
    monkeypatch.setattr(Config, "from_dict", classmethod(lambda _cls, _data: config))
    monkeypatch.setattr(RuntimeSettings, "from_dict", classmethod(lambda _cls, _data, **_kwargs: settings))
    monkeypatch.setattr(
        Session,
        "load_snapshot",
        classmethod(lambda _cls, uid, config=None, settings=None, cwd="": selected.append((uid, config, settings, cwd)) or resumed),
    )

    class Loop:
        resume_request = ""

        def run(self):
            return 0

        def close_background_output(self):
            pass

    monkeypatch.setattr(cli, "CommandLoop", lambda _agent: Loop())
    monkeypatch.chdir(tmp_path)

    assert main([flag]) == 0
    # The alias is resolved against the current project, not a global pointer.
    assert selected == [("latest", config, settings, str(tmp_path))]


def test_resume_request_starts_the_next_run_on_the_chosen_session(tmp_path, monkeypatch):
    """/sessions ends one run and main starts the next on the session it named, instead of
    re-pointing a live object graph at a different Session."""
    config = Config(data_dir=str(tmp_path / "data"))
    settings = RuntimeSettings()
    loaded = []

    monkeypatch.setattr(ConfigFile, "load", lambda _path: {})
    monkeypatch.setattr(Config, "from_dict", classmethod(lambda _cls, _data: config))
    monkeypatch.setattr(RuntimeSettings, "from_dict", classmethod(lambda _cls, _data, **_kwargs: settings))
    monkeypatch.setattr(
        Session,
        "load_snapshot",
        classmethod(lambda _cls, uid, config=None, settings=None, cwd="": loaded.append(uid) or SimpleNamespace(settings=settings, mcp=None)),
    )
    closed = []
    handovers = iter(["second-uid", ""])

    class Loop:
        def __init__(self, _agent):
            self.resume_request = ""

        def run(self):
            self.resume_request = next(handovers)
            return 3

        def close_background_output(self):
            closed.append(self.resume_request)

    monkeypatch.setattr(cli, "CommandLoop", Loop)
    monkeypatch.chdir(tmp_path)

    assert main(["--resume", "first-uid"]) == 3
    assert loaded == ["first-uid", "second-uid"]
    # Each run is torn down before the next is built; nothing is carried across.
    assert closed == ["second-uid", ""]


def test_runtime_settings_reads_limits_and_yolo_override():
    settings = RuntimeSettings.from_dict(
        {"runtime": {"shell_timeout": 7, "max_agent_steps": 0, "max_context_tokens": 0, "yolo": False}},
        yolo=True,
    )

    assert settings.shell_timeout == 7
    assert settings.max_steps == 1
    assert settings.max_context_tokens == 1
    assert settings.yolo is True


def test_runtime_settings_default_context_budget_is_240k():
    assert RuntimeSettings().max_context_tokens == 240 * 1024
    assert RuntimeSettings.from_dict({}).max_context_tokens == 240 * 1024


def test_provider_timeout_defaults_distinguish_inactivity_from_total_generation():
    assert ProviderConfig().timeout == 120
    assert Config.from_dict({}).provider.timeout == 120
    assert ProviderConfig().response_timeout == 600
    assert Config.from_dict({}).provider.response_timeout == 600
    assert ProviderConfig.from_dict({"response_timeout": 0}).response_timeout == 0
    assert "# response_timeout = 600" in ConfigFile.DEFAULT_TEXT


def test_provider_stream_defaults_on_and_can_be_disabled():
    assert ProviderConfig().stream is True
    assert ProviderConfig.from_dict({"stream": False}).stream is False
    assert "# stream = true" in ConfigFile.DEFAULT_TEXT


def test_provider_image_input_defaults_to_auto_and_validates_values():
    assert ProviderConfig().image_input == "auto"
    assert ProviderConfig.from_dict({"image_input": "on"}).image_input == "on"
    with pytest.raises(ConfigError, match="provider.image_input"):
        ProviderConfig.from_dict({"image_input": "unknown"})


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


def test_chat_provider_params_cover_reasoning_variants(tmp_path):
    client = ModelClient(session(tmp_path))

    params = {}
    client.apply_provider_params(params, ProviderConfig(url="https://openrouter.ai/api/v1", model="x", reasoning="high"))
    assert params["extra_body"] == {"reasoning": {"effort": "high"}}

    params = {}
    client.apply_provider_params(params, ProviderConfig(url="https://api.openai.com/v1", model="gpt-5-mini", reasoning="low"))
    assert params["reasoning_effort"] == "low"

    params = {}
    client.apply_provider_params(params, ProviderConfig(url="https://api.deepseek.com/v1", model="deepseek-chat", reasoning="off"))
    assert params["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in params


def test_every_resolvable_chat_reasoning_mode_is_configurable_by_hand():
    """`chat_reasoning` is the escape hatch when auto guesses wrong for a gateway or an
    unrecognized model name, so every mode the compatibility rules can resolve to must also be
    accepted from config."""
    resolvable = {rule.value for compatibility in ProviderConfig.COMPATIBILITY.values() for rule in compatibility.chat_reasoning_rules} | {
        compatibility.chat_reasoning for compatibility in ProviderConfig.COMPATIBILITY.values() if compatibility.chat_reasoning is not None
    }

    assert resolvable <= set(CHAT_REASONING_CHOICES), sorted(resolvable - set(CHAT_REASONING_CHOICES))
    for mode in resolvable:
        assert ProviderConfig.from_dict({"chat_reasoning": mode}).chat_reasoning == mode


def test_openai_suppresses_temperature_only_for_reasoning_families(tmp_path):
    """Reasoning models reject temperature outright, while sibling chat models still take it."""
    client = ModelClient(session(tmp_path))
    reasoning = ProviderConfig(url="https://api.openai.com/v1", model="gpt-5", reasoning="medium", temperature=0.7)
    assert reasoning.resolve().suppress_temperature is True
    params = {}
    client.apply_provider_params(params, reasoning)
    assert params == {"reasoning_effort": "medium"}

    chat = ProviderConfig(url="https://api.openai.com/v1", model="gpt-4o", temperature=0.7)
    assert chat.resolve().suppress_temperature is False
    params = {}
    client.apply_provider_params(params, chat)
    assert params == {"temperature": 0.7}


def test_opencode_routes_each_model_family_to_its_documented_protocol():
    """One base URL multiplexes three wire protocols by model, so api=auto cannot read the URL."""

    def api(model):
        return ProviderConfig(url="https://opencode.ai/zen/v1", model=model).resolve().api

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

    params = client.anthropic_params([{"role": "user", "content": "hi"}], None)
    assert params["thinking"]["type"] == "enabled"
    assert "temperature" not in params

    provider.reasoning = "off"
    params = client.anthropic_params([{"role": "user", "content": "hi"}], None)
    assert "thinking" not in params
    assert params["temperature"] == 0.3

    provider.model = "claude-fable-5"
    params = client.anthropic_params([{"role": "user", "content": "hi"}], None)
    assert "thinking" not in params
    assert "temperature" not in params

    provider.model, provider.reasoning = "claude-sonnet", "medium"
    params = client.anthropic_params([{"role": "user", "content": "hi"}], None)
    assert "thinking" not in params
    assert params["temperature"] == 0.3


@pytest.mark.parametrize(
    ("model", "expected"),
    (
        # Extended thinking is the only mode at 4.5 and earlier.
        ("claude-sonnet-4-5", {"thinking": {"type": "enabled", "budget_tokens": 8192}}),
        (
            "claude-opus-4-5-20251101",
            {"thinking": {"type": "enabled", "budget_tokens": 8192}, "output_config": {"effort": "high"}},
        ),
        ("anthropic.claude-haiku-4-5-20251001-v1:0", {"thinking": {"type": "enabled", "budget_tokens": 8192}}),
        ("claude-3-7-sonnet-20250219", {"thinking": {"type": "enabled", "budget_tokens": 8192}}),
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

    params = client.anthropic_params([{"role": "user", "content": "hi"}], None)

    assert {key: params[key] for key in ("thinking", "output_config") if key in params} == expected


def test_anthropic_reasoning_off_respects_models_that_cannot_stop_thinking(tmp_path):
    """Adaptive models think by default, so "off" has to say so — except on the always-thinking
    families, which reject `disabled` with a 400 and have to be left unconfigured."""
    client = ModelClient(session(tmp_path))
    provider = client.session.config.provider
    provider.url, provider.api, provider.reasoning = "https://api.anthropic.com", "anthropic", "off"

    def thinking(model):
        provider.model = model
        params = client.anthropic_params([{"role": "user", "content": "hi"}], None)
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
        return client.anthropic_params([{"role": "user", "content": "hi"}], None)["output_config"]["effort"]

    assert effort("claude-sonnet-4-6") == "max"
    assert effort("claude-opus-4-7") == "xhigh"
    assert effort("claude-opus-5") == "xhigh"

    provider.reasoning = "minimal"
    assert effort("claude-opus-5") == "low"

    # Opus 4.5 is the one manual-thinking generation that also accepts output_config.effort.
    provider.reasoning = "medium"
    provider.model = "claude-opus-4-5"
    params = client.anthropic_params([{"role": "user", "content": "hi"}], None)
    assert params["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert params["output_config"] == {"effort": "medium"}


def test_anthropic_assistant_turns_are_echoed_back_verbatim(tmp_path):
    """The API verifies that thinking blocks return exactly as it produced them, signature
    included, so a rebuilt assistant turn breaks any tool loop that thought."""
    client = ModelClient(session(tmp_path))
    blocks = [
        {"type": "thinking", "thinking": "", "signature": "sig-abc"},
        {"type": "text", "text": "checking"},
        {"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"command": "ls"}},
    ]
    assistant, calls, _ = client.anthropic_result({"content": blocks})
    assert [call.name for call in calls] == ["Bash"]

    params = client.anthropic_params(
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

    blocks = client.anthropic_messages(history)[1]["content"]
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

    assert ModelClient(s).anthropic_messages(history)[1]["content"] == blocks


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

    for reasoning in ("minimal", "low", "medium", "high", "xhigh"):
        provider.reasoning = reasoning
        params = {}
        client.apply_provider_params(params, provider)
        assert params == {"reasoning_effort": reasoning}

    provider.reasoning = "off"
    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"reasoning_effort": "none"}

    provider.url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert provider.resolve().chat_reasoning == "reasoning_effort"

    provider.url = "https://notaliyuncs.com/compatible-mode/v1"
    assert provider.resolve().chat_reasoning == "off"

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
def test_zai_regional_endpoints_share_documented_reasoning_effort(url, tmp_path):
    client = ModelClient(session(tmp_path))
    provider = ProviderConfig(url=url, model="glm-5.2", reasoning="xhigh", temperature=0.6)
    resolved = provider.resolve()
    assert resolved.chat_reasoning == "thinking_effort"
    assert resolved.prompt_cache_key is False
    assert resolved.chat_reasoning_history == "current_turn"
    assert client.prompt_cache_key(provider, None) == ""

    params = {}
    client.apply_provider_params(params, provider)
    assert params == {
        "temperature": 0.6,
        "reasoning_effort": "xhigh",
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
def test_provider_compatibility_requires_a_real_domain_boundary(url, model, tmp_path):
    client = ModelClient(session(tmp_path))
    provider = ProviderConfig(url=url, model=model, reasoning="high", temperature=0.4)
    resolved = provider.resolve()
    assert resolved.chat_reasoning == "off"
    assert resolved.prompt_cache_key is True

    params = {}
    client.apply_provider_params(params, provider)
    assert params == {"temperature": 0.4}


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


def test_chat_provider_extra_body_passthrough(tmp_path):
    client = ModelClient(session(tmp_path))

    # Vendor extensions (e.g. Qianwen web search) pass through verbatim into extra_body.
    params = {}
    search = {"enable_search": True, "search_options": {"forced_search": True, "search_strategy": "max"}}
    provider = ProviderConfig(url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen3-max", reasoning="off", extra_body=search)
    client.apply_provider_params(params, provider)
    assert params["extra_body"] == search

    # Configured extra_body merges with minacode-managed reasoning fields...
    params = {}
    client.apply_provider_params(params, ProviderConfig(url="https://openrouter.ai/api/v1", model="x", reasoning="high", extra_body={"enable_search": True}))
    assert params["extra_body"] == {"enable_search": True, "reasoning": {"effort": "high"}}

    # ...and reasoning wins on key conflict so minacode stays in control of its own fields.
    params = {}
    client.apply_provider_params(
        params, ProviderConfig(url="https://openrouter.ai/api/v1", model="x", reasoning="high", extra_body={"reasoning": {"effort": "low"}})
    )
    assert params["extra_body"] == {"reasoning": {"effort": "high"}}

    # Managed thinking.type remains authoritative without discarding documented history options.
    params = {}
    client.apply_provider_params(
        params,
        ProviderConfig(
            url="https://api.z.ai/api/paas/v4",
            model="glm-5.1",
            reasoning="high",
            extra_body={"thinking": {"clear_thinking": False}},
        ),
    )
    assert params["extra_body"] == {"thinking": {"clear_thinking": False, "type": "enabled"}}

    # extra_body round-trips through config; non-object values are ignored.
    assert ProviderConfig.from_dict({"extra_body": search}).extra_body == search
    assert ProviderConfig.from_dict({"extra_body": "nope"}).extra_body == {}
    assert ProviderConfig().extra_body == {}


def _strict_check(node, path="root"):
    if isinstance(node, dict):
        for key in ("minItems", "maxItems", "minLength", "maxLength"):
            assert key not in node, f"{path}: leftover {key}"
        kind = node.get("type")
        if isinstance(kind, list):
            # DeepSeek strict rejects object/array inside a type union; only scalars + null allowed.
            assert all(item in ("string", "number", "integer", "boolean", "null") for item in kind), f"{path}: non-scalar in type union {kind}"
        if isinstance(node.get("properties"), dict):
            assert node.get("additionalProperties") is False, f"{path}: additionalProperties"
            assert set(node["required"]) == set(node["properties"]), f"{path}: required != properties"
            for key, sub in node["properties"].items():
                _strict_check(sub, f"{path}.{key}")
        if "items" in node:
            _strict_check(node["items"], f"{path}[]")
        for combiner in ("anyOf", "oneOf", "allOf"):
            for index, sub in enumerate(node.get(combiner, [])):
                _strict_check(sub, f"{path}.{combiner}[{index}]")
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _strict_check(item, f"{path}[{index}]")


def test_strict_tools_off_path_emits_non_strict_schema():
    for tool in TOOL_REGISTRY.values():
        legacy = {
            "type": "function",
            "function": {
                "name": tool.NAME,
                "description": "\n".join([tool.DESCRIPTION, *(("- " + item) for item in tool.EXAMPLE if item)]),
                "parameters": tool.params_schema(),
            },
        }
        assert tool.schema(False) == legacy
        assert "strict" not in tool.schema(False)["function"]


def test_strict_tools_gating_and_beta_routing():
    def resolved(url, strict=False):
        return ProviderConfig(url=url, strict_tools=strict).resolve()

    # Unsupported hosts never activate strict, even when requested, and stay on their endpoint.
    for url in ("https://openrouter.ai/api/v1", "https://api.together.xyz/v1", "http://localhost:1234/v1"):
        assert resolved(url, strict=True).strict_tools_active is False
        assert resolved(url, strict=True).base_url == url

    # DeepSeek: off keeps the stable endpoint; on activates strict and routes to /beta (idempotently).
    assert resolved("https://api.deepseek.com").strict_tools_active is False
    assert resolved("https://api.deepseek.com").base_url == "https://api.deepseek.com"
    assert resolved("https://api.deepseek.com", strict=True).strict_tools_active is True
    assert resolved("https://api.deepseek.com", strict=True).base_url == "https://api.deepseek.com/beta"
    assert resolved("https://api.deepseek.com/beta", strict=True).base_url == "https://api.deepseek.com/beta"

    # OpenAI supports strict but not the beta endpoint, so it stays on the normal URL.
    assert resolved("https://api.openai.com/v1", strict=True).strict_tools_active is True
    assert resolved("https://api.openai.com/v1", strict=True).base_url == "https://api.openai.com/v1"


def test_resolved_base_url_removes_known_protocol_suffixes():
    def p(url):
        return ProviderConfig(url=url).resolve().base_url

    assert p("https://api.openai.com/v1/chat/completions") == "https://api.openai.com/v1"
    assert p("https://api.openai.com/v1/responses") == "https://api.openai.com/v1"
    assert p("https://api.openai.com/v1/messages") == "https://api.openai.com/v1"
    assert p("https://api.openai.com/v1") == "https://api.openai.com/v1"
    assert p("https://api.openai.com/v1/") == "https://api.openai.com/v1"
    assert p("https://api.openai.com/v1/chat/completions/") == "https://api.openai.com/v1"


def test_provider_api_auto_recognizes_explicit_endpoint_suffixes():
    assert ProviderConfig.from_dict({"api": "responses"}).api == "responses"
    assert ProviderConfig(url="https://api.openai.com/v1/responses").resolve().api == "responses"
    assert ProviderConfig(url="https://api.openai.com/v1/chat/completions").resolve().api == "chat"
    assert ProviderConfig(url="https://api.anthropic.com/v1/messages").resolve().api == "anthropic"
    assert ProviderConfig(url="https://api.openai.com/v1").resolve().api == "chat"
    assert ProviderConfig(url="https://api.openai.com/v1/responses", api="chat").resolve().api == "chat"


def test_openai_responses_path_supports_strict_tools():
    provider = ProviderConfig(url="https://api.openai.com/v1", api="responses", strict_tools=True)
    assert provider.resolve().strict_tools_active is True


def test_strict_tools_schema_is_valid_and_does_not_mutate_classvars():
    before = {name: json.dumps(tool.params_schema()) for name, tool in TOOL_REGISTRY.items()}
    for name, tool in TOOL_REGISTRY.items():
        function = tool.schema(True)["function"]
        if function.get("strict"):
            _strict_check(function["parameters"], name)
        else:
            # Only free-form schemas (open objects) may skip strict; they stay untransformed.
            assert Tool._strictifiable(tool.params_schema()) is False, name
            assert function["parameters"] == tool.params_schema()
    after = {name: json.dumps(tool.params_schema()) for name, tool in TOOL_REGISTRY.items()}
    assert before == after  # deepcopy keeps shared ClassVar schemas intact

    search_context = TOOL_REGISTRY["Search"].schema(True)["function"]["parameters"]["properties"]["context"]
    assert "null" in search_context["type"]
    # Optional array/object params use anyOf (never object/array inside a type union).
    search_queries = TOOL_REGISTRY["Search"].schema(True)["function"]["parameters"]["properties"]["queries"]
    assert search_queries["anyOf"][1] == {"type": "null"}


def test_strict_tools_skips_free_form_object_schemas():
    # MCP.arguments is a free-form object; strict cannot close it, so MCP stays non-strict.
    mcp = TOOL_REGISTRY["MCP"].schema(True)["function"]
    assert "strict" not in mcp
    assert Tool._strictifiable(TOOL_REGISTRY["MCP"].params_schema()) is False
    assert Tool._strictifiable(TOOL_REGISTRY["Read"].params_schema()) is True


def test_drop_nulls_strips_omitted_strict_arguments():
    assert ModelClient.drop_nulls({"a": 1, "b": None, "c": {"d": None, "e": 2}, "f": [{"g": None, "h": 3}]}) == {"a": 1, "c": {"e": 2}, "f": [{"h": 3}]}


def test_chat_tool_call_parsing_handles_valid_invalid_and_non_object_payloads(tmp_path):
    client = ModelClient(session(tmp_path))
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(id="ok", function=SimpleNamespace(name="Bash", arguments=json.dumps({"command": "pwd"}))),
            SimpleNamespace(id="second", function=SimpleNamespace(name="Bash", arguments=json.dumps({"command": "whoami"}))),
            SimpleNamespace(id="bad-json", function=SimpleNamespace(name="Read", arguments="{")),
            SimpleNamespace(id="list-payload", function=SimpleNamespace(name="Recall", arguments=json.dumps(["tr.1"]))),
        ]
    )

    calls = client.tool_calls(message)

    assert calls[0] == ToolCall(id="ok", name="Bash", args=["pwd"])
    assert calls[1] == ToolCall(id="second", name="Bash", args=["whoami"])
    assert calls[2].id == "bad-json"
    assert calls[2].name == "Read"
    assert calls[2].args == []
    assert calls[3] == ToolCall(id="list-payload", name="Recall", args=[["tr.1"]])


def test_model_request_retries_retryable_errors_and_reports_attempts(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.config.provider.url = "https://example.test/v1"
    s.config.provider.key = "key"
    s.config.provider.model = "model"
    client = ModelClient(s)
    calls = []
    retries = []

    def fail(_messages, _tools):
        calls.append(1)
        raise ModelError("Error code: 500 - provider failed")

    monkeypatch.setattr(client, "chat_request", fail)
    monkeypatch.setattr(
        time,
        "sleep",
        lambda _seconds: retries.append((s.state.current_model_attempt, s.state.model_retry_reason)),
    )

    with pytest.raises(ModelError, match="after 6 attempts"):
        client.request([{"role": "user", "content": "hi"}])

    assert len(calls) == 6
    assert retries == [(2, "500"), (3, "500"), (4, "500"), (5, "500"), (6, "500")]
    assert s.state.model_retry_count == 5
    assert s.state.current_model_attempt == 0
    assert s.state.model_retry_reason == ""


def test_retryable_error_detects_status_codes_in_text(tmp_path):
    client = ModelClient(session(tmp_path))

    assert client.retryable_error(ModelError("Error code: 500 - provider failed"))
    assert client.retryable_error(ModelError("{'error': {'code': 503, 'message': 'busy'}}"))
    assert not client.retryable_error(ModelError("Error code: 400 - bad request"))


def test_retry_reason_is_short_and_safe(tmp_path):
    client = ModelClient(session(tmp_path))

    assert client.retry_reason(ModelError("Error code: 429 - secret provider payload")) == "429"
    assert client.retry_reason(ModelError("request timed out with secret provider payload")) == "timeout"
    assert client.retry_reason(ModelError("connection reset by peer")) == "connection"


def test_model_usage_counts_cached_tokens_from_multiple_shapes():
    usage = ModelUsage()

    usage.add(SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=20, prompt_tokens_details=SimpleNamespace(cached_tokens=4)))
    usage.add({"input_tokens": 7, "output_tokens": 3, "input_tokens_details": {"cached_tokens": 2}})

    assert usage.calls == 2
    assert usage.prompt_tokens == 17
    assert usage.completion_tokens == 8
    assert usage.total_tokens == 30
    assert usage.cached_prompt_tokens == 6
    assert usage.last_cached_prompt_tokens == 2


def test_model_usage_folds_anthropic_cache_legs_into_prompt_tokens():
    usage = ModelUsage()

    # Anthropic reports input_tokens without the cached legs, so a cache hit must not read as a
    # ratio above 100% or shrink the request's token total to the uncached remainder.
    usage.add(SimpleNamespace(input_tokens=20, output_tokens=5, cache_read_input_tokens=30_000, cache_creation_input_tokens=1_000))

    assert usage.last_prompt_tokens == 31_020
    assert usage.last_cached_prompt_tokens == 30_000
    assert usage.last_cached_prompt_tokens * 100 // usage.last_prompt_tokens == 96
    assert usage.prompt_tokens == 31_020
    assert usage.total_tokens == 31_025


def test_context_cleans_surrogate_text(tmp_path):
    bad = "bad \udce5 text"
    s = session(tmp_path)
    s.store_tool_result("Bash", [bad], bad)
    s.record_tool_error("tr.1", "Bash", [bad], bad)

    messages = ContextManager(s).model_messages("sys", [{"role": "user", "content": bad}])

    json.dumps(messages, ensure_ascii=False).encode("utf-8")
    assert "\udce5" not in str(messages)


def test_code_index_update_paths_only_keeps_workspace_files(tmp_path):
    s = session(tmp_path)
    inside = tmp_path / "inside.py"
    outside = tmp_path.parent / "outside.py"
    directory = tmp_path / "pkg"
    inside.write_text("x = 1\n", encoding="utf-8")
    outside.write_text("x = 2\n", encoding="utf-8")
    directory.mkdir()

    paths = CodeIndex(s).update_paths([str(inside), str(outside), str(directory), str(tmp_path / "missing.py")])

    assert paths == [str(inside)]


def test_code_index_update_pending_updates_small_batches_and_skips_large_batches(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    updates = []

    def status(root, *, check=False, max_pending_files=20):
        if check:
            return SimpleNamespace(status="stale", message="", reason="changed", pending_changes=1, pending_files=("a.py",))
        return SimpleNamespace(status="ready", message="", reason="", pending_changes="unknown", pending_files=())

    monkeypatch.setattr(csi, "status", status)
    monkeypatch.setattr(csi, "update", lambda paths, *, root: updates.append((root, list(paths))))

    assert CodeIndex(session(tmp_path)).update_pending() == "updated 1 file(s)"
    assert updates == [(str(tmp_path), [str(tmp_path / "a.py")])]

    updates.clear()
    monkeypatch.setattr(
        csi,
        "status",
        lambda root, *, check=False, max_pending_files=20: SimpleNamespace(
            status="stale", message="", reason="changed", pending_changes=CodeIndex.AUTO_UPDATE_LIMIT + 1, pending_files=("a.py",) * 21
        ),
    )
    assert CodeIndex(session(tmp_path)).update_pending() == ""
    assert updates == []


def test_code_index_sync_uses_python_api_and_updates_status(tmp_path, monkeypatch):
    calls = []

    monkeypatch.setattr(csi, "clean", lambda root: calls.append(("clean", root)))
    monkeypatch.setattr(csi, "index", lambda root: calls.append(("index", root)))
    monkeypatch.setattr(
        csi,
        "status",
        lambda root, *, check=False, max_pending_files=20: SimpleNamespace(status="ready", message="", reason="", pending_changes=0, pending_files=()),
    )

    s = session(tmp_path)
    result = CodeIndex(s).sync(force=True)

    assert calls == [("clean", str(tmp_path)), ("index", str(tmp_path))]
    assert "code_index: rebuilt" in result
    assert s.state.code_index_status == "synced"


def test_code_index_refresh_existing_uses_library_async_refresh(tmp_path, monkeypatch):
    calls = []

    class Worker:
        def join(self):
            calls.append(("join",))

    monkeypatch.setattr(
        csi,
        "status",
        lambda root, *, check=False, max_pending_files=20: (
            calls.append(("status", check)) or SimpleNamespace(status="ready", message="", reason="", pending_changes=0, pending_files=())
        ),
    )
    monkeypatch.setattr(csi, "refresh_async", lambda root: calls.append(("refresh_async", root)) or Worker())

    s = session(tmp_path)
    assert CodeIndex(s).refresh_existing_async() is True
    for _ in range(50):
        if ("join",) in calls and not s.state.code_index_refreshing:
            break
        time.sleep(0.01)

    assert ("refresh_async", str(tmp_path)) in calls
    assert ("join",) in calls
    assert ("status", True) in calls
    assert s.state.code_index_refreshing is False
    assert s.state.code_index_status == "synced"


def test_status_bar_animates_refreshing_code_index(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.state.code_index_refreshing = True
    s.state.code_index_notice = "syncing"
    bar = StatusBar(s)

    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    first = bar.index_status()
    monkeypatch.setattr(time, "monotonic", lambda: StatusBar.INTERVAL)
    second = bar.index_status()

    assert first != second
    assert first in StatusBar.INDEX_SPINNER
    assert second in StatusBar.INDEX_SPINNER


def test_update_checker_start_spawns_daemon_thread(tmp_path, monkeypatch):
    started = []

    class FakeThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            started.append((self.target, self.daemon))

    s = data_session(tmp_path)
    monkeypatch.setattr(threading, "Thread", FakeThread)

    UpdateChecker(s).start()
    assert len(started) == 1
    assert started[0][1] is True  # daemon
    assert s.update.checking is True

    # start() is a no-op while a check is already in flight so we don't stack duplicates.
    UpdateChecker(s).start()
    assert len(started) == 1


def test_update_status_signals_newer_version_in_status_bar(tmp_path):
    s = data_session(tmp_path)
    s.update.latest = "99.0.0"
    assert UpdateStatus.version_tuple("1.2") == (1, 2, 0)
    assert s.update.newer_than(__version__)
    assert s.update.latest in StatusBar(s).update_status()


def test_update_checker_fetch_latest_uses_bounded_timeout(tmp_path, monkeypatch):
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'{"info":{"version":"9.8.7"}}'

    def fake_urlopen(request, timeout):
        seen["timeout"] = timeout
        seen["user_agent"] = request.get_header("User-agent")
        return Response()

    monkeypatch.setattr(update_module, "urlopen", fake_urlopen)

    assert UpdateChecker(data_session(tmp_path)).fetch_latest() == "9.8.7"
    assert seen == {"timeout": UpdateChecker.TIMEOUT, "user_agent": HTTP_USER_AGENT}


def test_start_session_announces_detected_upgrade_command(tmp_path, monkeypatch):
    s = data_session(tmp_path)
    s.update.latest = "999.0.0"
    emitted = []
    monkeypatch.setattr(UpdateChecker, "start", lambda _checker: None)
    monkeypatch.setattr(UpdateChecker, "upgrade_command", lambda: ["uv", "tool", "upgrade", "minacode"])
    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 0)
    monkeypatch.setattr(CodeIndex, "refresh_existing_async", lambda _index: False)

    CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=emitted.append).start_session()

    assert any("upgrade with `uv tool upgrade minacode`" in line for line in emitted)


def test_tool_runner_unknown_tool_records_concise_error(tmp_path):
    s = session(tmp_path)
    ToolRunner(s, ContextManager(s), output_fn=lambda text: None).run([ToolCall("x", "MissingTool", [])])
    assert s.tool_records == []
    assert s.tool_results == {}
    assert len(s.tool_errors) == 1


def test_tool_runner_non_refusal_failures_do_not_stop_batch(tmp_path):
    s = session(tmp_path)
    s.settings.yolo = True
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run([ToolCall("bad", "Bash", []), ToolCall("create", "Edit", ["ok.txt", [{"op": "create", "content": "ok\n"}]])])

    assert len(s.tool_errors) == 1
    assert len(s.tool_records) == 1
    assert (tmp_path / "ok.txt").read_text(encoding="utf-8") == "ok\n"
