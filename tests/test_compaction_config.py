"""compaction config (split from tests/test_core_logic.py)."""

import time

import pytest
from catalog_harness import resolve
from model_harness import _MockClientFactory

from minacode import compaction
from minacode.base import (
    ConfigError,
    ModelError,
)
from minacode.config import (
    Config,
    ProviderConfig,
)
from minacode.context import ContextManager
from minacode.model import ModelClient
from minacode.render import StatusBar
from minacode.session import Session, SessionSnapshotCodec


def test_compaction_config_fields_parse_and_default_empty():
    config = Config.from_dict(
        {
            "compaction": {"provider": "fast", "model": "m-x", "reasoning": "high", "api": "responses"},
            "provider": {"active": "default", "default": {"model": "d"}, "fast": {"model": "m"}},
        }
    )
    assert config.compaction_provider == "fast"
    assert config.compaction_model == "m-x"
    assert config.compaction_reasoning == "high"
    assert config.compaction_api == "responses"

    # Defaults: no [compaction] fields means "inherit the active provider entry" at call time.
    plain = Config.from_dict({"provider": {"default": {"model": "d"}}})
    assert plain.compaction_provider == ""
    assert plain.compaction_model == ""
    assert plain.compaction_reasoning == ""
    assert plain.compaction_api == ""


def test_compaction_config_rejects_invalid_values():
    with pytest.raises(ConfigError, match="compaction.provider"):
        Config.from_dict({"compaction": {"provider": "nope"}, "provider": {"default": {}}})
    with pytest.raises(ConfigError, match="compaction.reasoning"):
        Config.from_dict({"compaction": {"reasoning": "turbo"}, "provider": {"default": {}}})
    with pytest.raises(ConfigError, match="compaction.api"):
        Config.from_dict({"compaction": {"api": "oai"}, "provider": {"default": {}}})


def test_compaction_provider_config_folds_overrides_without_sharing():
    from minacode.config import compaction_provider_config

    config = Config.from_dict(
        {
            "compaction": {"provider": "fast", "model": "m-x", "reasoning": "off", "api": "chat"},
            "provider": {"active": "default", "default": {"model": "d"}, "fast": {"model": "m", "reasoning": "high", "api": "anthropic"}},
        }
    )
    entry = compaction_provider_config(config)
    assert entry.model == "m-x"
    assert entry.reasoning == "off"
    assert entry.api == "chat"
    assert entry is not config.providers["fast"]
    assert config.providers["fast"].model == "m"  # the base entry is untouched

    config.compaction_model = ""
    entry = compaction_provider_config(config)
    assert entry.model == "m"  # empty override inherits the entry's model

    config.compaction_provider = ""
    entry = compaction_provider_config(config)
    assert entry.model == "d"  # empty provider = the active entry
    assert entry is not config.provider


def test_provider_compaction_fields_parse_and_default_empty():
    config = Config.from_dict(
        {
            "provider": {
                "active": "default",
                "default": {"model": "d"},
                "fast": {"model": "m", "compaction": {"model": "m-x", "reasoning": "off", "api": "chat"}},
            }
        }
    )
    nested = config.providers["fast"]
    assert nested.compaction_model == "m-x"
    assert nested.compaction_reasoning == "off"
    assert nested.compaction_api == "chat"
    # No nested table: all three stay empty (inherit), and the entry parses unchanged.
    assert config.providers["default"].compaction_model == ""
    assert config.providers["default"].compaction_reasoning == ""
    assert config.providers["default"].compaction_api == ""


def test_provider_compaction_rejects_invalid_values():
    with pytest.raises(ConfigError, match="provider.compaction.reasoning"):
        Config.from_dict({"provider": {"default": {"compaction": {"reasoning": "turbo"}}}})
    with pytest.raises(ConfigError, match="provider.compaction.api"):
        Config.from_dict({"provider": {"default": {"compaction": {"api": "oai"}}}})


def test_compaction_provider_config_per_provider_wins_over_global():
    from minacode.config import compaction_provider_config

    config = Config.from_dict(
        {
            "compaction": {"model": "global-m", "reasoning": "high", "api": "responses"},
            "provider": {
                "active": "default",
                "default": {"model": "d", "compaction": {"model": "per-m"}},
                "fast": {"model": "f"},
            },
        }
    )
    # Per-provider (default) wins per field; unset per fields inherit the global section.
    entry = compaction_provider_config(config)
    assert entry.model == "per-m"
    assert entry.reasoning == "high"
    assert entry.api == "responses"

    # Per-provider empty: the global value applies; both empty: the entry's own value.
    config.providers["default"].compaction_model = ""
    config.compaction_model = ""
    entry = compaction_provider_config(config)
    assert entry.model == "d"
    assert entry.reasoning == "high"
    assert entry.api == "responses"


def test_compaction_provider_config_per_provider_follows_base_entry():
    from minacode.config import compaction_provider_config

    config = Config.from_dict(
        {
            "compaction": {"provider": "fast"},
            "provider": {
                "active": "default",
                "default": {"model": "d", "compaction": {"model": "active-per"}},
                "fast": {"model": "f", "compaction": {"model": "fast-per"}},
            },
        }
    )
    # The base entry is "fast": its nested table wins, not the active entry's.
    entry = compaction_provider_config(config)
    assert entry.model == "fast-per"
    assert config.providers["fast"].compaction_model == "fast-per"
    assert config.providers["default"].compaction_model == "active-per"  # untouched
    assert entry is not config.providers["fast"]


def _compaction_bar_session(tmp_path, **compaction):
    config = Config.from_dict(
        {
            "compaction": compaction,
            "provider": {
                "active": "default",
                "default": {"model": "big-model", "url": "http://test", "key": "k", "reasoning": "high"},
                "cheap": {"model": "small-model", "url": "http://test", "key": "k"},
            },
        }
    )
    s = Session(cwd=str(tmp_path), config=config)
    s.config.data_dir = str(tmp_path / "data")
    return s


def test_status_bar_names_the_entry_a_summary_runs_on(tmp_path):
    """A summary on its own provider entry is the same situation as an in-flight worker: the
    request on the wire is not the row's model, so the bar names the one that is."""
    s = _compaction_bar_session(tmp_path, provider="cheap", model="haiku", reasoning="off")
    bar = StatusBar(s)

    assert " | ".join(text for text, _ in bar.entries(show_elapsed=False)).startswith("default/big-model | high")

    s.state.compaction_entry = "cheap/haiku"
    entries = bar.entries(show_elapsed=False)
    assert entries[:3] == [("[compaction]", "ctx"), ("cheap/haiku", "warn"), ("off", "reason")]

    # Cleared when the request ends: the row goes back to the conversation's own model.
    s.state.compaction_entry = ""
    assert bar.entries(show_elapsed=False)[0] == ("default/big-model", "provider")


def test_status_bar_marks_a_compaction_running_on_the_row_own_entry(tmp_path):
    """With no [compaction] overrides the resolved entry is the active one, and the row keeps its
    own provider segments -- but it still says a summary is what the wait is for. Naming the phase
    is the point: a compaction otherwise looks exactly like an ordinary request."""
    s = _compaction_bar_session(tmp_path)
    bar = StatusBar(s)
    s.state.compaction_entry = "default/big-model"

    assert bar.entries(show_elapsed=False)[:3] == [("[compaction]", "ctx"), ("default/big-model", "provider"), ("high", "reason")]

    s.state.compaction_entry = ""
    assert bar.entries(show_elapsed=False)[0] == ("default/big-model", "provider")


def test_status_bar_output_rate_reads_the_stream_that_is_running(tmp_path):
    """The rate belongs to the response being watched: no stream, no number. It is an estimate from
    streamed characters, because token deltas are not on the wire."""
    s = _compaction_bar_session(tmp_path)
    bar = StatusBar(s)

    assert bar.output_rate() == ""  # nothing streaming

    s.state.stream_started_at = time.monotonic() - 2.0
    s.state.stream_chars = 400
    assert bar.output_rate() == "↓ 50 tok/s"

    # Suppressed inside the first second, where a chunk over a near-zero elapsed reads as a wild number.
    s.state.stream_started_at = time.monotonic() - 0.2
    assert bar.output_rate() == ""

    # Cleared when the request ends, so a finished response does not freeze a rate on the divider.
    s.state.stream_started_at = 0.0
    assert bar.output_rate() == ""


def test_status_bar_output_rate_follows_an_in_flight_worker(tmp_path):
    """Same in-flight predicate as every other value on the row: while a delegation runs, the speed
    shown is the worker's, and it goes back to the parent's the moment the worker answers."""
    s = _compaction_bar_session(tmp_path)
    worker = Session(cwd=str(tmp_path), config=s.config)
    s.worker = worker
    bar = StatusBar(s)

    worker.state.stream_started_at = time.monotonic() - 2.0
    worker.state.stream_chars = 800
    assert bar.output_rate() == ""  # an idle worker never shadows the parent

    worker._active_turn_messages = [{"role": "user", "content": "order"}]
    assert bar.output_rate() == "↓ 100 tok/s"


def test_model_client_counts_streamed_output_per_request(tmp_path):
    """One funnel for every API shape, reasoning deltas included: the wait is made of both."""
    s = _compaction_bar_session(tmp_path)
    model = ModelClient(s)

    model._emit_stream("reasoning", "abcd")
    model._emit_stream("output", "efgh")
    assert s.state.stream_chars == 8
    assert s.state.stream_started_at > 0  # started at the first delta, not at the request

    # The next attempt starts from zero: a rate must not blend two responses.
    s.state.stream_started_at, s.state.stream_chars = 0.0, 0
    model._emit_stream("output", "ij")
    assert s.state.stream_chars == 2


def test_compaction_entry_is_cleared_when_the_summary_fails(tmp_path, monkeypatch):
    """The label is live display state: a timeout, a cancel, or a provider error must not leave a
    stale row naming a request that is no longer running."""
    s = _compaction_bar_session(tmp_path, provider="cheap", model="haiku")
    model = ModelClient(s)

    def explode(*_args, **_kwargs):
        assert s.state.compaction_entry == "cheap/haiku"  # set while the request is in flight
        raise ModelError("provider said no")

    monkeypatch.setattr(model, "api_request", explode)

    with pytest.raises(ModelError):
        compaction.Compactor(ContextManager(s), model).compact("context")
    assert s.state.compaction_entry == ""


def test_compaction_refuses_an_incomplete_entry_by_name(tmp_path):
    """The client's own gate checks the active provider, which is the wrong entry when a summary
    runs elsewhere. Without this the SDK reports "Missing credentials", naming nothing the user
    can act on, and compaction silently degrades to trimming on every pass."""
    config = Config.from_dict(
        {
            "provider": {
                "active": "main",
                "main": {"url": "http://test", "key": "k", "model": "big"},
                "cheap": {"url": "http://test"},  # no key, no model
            },
            "compaction": {"provider": "cheap"},
        }
    )
    s = Session(cwd=str(tmp_path), config=config)
    s.config.data_dir = str(tmp_path / "data")

    assert s.missing_config() == []  # the active entry is complete; only the compaction one is not
    with pytest.raises(ModelError, match=r"compaction provider `cheap` is missing key, model"):
        compaction.Compactor(ContextManager(s), ModelClient(s)).compact("context")
    assert s.state.compaction_entry == ""  # refused before the request, so no stale status row


def test_provider_entry_reports_its_own_missing_fields(tmp_path):
    """One definition of a usable entry, shared by the active-provider gate and compaction."""
    config = Config.from_dict({"provider": {"active": "p", "p": {"url": "", "key": "k", "model": ""}}})
    s = Session(cwd=str(tmp_path), config=config)

    assert config.providers["p"].missing_fields() == ["url", "model"]
    assert s.missing_config() == ["provider.url", "provider.model"]


def test_summary_tokens_are_counted_apart_from_the_conversation(tmp_path, monkeypatch):
    """A summary can be billed to another account at another price, and is a fresh prefix that
    never hits the conversation's cache. One blended total can be multiplied by neither price."""
    config = Config.from_dict(
        {
            "provider": {
                "active": "main",
                "main": {"url": "http://test", "key": "k", "model": "big"},
                "cheap": {"url": "http://test", "key": "k", "model": "small"},
            },
            "compaction": {"provider": "cheap", "model": "small-flash"},
        }
    )
    s = Session(cwd=str(tmp_path), config=config)
    s.config.data_dir = str(tmp_path / "data")
    s.usage.add({"prompt_tokens": 120_000, "completion_tokens": 900, "total_tokens": 120_900}, 200_000)
    model = ModelClient(s)
    monkeypatch.setattr(
        model,
        "client",
        _MockClientFactory(
            [
                (
                    200,
                    {
                        "id": "c",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "small-flash",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"summary":"s"}'}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 95_000, "completion_tokens": 700, "total_tokens": 95_700},
                    },
                )
            ]
        ),
    )

    compaction.Compactor(ContextManager(s), model).compact("long context")

    assert (s.usage.calls, s.usage.total_tokens) == (1, 120_900)  # the conversation's row is untouched
    assert (s.compaction_usage.calls, s.compaction_usage.total_tokens) == (1, 95_700)
    # The summary request also refreshes its own counter's last-request snapshot, which the status
    # bar's compaction row reads; the conversation row's snapshot is not overwritten.
    assert s.compaction_usage.last_prompt_tokens == 95_000
    assert (s.usage.last_prompt_tokens, s.usage.last_prompt_budget) == (120_000, 200_000)


def test_compaction_usage_survives_a_resume(tmp_path):
    config = Config.from_dict({"provider": {"active": "p", "p": {"url": "http://test", "key": "k", "model": "m"}}})
    config.data_dir = str(tmp_path / "data")
    s = Session(cwd=str(tmp_path), config=config)
    s.messages.append({"role": "user", "content": "hello"})
    s.compaction_usage.add({"prompt_tokens": 95_000, "completion_tokens": 700, "total_tokens": 95_700}, 200_000)
    s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=config, cwd=str(tmp_path))

    assert restored.compaction_usage.total_tokens == 95_700
    assert restored.compaction_usage.calls == 1
    # A snapshot written before the field existed decodes to zeros, not an error.
    assert SessionSnapshotCodec.model_usage({}).total_tokens == 0


def test_deepseek_tool_call_replay_travels_with_the_model_not_the_endpoint():
    """DeepSeek returns 400 when a tool-call turn comes back without its reasoning, and ignores it
    everywhere else. That is a property of the model, so a gateway serving it inherits both halves."""
    direct = ProviderConfig.from_dict({"url": "https://api.deepseek.com/v1", "model": "deepseek-v4-chat"})
    gateway = ProviderConfig.from_dict({"url": "https://opencode.ai/zen/v1", "model": "deepseek-v4-chat"})

    assert resolve(direct).chat_reasoning_history == "tool_calls"
    assert resolve(gateway).chat_reasoning_history == "tool_calls"
    # A model the gateway does not document keeps the generic full-replay default.
    assert resolve(ProviderConfig.from_dict({"url": "https://opencode.ai/zen/v1", "model": "vendor/model"})).chat_reasoning_history == "all"
