"""context budget (split from tests/test_context.py)."""
import json
import threading
from types import SimpleNamespace

from agent_harness import session, session_with_provider

from wizolt.config import (
    DEFAULT_OUTPUT_RESERVE_TOKENS,
    MIN_CONTEXT_SAFETY_TOKENS,
)
from wizolt.context import ContextManager
from wizolt.engine import Agent
from wizolt.prompts import (
    COMPACTION_SUMMARY_TITLE,
)
from wizolt.session import AgentState


def test_provider_context_limit_overrides_the_runtime_default(tmp_path):
    # Provider entries are effectively per-model, so a 1M-window model should not have to share one
    # global number with a 128K one. Unset (0) inherits, and the budget is resolved per call so
    # switching the active entry moves it.
    from wizolt.config import Config, request_budget_for

    config = Config.from_dict(
        {
            "provider": {
                "active": "big",
                "big": {"model": "wide", "max_context_tokens": 1_048_576},
                "small": {"model": "narrow", "max_context_tokens": 131_072},
                "plain": {"model": "default"},
            },
            "runtime": {"max_context_tokens": 262_144},
        }
    )
    runtime_default = 262_144
    assert config.providers["big"].context_token_limit(runtime_default) == 1_048_576
    assert config.providers["small"].context_token_limit(runtime_default) == 131_072
    assert config.providers["plain"].context_token_limit(runtime_default) == runtime_default  # 0 inherits

    s = session(tmp_path)
    s.config = config
    s.settings.max_context_tokens = runtime_default
    context = ContextManager(s)
    wide = context.request_token_budget()

    s.config.active_provider = "small"  # what /provider does
    narrow = context.request_token_budget()

    assert wide > narrow, "the budget must follow the active provider entry"
    assert wide == request_budget_for(1_048_576, config.providers["big"].output_token_budget())
    assert narrow == request_budget_for(131_072, config.providers["small"].output_token_budget())

def test_the_context_budget_has_exactly_one_definition():
    """Every consumer must derive the budget from Session.request_token_budget().

    This is the regression guard, not a style rule. The budget is read by compaction, the usage
    recorder, and two renderers; each one that computes it itself is a place the provider override
    can silently stop applying, which is exactly how `runtime.max_context_tokens` used to leak back
    in. `request_budget_for` is pure and takes a plain int, so a new call site that passes
    `settings.max_context_tokens` type-checks, passes its own tests, and quietly ignores the entry's
    limit. Keep the call sites at one; route new readers through the session.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "wizolt"
    callers = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "request_budget_for(" in path.read_text(encoding="utf-8") and path.name != "config.py"  # config.py defines it
    )
    assert callers == ["session/__init__.py"], (
        f"the context budget is computed in {callers}; call Session.request_token_budget() instead so the "
        "per-provider max_context_tokens override cannot be bypassed"
    )

def test_provider_context_limit_reaches_every_budget_reader(tmp_path):
    # End to end, through the real Agent: a provider-level limit must drive the compaction budget and
    # the budget recorded for the status row, with no path still reading the runtime default.
    from wizolt.config import Config, request_budget_for

    s = session(tmp_path)
    s.config = Config.from_dict({"provider": {"active": "wide", "wide": {"model": "m", "max_context_tokens": 1_048_576}}})
    s.settings.max_context_tokens = 262_144  # the runtime default, deliberately much smaller
    expected = request_budget_for(1_048_576, s.config.provider.output_token_budget())
    runtime_only = request_budget_for(262_144, s.config.provider.output_token_budget())
    assert expected != runtime_only

    agent = Agent(s, output_fn=lambda text: None)
    assert s.request_token_budget() == expected
    assert agent.context.request_token_budget() == expected

    # The recorded budget is what /status and the status bar divide by, so it has to agree.
    type(agent.model)._record_usage(agent.model, SimpleNamespace(prompt_tokens=1_000, completion_tokens=1))
    assert s.usage.last_prompt_budget == expected

    # And the fill really is measured against the wide window, not the runtime default.
    assert s.usage.last_prompt_tokens * 100 // s.usage.last_prompt_budget == 1_000 * 100 // expected

def test_provider_context_limit_shares_one_denominator_with_usage(tmp_path):
    # The compaction trigger and the status-bar fill must be measured against the same number, or
    # the bar reads 60% while the context manager is already compacting.
    from types import SimpleNamespace

    from wizolt.config import Config, request_budget_for
    from wizolt.model import ModelClient

    s = session(tmp_path)
    s.config = Config.from_dict({"provider": {"active": "big", "big": {"model": "wide", "max_context_tokens": 1_048_576}}})
    s.settings.max_context_tokens = 262_144
    ModelClient._record_usage(SimpleNamespace(session=s), SimpleNamespace(prompt_tokens=10, completion_tokens=1))

    assert s.usage.last_prompt_budget == ContextManager(s).request_token_budget()
    assert s.usage.last_prompt_budget > request_budget_for(262_144, 16_384), "the runtime default was used, not the entry's"

def test_compaction_uses_configured_context_budget(tmp_path):
    s = session_with_provider(tmp_path)
    s.settings.max_context_tokens = 1
    # Note's, and a compaction must leave them exactly as they are even when the summarizer
    # volunteers replacements: they survive eviction on their own, so handing them over only lets
    # a prose re-derivation replace a validated structure -- and compound on the next pass.
    s.state.plan = AgentState.plan_items([{"status": "doing", "text": "the agent's own step"}])
    s.state.known = ["the agent's own fact"]
    s.messages = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old answer"},
        *({"role": "assistant", "content": f"recent {index}"} for index in range(8)),
        {"role": "user", "content": "latest"},
        {"role": "tool", "content": "tool kept"},
    ]
    context = ContextManager(s)
    compaction_phases = []
    context.on_compaction = lambda active, _error: compaction_phases.append(active)

    class FakeModel:
        def __init__(self, session):
            self.session = session
            self.input = None
            self.cancel_requested = threading.Event()

        def api_request(self, messages, _tools, **_kwargs):
            # The inline form carries the conversation as messages; the flattened text is only
            # built when that form cannot serve, so this reads whichever one was actually sent.
            self.input = "\n".join(str(message.get("content") or "") for message in messages)
            return "", "", json.dumps({"summary": "compact summary", "plan": ["next"], "known": ["fact"]})

        @staticmethod
        def parse_json_object(content):
            return json.loads(content)

    model = FakeModel(s)
    context.prepare_messages(model, "system", [{"role": "user", "content": "request"}])
    assert compaction_phases == [True, False]
    assert model.input is not None
    assert "old answer" in model.input
    assert "recent 7" in model.input  # this budget is 1 token, so the size bound collapses the tail
    assert "Compact the wizolt working context." in model.input  # the appended instruction
    assert "tool kept" not in model.input  # the kept tail is not handed to the summarizer
    assert "\nrequest" not in model.input  # nor the turn message; "request" alone occurs in the system prompt
    assert s.state.summary == "compact summary"
    assert [vars(item) for item in s.state.plan] == [{"status": "doing", "text": "the agent's own step"}]
    assert s.state.known == ["the agent's own fact"]
    assert [message["role"] for message in s.messages] == ["user", "user", "tool"]
    assert s.messages[0]["content"].startswith(COMPACTION_SUMMARY_TITLE)
    assert "compact summary" in s.messages[0]["content"]
    assert s.messages[1]["content"] == "latest"
    assert s.messages[2]["content"] == "tool kept"
    assert all("recent 7" not in str(message.get("content") or "") for message in s.messages)

def test_default_budget_leaves_more_input_room_than_the_previous_240k_ceiling(tmp_path):
    """The output reserve trades against the input budget one for one, so the two defaults are one
    decision: doubling the output cap only pays off because the ceiling rose further."""
    s = session(tmp_path)
    context = ContextManager(s)

    assert s.settings.max_context_tokens == 256 * 1024
    assert s.config.provider.output_token_budget() == DEFAULT_OUTPUT_RESERVE_TOKENS
    assert context.request_token_budget() > 240 * 1024 - 8_192 - MIN_CONTEXT_SAFETY_TOKENS

def test_compaction_budget_reserves_output_and_safety(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 100_000
    context = ContextManager(s)

    assert context.request_token_budget() == 100_000 - DEFAULT_OUTPUT_RESERVE_TOKENS - MIN_CONTEXT_SAFETY_TOKENS

    s.config.provider.max_tokens = 10_000
    assert context.request_token_budget() == 100_000 - 10_000 - MIN_CONTEXT_SAFETY_TOKENS

def test_tool_schemas_can_trigger_compaction_before_context_ceiling(tmp_path):
    s = session_with_provider(tmp_path)
    s.settings.max_context_tokens = 30_000
    s.messages = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "latest request"},
    ]
    context = ContextManager(s)
    turn = [{"role": "user", "content": "continue"}]
    tools = [{"type": "function", "function": {"name": "Large", "description": "x" * 80_000, "parameters": {}}}]
    messages = context.model_messages("system", turn)
    assert context.request_tokens(messages) < context.request_token_budget()
    assert context.request_token_budget() <= context.request_tokens(messages, tools) < s.settings.max_context_tokens

    class FakeModel:
        def __init__(self, session):
            self.session = session
            self.called = False
            self.cancel_requested = threading.Event()

        def api_request(self, _messages, _tools, **_kwargs):
            self.called = True
            return "", "", json.dumps({"summary": "summary"})

        @staticmethod
        def parse_json_object(content):
            return json.loads(content)

    model = FakeModel(s)
    context.prepare_messages(model, "system", turn, tools)

    assert model.called is True
