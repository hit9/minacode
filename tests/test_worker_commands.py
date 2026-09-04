"""worker commands (split from tests/test_worker_handoff.py)."""

import json

import pytest
from agent_harness import session
from test_worker_handoff import FakeModelClient, _delegate_call, _delegate_runner, _delegate_session, _worker_history_for_compaction

from wizolt.cli.worker import worker_command
from wizolt.prompts import WORKER_PROMPT


def test_worker_config_parses_model_and_reasoning(tmp_path):
    from wizolt.config import (
        Config,
    )

    config = Config.from_dict(
        {
            "worker": {"provider": "fast", "model": "m-x", "reasoning": "high", "api": "responses"},
            "provider": {"active": "default", "default": {"model": "d"}, "fast": {"model": "m"}},
        }
    )
    assert config.worker_provider == "fast"
    assert config.worker_model == "m-x"
    assert config.worker_reasoning == "high"
    assert config.worker_api == "responses"

    # Defaults: no [worker] model/reasoning/api means "inherit the entry's value" at spawn time.
    plain = Config.from_dict({"provider": {"default": {"model": "d"}}})
    assert plain.worker_model == "" and plain.worker_reasoning == "" and plain.worker_api == ""


def test_worker_config_rejects_invalid_worker_reasoning(tmp_path):
    from wizolt.base import ConfigError
    from wizolt.config import (
        Config,
    )

    with pytest.raises(ConfigError, match="worker.reasoning"):
        Config.from_dict({"worker": {"reasoning": "turbo"}, "provider": {"default": {}}})


def test_worker_config_rejects_invalid_worker_api(tmp_path):
    from wizolt.base import ConfigError
    from wizolt.config import (
        Config,
    )

    with pytest.raises(ConfigError, match="worker.api"):
        Config.from_dict({"worker": {"api": "oai"}, "provider": {"default": {}}})


def test_worker_provider_config_applies_api_override(tmp_path):
    """worker_provider_config folds an explicit worker.api into the detached entry; an empty
    worker_api inherits the entry's own protocol (the worker never shares the parent's object)."""
    from wizolt.config import (
        Config,
    )
    from wizolt.tools.delegate import worker_provider_config

    config = Config.from_dict(
        {
            "worker": {"provider": "fast", "api": "chat"},
            "provider": {"active": "default", "default": {"model": "d", "api": "auto"}, "fast": {"model": "m", "api": "anthropic"}},
        }
    )
    entry = worker_provider_config(config, "fast")
    assert entry.api == "chat"  # the [worker] api override wins
    assert entry is not config.providers["fast"]
    assert config.providers["fast"].api == "anthropic"  # the parent's entry is untouched

    config.worker_api = ""
    entry = worker_provider_config(config, "fast")
    assert entry.api == "anthropic"  # empty override inherits the entry's api

    entry = worker_provider_config(config, "default")
    assert entry.api == "auto"


def test_worker_provider_command_does_not_flip_registration_gate(tmp_path):
    from wizolt.cli import CommandLoop
    from wizolt.config import (
        ProviderConfig,
    )
    from wizolt.engine import Agent
    from wizolt.session import Session
    from wizolt.tools import Tool

    parent = session(tmp_path)
    parent.config.providers["alt"] = ProviderConfig(model="m")
    agent = Agent(parent, output_fn=lambda text: None)
    loop = CommandLoop(agent, input_fn=lambda prompt: "", output_fn=lambda text: None)

    def names(s):
        return {schema["function"]["name"] for schema in Tool.resolved_schemas(s)}

    parent.settings.worker = True
    assert parent.worker_tool_enabled is False
    assert "Delegate" not in names(parent)
    # Frozen off: the command stores the value for the next spawn and says a restart is needed;
    # the tool block is unchanged mid-session.
    assert worker_command(loop, "provider alt") == "Set worker provider = alt (delegation is off this session; takes effect after a restart)"
    assert parent.config.worker_provider == "alt"
    assert "Delegate" not in names(parent)
    # "off" clears quietly when the gate is frozen off.
    assert worker_command(loop, "provider off") == "worker provider: off"
    assert parent.config.worker_provider == ""

    before = parent.config.worker_provider
    assert worker_command(loop, "provider nope") == "Unknown provider: nope"
    assert parent.config.worker_provider == before

    # Simulating a restart: a freshly constructed session over the same config re-evaluates the
    # frozen gate, so the stored value registers Delegate...
    parent.config.worker_provider = "alt"
    fresh = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings)
    fresh.settings.worker = True
    assert fresh.worker_tool_enabled is True
    assert "Delegate" in names(fresh)
    # ...and the frozen-on gate stays registered across runtime changes, including clearing the
    # provider; only the next session re-evaluates it.
    fresh_agent = Agent(fresh, output_fn=lambda text: None)
    fresh_loop = CommandLoop(fresh_agent, input_fn=lambda prompt: "", output_fn=lambda text: None)
    assert worker_command(fresh_loop, "provider off") == "worker provider: off"
    assert fresh.config.worker_provider == ""
    assert "Delegate" in names(fresh)


def test_worker_provider_off_selects_literal_off_entry(tmp_path):
    from wizolt.cli import CommandLoop
    from wizolt.config import (
        ProviderConfig,
    )
    from wizolt.engine import Agent

    parent = session(tmp_path)
    parent.config.providers["off"] = ProviderConfig(model="m")
    agent = Agent(parent, output_fn=lambda text: None)
    loop = CommandLoop(agent, input_fn=lambda prompt: "", output_fn=lambda text: None)

    assert worker_command(loop, "provider off") == "Set worker provider = off (delegation is off this session; takes effect after a restart)"
    assert parent.config.worker_provider == "off"


def test_worker_model_and_reason_overrides(tmp_path):
    from wizolt.cli import CommandLoop
    from wizolt.engine import Agent

    parent = session(tmp_path)
    agent = Agent(parent, output_fn=lambda text: None)
    loop = CommandLoop(agent, input_fn=lambda prompt: "", output_fn=lambda text: None)

    assert worker_command(loop, "model") == "worker model: (inherit)"
    assert worker_command(loop, "model gpt-5.2") == "Set worker.model = gpt-5.2"
    assert parent.config.worker_model == "gpt-5.2"
    assert worker_command(loop, "model") == "worker model: gpt-5.2"
    assert worker_command(loop, "model default") == "worker model: (inherit)"
    assert parent.config.worker_model == ""

    assert worker_command(loop, "reason high") == "Set worker.reasoning = high"
    assert parent.config.worker_reasoning == "high"
    assert worker_command(loop, "reason off") == "Set worker.reasoning = off"  # a valid effort
    assert parent.config.worker_reasoning == "off"
    assert worker_command(loop, "reason default") == "worker reasoning: (inherit)"
    assert parent.config.worker_reasoning == ""

    choices = ("off", *parent.policy.effort_order)
    assert worker_command(loop, "reason turbo") == "Usage: /worker reason " + "|".join(choices)
    assert worker_command(loop, "provider a b") == "Usage: /worker provider [NAME]"
    assert worker_command(loop, "model a b") == "Usage: /worker model [MODEL]"
    assert worker_command(loop, "reason a b") == "Usage: /worker reason [EFFORT]"


async def test_delegate_spawn_isolates_provider_and_applies_overrides(tmp_path, monkeypatch):
    from wizolt.config import (
        ProviderConfig,
    )
    from wizolt.session import SessionSnapshotStore

    parent = _delegate_session(tmp_path)
    parent.config.providers["alt"] = ProviderConfig(model="m")
    parent.config.worker_provider = "alt"
    parent.config.worker_model = "worker-model"
    parent.config.worker_reasoning = "high"
    parent.messages.append({"role": "user", "content": "parent request"})
    parent.save_snapshot()
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    await _delegate_call(parent, runner, action="send", order="o")

    worker_provider = parent.worker.config.provider
    assert worker_provider is not parent.config.providers["alt"]
    assert parent.worker.config.providers is not parent.config.providers
    assert worker_provider.model == "worker-model"
    assert worker_provider.reasoning == "high"
    assert parent.config.providers["alt"].model == "m"

    # Mutating the worker's active entry never leaks into the parent's providers entry.
    worker_provider.model = "mutated"
    assert parent.config.providers["alt"].model == "m"

    # Resume: the worker comes back through SessionSnapshotStore.load with the same freshly built
    # config, so a current override applies to the restored worker too.
    parent.worker.messages.append({"role": "user", "content": "worker request"})
    parent.worker.save_snapshot()
    model.script.append(({"role": "assistant", "content": "two"}, [], "two"))
    parent.config.worker_model = "resumed-model"
    fresh = SessionSnapshotStore.load(parent.uid, config=parent.config, settings=parent.settings, cwd=str(tmp_path))
    runner = _delegate_runner(fresh)
    await _delegate_call(fresh, runner, action="send", order="o")
    assert fresh.worker.config.provider.model == "resumed-model"


async def test_worker_model_switch_applies_to_live_worker(tmp_path, monkeypatch):
    from wizolt.cli import CommandLoop
    from wizolt.engine import Agent

    parent = _delegate_session(tmp_path)
    parent.config.providers["default"].model = "parent-model"
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    await _delegate_call(parent, runner, action="send", order="o")

    agent = Agent(parent, output_fn=lambda text: None)
    loop = CommandLoop(agent, input_fn=lambda prompt: "", output_fn=lambda text: None)
    worker = parent.worker
    assert worker.config.provider.model == "parent-model"

    worker_command(loop, "model worker-model")
    assert worker.config.provider.model == "worker-model"
    assert parent.config.providers["default"].model == "parent-model"  # untouched

    worker_command(loop, "model default")
    assert worker.config.provider.model == "parent-model"  # restores the entry's model
    assert parent.config.providers["default"].model == "parent-model"


async def test_worker_provider_switch_applies_to_live_worker(tmp_path, monkeypatch):
    from wizolt.cli import CommandLoop
    from wizolt.config import (
        ProviderConfig,
    )
    from wizolt.engine import Agent

    parent = _delegate_session(tmp_path)
    parent.config.providers["alt"] = ProviderConfig(model="m")
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    await _delegate_call(parent, runner, action="send", order="o")

    agent = Agent(parent, output_fn=lambda text: None)
    loop = CommandLoop(agent, input_fn=lambda prompt: "", output_fn=lambda text: None)
    worker = parent.worker

    worker_command(loop, "provider alt")
    assert worker.config.active_provider == "alt"
    assert worker.config.provider is not parent.config.providers["alt"]
    assert worker.config.provider.model == "m"
    assert parent.config.providers["alt"].model == "m"  # untouched


async def test_delegate_send_finish_display_summary_and_preview(tmp_path, monkeypatch):
    from wizolt.base import LogBlock, LogRole, ToolCall
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    model = FakeModelClient([({"role": "assistant", "content": "the worker answer"}, [], "the worker answer")])
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    outputs = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=outputs.append)
    status, _, _ = await runner.run_one(ToolCall("delegate-1", "Delegate", [{"action": "send", "order": "o"}]))
    assert status == "ok"

    blocks = [item for item in outputs if isinstance(item, LogBlock)]
    # The confirmation line shows the short root (`Delegate send`, not the order blob); the finish
    # block is the one with OUTPUT children (the worker's answer preview).
    assert any(block.items and block.items[0].label == "Delegate" and block.items[0].text == "send" for block in blocks)
    finish = next(block for block in blocks if any(item.role is LogRole.OUTPUT for item, _ in block.walk()))
    # The finish block is the closing marker of the delegation bracket, so it carries the same
    # yellow [worker] identity as the start marker: a root line whose label is the bracket tag.
    assert finish.items[0].label == "[worker]" and finish.items[0].text == "◀"
    rendered = str(finish)
    assert "steps 1" in rendered and "(none)" in rendered
    assert "the worker answer" in rendered
    assert "<Delegate" not in rendered and "<worker>" not in rendered and "</worker>" not in rendered
    assert any(item.label == "stored" for item, _ in finish.walk())


async def test_delegate_send_finish_worker_rule_label_and_preview(tmp_path, monkeypatch):
    from wizolt.base import LogBlock, LogRole, ToolCall
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    model = FakeModelClient([({"role": "assistant", "content": "the worker answer"}, [], "the worker answer")])
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    outputs = []
    labels = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=outputs.append)
    runner.worker_rule = lambda label: labels.append(label)
    status, _, _ = await runner.run_one(ToolCall("delegate-1", "Delegate", [{"action": "send", "order": "o"}]))
    assert status == "ok"

    done = [label for label in labels if label.startswith("worker done · ")]
    assert done, "the finish worker_rule callback never fired"
    assert "worker done · steps 1" in done[0] and " in / " in done[0]
    assert "(none)" not in done[0]  # no files touched: the files segment is omitted, not '(none)'

    blocks = [item for item in outputs if isinstance(item, LogBlock)]
    finish = next(block for block in blocks if any(item.role is LogRole.OUTPUT for item, _ in block.walk()))
    rendered = str(finish)
    assert "the worker answer" in rendered
    assert any(item.label == "stored" for item, _ in finish.walk())
    # The done summary lives in the rule label now, not as a child line of the finish block.
    assert not any(item.label == "done" and item.text.startswith("steps ") for item, _ in finish.walk())


async def test_delegate_send_finish_worker_rule_label_carries_title(tmp_path, monkeypatch):
    from wizolt.base import ToolCall
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    model = FakeModelClient([({"role": "assistant", "content": "the worker answer"}, [], "the worker answer")])
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    labels = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=lambda text: None)
    runner.worker_rule = lambda label: labels.append(label)
    status, _, _ = await runner.run_one(ToolCall("delegate-1", "Delegate", [{"action": "send", "order": "o", "title": "fix /status blank line"}]))
    assert status == "ok"

    done = [label for label in labels if label.startswith("worker done · ")]
    assert done, "the finish worker_rule callback never fired"
    assert done[0].startswith("worker done · fix /status blank line · steps 1")


async def test_delegate_send_finish_display_prints_full_answer_and_folded_preview(tmp_path, monkeypatch):
    from wizolt.base import LogBlock, LogRole, ToolCall
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    answer = "\n".join(f"report line {i}" for i in range(40))
    model = FakeModelClient([({"role": "assistant", "content": answer}, [], answer)])
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    outputs = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=outputs.append)
    status, _, _ = await runner.run_one(ToolCall("delegate-1", "Delegate", [{"action": "send", "order": "o"}]))
    assert status == "ok"

    blocks = [item for item in outputs if isinstance(item, LogBlock)]
    # The worker's own output (agent.output_fn) prints the whole answer as one AUTO block.
    full = next(
        block
        for block in blocks
        if any(item.role is LogRole.AUTO and "report line 0" in item.text and "report line 39" in item.text for item, _ in block.walk())
    )
    assert "report line 20" in str(full)  # the middle of the answer survived
    # The finish block's preview is still the folded three-line form (head, omitted marker, tail).
    finish = next(block for block in blocks if block is not full and any(item.role is LogRole.OUTPUT for item, _ in block.walk()))
    rendered = str(finish)
    assert "lines omitted" in rendered
    assert "report line 20" not in rendered  # the folded preview only carries the head and tail


async def test_delegate_send_routes_the_final_report_through_worker_answer(tmp_path, monkeypatch):
    from wizolt.base import LogBlock, LogRole, ToolCall
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    model = FakeModelClient([({"role": "assistant", "content": "the report"}, [], "the report")])
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    answers = []
    outputs = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=outputs.append)
    runner.worker_answer = answers.append
    status, _, _ = await runner.run_one(ToolCall("delegate-1", "Delegate", [{"action": "send", "order": "o"}]))
    assert status == "ok"

    assert answers == ["the report"]  # the report went through the markdown hook, exactly once
    auto = [block for block in outputs if isinstance(block, LogBlock) and any(item.role is LogRole.AUTO for item, _ in block.walk())]
    assert not auto  # nothing on the plain output channel carries the final report


async def test_delegate_reset_finish_display_worker_root_and_cleared_notice(tmp_path):
    from wizolt.base import LogBlock, ToolCall
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner
    from wizolt.session import Session

    parent = _delegate_session(tmp_path)
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    worker.save_snapshot()
    parent.worker = worker
    outputs = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=outputs.append)

    status, _, _ = await runner.run_one(ToolCall("delegate-r", "Delegate", [{"action": "reset"}]))
    assert status == "ok"

    blocks = [item for item in outputs if isinstance(item, LogBlock)]
    finish = next(block for block in blocks if any(item.label == "done" for item, _ in block.walk()))
    # Reset keeps its ordinary tool root (the short_call, not [worker] ◀) and a done child.
    assert finish.items[0].label != "[worker]"
    assert "worker context cleared" in str(finish)


async def test_delegate_reset_finish_worker_rule_label(tmp_path):
    from wizolt.base import ToolCall
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner
    from wizolt.session import Session

    parent = _delegate_session(tmp_path)
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    worker.save_snapshot()
    parent.worker = worker
    outputs = []
    labels = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=outputs.append)
    runner.worker_rule = lambda label: labels.append(label)

    status, _, _ = await runner.run_one(ToolCall("delegate-r", "Delegate", [{"action": "reset"}]))
    assert status == "ok"

    # Reset is a one-shot tool call, not a delegation bracket: no full-width worker_rule rule
    # fires; the reset shows as an ordinary tool root with a plain done child.
    assert labels == [], "reset must not emit a worker_rule divider"
    from wizolt.base import LogBlock

    done = [item for block in outputs if isinstance(block, LogBlock) for item, _ in block.walk() if item.label == "done"]
    assert done, "reset should keep its ordinary tool root with a done child"
    assert "worker context cleared" in next(item.text for block in outputs if isinstance(block, LogBlock) for item, _ in block.walk() if item.label == "done")


def test_worker_stream_forwards_output_and_suppresses_output_done_promote():
    from wizolt.tools.delegate import _worker_stream

    calls: list[tuple[str, str]] = []

    class StubRunner:
        def __init__(self):
            self.model_stream = lambda kind, text: calls.append((kind, text))

    stream = _worker_stream(StubRunner())

    stream("output", "x")
    stream("output_done", "t")
    stream("", "")
    stream("tool", "Bash")

    assert calls == [("output", "x"), ("", ""), ("", ""), ("tool", "Bash")]
    assert all(kind != "output_done" for kind, _ in calls)


async def test_worker_compaction_triggers_on_budget_overrun(tmp_path, monkeypatch):
    from wizolt.prompts import COMPACTION_SUMMARY_TITLE, PREVIOUS_CONTEXT_TRIMMED

    parent = _delegate_session(tmp_path)
    # A real delegation first: the worker is spawned through DelegateTool._send and keeps its
    # agent (ModelClient + ContextManager). The budget must be tight enough that the estimate
    # overruns it: 40k context -> 19_520 request budget (16_384 output reserve + 4_096 safety).
    parent.settings.max_context_tokens = 40_000
    model = FakeModelClient([({"role": "assistant", "content": "answer one"}, [], "answer one")])
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    await _delegate_call(parent, runner, action="send", order="order one")
    worker = _worker_history_for_compaction(parent)

    calls = []
    worker._agent.context.on_compaction = lambda active, _error: calls.append(active)
    messages = await worker._agent.context.prepare_messages(worker._agent.model, WORKER_PROMPT, turn_messages=None)
    # One compaction, with the lifecycle callback bracketing the phase (True then False).
    assert worker.state.compaction_count == 1
    assert calls == [True, False]
    # FakeModelClient has no `compact`, so the deterministic-trim fallback leaves its note in the
    # session summary instead of calling a model.
    assert PREVIOUS_CONTEXT_TRIMMED in worker.state.summary
    # The compacted head is replaced by a single checkpoint marker: 9 messages -> summary + keep.
    assert len(worker.messages) == 2
    assert any(str(m.get("content") or "").startswith(COMPACTION_SUMMARY_TITLE) for m in worker.messages)
    # The oversized history is gone from the projection, while the checkpoint survives; the
    # compacted head was stored as a recallable history segment.
    projected = [str(m.get("content") or "") for m in messages]
    assert any(content.startswith(COMPACTION_SUMMARY_TITLE) for content in projected)
    assert not any("u1" in content for content in projected)
    assert worker.history and worker.history[-1].key == "seg.1"
    # The overdue-by-usage guard resets: compaction clears the last-* usage fields.
    assert worker.usage.last_prompt_budget == 0


async def test_worker_compaction_persists_and_flows_into_next_delegation(tmp_path, monkeypatch):
    from wizolt.prompts import COMPACTION_SUMMARY_TITLE, PREVIOUS_CONTEXT_TRIMMED
    from wizolt.session import SessionSnapshotStore

    parent = _delegate_session(tmp_path)
    parent.settings.max_context_tokens = 40_000
    model = FakeModelClient(
        [
            ({"role": "assistant", "content": "answer one"}, [], "answer one"),
            ({"role": "assistant", "content": "answer two"}, [], "answer two"),
        ]
    )
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    await _delegate_call(parent, runner, action="send", order="order one")
    worker = _worker_history_for_compaction(parent)

    await worker._agent.context.prepare_messages(worker._agent.model, WORKER_PROMPT, turn_messages=None)
    assert worker.state.compaction_count == 1

    # Persistence: the snapshot holds the compacted state, not the pre-compaction history.
    worker.save_snapshot()
    loaded = SessionSnapshotStore.load(worker.uid, config=worker.config, settings=worker.settings, cwd=worker.cwd)
    assert loaded.state.compaction_count == 1
    assert PREVIOUS_CONTEXT_TRIMMED in loaded.state.summary
    assert [m.get("content") for m in loaded.messages[:2]] == [m.get("content") for m in worker.messages]
    assert not any("x" * 1000 in str(m.get("content") or "") for m in loaded.messages)

    # Continuity: the next delegation runs on the compacted context (summary in, oversized
    # history out) and does not re-compact.
    calls = []
    worker._agent.context.on_compaction = lambda active, _error: calls.append(active)
    await _delegate_call(parent, runner, action="send", order="order two")
    assert calls == []
    assert worker.state.compaction_count == 1
    second = json.dumps(model.requests[1])
    assert COMPACTION_SUMMARY_TITLE in second
    assert "order two" in second
    assert "x" * 200 not in second


def test_worker_model_discovery_shows_loading_state(tmp_path, monkeypatch):
    """The /worker model stage shows the same dispatch note as /model while remote discovery
    runs, and drops it afterwards; without credentials the note never appears."""
    from wizolt.cli import CommandLoop
    from wizolt.cli import commands as commands_mod
    from wizolt.config import ProviderConfig
    from wizolt.engine import Agent
    from wizolt.tui.app import TuiApp

    parent = session(tmp_path)
    parent.config.providers["fast"] = ProviderConfig(model="m", url="https://example.com/v1", key="key")
    parent.config.worker_provider = "fast"
    agent = Agent(parent, output_fn=lambda text: None)
    loop = CommandLoop(agent, input_fn=lambda prompt: "", output_fn=lambda text: None)
    loop.interactive_input = True
    transitions = []
    loop.tui = TuiApp()
    loop.tui.set_dispatching = lambda prompt="": transitions.append(prompt)
    monkeypatch.setattr(commands_mod, "remote_models", lambda _loop, _provider: ("remote-model",))

    # Non-interactive select_choice yields nothing, but remote discovery still runs (and
    # still shows the loading note while it does): the entry has credentials.
    assert worker_command(loop, "model") == "worker model: (inherit)"
    assert transitions == ["Loading models...", ""]
    transitions.clear()

    selected = iter(["remote-model"])
    monkeypatch.setattr("wizolt.cli.worker.select_choice", lambda *_args, **_kwargs: next(selected))
    assert "Set worker.model = remote-model" in worker_command(loop, "model")
    assert transitions == ["Loading models...", ""]

    # No url/key on the entry: no remote call, so no loading note either.
    parent.config.providers["fast"].url = ""
    parent.config.providers["fast"].key = ""
    selected = iter(["default"])
    monkeypatch.setattr("wizolt.cli.worker.select_choice", lambda *_args, **_kwargs: next(selected))
    transitions.clear()
    assert "worker model: (inherit)" in worker_command(loop, "model")
    assert transitions == []
