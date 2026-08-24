"""delegate send (split from tests/test_worker_handoff.py)."""

import pytest
from test_worker_handoff import FakeModelClient, _delegate_call, _delegate_runner, _delegate_session

from minacode.tools import tooloutput
from minacode.prompts import WORKER_PROMPT


def test_delegate_send_logs_a_worker_start_marker(tmp_path, monkeypatch):
    from minacode.base import LogBlock, LogRole, oneline
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    parent.config.providers["default"].model = "worker-model-x"
    order = "Rewrite the worker handoff plan to cover the start marker, then check it. " * 8
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    outputs = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=outputs.append)
    _delegate_call(parent, runner, action="send", order=order)

    blocks = [block for block in outputs if isinstance(block, LogBlock)]
    marker = next(block for block in blocks if any(item.role is LogRole.WORKER for item, _ in block.walk()))
    rendered = str(marker)
    assert "[worker]" in rendered
    assert "▶" in rendered
    assert "default/worker-model-x" in rendered
    assert oneline(order, 200) in rendered


def test_delegate_send_worker_rule_start_label(tmp_path, monkeypatch):
    from minacode.base import oneline
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    parent.config.providers["default"].model = "worker-model-x"
    order = "Rewrite the worker handoff plan to cover the start rule, then check it. " * 8
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    labels = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=lambda text: None)
    runner.worker_rule = lambda label: labels.append(label)
    _delegate_call(parent, runner, action="send", order=order)

    assert labels, "the worker_rule callback never fired"
    assert labels[0].startswith("worker start · default/worker-model-x · ")
    assert oneline(order, 60) in labels[0]
    assert not any("[worker]" in rendered for rendered in labels)  # the rule label replaces the [worker] ▶ line


def test_delegate_send_worker_rule_start_label_with_title(tmp_path, monkeypatch):
    from minacode.base import oneline
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    parent.config.providers["default"].model = "worker-model-x"
    order = "Rewrite the worker handoff plan to cover the start rule, then check it. " * 8
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    labels = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=lambda text: None)
    runner.worker_rule = lambda label: labels.append(label)
    _delegate_call(parent, runner, action="send", order=order, title="fix /status blank line")

    assert labels, "the worker_rule callback never fired"
    assert labels[0].startswith("worker start · default/worker-model-x · ")
    assert "fix /status blank line" in labels[0]
    assert oneline(order, 60) not in labels[0]


def test_delegate_send_worker_start_marker_with_title(tmp_path, monkeypatch):
    from minacode.base import LogBlock, LogRole, oneline
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    parent.config.providers["default"].model = "worker-model-x"
    order = "Rewrite the worker handoff plan to cover the start marker, then check it. " * 8
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    outputs = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=outputs.append)
    _delegate_call(parent, runner, action="send", order=order, title="fix /status blank line")

    blocks = [block for block in outputs if isinstance(block, LogBlock)]
    marker = next(block for block in blocks if any(item.role is LogRole.WORKER for item, _ in block.walk()))
    rendered = str(marker)
    assert "[worker]" in rendered and "▶" in rendered
    assert "default/worker-model-x" in rendered
    assert "fix /status blank line" in rendered
    assert oneline(order, 200) not in rendered


def test_delegate_send_worker_rule_start_label_falls_back_to_order(tmp_path, monkeypatch):
    from minacode.base import oneline
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    parent.config.providers["default"].model = "worker-model-x"
    order = "Rewrite the worker handoff plan to cover the start rule, then check it. " * 8
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    labels = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=lambda text: None)
    runner.worker_rule = lambda label: labels.append(label)
    _delegate_call(parent, runner, action="send", order=order)

    assert labels, "the worker_rule callback never fired"
    assert labels[0].startswith("worker start · default/worker-model-x · ")
    assert oneline(order, 60) in labels[0]


def test_delegate_rejects_empty_title(tmp_path):
    from minacode.base import ToolError
    from minacode.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    with pytest.raises(ToolError, match="non-empty string"):
        DelegateTool(parent, [{"action": "send", "order": "work", "title": "   "}]).call()


def test_delegate_send_language_directive_is_injected_into_the_order(tmp_path, monkeypatch):
    parent = _delegate_session(tmp_path)
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    _delegate_call(parent, runner, action="send", order="fix the parser", language="Chinese")

    worker_order = model.requests[0][-1]["content"]
    assert worker_order.startswith("fix the parser")
    assert "Reply language: Chinese" in worker_order and "live stream" in worker_order


def test_delegate_send_rejects_a_blank_language(tmp_path):
    from minacode.base import ToolError

    parent = _delegate_session(tmp_path)
    runner = _delegate_runner(parent)
    with pytest.raises(ToolError, match="language"):
        _delegate_call(parent, runner, action="send", order="o", language="   ")


def test_worker_inherits_forced_reply_language_from_parent(tmp_path, monkeypatch):
    from minacode.context import ContextManager

    parent = _delegate_session(tmp_path)
    parent.settings.language = "Chinese"
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    _delegate_call(parent, runner, action="send", order="fix the parser")

    worker = parent.worker
    assert worker.settings.language == "Chinese"
    system = model.requests[0][0]["content"]
    assert system.startswith(WORKER_PROMPT.strip())
    assert "LANGUAGE OVERRIDE:" in system and "Chinese" in system
    # the projection the worker uses matches the request the fake model received
    assert ContextManager(worker).model_messages(worker.system_prompt)[0]["content"] == system


def test_delegate_envelope_reports_max_steps_from_runtime_fact(tmp_path, monkeypatch):
    from minacode.engine import Agent

    parent = _delegate_session(tmp_path)
    runner = _delegate_runner(parent)

    def run_stopped(self, order):
        self.stopped_at_max_steps = True
        return "done"

    def run_normal(self, order):
        self.stopped_at_max_steps = False
        return "Stopped after max_agent_steps=3 (cosmetic wording only)"

    monkeypatch.setattr(Agent, "run", run_stopped)
    result = _delegate_call(parent, runner, action="send", order="o")
    assert 'stopped_at_max_steps="true"' in result

    monkeypatch.setattr(Agent, "run", run_normal)
    result = _delegate_call(parent, runner, action="send", order="o")
    assert 'stopped_at_max_steps="false"' in result  # the words are irrelevant; the fact is not set


def test_delegate_envelope_reports_token_spend_and_summary_renders(tmp_path, monkeypatch):
    from minacode.engine import Agent

    parent = _delegate_session(tmp_path)
    runner = _delegate_runner(parent)

    def run_quiet(self, order):
        self.stopped_at_max_steps = False
        return "done"

    monkeypatch.setattr(Agent, "run", run_quiet)
    result = _delegate_call(parent, runner, action="send", order="o")
    assert 'tokens="' in result
    assert 'rounds="' in result
    assert 'context_percent="0"' in result  # no usage budget with the fake model: the state fallback
    parent.worker.state.context_percent = 42  # the envelope reports the live fill, not a delta
    result = _delegate_call(parent, runner, action="send", order="o")
    assert 'context_percent="42"' in result
    summary = tooloutput.delegate_result_summary(result)
    assert " in / " in summary and " out" in summary
    assert "0 in / 0 out" in summary


def test_delegate_summary_formats_tokens_and_tolerates_old_envelopes(tmp_path):
    parent = _delegate_session(tmp_path)
    summary = tooloutput.delegate_result_summary(
        '<Delegate action="send" steps="3" elapsed="2.5s" files="a.txt, b.txt" stopped_at_max_steps="false" tokens="8200/1300">'
    )
    assert "8.2K in / 1.3K out" in summary

    legacy = tooloutput.delegate_result_summary('<Delegate action="send" steps="3" elapsed="2.5s" files="a.txt, b.txt" stopped_at_max_steps="false">')
    assert "steps 3" in legacy
    assert "2.5s" in legacy
    assert "a.txt, b.txt" in legacy
    assert " in / " not in legacy
    assert "round " not in legacy and "ctx " not in legacy  # neither attribute existed back then


def test_delegate_summary_shows_rounds_and_context_fill(tmp_path):
    parent = _delegate_session(tmp_path)
    envelope = '<Delegate action="send" steps="3" elapsed="2.5s" files="a.txt" stopped_at_max_steps="false" tokens="10/20" rounds="4" context_percent="73">'
    fields = tooloutput.delegate_result_fields(envelope)
    assert fields is not None
    assert (fields.rounds, fields.context_percent) == ("4", "73")

    summary = tooloutput.delegate_result_summary(envelope)
    assert "round 4" in summary
    assert "ctx 73%" in summary


def test_send_rejects_worker_calls_to_excluded_tools(tmp_path, monkeypatch):
    """End to end at the Delegate boundary: the worker's real tool block has ViewImage and
    ToolScript but not Ask/NextHints/Delegate, a hallucinated call to an excluded tool is
    rejected with a tool message instead of blocking on user input, and ViewImage executes
    as an ordinary tool (its failure here is a plain missing-file error)."""
    from minacode.base import ToolCall
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    prompts = []

    def fail_on_user(*args):
        prompts.append(args)
        raise AssertionError("worker blocked on user input")

    model = FakeModelClient(
        [
            (
                {"role": "assistant", "content": ""},
                [ToolCall("a1", "Ask", [{"questions": [{"question": "hi?"}]}]), ToolCall("n1", "NextHints", [{"inputs": ["x"]}])],
                "",
            ),
            ({"role": "assistant", "content": ""}, [ToolCall("v1", "ViewImage", ["missing.png"])], ""),
            ({"role": "assistant", "content": "done"}, [], "done"),
        ]
    )
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = ToolRunner(parent, ContextManager(parent), input_fn=fail_on_user, output_fn=lambda text: None)
    result = _delegate_call(parent, runner, action="send", order="do the thing")

    names = {(schema.get("function") or schema).get("name") for schema in model.received_tools[0] or [] if isinstance(schema, dict)}
    assert {"ViewImage", "ToolScript"} <= names
    assert not {"Ask", "NextHints", "Delegate"} & names

    assert prompts == []  # no user prompt ever fired: the calls were rejected, not executed
    second = str(model.requests[1])
    assert "Ask is not available in this session" in second
    assert "NextHints is not available in this session" in second
    assert "Cannot read image" in str(model.requests[2])
    assert "done" in result
