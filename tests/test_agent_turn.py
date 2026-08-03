"""The agent turn: the tool loop, interrupts, textual tool-call correction, live follow-ups,
parallel execution, and provider message conversion."""

import json
import threading
import time
from types import SimpleNamespace

import pytest
from agent_harness import call, queue, session

import minacode.engine as engine_module
from minacode.base import (
    DEFAULT_MAX_TOKENS,
    Config,
    LogBlock,
    MalformedToolCallError,
    ModelError,
    ProviderConfig,
    ToolCall,
)
from minacode.context import ContextManager
from minacode.engine import Agent
from minacode.model import ModelClient
from minacode.prompts import INTERRUPT_MARKER, LIVE_FOLLOWUP_PREFIX, SYSTEM_PROMPT
from minacode.runner import ToolRunner
from minacode.session import Session, SessionSnapshotCodec
from minacode.skill import SkillLibrary
from minacode.tools import BashTool, ReadTool, Tool


def _runner(tmp_path, input_reply=""):
    s = Session(cwd=str(tmp_path))
    return s, ToolRunner(s, ContextManager(s), input_fn=lambda *a: input_reply, output_fn=lambda *a: None)


def test_tool_runner_refusal_stops_batch_and_invalid_args_are_not_stored(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: "skip it", output_fn=lambda text: None)
    runner.run([call("Bash", [":"]), call("Edit", ["second.txt", [{"op": "create", "content": "second"}]])])

    assert s.tool_records == []
    assert len(s.tool_errors) == 1
    assert "skip it" in s.tool_errors[0].error
    assert not (tmp_path / "second.txt").exists()

    outputs = []
    bad = session(tmp_path)
    ToolRunner(bad, ContextManager(bad), output_fn=lambda text: outputs.append(str(text))).run([call("Bash", [])])
    assert bad.tool_records == []
    assert len(bad.tool_errors) == 1
    assert outputs and "· rejected:" in outputs[0]  # argument errors collapse to a quiet line


def test_tool_runner_refuses_without_reason_on_n(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: "n", output_fn=lambda text: None)

    runner.run([call("Bash", [":"])])

    assert s.tool_errors[0].error == "Cancelled: user refused tool call"


def test_tool_runner_refuses_with_direct_reason_input(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: "not now", output_fn=lambda text: None)

    runner.run([call("Bash", [":"])])

    assert s.tool_records == []
    assert len(s.tool_errors) == 1
    assert "not now" in s.tool_errors[0].error


def test_recall_tool_runner_does_not_create_new_result_keys(tmp_path):
    s = session(tmp_path)
    key = s.store_tool_result("Read", ["a.txt"], "result")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run([call("Recall", [key])])
    assert [record.key for record in s.tool_records] == [key]


def test_agent_runs_tool_loop_and_stops_at_max_steps(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    s = session(tmp_path)
    s.skills = SkillLibrary({})  # no skills: assert the base frame layout
    agent = Agent(s, output_fn=lambda text: None)

    class FakeModel:
        def __init__(self):
            self.messages = []

        def request(self, messages, tools=None):
            self.messages.append(messages)
            if len(self.messages) == 1:
                return {}, [call("Read", [{"path": "a.txt", "ranges": [[0, 1]]}])], ""
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = FakeModel()
    assert agent.run("read file") == "done"
    assert len(agent.model.messages) == 2
    assert [len(messages) for messages in agent.model.messages] == [3, 5]
    assert agent.model.messages[1][3]["role"] == "assistant"
    assert agent.model.messages[1][3]["tool_calls"][0]["id"] == "Read-id"
    assert agent.model.messages[1][4]["role"] == "tool"
    assert agent.model.messages[1][4]["tool_call_id"] == "Read-id"
    assert any("tool tr.1 Read a.txt 0:1" in (message.get("content") or "") for message in agent.model.messages[1])
    assert any(message["role"] == "tool" and "<Read" in message["content"] for message in agent.model.messages[1])
    assert not any("FILE STATE" in (message.get("content") or "") for message in agent.model.messages[1])
    assert len(s.tool_records) == 1
    assert s.messages[-1]["content"] == "done"
    assert s.state.goal == ""

    limited = session(tmp_path)
    limited.skills = SkillLibrary({})
    limited.settings.max_steps = 2
    limited_agent = Agent(limited, output_fn=lambda text: None)

    class LoopingModel:
        def request(self, messages, tools=None):
            return {}, [call("Read", [{"path": "a.txt", "ranges": [[0, 0]]}])], ""

    limited_agent.model = LoopingModel()
    answer = limited_agent.run("keep going")
    assert limited.state.turn_step == 2
    assert len(limited.tool_records) == 2
    assert limited.messages[-1]["content"] == answer


def test_agent_persists_responses_output_on_final_assistant_message(tmp_path):
    s = session(tmp_path)
    s.skills = SkillLibrary({})
    agent = Agent(s, output_fn=lambda _text: None)

    class FakeModel:
        def request(self, messages, tools=None):
            return (
                {
                    "role": "assistant",
                    "content": "done",
                    "_responses_output": [
                        {"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque", "summary": []},
                        {
                            "id": "msg_1",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "done", "annotations": []}],
                        },
                    ],
                },
                [],
                "done",
            )

    agent.model = FakeModel()

    assert agent.run("finish") == "done"
    assert s.messages[-1]["_responses_output"][0]["type"] == "reasoning"
    s.save_snapshot()
    restored = Session.load_snapshot(s.uid, config=s.config, settings=s.settings)
    restored_assistant = next(message for message in reversed(restored.messages) if message.get("role") == "assistant")
    assert restored_assistant["_responses_output"] == s.messages[-1]["_responses_output"]


def test_interrupted_turn_persists_completed_tool_batches_for_resume(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    s = session(tmp_path)
    s.skills = SkillLibrary({})
    agent = Agent(s, output_fn=lambda text: None)

    class InterruptingModel:
        calls = 0

        def request(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                return {}, [call("Read", [{"path": "a.txt", "ranges": [[0, 1]]}])], ""
            raise KeyboardInterrupt

    agent.model = InterruptingModel()
    with pytest.raises(KeyboardInterrupt):
        agent.run("read file")

    assert [message["role"] for message in s.messages] == ["user", "assistant", "tool", "user"]
    assert s.messages[-1]["content"] == INTERRUPT_MARKER
    assert s._active_turn_messages == []
    restored = Session.load_snapshot(s.uid, config=s.config, settings=s.settings)
    messages = [message for message in restored.messages if not SessionSnapshotCodec.is_internal_message(message)]
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "user"]
    assert messages[-1]["content"] == INTERRUPT_MARKER
    assert messages[1]["tool_calls"][0]["id"] == "Read-id"
    assert messages[2]["tool_call_id"] == "Read-id"
    assert "<Read" in messages[2]["content"]
    assert [record.name for record in restored.tool_records] == ["Read"]


def test_interrupted_turn_before_any_output_is_retracted(tmp_path):
    s = session(tmp_path)
    s.skills = SkillLibrary({})
    agent = Agent(s, output_fn=lambda text: None)

    class InterruptingModel:
        def request(self, messages, tools=None):
            raise KeyboardInterrupt

    agent.model = InterruptingModel()
    with pytest.raises(KeyboardInterrupt):
        agent.run("never sent")

    # Retract: the agent produced nothing, so the turn leaves no trace in the context or on disk,
    # while the input history (a separate FileHistory) still recalls it for Ctrl-P.
    assert s.messages == []
    assert s._active_turn_messages == []
    restored = Session.load_snapshot(s.uid, config=s.config, settings=s.settings)
    messages = [message for message in restored.messages if not SessionSnapshotCodec.is_internal_message(message)]
    assert messages == []


def test_interrupted_turn_completes_dangling_tool_calls(tmp_path):
    s = session(tmp_path)
    s.skills = SkillLibrary({})
    agent = Agent(s, output_fn=lambda text: None)

    class Model:
        def request(self, messages, tools=None):
            return {}, [call("Read", [{"path": "missing", "ranges": [[0, 0]]}])], ""

        def cancel(self):
            pass

    class Tools:
        def run(self, calls, batch_suffix=""):
            agent.cancel()
            return []

        def cancel(self):
            pass

    agent.model = Model()
    agent.tools = Tools()
    with pytest.raises(KeyboardInterrupt):
        agent.run("read missing")

    # Interrupt: the partial turn stands, the unanswered tool call gets a cancelled result so the
    # next request stays valid, and the marker records where the turn ended.
    assert [message["role"] for message in s.messages] == ["user", "assistant", "tool", "user"]
    assert s.messages[2]["tool_call_id"] == "Read-id"
    assert "Cancelled" in s.messages[2]["content"]
    assert s.messages[3]["content"] == INTERRUPT_MARKER


def test_agent_cancel_stops_after_active_tool_batch(tmp_path):
    agent = Agent(session(tmp_path), output_fn=lambda text: None)

    class Model:
        calls = 0

        def request(self, messages, tools=None):
            self.calls += 1
            return {}, [call("Read", [{"path": "missing", "ranges": [[0, 0]]}])], ""

        def cancel(self):
            pass

    class Tools:
        def run(self, calls, batch_suffix=""):
            agent.cancel()
            return []

        def cancel(self):
            pass

    agent.model = Model()
    agent.tools = Tools()

    with pytest.raises(KeyboardInterrupt):
        agent.run("stop after the tool")

    assert agent.model.calls == 1


def test_agent_rejects_empty_final_response(tmp_path):
    agent = Agent(session(tmp_path), output_fn=lambda text: None)

    class EmptyModel:
        def request(self, messages, tools=None):
            return {"role": "assistant", "content": ""}, [], ""

    agent.model = EmptyModel()
    with pytest.raises(ModelError, match="empty final response"):
        agent.run("answer me")


def test_agent_corrects_textual_tool_call_with_a_committed_message(tmp_path):
    s = session(tmp_path)
    agent = Agent(s, output_fn=lambda _text: None)
    pseudo = 'course\n<invoke name="Bash">\n<parameter name="command">secret command</parameter>\n</invoke>'

    class Model:
        def __init__(self):
            self.requests = []
            self.on_stream = None

        def request(self, messages, tools=None):
            self.requests.append((messages, tools))
            if len(self.requests) == 1:
                return {"role": "assistant", "content": pseudo}, [], pseudo
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = Model()

    assert agent.run("continue") == "done"
    assert len(agent.model.requests) == 2
    first_messages = agent.model.requests[0][0]
    correction_messages = agent.model.requests[1][0]
    assert correction_messages[:-1] == first_messages
    correction = correction_messages[-1]
    assert correction["role"] == "user"
    assert correction["content"] == Agent.tool_call_correction("Bash")
    assert "secret command" not in correction["content"]
    # Sent means durable: the correction is a real turn message, not a request-local one.
    assert [message["role"] for message in s.messages] == ["user", "user", "assistant"]
    assert s.messages[1] == correction
    assert s.messages[-1]["content"] == "done"
    assert all(pseudo not in str(message.get("content") or "") for message in s.messages)  # the markup itself is never replayed
    assert s.tool_records == []


def test_agent_executes_native_call_after_textual_tool_correction_and_replays_the_correction(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    s = session(tmp_path)
    agent = Agent(s, output_fn=lambda _text: None)
    pseudo = 'course\n<invoke name="Read">\n<parameter name="path">ignored.txt</parameter>\n</invoke>'

    class Model:
        def __init__(self):
            self.requests = []
            self.on_stream = None

        def request(self, messages, tools=None):
            self.requests.append(messages)
            if len(self.requests) == 1:
                return {}, [], pseudo
            if len(self.requests) == 2:
                return {}, [call("Read", [{"path": "a.txt", "ranges": [[0, 1]]}])], ""
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = Model()

    assert agent.run("read it") == "done"
    assert len(agent.model.requests) == 3
    assert agent.model.requests[1][:-1] == agent.model.requests[0]
    assert "[Runtime protocol correction]" in agent.model.requests[1][-1]["content"]
    replayed = [message for message in agent.model.requests[2] if "[Runtime protocol correction]" in str(message.get("content") or "")]
    assert replayed == [agent.model.requests[1][-1]]  # carried forward once, from history
    assert [record.name for record in s.tool_records] == ["Read"]
    assert all(pseudo not in str(message.get("content") or "") for message in s.messages)


def test_agent_recovers_after_five_textual_tool_corrections_that_stack_in_history(tmp_path):
    s = session(tmp_path)
    agent = Agent(s, output_fn=lambda _text: None)
    names = ["Edit", "Job", "Bash", "Note", "Read"]
    statuses = []

    class Model:
        def __init__(self):
            self.requests = []
            self.on_stream = lambda kind, text: statuses.append((kind, text))

        def request(self, messages, tools=None):
            self.requests.append(messages)
            if len(self.requests) <= len(names):
                name = names[len(self.requests) - 1]
                pseudo = f'course\n<invoke name="{name}"><parameter name="args">untrusted</parameter></invoke>'
                return {}, [], pseudo
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = Model()

    assert agent.run("continue") == "done"
    assert len(agent.model.requests) == engine_module.MAX_TEXTUAL_TOOL_CORRECTIONS + 1
    base_messages = agent.model.requests[0]
    corrections = [{"role": "user", "content": Agent.tool_call_correction(name)} for name in names]
    for index, name in enumerate(names, start=1):
        correction_request = agent.model.requests[index]
        # Corrections stack instead of replacing each other: nothing already sent is withdrawn.
        assert correction_request == [*base_messages, *corrections[:index]]
        assert "untrusted" not in correction_request[-1]["content"]
    assert statuses == [
        (f"correcting malformed tool call {index}/{engine_module.MAX_TEXTUAL_TOOL_CORRECTIONS} · {name}", "") for index, name in enumerate(names, start=1)
    ]
    assert [message["role"] for message in s.messages] == ["user", *["user"] * len(names), "assistant"]
    assert s.messages[1 : 1 + len(names)] == corrections
    assert s.messages[-1]["content"] == "done"


def test_agent_stops_after_sixth_textual_tool_call_without_persisting_responses(tmp_path):
    s = session(tmp_path)
    agent = Agent(s, output_fn=lambda _text: None)
    pseudo = 'course\n<invoke name="Bash">\n<parameter name="command">never run</parameter>\n</invoke>'

    class Model:
        def __init__(self):
            self.requests = []
            self.on_stream = None

        def request(self, messages, tools=None):
            self.requests.append(messages)
            return {}, [], pseudo

    agent.model = Model()

    with pytest.raises(
        MalformedToolCallError,
        match=r"Model emitted Bash as text 6 times; none of the textual calls were executed\.",
    ):
        agent.run("continue")

    assert len(agent.model.requests) == engine_module.MAX_TEXTUAL_TOOL_CORRECTIONS + 1
    assert s.tool_records == []
    # The turn aborts, but the corrections it already sent survive: history is append-only.
    assert s.messages == [
        {"role": "user", "content": "continue"},
        *[{"role": "user", "content": Agent.tool_call_correction("Bash")}] * engine_module.MAX_TEXTUAL_TOOL_CORRECTIONS,
    ]
    assert s._active_turn_messages == []
    restored = Session.load_snapshot(s.uid, config=s.config)
    restored_messages = [message for message in restored.messages if not SessionSnapshotCodec.is_internal_message(message)]
    assert restored_messages == s.messages


@pytest.mark.parametrize(
    "content",
    [
        '```xml\n<invoke name="Bash">\n<parameter name="command">echo safe</parameter>\n</invoke>',
        'Example only:\n> <invoke name="Bash"><parameter name="command">echo safe</parameter></invoke>',
        'Example only:\n    <invoke name="Bash"><parameter name="command">echo safe</parameter></invoke>',
        '<invoke name="Unknown">\n<parameter name="command">echo safe</parameter>\n</invoke>',
        '<invoke name="Bash">\n<parameter name="command">echo incomplete</parameter>',
        '<invoke name="Bash"><parameter name="command">echo middle</parameter></invoke>\nordinary tail',
    ],
)
def test_textual_tool_call_detector_rejects_non_executable_boundaries(content):
    tools = [{"type": "function", "function": {"name": "Bash", "parameters": {}}}]

    assert Agent.textual_tool_call(content, tools) is None


def test_agent_does_not_reclassify_content_when_native_tool_call_exists(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    s = session(tmp_path)
    output = []
    agent = Agent(s, output_fn=output.append)
    pseudo = '<invoke name="Bash"><parameter name="command">not trusted</parameter></invoke>'

    class Model:
        def __init__(self):
            self.requests = []

        def request(self, messages, tools=None):
            self.requests.append(messages)
            if len(self.requests) == 1:
                return {}, [call("Read", [{"path": "a.txt", "ranges": [[0, 1]]}])], pseudo
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = Model()

    assert agent.run("read it") == "done"
    assert len(agent.model.requests) == 2
    assert all("[Runtime protocol correction]" not in str(message.get("content") or "") for message in agent.model.requests[1])
    assert [record.name for record in s.tool_records] == ["Read"]
    assert output[0] == pseudo
    assert len(output) == 2


def test_system_prompt_requires_native_tool_calls():
    assert "Use native tool calls; never print tool XML or tool-call JSON." in SYSTEM_PROMPT


def test_model_cancel_closes_active_client_and_interrupts_request(tmp_path):
    s = session(tmp_path)
    s.config.provider.url = "https://example.test/v1"
    s.config.provider.key = "test"
    s.config.provider.model = "model"
    started = threading.Event()
    closed = threading.Event()

    class Completions:
        def create(self, **_params):
            started.set()
            closed.wait(timeout=1)
            raise RuntimeError("connection closed")

    class Client:
        chat = SimpleNamespace(completions=Completions())

        def close(self):
            closed.set()

    model = ModelClient(s)
    model.client = Client
    errors = []

    def request():
        try:
            model.request([{"role": "user", "content": "hello"}], [])
        except BaseException as error:  # noqa: BLE001 - harness collects every thread failure, KeyboardInterrupt included
            errors.append(error)

    thread = threading.Thread(target=request)
    thread.start()
    assert started.wait(timeout=1)
    model.cancel()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], KeyboardInterrupt)


def test_agent_injects_pending_user_input_once(tmp_path):
    s = session(tmp_path)
    queue(s, "extra instruction")
    output = []
    agent = Agent(s, output_fn=output.append)

    class FakeModel:
        def __init__(self):
            self.messages = []

        def request(self, messages, tools=None):
            self.messages.append(messages)
            if len(self.messages) == 1:
                s.enqueue_user_input("second instruction")
                return {}, [call("Bash", ["wc -l missing.txt"])], "checking"
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = FakeModel()
    assert agent.run("initial request") == "done"

    first = "\n\n".join(message.get("content") or "" for message in agent.model.messages[0])
    second = "\n\n".join(message.get("content") or "" for message in agent.model.messages[1])
    first_followup = next(message["content"] for message in agent.model.messages[0] if "extra instruction" in (message.get("content") or ""))
    second_followup = next(message["content"] for message in agent.model.messages[1] if "second instruction" in (message.get("content") or ""))
    assert "[Live follow-up received while you were working]" in LIVE_FOLLOWUP_PREFIX
    assert "[Live follow-up received while you were working]" in SYSTEM_PROMPT
    assert "Answer this in visible text in your next assistant message" in LIVE_FOLLOWUP_PREFIX
    assert "in the same message as its tool calls" in SYSTEM_PROMPT
    assert first_followup == LIVE_FOLLOWUP_PREFIX + "extra instruction"
    assert second_followup == LIVE_FOLLOWUP_PREFIX + "second instruction"
    assert "extra instruction" in first
    assert "extra instruction" in second
    assert "checking" in second
    assert "second instruction" in second
    assert s.messages[0]["content"] == "initial request"
    assert s.messages[1]["content"] == "extra instruction"
    assert s.messages[2]["content"] == "checking"
    assert s.messages[3]["role"] == "tool"
    assert s.messages[3]["content"].startswith("tool tr.1 Bash wc -l missing.txt")
    assert s.messages[4]["content"] == "second instruction"
    assert "checking" in output
    assert s.messages[5]["role"] == "assistant"
    assert s.pending_user_inputs == []


def test_agent_never_reshapes_tools_for_a_live_followup(tmp_path):
    """A live follow-up may not change the shape of a request. The tool list is part of the cached
    prefix, so a tools-only response is accepted as-is: the batch runs, the turn continues, and no
    extra request is made to extract an acknowledgement first."""
    s = session(tmp_path)
    queue(s, "first follow-up", "second follow-up")
    output = []
    agent = Agent(s, output_fn=output.append)

    class FakeModel:
        def __init__(self):
            self.requests = []

        def request(self, messages, tools=None):
            self.requests.append((messages, tools))
            if len(self.requests) == 1:
                return {}, [call("Bash", ["echo hi"])], ""
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = FakeModel()

    assert agent.run("initial request") == "done"
    assert len(agent.model.requests) == 2  # one request per step, none inserted for the follow-ups
    assert all(tools for _messages, tools in agent.model.requests)
    assert agent.model.requests[0][1] == agent.model.requests[1][1]
    first_request = "\n".join(message.get("content") or "" for message in agent.model.requests[0][0])
    assert "first follow-up" in first_request and "second follow-up" in first_request
    assert [message["role"] for message in s.messages] == ["user", "user", "user", "assistant", "tool", "assistant"]
    assert len(s.tool_records) == 1
    assert s.pending_user_inputs == []
    assert all(isinstance(item, LogBlock) for item in output)  # only the tool log; no forced acknowledgement text


def test_agent_keeps_one_tool_block_for_the_whole_turn(tmp_path):
    """The cached prefix must survive a turn that mixes tool batches, live follow-ups, and a
    protocol correction: every request carries the same non-empty tool block."""
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    s = session(tmp_path)
    queue(s, "an early follow-up")
    agent = Agent(s, output_fn=lambda _text: None)
    pseudo = '<invoke name="Read"><parameter name="path">ignored.txt</parameter></invoke>'
    read = call("Read", [{"path": "a.txt", "ranges": [[0, 1]]}])

    class FakeModel:
        def __init__(self):
            self.requests = []
            self.on_stream = None

        def request(self, messages, tools=None):
            self.requests.append((messages, tools))
            if len(self.requests) == 1:
                s.enqueue_user_input("a later follow-up")
                return {}, [read], "on it"
            if len(self.requests) == 2:
                return {"role": "assistant", "content": pseudo}, [], pseudo  # triggers a correction
            if len(self.requests) == 3:
                return {}, [read], "still going"
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = FakeModel()

    assert agent.run("initial request") == "done"
    assert len(agent.model.requests) == 4
    tool_blocks = [tools for _messages, tools in agent.model.requests]
    assert all(tools and tools == tool_blocks[0] for tools in tool_blocks)

    # Messages only ever grow: each request is a prefix of the next, once the one-shot follow-up
    # marker (dropped when the queued input is committed) is normalized away.
    def normalized(messages):
        return [str(message.get("content") or "").replace(LIVE_FOLLOWUP_PREFIX, "") for message in messages]

    lengths = [len(messages) for messages, _tools in agent.model.requests]
    assert lengths == sorted(lengths) and len(set(lengths)) == len(lengths)
    for earlier, later in zip(agent.model.requests, agent.model.requests[1:]):
        assert normalized(later[0])[: len(earlier[0])] == normalized(earlier[0])
    assert s.pending_user_inputs == []


def test_agent_commits_textual_tool_call_correction_to_history(tmp_path):
    """The correction is a real message, not a request-local one: what reached the provider must
    reach durable history, and the retry keeps the same tool list."""
    s = session(tmp_path)
    queue(s, "live follow-up")
    agent = Agent(s, output_fn=lambda text: None)
    pseudo = '<invoke name="Bash"><parameter name="command">should-not-run</parameter></invoke>'
    correction = {"role": "user", "content": Agent.tool_call_correction("Bash")}

    class FakeModel:
        def __init__(self):
            self.requests = []
            self.on_stream = None

        def request(self, messages, tools=None):
            self.requests.append(([dict(message) for message in messages], tools))
            if len(self.requests) == 1:
                return {"role": "assistant", "content": pseudo}, [], pseudo
            if len(self.requests) == 2:
                return {}, [call("Bash", ["echo hi"])], "on it"
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = FakeModel()

    assert agent.run("initial request") == "done"
    assert len(agent.model.requests) == 3
    assert all(tools == agent.model.requests[0][1] for _messages, tools in agent.model.requests)
    retry_messages = agent.model.requests[1][0]
    assert retry_messages[-1] == correction
    assert retry_messages[:-1] == agent.model.requests[0][0]

    # Committed to history, after the follow-up it followed on the wire, and replayed on the next step.
    assert correction in s.messages
    followup = next(message for message in s.messages if "live follow-up" in (message.get("content") or ""))
    assert s.messages.index(followup) < s.messages.index(correction)
    assert correction in agent.model.requests[2][0]
    assert all(pseudo not in str(message.get("content") or "") for message in s.messages)
    assert s.pending_user_inputs == []


def test_agent_shares_textual_tool_call_limit_across_corrections(tmp_path):
    s = session(tmp_path)
    output = []
    agent = Agent(s, output_fn=output.append)
    pseudo = '<invoke name="Bash"><parameter name="command">never-run</parameter></invoke>'

    class FakeModel:
        def __init__(self):
            self.requests = []
            self.on_stream = None

        def request(self, messages, tools=None):
            self.requests.append((messages, tools))
            return {"role": "assistant", "content": pseudo}, [], pseudo

    agent.model = FakeModel()

    with pytest.raises(
        MalformedToolCallError,
        match=r"Model emitted Bash as text 6 times; none of the textual calls were executed\.",
    ):
        agent.run("initial request")

    assert len(agent.model.requests) == engine_module.MAX_TEXTUAL_TOOL_CORRECTIONS + 1
    assert all(tools == agent.model.requests[0][1] and tools for _messages, tools in agent.model.requests)
    # Each correction stacks onto the previous one, so the retries grow by exactly one message.
    lengths = [len(messages) for messages, _tools in agent.model.requests]
    assert lengths == [lengths[0] + index for index in range(len(lengths))]
    assert output == []
    assert all(pseudo not in str(message.get("content") or "") for message in s.messages)
    assert s.tool_records == []


def test_agent_shares_resolved_tools_with_model_request(tmp_path, monkeypatch):
    s = session(tmp_path)
    agent = Agent(s, output_fn=lambda text: None)
    tools = [{"type": "function", "function": {"name": "Test", "parameters": {}}}]
    resolved = []

    def resolve(session):
        resolved.append(session)
        return tools

    class FakeModel:
        received_tools = None

        def request(self, messages, request_tools=None):
            self.received_tools = request_tools
            return {"role": "assistant", "content": "done"}, [], "done"

    monkeypatch.setattr(Tool, "resolved_schemas", staticmethod(resolve))
    agent.model = FakeModel()

    assert agent.run("hello") == "done"
    assert resolved == [s]
    assert agent.model.received_tools is tools


def test_agent_emits_and_records_intermediate_content_before_tools(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    s = session(tmp_path)
    output = []
    agent = Agent(s, output_fn=output.append)

    class TalkingModel:
        def __init__(self):
            self.messages = []

        def request(self, messages, tools=None):
            self.messages.append(messages)
            if len(self.messages) == 1:
                return {}, [call("Read", [{"path": "a.txt", "ranges": [[0, 1]]}])], "I'll inspect that first."
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = TalkingModel()
    assert agent.run("read file") == "done"
    assert output[0] == "I'll inspect that first."
    assert any(isinstance(line, LogBlock) and str(line).startswith("  Read  ") for line in output)
    assert [message["role"] for message in s.messages] == ["user", "assistant", "tool", "assistant"]
    assert s.messages[0]["content"] == "read file"
    assert s.messages[1]["content"] == "I'll inspect that first."
    assert s.messages[2]["content"].startswith("tool tr.1 Read a.txt 0:1")
    assert "<Read" in s.messages[2]["content"]
    assert "-> FILE STATE" not in s.messages[2]["content"]
    assert s.messages[3]["content"] == "done"
    assert any("I'll inspect that first." in (message.get("content") or "") for message in agent.model.messages[1])


def test_terminal_next_hints_recognizes_all_next_hints_batch(tmp_path):
    agent = Agent(session(tmp_path), output_fn=lambda text: None)
    assert agent.terminal_next_hints([call("NextHints", [{"inputs": ["x"]}])])
    assert agent.terminal_next_hints([call("NextHints", [{"inputs": ["x"]}]), call("NextHints", [{"inputs": ["y"]}])])
    assert not agent.terminal_next_hints([call("NextHints", [{"inputs": ["x"]}]), call("Read", [{"path": "f"}])])
    assert not agent.terminal_next_hints([])


def test_finish_with_next_hints_runs_tool_and_finishes_without_dup_answer(tmp_path):
    s = session(tmp_path)
    agent = Agent(s, output_fn=lambda text: None)
    turn_messages = [{"role": "user", "content": "hi"}]
    assistant = {
        "role": "assistant",
        "content": "the answer",
        "reasoning_content": "reasoning",
        "_responses_output": [
            {"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque"},
            {"id": "msg_1", "type": "message", "content": [{"type": "output_text", "text": "the answer"}]},
            {"id": "fc_1", "type": "function_call", "call_id": "NextHints-id", "name": "NextHints", "arguments": "{}"},
        ],
    }
    calls = [call("NextHints", [{"inputs": ["run tests", "show diff"]}])]

    assert agent.finish_with_next_hints(turn_messages, assistant, calls, "the answer", 0) == "the answer"
    assert s.quick_hints == ("run tests", "show diff")
    # user, tool-bearing assistant (no content), tool result, plain final answer
    assert [m["role"] for m in s.messages] == ["user", "assistant", "tool", "assistant"]
    assert s.messages[-1] == {"role": "assistant", "content": "the answer"}
    assert s.messages[-3].get("content") is None
    assert [c["function"]["name"] for c in s.messages[-3]["tool_calls"]] == ["NextHints"]
    assert [m.get("content") for m in s.messages if m.get("role") == "assistant" and m.get("content")] == ["the answer"]
    replayed = ModelClient(s).responses_input(s.messages)
    assert [item.get("id") for item in replayed if item.get("id")] == ["rs_1", "fc_1"]


def test_all_next_hints_batch_with_answer_ends_turn_in_single_model_call(tmp_path):
    s = session(tmp_path)
    s.skills = SkillLibrary({})
    agent = Agent(s, output_fn=lambda text: None)

    class FakeModel:
        def __init__(self):
            self.messages = []

        def request(self, messages, tools=None):
            self.messages.append(messages)
            return {"role": "assistant", "content": "all done"}, [call("NextHints", [{"inputs": ["run tests"]}])], "all done"

    agent.model = FakeModel()
    assert agent.run("do it") == "all done"
    assert len(agent.model.messages) == 1  # finished on the first call, no extra round trip
    assert s.quick_hints == ("run tests",)
    assert [m["role"] for m in s.messages] == ["user", "assistant", "tool", "assistant"]
    assert s.messages[-1]["content"] == "all done"
    assert "tool_calls" not in s.messages[-1]


def test_non_terminal_next_hints_do_not_leak_into_a_later_answer(tmp_path):
    """A NextHints batch without answer text runs as a normal tool batch; the next step supersedes
    its hints, so a later final answer never displays stale suggestions beside it."""
    s = session(tmp_path)
    s.skills = SkillLibrary({})
    agent = Agent(s, output_fn=lambda text: None)

    class FakeModel:
        def __init__(self):
            self.messages = []

        def request(self, messages, tools=None):
            self.messages.append(messages)
            if len(self.messages) == 1:
                # No answer text: not terminal, so it runs as an ordinary batch and publishes hints.
                return {"role": "assistant", "content": ""}, [call("NextHints", [{"inputs": ["stale suggestion"]}])], ""
            return {"role": "assistant", "content": "different final answer"}, [], "different final answer"

    agent.model = FakeModel()
    assert agent.run("do it") == "different final answer"
    assert len(agent.model.messages) == 2  # the turn continued past the non-terminal batch
    assert s.quick_hints == ()  # the stale hints were cleared, not shown next to the answer


def test_agent_tool_error_feedback_is_visible_on_next_model_request(tmp_path):
    s = session(tmp_path)
    agent = Agent(s, output_fn=lambda text: None)

    class FeedbackModel:
        def __init__(self):
            self.messages = []

        def request(self, messages, tools=None):
            self.messages.append(messages)
            if len(self.messages) == 1:
                return {}, [call("Bash", [])], ""
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = FeedbackModel()
    assert agent.run("run bad tool") == "done"
    assert len(s.tool_errors) == 1
    assert s.tool_records == []
    second_context = "\n\n".join(message.get("content") or "" for message in agent.model.messages[1])
    assert "tool - Bash" in second_context
    assert "status: failed" in second_context
    assert "Bash" in second_context


def test_provider_compatibility_and_prompt_cache_key(tmp_path):
    opencode_claude = ProviderConfig(url="https://opencode.ai/zen/go/v1", key="k", model="claude-sonnet", api="auto")
    assert opencode_claude.resolve().api == "anthropic"

    opencode_qwen = ProviderConfig(url="https://opencode.ai/zen/go/v1", key="k", model="qwen3.7-max", api="auto")
    assert opencode_qwen.resolve().api == "anthropic"

    opencode_deepseek = ProviderConfig(url="https://opencode.ai/zen/go/v1", key="k", model="deepseek-v4-flash", api="auto")
    resolved = opencode_deepseek.resolve()
    assert resolved.api == "chat"
    assert resolved.chat_reasoning == "thinking"

    provider = ProviderConfig(url="https://api.openai.com/v1", key="k", model="gpt-5-mini", prompt_cache_key="auto")
    s = Session(cwd=str(tmp_path), config=Config(active_provider="p", providers={"p": provider}))
    client = ModelClient(s)
    first = client.prompt_cache_key(provider, [BashTool.schema(), ReadTool.schema()])
    second = client.prompt_cache_key(provider, [ReadTool.schema(), BashTool.schema()])
    assert first == second
    assert first.startswith("minacode-")

    provider.prompt_cache_key = "fixed-key"
    assert client.prompt_cache_key(provider, None) == "fixed-key"
    provider.prompt_cache_key = "off"
    assert client.prompt_cache_key(provider, None) == ""


def test_anthropic_message_conversion_and_tool_result_parsing(tmp_path):
    provider = ProviderConfig(url="https://api.anthropic.com/v1/messages", key="k", model="claude-sonnet", api="anthropic", reasoning="off", temperature=0.2)
    s = Session(cwd=str(tmp_path), config=Config(active_provider="p", providers={"p": provider}))
    client = ModelClient(s)
    arguments = json.dumps({"files": [{"path": "a.txt", "ranges": [[0, 1]]}]})
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "thinking", "tool_calls": [{"id": "tc.1", "function": {"name": "Read", "arguments": arguments}}]},
        {"role": "tool", "tool_call_id": "tc.1", "content": "tool output"},
    ]

    params = client.anthropic_params(messages, [ReadTool.schema()])
    # system is a cache_control-marked block so the tools+system prefix is cached across turns.
    assert params["system"] == [{"type": "text", "text": "system", "cache_control": {"type": "ephemeral"}}]
    assert params["temperature"] == 0.2
    assert params["max_tokens"] == DEFAULT_MAX_TOKENS
    # An unversioned gateway alias remains generic rather than guessing a thinking generation.
    assert "thinking" not in params
    assert params["messages"][0] == {"role": "user", "content": "first\n\nsecond"}
    assert params["messages"][1]["content"][1]["type"] == "tool_use"
    assert params["messages"][2]["content"][0]["type"] == "tool_result"
    assert params["tools"][0]["name"] == "Read"
    assert params["tools"][0]["input_schema"]["additionalProperties"] is False

    provider.max_tokens = 2_048
    assert client.anthropic_params(messages, None)["max_tokens"] == 2_048
    provider.temperature = None
    provider.reasoning = "minimal"
    provider.model = "claude-sonnet-4-5"
    assert client.anthropic_params(messages, None)["thinking"] == {"type": "enabled", "budget_tokens": 1_024}

    result = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="answer"),
            SimpleNamespace(type="tool_use", id="tc.2", name="Bash", input={"command": "pwd"}),
        ],
        usage={},
    )
    assistant, calls, text = client.anthropic_result(result)
    assert text == "answer"
    assert assistant["tool_calls"][0]["function"]["name"] == "Bash"
    assert calls == [ToolCall(id="tc.2", name="Bash", args=["pwd"])]


def test_malformed_tool_args_defer_to_execution_chat(tmp_path):
    """A live chat tool call whose args fail payload validation (Bash with empty command) must not
    raise out of parsing; the error is deferred onto the call so the turn is not aborted."""
    s = Session(cwd=str(tmp_path))
    client = ModelClient(s)
    raw = SimpleNamespace(id="x1", function=SimpleNamespace(name="Bash", arguments='{"command": ""}'))
    message = SimpleNamespace(tool_calls=[raw])
    calls = client.tool_calls(message)  # must not raise ToolError
    assert len(calls) == 1
    assert calls[0].args == []
    assert "non-empty" in calls[0].error


def test_malformed_tool_args_defer_to_execution_anthropic(tmp_path):
    """Same deferral on the anthropic path: a tool_use with invalid input is captured, not raised."""
    s = Session(cwd=str(tmp_path))
    client = ModelClient(s)
    result = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id="a1", name="Bash", input={"command": ""})],
        usage={},
    )
    _, calls, _ = client.anthropic_result(result)  # must not raise ToolError
    assert len(calls) == 1
    assert calls[0].error


def test_deferred_tool_error_surfaces_as_tool_result(tmp_path):
    """A deferred-error call runs through ToolRunner and is reported back to the model as a failed
    tool result (so it can self-correct), rather than escaping to abort the turn."""
    s = Session(cwd=str(tmp_path))
    ctx = ContextManager(s)
    runner = ToolRunner(s, ctx, input_fn=lambda *a: "", output_fn=lambda *a: None)
    call = ToolCall(id="x1", name="Bash", args=[], error="Bash command must be non-empty")
    results = runner.run([call])
    assert len(results) == 1
    assert results[0]["role"] == "tool"
    assert "non-empty" in results[0]["content"]


def test_parallel_safe_classification(tmp_path):
    _, runner = _runner(tmp_path)

    def safe(name, args):
        return runner.parallel_safe(ToolCall(id="x", name=name, args=args))

    assert safe("Read", [{"path": "f.txt"}])
    assert safe("Search", [{"pattern": "x"}])
    assert not safe("Bash", ["git status --short"])  # Bash streams live output, so it stays serial
    assert not safe("Bash", ["git commit -m x"])  # mutating command
    assert not safe("Bash", ["echo hi"])  # live-output command
    assert not safe("Edit", ["f.txt", [{"op": "insert_after", "start": "0:a", "content": "x"}]])
    assert not safe("Ask", [{"question": "q?"}])  # interactive
    assert not safe("NextHints", [{"inputs": ["x"]}])  # writes session state; serial so model order wins
    assert not safe("Nope", [])  # unknown tool


def test_parallel_readonly_preserves_request_order(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text(f"content-{i}\n")
    s, runner = _runner(tmp_path)
    s.settings.max_parallel_tools = 4
    calls = [ToolCall(id=f"r{i}", name="Read", args=[{"path": f"f{i}.txt", "ranges": [[0, 0]]}]) for i in range(5)]

    # Force overlapping execution and record peak concurrency.
    active = {"cur": 0, "max": 0}
    guard = threading.Lock()
    original = ReadTool.call

    def traced(self):
        with guard:
            active["cur"] += 1
            active["max"] = max(active["max"], active["cur"])
        time.sleep(0.03)
        try:
            return original(self)
        finally:
            with guard:
                active["cur"] -= 1

    ReadTool.call = traced
    try:
        messages = runner.run(calls)
    finally:
        ReadTool.call = original

    assert [m["tool_call_id"] for m in messages] == [f"r{i}" for i in range(5)]
    assert active["max"] >= 2  # actually ran concurrently


def test_parallel_disabled_runs_serial(tmp_path):
    for i in range(3):
        (tmp_path / f"f{i}.txt").write_text(f"c{i}\n")
    s, runner = _runner(tmp_path)
    s.settings.max_parallel_tools = 1  # disabled -> identical to legacy serial behavior
    calls = [ToolCall(id=f"r{i}", name="Read", args=[{"path": f"f{i}.txt", "ranges": [[0, 0]]}]) for i in range(3)]

    active = {"cur": 0, "max": 0}
    guard = threading.Lock()
    original = ReadTool.call

    def traced(self):
        with guard:
            active["cur"] += 1
            active["max"] = max(active["max"], active["cur"])
        time.sleep(0.02)
        try:
            return original(self)
        finally:
            with guard:
                active["cur"] -= 1

    ReadTool.call = traced
    try:
        messages = runner.run(calls)
    finally:
        ReadTool.call = original

    assert [m["tool_call_id"] for m in messages] == ["r0", "r1", "r2"]
    assert active["max"] == 1  # never overlapped


def test_refusal_short_circuits_across_parallel_and_serial(tmp_path):
    for i in range(3):
        (tmp_path / f"f{i}.txt").write_text(f"c{i}\n")
    s, runner = _runner(tmp_path, input_reply="no")  # decline confirmation
    s.settings.max_parallel_tools = 4
    calls = [
        ToolCall(id="r0", name="Read", args=[{"path": "f0.txt", "ranges": [[0, 0]]}]),
        ToolCall(id="r1", name="Read", args=[{"path": "f1.txt", "ranges": [[0, 0]]}]),
        ToolCall(id="b0", name="Bash", args=[":"]),  # confirmation required, refused
        ToolCall(id="r2", name="Read", args=[{"path": "f2.txt", "ranges": [[0, 0]]}]),  # skipped
    ]
    messages = runner.run(calls)
    by_id = {m["tool_call_id"]: m["content"] for m in messages}
    assert [m["tool_call_id"] for m in messages] == ["r0", "r1", "b0", "r2"]
    assert "refused" in by_id["b0"].lower()
    assert "Skipped" in by_id["r2"]


def test_silent_tool_success_emits_no_log_line(tmp_path):
    # NextHints is a pure-UI tool: its effect (the chips) shows at the idle prompt, so a successful
    # call must not print a call/result log line at all. The model still gets its tool result.
    s = session(tmp_path)
    outputs: list[str] = []
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda *a: "", output_fn=lambda text: outputs.append(str(text)))
    messages = runner.run([call("NextHints", [{"inputs": ["run the tests", "show the diff"]}])])

    assert outputs == []  # no log line for a successful pure-UI tool
    assert len(messages) == 1  # the model still receives its tool result
    assert s.quick_hints == ("run the tests", "show the diff")


def test_silent_tool_failure_still_emits_a_log_line(tmp_path):
    # A failed silent-tool call is a real error the user must see, so the suppression does not apply.
    s = session(tmp_path)
    outputs: list[str] = []
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda *a: "", output_fn=lambda text: outputs.append(str(text)))
    messages = runner.run([call("NextHints", [{"inputs": []}])])

    assert outputs and "rejected" in outputs[0]  # argument error is surfaced, not swallowed
    assert len(messages) == 1
    assert "at least one non-empty" in messages[0]["content"]


def test_agent_followup_turn_snapshot_resume_invariant(tmp_path, monkeypatch):
    """Save and reload a turn that took a live follow-up and a protocol correction: both appear once
    as durable user messages, pending is empty, and no assistant tool call lacks its result."""
    s = session(tmp_path)
    s.config.provider.url = "http://test"
    s.config.provider.key = "k"
    s.config.provider.model = "m"
    queue(s, "live follow-up")
    agent = Agent(s, output_fn=lambda _text: None)
    pseudo = '<invoke name="Bash"><parameter name="command">never-run</parameter></invoke>'

    responses = [
        ({"role": "assistant", "content": pseudo}, [], pseudo),
        (
            {
                "role": "assistant",
                "content": "acknowledged",
                "tool_calls": [{"id": "Read-id", "type": "function", "function": {"name": "Read", "arguments": "{}"}}],
                "_responses_output": [
                    {"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque"},
                    {"id": "fc_1", "type": "function_call", "call_id": "Read-id", "name": "Read", "arguments": "{}"},
                ],
            },
            [call("Read", [{"path": "a.txt", "ranges": [[0, 1]]}])],
            "acknowledged",
        ),
        ({"role": "assistant", "content": "done"}, [], "done"),
    ]
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")

    def fake_api_request(messages, tools, *, allow_stream=True):
        assert tools, "the tool list must never be emptied for a request"
        return responses.pop(0)

    monkeypatch.setattr(agent.model, "api_request", fake_api_request)
    assert agent.run("initial request") == "done"

    s.save_snapshot()
    restored = Session.load_snapshot(s.uid, config=s.config, settings=s.settings)

    # The live follow-up and the correction each appear once as durable user messages
    followup_messages = [m for m in restored.messages if "live follow-up" in (m.get("content") or "")]
    corrections = [m for m in restored.messages if "[Runtime protocol correction]" in (m.get("content") or "")]
    assert len(followup_messages) == 1
    assert len(corrections) == 1
    assert all(pseudo not in (m.get("content") or "") for m in restored.messages)

    # pending_user_inputs is empty
    assert restored.pending_user_inputs == []

    # There are no assistant local-tool calls without matching tool results
    tool_result_ids = {m.get("tool_call_id") for m in restored.messages if m.get("role") == "tool"}
    for msg in restored.messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            assert tc.get("id") in tool_result_ids, f"dangling tool call {tc.get('id')}"

    # Every function_call replayed into the next Responses request has its output beside it
    client = ModelClient(restored)
    replayed = client.responses_input(restored.messages)
    outputs = {item.get("call_id") for item in replayed if item.get("type") == "function_call_output"}
    assert [item.get("call_id") for item in replayed if item.get("type") == "function_call"] == ["Read-id"]
    assert outputs == {"Read-id"}
