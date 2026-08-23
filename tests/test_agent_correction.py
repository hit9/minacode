"""agent correction (split from tests/test_agent_turn.py)."""
import pytest
from agent_harness import call, session
from test_agent_turn import _correction

import minacode.engine as engine_module
from minacode.base import (
    SESSION_EVENT_KEY,
    MalformedToolCallError,
    ModelError,
)
from minacode.engine import Agent
from minacode.prompts import FAILED_TURN_MARKER, SYSTEM_PROMPT
from minacode.session import Session, SessionSnapshotCodec


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
    corrections = [_correction(name) for name in names]
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
    # The turn aborts, but the corrections it already sent survive: history is append-only, and
    # the error marker records where the turn ended (the failure-path counterpart of the interrupt
    # marker, keeping the two settling paths the same shape).
    assert s.messages == [
        {"role": "user", "content": "continue"},
        *[_correction("Bash")] * engine_module.MAX_TEXTUAL_TOOL_CORRECTIONS,
        {"role": "user", "content": FAILED_TURN_MARKER.format(error="Model emitted Bash as text 6 times; none of the textual calls were executed.")},
    ]
    assert s._active_turn_messages == []
    restored = Session.load_snapshot(s.uid, config=s.config)
    # Drop only what the load itself appended: the corrections are session events too, and they are
    # exactly what this asserts survived.
    restored_messages = [
        message for message in restored.messages if message.get(SESSION_EVENT_KEY) != "resumed" and not SessionSnapshotCodec.is_legacy_internal_message(message)
    ]
    assert restored_messages == s.messages

def test_failed_first_request_leaves_a_marked_legal_history_and_the_next_turn_runs(tmp_path):
    """The failure-path settling also covers a turn that died before the model ever answered: the
    user message stays, a bounded marker records where the turn stopped, and the settled history
    (two consecutive user messages) is a shape the protocol codecs merge, so the next turn on the
    same session goes out normally."""
    s = session(tmp_path)
    agent = Agent(s, output_fn=lambda _text: None)

    class Model:
        fail = True
        on_stream = None

        def request(self, messages, tools=None):
            if self.fail:
                self.fail = False
                raise ModelError("provider exploded")
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = Model()

    with pytest.raises(ModelError, match="provider exploded"):
        agent.run("continue")

    # Nothing was settled (no tool calls ever issued) and nothing was retracted: the user message
    # and the marker are both permanent history, with nothing live left behind.
    assert s.messages == [
        {"role": "user", "content": "continue"},
        {"role": "user", "content": FAILED_TURN_MARKER.format(error="provider exploded")},
    ]
    assert s._active_turn_messages == []
    assert s.state.turn_messages == 0

    # The next turn on the same session runs to completion on the marked history.
    assert agent.run("continue again") == "done"
    assert [message["role"] for message in s.messages] == ["user", "user", "user", "assistant"]
    assert s.messages[-1]["content"] == "done"

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
    assert output[-1] == "done"  # the engine publishes the final answer through output_fn
    assert len(output) == 3

def test_system_prompt_requires_native_tool_calls():
    assert "Use native tool calls; never print tool XML or tool-call JSON." in SYSTEM_PROMPT
