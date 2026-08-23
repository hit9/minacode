"""next hints batches (split from tests/test_agent_turn.py)."""
import json
import threading
import time
from types import SimpleNamespace
import pytest
from agent_harness import call, queue, session
import minacode.engine as engine_module
from minacode.base import (
    SESSION_EVENT_KEY,
    LogBlock,
    MalformedToolCallError,
    ModelError,
    ToolCall,
)
from minacode.config import (
    ANTHROPIC_DEFAULT_MAX_TOKENS,
    Config,
    ProviderConfig,
)
from minacode.context import ContextManager
from minacode.engine import Agent
from minacode.model import ModelClient
from minacode.prompts import FAILED_TURN_MARKER, INTERRUPT_MARKER, LIVE_FOLLOWUP_PREFIX, SYSTEM_PROMPT
from minacode.runner import ToolRunner
from minacode.session import Session, SessionSnapshotCodec
from minacode.skill import SkillLibrary
from minacode.tools import BashTool, ReadTool, Tool

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

def test_all_next_hints_batch_without_answer_ends_turn_in_single_model_call(tmp_path):
    """An all-NextHints batch is terminal even with no answer text: exactly one model request,
    no invented answer, no empty visible output callback, and every hint survives to the idle
    prompt instead of being superseded by a follow-up step."""
    s = session(tmp_path)
    s.skills = SkillLibrary({})
    outputs: list[str] = []
    agent = Agent(s, output_fn=outputs.append)

    class FakeModel:
        def __init__(self):
            self.messages = []

        def request(self, messages, tools=None):
            self.messages.append(messages)
            return {"role": "assistant", "content": ""}, [call("NextHints", [{"inputs": ["run the tests", "show the diff"]}])], ""

    agent.model = FakeModel()
    assert agent.run("do it") == ""  # nothing was invented as an answer
    assert len(agent.model.messages) == 1  # finished on the first call, no extra round trip
    assert s.quick_hints == ("run the tests", "show the diff")  # all hints survived
    assert outputs == []  # no empty visible output callback was emitted
    # The tool call and its result are recorded; no empty closing assistant message is stored.
    assert [m["role"] for m in s.messages] == ["user", "assistant", "tool"]
    assert s.messages[1]["tool_calls"][0]["function"]["name"] == "NextHints"
    assert s.messages[2]["tool_call_id"] == s.messages[1]["tool_calls"][0]["id"]

def test_tool_only_history_replays_across_all_protocols(tmp_path):
    """A tool-only terminal turn stays replayable on the next user turn through every adapter:
    each NextHints call is matched by exactly one tool result, and the next user message lands
    after them."""
    s = session(tmp_path)
    s.skills = SkillLibrary({})
    agent = Agent(s, output_fn=lambda text: None)

    class FakeModel:
        def __init__(self):
            self.messages = []

        def request(self, messages, tools=None):
            self.messages.append(messages)
            return (
                {"role": "assistant", "content": ""},
                [ToolCall("nh-1", "NextHints", [{"inputs": ["run the tests"]}]), ToolCall("nh-2", "NextHints", [{"inputs": ["show the diff"]}])],
                "",
            )

    agent.model = FakeModel()
    assert agent.run("do it") == ""
    assert [m["role"] for m in s.messages] == ["user", "assistant", "tool", "tool"]

    # The next user turn appends its opening user message to the tool-only history.
    history = [*s.messages, {"role": "user", "content": "next turn"}]
    call_ids = {call["id"] for message in s.messages if message.get("role") == "assistant" for call in message.get("tool_calls") or []}
    assert len(call_ids) == 2
    client = ModelClient(s)

    chat = client.chat_messages(history)
    tool_results = [message for message in chat if message.get("role") == "tool"]
    chat_call_ids = {call["id"] for message in chat if message.get("role") == "assistant" for call in message.get("tool_calls") or []}
    assert len(tool_results) == 2
    assert {message["tool_call_id"] for message in tool_results} == chat_call_ids == call_ids
    assert chat[-1]["content"] == "next turn"

    responses = client.responses_input(history)
    outputs = {item["call_id"] for item in responses if item.get("type") == "function_call_output"}
    inputs = {item["call_id"] for item in responses if item.get("type") == "function_call"}
    assert len(outputs) == 2
    assert outputs == inputs == call_ids

    anthropic = client.anthropic_messages(history)
    tool_use_ids = {
        block["id"] for message in anthropic if message.get("role") == "assistant" for block in message.get("content") or [] if block.get("type") == "tool_use"
    }
    tool_result_ids = {
        block["tool_use_id"]
        for message in anthropic
        if message.get("role") == "user"
        for block in message.get("content") or []
        if block.get("type") == "tool_result"
    }
    assert len(tool_use_ids) == 2
    assert tool_result_ids == tool_use_ids == call_ids

    # Several legal NextHints calls in one batch merge their suggestions instead of the last
    # call overwriting the rest.
    assert s.quick_hints == ("run the tests", "show the diff")

def test_failed_tool_only_next_hints_batch_continues_turn(tmp_path):
    """An all-NextHints batch whose calls all fail (no answer text, no suggestions) must not end
    the turn as a blank reply: the error results stay in history and the next step gets to
    correct them."""
    s = session(tmp_path)
    s.skills = SkillLibrary({})
    outputs: list[str] = []
    agent = Agent(s, output_fn=outputs.append)

    class FakeModel:
        def __init__(self):
            self.messages = []

        def request(self, messages, tools=None):
            self.messages.append(messages)
            if len(self.messages) == 1:
                # Empty inputs: the NextHints call fails, so no hints are produced.
                return {"role": "assistant", "content": ""}, [call("NextHints", [{"inputs": []}])], ""
            return {"role": "assistant", "content": "here is the answer"}, [], "here is the answer"

    agent.model = FakeModel()
    assert agent.run("do it") == "here is the answer"
    assert len(agent.model.messages) == 2  # the turn continued past the failed batch
    # The failed batch surfaced its own rejection line; nothing blank was published.
    assert outputs[-1] == "here is the answer"
    assert any(isinstance(item, LogBlock) and "rejected" in str(item) for item in outputs)
    assert s.quick_hints == ()  # no hints were stored
    # The failed tool result reached the second request, so the model could read and correct.
    second_context = "\n\n".join(str(message.get("content") or "") for message in agent.model.messages[1])
    assert "status: failed" in second_context
    assert "at least one non-empty" in second_context

def test_failed_next_hints_batch_counts_as_tool_batch(tmp_path):
    """A failed all-NextHints batch still counts as a tool batch: the next ordinary tool batch
    shows the ·2 suffix instead of presenting as the first batch."""
    s = session(tmp_path)
    s.skills = SkillLibrary({})
    suffixes: list[str] = []
    agent = Agent(s, output_fn=lambda text: None)

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def request(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                return {"role": "assistant", "content": ""}, [call("NextHints", [{"inputs": []}])], ""
            if self.calls == 2:
                return {"role": "assistant", "content": ""}, [call("Read", [{"path": "missing"}])], ""
            return {"role": "assistant", "content": "done"}, [], "done"

    class Tools:
        def run(self, calls, batch_suffix=""):
            suffixes.append(batch_suffix)
            return [{"role": "tool", "tool_call_id": calls[0].id, "name": calls[0].name, "content": "ok"}]

    agent.model = FakeModel()
    agent.tools = Tools()

    assert agent.run("do it") == "done"
    # The failed NextHints batch was the first batch (no suffix); the ordinary batch that
    # follows it is the second tool batch and carries ·2.
    assert suffixes == ["", "·2"]

def test_all_next_hints_batch_with_whitespace_content_ends_turn(tmp_path):
    """Whitespace-only content counts as no answer text: the all-NextHints batch still ends the
    turn in one model call, storing no empty closing message and publishing nothing."""
    s = session(tmp_path)
    s.skills = SkillLibrary({})
    outputs: list[str] = []
    agent = Agent(s, output_fn=outputs.append)

    class FakeModel:
        def __init__(self):
            self.messages = []

        def request(self, messages, tools=None):
            self.messages.append(messages)
            return {"role": "assistant", "content": "   \n\t "}, [call("NextHints", [{"inputs": ["run the tests"]}])], "   \n\t "

    agent.model = FakeModel()
    assert agent.run("do it") == ""
    assert len(agent.model.messages) == 1
    assert s.quick_hints == ("run the tests",)
    assert outputs == []
    assert [m["role"] for m in s.messages] == ["user", "assistant", "tool"]

def test_mixed_next_hints_batch_do_not_leak_into_a_later_answer(tmp_path):
    """A batch mixing NextHints with another tool is not terminal; its hints are transient
    intermediate state and are superseded, so a later final answer never displays them."""
    s = session(tmp_path)
    s.skills = SkillLibrary({})
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    agent = Agent(s, output_fn=lambda text: None)

    class FakeModel:
        def __init__(self):
            self.messages = []

        def request(self, messages, tools=None):
            self.messages.append(messages)
            if len(self.messages) == 1:
                # Mixed batch: hints are only intermediate state, so the turn continues.
                return (
                    {"role": "assistant", "content": ""},
                    [call("NextHints", [{"inputs": ["stale suggestion"]}]), call("Read", [{"path": "a.txt", "ranges": [[0, 1]]}])],
                    "",
                )
            return {"role": "assistant", "content": "different final answer"}, [], "different final answer"

    agent.model = FakeModel()
    assert agent.run("do it") == "different final answer"
    assert len(agent.model.messages) == 2  # the turn continued past the non-terminal batch
    assert s.quick_hints == ()  # the stale hints were cleared, not shown next to the answer
