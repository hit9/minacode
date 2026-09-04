"""agent flow (split from tests/test_agent_turn.py)."""
import asyncio
from types import SimpleNamespace

import pytest
from agent_harness import call, queue, session
from test_agent_turn import _correction

import wizolt.engine as engine_module
from wizolt.base import (
    LogBlock,
    MalformedToolCallError,
)
from wizolt.engine import Agent
from wizolt.prompts import LIVE_FOLLOWUP_PREFIX, SYSTEM_PROMPT
from wizolt.tools import Tool


async def test_cancelling_a_turn_ends_the_in_flight_attempt_and_closes_its_client(tmp_path):
    """Cancelling the turn is the whole mechanism: no signal, no second cancellation channel.

    The turn's task is cancelled, that reaches the provider attempt by propagation, and the attempt
    closes the client it opened on its way out -- so nothing is left holding a connection."""
    s = session(tmp_path)
    s.config.provider.url = "https://example.test/v1"
    s.config.provider.key = "test"
    s.config.provider.model = "model"
    started = asyncio.Event()
    closed = []

    class Completions:
        async def create(self, **_params):
            started.set()
            await asyncio.sleep(30)

    class Client:
        chat = SimpleNamespace(completions=Completions())

        def __init__(self, provider=None):
            pass

        async def close(self):
            closed.append(True)

    agent = Agent(s, output_fn=lambda _text: None)
    agent.model.client = Client

    turn = asyncio.ensure_future(agent.run("hello"))
    await asyncio.wait_for(started.wait(), 2)
    agent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert closed  # the attempt closed its own client before the turn unwound
    assert agent._active_task is None  # and the turn released its cancellation handle


async def test_a_late_cancel_cannot_reach_the_next_turn(tmp_path):
    """The task and its loop are cleared together when a turn ends, so a Ctrl-C that arrives just
    after one finished is ignored rather than cancelling whatever runs next."""
    s = session(tmp_path)
    agent = Agent(s, output_fn=lambda _text: None)

    class Model:
        async def request(self, messages, tools=None):
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = Model()
    assert await agent.run("first") == "done"

    agent.cancel()  # late: the turn is over and there is nothing to cancel

    assert await agent.run("second") == "done"


def test_agent_injects_pending_user_input_once(tmp_path):
    s = session(tmp_path)
    queue(s, "extra instruction")
    output = []
    agent = Agent(s, output_fn=output.append)

    class FakeModel:
        def __init__(self):
            self.messages = []

        async def request(self, messages, tools=None):
            self.messages.append(messages)
            if len(self.messages) == 1:
                s.enqueue_user_input("second instruction")
                return {}, [call("Bash", ["wc -l missing.txt"])], "checking"
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = FakeModel()
    assert agent.run_sync("initial request") == "done"

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
    # Committed exactly as sent, marker included: the second request replays the same bytes.
    assert s.messages[1]["content"] == LIVE_FOLLOWUP_PREFIX + "extra instruction"
    assert s.messages[1] in agent.model.messages[1]  # byte-identical in the next request's prefix
    assert s.messages[2]["content"] == "checking"
    assert s.messages[3]["role"] == "tool"
    assert s.messages[3]["content"].startswith("tool tr.1 Bash wc -l missing.txt")
    assert s.messages[4]["content"] == LIVE_FOLLOWUP_PREFIX + "second instruction"
    assert "checking" in output
    assert s.messages[5]["role"] == "assistant"
    assert s.pending_user_inputs == []

def test_agent_expands_file_mentions_in_queued_input_before_sending(tmp_path):
    (tmp_path / "queued.txt").write_text("queued context\n", encoding="utf-8")
    s = session(tmp_path)
    queue(s, "inspect @file:queued.txt")
    agent = Agent(s, output_fn=lambda _text: None)

    class FakeModel:
        def __init__(self):
            self.messages = []

        async def request(self, messages, tools=None):
            del tools
            self.messages.append(messages)
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = FakeModel()
    assert agent.run_sync("initial request") == "done"

    sent = agent.model.messages[0]
    assert any(message.get("content") == LIVE_FOLLOWUP_PREFIX + "inspect @file:queued.txt" for message in sent)
    assert any("--- FILE MENTIONS ---" in str(message.get("content") or "") and "queued context" in str(message.get("content") or "") for message in sent)
    assert sum("--- FILE MENTIONS ---" in str(message.get("content") or "") for message in s.messages) == 1
    assert not any("--- FILE MENTIONS ---" in str(message.get("content") or "") for message in s.transcript_messages)

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

        async def request(self, messages, tools=None):
            self.requests.append((messages, tools))
            if len(self.requests) == 1:
                return {}, [call("Bash", ["echo hi"])], ""
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = FakeModel()

    assert agent.run_sync("initial request") == "done"
    assert len(agent.model.requests) == 2  # one request per step, none inserted for the follow-ups
    assert all(tools for _, tools in agent.model.requests)
    assert agent.model.requests[0][1] == agent.model.requests[1][1]
    first_request = "\n".join(message.get("content") or "" for message in agent.model.requests[0][0])
    assert "first follow-up" in first_request and "second follow-up" in first_request
    assert [message["role"] for message in s.messages] == ["user", "user", "user", "assistant", "tool", "assistant"]
    assert len(s.tool_records) == 1
    assert s.pending_user_inputs == []
    assert all(isinstance(item, LogBlock) for item in output[:-1])  # tool logs only
    assert output[-1] == "done"  # the engine publishes the final answer through output_fn

def test_agent_never_rewrites_a_sent_followup_message(tmp_path):
    """The follow-up marker travels with the message it marked. Committing the bare text instead
    would change bytes the provider already cached, ending the shared prefix at that message."""
    s = session(tmp_path)
    queue(s, "an early follow-up")
    agent = Agent(s, output_fn=lambda _text: None)
    read = call("Read", [{"path": "a.txt", "ranges": [[0, 1]]}])
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModel:
        def __init__(self):
            self.requests = []

        async def request(self, messages, tools=None):
            self.requests.append([dict(message) for message in messages])
            if len(self.requests) == 1:
                return {}, [read], "on it"
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = FakeModel()

    assert agent.run_sync("initial request") == "done"
    sent = next(message for message in agent.model.requests[0] if "an early follow-up" in str(message.get("content") or ""))
    assert sent["content"] == LIVE_FOLLOWUP_PREFIX + "an early follow-up"
    replayed = [message for message in agent.model.requests[1] if "an early follow-up" in str(message.get("content") or "")]
    assert replayed == [sent]  # same bytes, once
    assert [message for message in s.messages if "an early follow-up" in str(message.get("content") or "")] == [sent]

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

        async def request(self, messages, tools=None):
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

    assert agent.run_sync("initial request") == "done"
    assert len(agent.model.requests) == 4
    tool_blocks = [tools for _, tools in agent.model.requests]
    assert all(tools and tools == tool_blocks[0] for tools in tool_blocks)

    # Messages only ever grow: each request is a prefix of the next, once the one-shot follow-up
    # marker (dropped when the queued input is committed) is normalized away.
    def normalized(messages):
        return [str(message.get("content") or "").replace(LIVE_FOLLOWUP_PREFIX, "") for message in messages]

    lengths = [len(messages) for messages, _ in agent.model.requests]
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
    correction = _correction("Bash")

    class FakeModel:
        def __init__(self):
            self.requests = []
            self.on_stream = None

        async def request(self, messages, tools=None):
            self.requests.append(([dict(message) for message in messages], tools))
            if len(self.requests) == 1:
                return {"role": "assistant", "content": pseudo}, [], pseudo
            if len(self.requests) == 2:
                return {}, [call("Bash", ["echo hi"])], "on it"
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = FakeModel()

    assert agent.run_sync("initial request") == "done"
    assert len(agent.model.requests) == 3
    assert all(tools == agent.model.requests[0][1] for _, tools in agent.model.requests)
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

        async def request(self, messages, tools=None):
            self.requests.append((messages, tools))
            return {"role": "assistant", "content": pseudo}, [], pseudo

    agent.model = FakeModel()

    with pytest.raises(
        MalformedToolCallError,
        match=r"Model emitted Bash as text 6 times; none of the textual calls were executed\.",
    ):
        agent.run_sync("initial request")

    assert len(agent.model.requests) == engine_module.MAX_TEXTUAL_TOOL_CORRECTIONS + 1
    assert all(tools == agent.model.requests[0][1] and tools for _, tools in agent.model.requests)
    # Each correction stacks onto the previous one, so the retries grow by exactly one message.
    lengths = [len(messages) for messages, _ in agent.model.requests]
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

        async def request(self, messages, request_tools=None):
            self.received_tools = request_tools
            return {"role": "assistant", "content": "done"}, [], "done"

    monkeypatch.setattr(Tool, "resolved_schemas", staticmethod(resolve))
    agent.model = FakeModel()

    assert agent.run_sync("hello") == "done"
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

        async def request(self, messages, tools=None):
            self.messages.append(messages)
            if len(self.messages) == 1:
                return {}, [call("Read", [{"path": "a.txt", "ranges": [[0, 1]]}])], "I'll inspect that first."
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = TalkingModel()
    assert agent.run_sync("read file") == "done"
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
