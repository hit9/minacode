"""agent loop (split from tests/test_agent_turn.py)."""
import pytest
from agent_harness import call, session

from minacode.engine import Agent
from minacode.prompts import INTERRUPT_MARKER
from minacode.session import Session, SessionSnapshotCodec
from minacode.skill import SkillLibrary


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

def test_file_mentions_land_as_their_own_user_message(tmp_path):
    """The FILE MENTIONS block is appended as its own user message after the user's text, which
    is never rewritten."""
    (tmp_path / "small.py").write_text("print(1)\n", encoding="utf-8")
    s = session(tmp_path)
    s.skills = SkillLibrary({})
    agent = Agent(s, output_fn=lambda _text: None)

    class FakeModel:
        def __init__(self):
            self.requests = []

        def request(self, messages, tools=None):
            self.requests.append(messages)
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = FakeModel()
    assert agent.run("fix @file:small.py") == "done"
    user_messages = [message for message in agent.model.requests[0] if message["role"] == "user"]
    contents = [str(message.get("content") or "") for message in user_messages]
    assert "fix @file:small.py" in contents  # the user's text is present and never rewritten
    mentions = [content for content in contents if "--- FILE MENTIONS ---" in content]
    assert len(mentions) == 1
    assert "[small.py] 1 lines" in mentions[0]

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
    assert s.transcript_messages[-1] == {"role": "assistant", "content": "done"}
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
    assert [message["role"] for message in s.transcript_messages] == ["user", "assistant", "tool"]
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
    assert [message["role"] for message in restored.transcript_messages] == ["user", "assistant", "tool"]
    assert restored.transcript_messages[-1] == {
        "role": "tool",
        "tool_call_id": "Read-id",
        "result_key": "tr.1",
        "status": "ok",
    }

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
    assert s.transcript_messages == []
    assert s._active_turn_messages == []
    assert s._active_transcript_messages == []
    restored = Session.load_snapshot(s.uid, config=s.config, settings=s.settings)
    messages = [message for message in restored.messages if not SessionSnapshotCodec.is_internal_message(message)]
    assert messages == []
    assert restored.transcript_messages == []

def test_interrupted_unfinished_tool_call_gets_semantic_transcript_result(tmp_path):
    s = session(tmp_path)
    s.skills = SkillLibrary({})
    agent = Agent(s, output_fn=lambda _text: None)

    class ToolCallingModel:
        def request(self, messages, tools=None):
            return {}, [call("Read", [{"path": "a.txt", "ranges": [[0, 1]]}])], ""

    agent.model = ToolCallingModel()

    def interrupt_tools(*_args, **_kwargs):
        raise KeyboardInterrupt

    agent.tools.run = interrupt_tools

    with pytest.raises(KeyboardInterrupt):
        agent.run("read file")

    assert s.transcript_messages[-1] == {
        "role": "tool",
        "tool_call_id": "Read-id",
        "result_key": "",
        "status": "failed",
    }
    restored = Session.load_snapshot(s.uid, config=s.config, settings=s.settings)
    assert restored.transcript_messages[-1] == s.transcript_messages[-1]

def test_current_turn_compaction_does_not_rewrite_visible_transcript(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    s = session(tmp_path)
    s.skills = SkillLibrary({})
    agent = Agent(s, output_fn=lambda _text: None)
    prepare_calls = 0

    def prepare_messages(_model, _system, turn_messages, _tools):
        nonlocal prepare_calls
        prepare_calls += 1
        if prepare_calls == 2:
            turn_messages[:] = [{"role": "user", "content": "compacted current turn"}]
        return list(turn_messages)

    monkeypatch.setattr(agent.context, "prepare_messages", prepare_messages)
    monkeypatch.setattr(agent.context, "update_percent", lambda _messages, _tools: 0)

    class Model:
        calls = 0

        def request(self, _messages, _tools=None):
            self.calls += 1
            if self.calls == 1:
                return {}, [call("Read", [{"path": "a.txt", "ranges": [[0, 1]]}])], "checking"
            return {}, [], "done"

    agent.model = Model()

    assert agent.run("read the file") == "done"
    assert [message.get("content") for message in s.messages] == ["compacted current turn", "done"]
    assert [message["role"] for message in s.transcript_messages] == ["user", "assistant", "tool", "assistant"]
    assert s.transcript_messages[0]["content"] == "read the file"
    assert s.transcript_messages[1]["content"] == "checking"
    assert s.transcript_messages[-1]["content"] == "done"

    s.save_snapshot()
    restored = Session.load_snapshot(s.uid, config=s.config, settings=s.settings)
    assert [message["role"] for message in restored.transcript_messages] == ["user", "assistant", "tool", "assistant"]
    assert restored.transcript_messages[0]["content"] == "read the file"

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
