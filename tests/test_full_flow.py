"""Full-flow tests: the real agent loop driving the real ModelClient over a mocked wire.

Existing tests cover the two halves separately — the `test_model_*.py` modules exercise ModelClient
against `httpx.MockTransport`, and `test_agent_turn.py` runs the agent loop against a hand-scripted
Python fake injected at `agent.model`. Neither crosses the seam between them: how the agent's
messages and tool schemas serialize onto the wire, and how a provider's response parses back into
tool calls that the runner then executes. These tests close that seam by pointing the real
ModelClient at a scripted in-process LLM (no sockets, no ports) and running `agent.run` end to end.
"""

import json

import pytest
from model_harness import _AnthropicMockClientFactory, _MockClientFactory
from openai_mock_server import OpenAIMockServer

from wizolt.base import SESSION_EVENT_KEY
from wizolt.config import (
    MIN_CONTEXT_SAFETY_TOKENS,
    Config,
    ProviderConfig,
)
from wizolt.context import ContextManager
from wizolt.engine import Agent
from wizolt.model import ModelClient
from wizolt.prompts import COMPACTION_SUMMARY_TITLE, SYSTEM_PROMPT
from wizolt.session import Session
from wizolt.skill import SkillLibrary
from wizolt.tools import Tool


def _tool_call_response(call_id: str, name: str, arguments: dict) -> tuple[int, dict]:
    return 200, {
        "id": "chatcmpl-call",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _answer_response(text: str) -> tuple[int, dict]:
    return 200, {
        "id": "chatcmpl-answer",
        "object": "chat.completion",
        "created": 2,
        "model": "gpt-4",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    }


def _session(tmp_path, *, api: str = "chat", model: str = "gpt-5.6", reasoning: str = "medium"):
    config = Config()
    config.data_dir = str(tmp_path / "data")
    config.providers = {"default": ProviderConfig(url="http://test", key="sk-test", model=model, api=api, reasoning=reasoning, stream=False)}
    session = Session(cwd=str(tmp_path), config=config)
    session.settings.yolo = True  # auto-approve mutating tools so the flow runs unattended
    session.skills = SkillLibrary({})  # no skills: keep the system frame deterministic
    return session


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_append_only_turns_reuse_implicit_cache_for_both_openai_protocols(tmp_path, monkeypatch, api):
    session = _session(tmp_path, api=api)
    server = OpenAIMockServer(["first answer", "second answer"])
    monkeypatch.setattr(ModelClient, "client", lambda self, **kwargs: server.client())
    agent = Agent(session, output_fn=lambda _text: None)

    assert agent.run("first request") == "first answer"
    assert agent.run("second request") == "second answer"

    assert len(server.requests) == 2
    first_prompt, first_read, first_write = server.cache_events[0]
    second_prompt, second_read, second_write = server.cache_events[1]
    assert first_prompt > 0 and first_read == 0 and first_write > 0
    assert second_prompt > first_prompt and second_read > 0 and second_write > 0
    assert session.usage.last_cached_prompt_tokens == second_read
    assert session.usage.last_cache_write_prompt_tokens == second_write

    key = "input" if api == "responses" else "messages"
    first_items = server.requests[0][key]
    second_items = server.requests[1][key]
    assert second_items[: len(first_items)] == first_items


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_note_tool_history_advances_the_longest_implicit_breakpoint(tmp_path, monkeypatch, api):
    session = _session(tmp_path, api=api)
    server = OpenAIMockServer(
        [
            {"tool": "Note", "arguments": {"set_goal": "preserve cache"}},
            "noted",
            "continued",
        ]
    )
    monkeypatch.setattr(ModelClient, "client", lambda self, **kwargs: server.client())
    agent = Agent(session, output_fn=lambda _text: None)

    assert agent.run("remember the cache goal") == "noted"
    assert agent.run("continue") == "continued"

    assert session.state.goal == "preserve cache"
    assert len(server.cache_events) == 3
    first_user_read = server.cache_events[1][1]
    tool_boundary_read = server.cache_events[2][1]
    assert first_user_read > 0
    assert tool_boundary_read > first_user_read
    if api == "responses":
        assert any(item.get("type") == "function_call_output" for item in server.requests[1]["input"])
    else:
        assert any(message.get("role") == "tool" for message in server.requests[1]["messages"])


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_compaction_starts_one_cache_epoch_then_the_checkpoint_warms(tmp_path, monkeypatch, api):
    session = _session(tmp_path, api=api)
    server = OpenAIMockServer(
        [
            "archived answer",
            "warm answer",
            json.dumps({"summary": "archived", "goal": "continue", "plan": [], "known": [], "check": "done"}),
            "continued",
            "after checkpoint",
        ]
    )
    monkeypatch.setattr(ModelClient, "client", lambda self, **kwargs: server.client())
    agent = Agent(session, output_fn=lambda _text: None)

    assert agent.run("archive request " + "x" * 8_000) == "archived answer"
    assert agent.run("keep working") == "warm answer"
    baseline = _session(tmp_path / "baseline", api=api)
    baseline_context = ContextManager(baseline)
    baseline_messages = baseline_context.model_messages(SYSTEM_PROMPT, [{"role": "user", "content": "continue"}])
    baseline_tokens = baseline_context.request_tokens(baseline_messages, Tool.resolved_schemas(baseline))
    original_limit = session.settings.max_context_tokens
    session.settings.max_context_tokens = baseline_tokens + 500 + session.config.provider.output_token_budget() + MIN_CONTEXT_SAFETY_TOKENS

    assert agent.run("continue") == "continued"
    session.settings.max_context_tokens = original_limit
    assert agent.run("after compaction") == "after checkpoint"

    assert session.state.compaction_count == 1
    assert len(server.cache_events) == 5
    assert server.cache_events[0][2] > 0  # cold: the first turn writes
    assert server.cache_events[1][1] > 0  # an appending turn reads what the last one wrote
    # [2] is the summary request, and it reads the cache: Compactor.request slices the very
    # projection the turn just sent and appends one instruction, so the conversation is a warm
    # prefix and only the tail is paid for. Rebuilding a lookalike out of `compacted` instead --
    # the regression this guards -- diverges at the first message and reads nothing back.
    assert server.cache_events[2][1] >= server.cache_events[1][1]
    # [3] is the first request after the rebuild, and it reads nothing: apply_compaction replaced
    # the head of the conversation, so one new epoch begins whatever the checkpoint says. [4] shows
    # the checkpoint is stable history rather than a per-turn rebuild -- the next turn warms from it.
    assert server.cache_events[3][1] == 0
    assert server.cache_events[3][2] > 0
    assert server.cache_events[4][1] > 0


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_resume_event_keeps_old_breakpoint_and_becomes_part_of_the_next_one(tmp_path, monkeypatch, api):
    session = _session(tmp_path, api=api)
    server = OpenAIMockServer(["before resume", "resumed answer", "next answer"])
    monkeypatch.setattr(ModelClient, "client", lambda self, **kwargs: server.client())

    assert Agent(session, output_fn=lambda _text: None).run("first request") == "before resume"
    session.save_snapshot()
    resumed = Session.load_snapshot(session.uid, config=session.config, cwd=session.cwd)
    resumed.skills = SkillLibrary({})
    agent = Agent(resumed, output_fn=lambda _text: None)
    assert agent.run("after resume") == "resumed answer"
    assert agent.run("next request") == "next answer"

    assert len(server.cache_events) == 3
    assert server.cache_events[1][1] > 0
    assert server.cache_events[2][1] > server.cache_events[1][1]
    assert sum(message.get(SESSION_EVENT_KEY) == "resumed" for message in resumed.messages) == 1
    key = "input" if api == "responses" else "messages"
    wire_items = server.requests[1][key]
    assert any(item.get("role") == "user" and str(item.get("content") or "").startswith("<session_event type=") for item in wire_items)


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_model_cache_scope_change_misses_once_then_warms(tmp_path, monkeypatch, api):
    session = _session(tmp_path, api=api)
    server = OpenAIMockServer(["first", "after switch", "warm again"])
    monkeypatch.setattr(ModelClient, "client", lambda self, **kwargs: server.client())
    agent = Agent(session, output_fn=lambda _text: None)

    assert agent.run("first request") == "first"
    session.config.provider.model = "gpt-5.6-new-scope"
    assert agent.run("after model switch") == "after switch"
    assert agent.run("same model again") == "warm again"

    assert server.cache_events[0][1] == 0
    assert server.cache_events[1][1] == 0
    assert server.cache_events[2][1] > 0
    assert server.requests[1]["prompt_cache_key"] != server.requests[0]["prompt_cache_key"]
    assert server.requests[2]["prompt_cache_key"] == server.requests[1]["prompt_cache_key"]


def test_full_flow_edit_then_answer(tmp_path, monkeypatch):
    """The model emits an Edit tool call, the runner applies it to disk, and the tool result rides
    back to the model on the next request before the final answer — the whole loop over the wire."""
    session = _session(tmp_path)
    edit_args = {"path": "hello.txt", "edits": [{"op": "create", "content": "hi\n"}]}
    factory = _MockClientFactory([_tool_call_response("call_1", "Edit", edit_args), _answer_response("Created hello.txt.")])
    monkeypatch.setattr(ModelClient, "client", lambda self, **kwargs: factory())

    answer = Agent(session, output_fn=lambda text: None).run("create hello.txt containing hi")

    # The tool really ran: the file exists on disk and the run returned the model's final answer.
    assert answer == "Created hello.txt."
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hi\n"
    assert [record.name for record in session.tool_records] == ["Edit"]

    # The wire round-trip: two model calls. The first carries the tool schemas; the second carries
    # the assistant's tool_calls and the tool result serialized back as a `tool` message.
    requests = [json.loads(call.content) for call in factory.calls]
    assert len(requests) == 2
    assert requests[0]["tools"]
    assert any(tool["function"]["name"] == "Edit" for tool in requests[0]["tools"])

    second = requests[1]["messages"]
    assistant_calls = [m for m in second if m["role"] == "assistant" and m.get("tool_calls")]
    assert assistant_calls and assistant_calls[0]["tool_calls"][0]["function"]["name"] == "Edit"
    tool_messages = [m for m in second if m["role"] == "tool"]
    assert tool_messages and tool_messages[0]["tool_call_id"] == "call_1"
    assert "<Edit" in tool_messages[0]["content"]


def test_full_flow_anthropic_tool_then_answer(tmp_path, monkeypatch):
    """Anthropic tool_use crosses the SDK boundary, runs, and returns as tool_result."""

    session = _session(tmp_path, api="anthropic", model="claude-3", reasoning="off")
    edit_args = {"path": "claude.txt", "edits": [{"op": "create", "content": "done\n"}]}
    factory = _AnthropicMockClientFactory(
        [
            (
                200,
                {
                    "id": "msg_tool",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3",
                    "content": [{"type": "tool_use", "id": "call_1", "name": "Edit", "input": edit_args}],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            ),
            (
                200,
                {
                    "id": "msg_answer",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3",
                    "content": [{"type": "text", "text": "Created claude.txt."}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 20, "output_tokens": 8},
                },
            ),
        ]
    )
    monkeypatch.setattr(ModelClient, "anthropic_client", lambda self, **kwargs: factory())

    answer = Agent(session, output_fn=lambda _text: None).run("create claude.txt")

    assert answer == "Created claude.txt."
    assert (tmp_path / "claude.txt").read_text(encoding="utf-8") == "done\n"
    assert len(factory.calls) == 2
    second = json.loads(factory.calls[1].content)
    assert any(message["role"] == "assistant" and message["content"][0]["type"] == "tool_use" for message in second["messages"])
    tool_results = [
        block
        for message in second["messages"]
        if message["role"] == "user" and isinstance(message["content"], list)
        for block in message["content"]
        if block["type"] == "tool_result"
    ]
    assert tool_results and tool_results[0]["tool_use_id"] == "call_1"


def test_full_flow_compacts_before_answering(tmp_path, monkeypatch):
    """An over-budget request crosses the compactor wire, then resumes from one full checkpoint."""
    session = _session(tmp_path)
    old_request = "archive request " + "x" * 200 + " OLD_BODY_SENTINEL " + "x" * 8000
    session.messages = [
        {"role": "user", "content": old_request},
        {"role": "assistant", "content": "archived answer " + "y" * 8000},
        {"role": "user", "content": "latest retained request"},
        {"role": "assistant", "content": "latest retained answer"},
    ]
    baseline = _session(tmp_path / "baseline")
    baseline_context = ContextManager(baseline)
    baseline_messages = baseline_context.model_messages(SYSTEM_PROMPT, [{"role": "user", "content": "continue"}])
    baseline_tokens = baseline_context.request_tokens(baseline_messages, Tool.resolved_schemas(baseline))
    session.settings.max_context_tokens = baseline_tokens + 500 + session.config.provider.output_token_budget() + MIN_CONTEXT_SAFETY_TOKENS

    # The agent's own working state, set the way `Note` sets it. The checkpoint has to carry it
    # across the eviction unchanged; the summarizer's volunteered replacements below are ignored.
    session.state.goal = "continue"
    session.state.known = ["durable fact"]
    session.state.check = "tests"
    compacted_state = json.dumps(
        {"summary": "Archived work was completed.", "goal": "invented", "plan": [{"status": "todo", "text": "invented"}], "known": ["invented"], "check": "invented"}
    )
    factory = _MockClientFactory([_answer_response(compacted_state), _answer_response("Continued successfully.")])
    monkeypatch.setattr(ModelClient, "client", lambda self, **kwargs: factory())

    answer = Agent(session, output_fn=lambda text: None).run("continue")

    assert answer == "Continued successfully."
    requests = [json.loads(call.content) for call in factory.calls]
    assert len(requests) == 2

    compactor_request, agent_request = requests
    # The summary rides the agent's own prefix: same system message, same tools, the conversation
    # as real messages, and the compaction instruction appended as the last message. That makes
    # this request a prefix of the agent's, so the provider's cache covers all but the tail.
    assert compactor_request["messages"][0]["content"] == agent_request["messages"][0]["content"]
    assert compactor_request["tools"] == agent_request["tools"]
    # Identical tool_choice, deliberately: changing it invalidates the messages cache, which is the
    # conversation this request exists to reuse. The instruction not to call tools is in the tail.
    assert compactor_request["tool_choice"] == agent_request["tool_choice"]
    assert "Compact the wizolt working context." in compactor_request["messages"][-1]["content"]
    assert "OLD_BODY_SENTINEL" in "\n".join(str(message.get("content") or "") for message in compactor_request["messages"])

    active_messages = agent_request["messages"]
    contents = [str(message.get("content") or "") for message in active_messages]
    conversation = next(index for index, content in enumerate(contents) if content.startswith(COMPACTION_SUMMARY_TITLE))
    current_turn = max(index for index, content in enumerate(contents) if content == "continue")
    assert conversation < current_turn
    # The working state is labelled a snapshot rather than kept current: correcting the checkpoint
    # later would rewrite the head of the conversation and start another cache epoch.
    assert "Working state (at this compaction; a later Note call supersedes it):\nGoal: continue" in contents[conversation]
    # The retained archive by range and count, so the model knows it exists without being told to
    # read it; naming only the newest segment left the older ones with no trace after a rebuild.
    assert "Recallable history: seg.1 (1 segment)" in contents[conversation]
    # goal/plan/known/check are Note's: a summarizer that volunteers replacements is ignored.
    assert "invented" not in contents[conversation]
    assert (session.state.goal, session.state.known, session.state.check) == ("continue", ["durable fact"], "tests")
    assert not any(content.startswith(("--- History index ---", "--- Memory ---")) for content in contents)
    assert "OLD_BODY_SENTINEL" not in "\n".join(contents)
    assert agent_request["tools"]

    assert session.state.summary == "Archived work was completed."
    assert session.state.compaction_count == 1
    assert [segment.key for segment in session.history] == ["seg.1"]
    assert "OLD_BODY_SENTINEL" in session.history[0].text


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_a_note_update_after_compaction_never_rewrites_the_checkpoint(tmp_path, monkeypatch, api):
    """The checkpoint is a snapshot, and keeping it a snapshot is what keeps the prefix cached.

    It bakes `state.format()` into message text at compaction time and holds no reference to the
    state it came from, so a later `Note` call changes `session.state` and leaves those bytes
    alone. Making the checkpoint stay current instead would rewrite the head of the conversation
    on every plan edit and start a fresh cache epoch each time -- the whole body at full price for
    one line. Nothing else in the suite would notice: the content assertions would still pass on
    fresher text, and no unit test can see a cache bill.
    """
    session = _session(tmp_path, api=api)
    server = OpenAIMockServer(
        [
            "archived answer",
            "warm answer",
            json.dumps({"title": "archived span", "summary": "archived"}),
            "continued",
            {"tool": "Note", "arguments": {"set_goal": "changed after the checkpoint was written"}},
            "noted",
        ]
    )
    monkeypatch.setattr(ModelClient, "client", lambda self, **kwargs: server.client())
    agent = Agent(session, output_fn=lambda _text: None)
    session.state.goal = "set before compaction"

    assert agent.run("archive request " + "x" * 8_000) == "archived answer"
    assert agent.run("keep working") == "warm answer"
    baseline = _session(tmp_path / "baseline", api=api)
    baseline_context = ContextManager(baseline)
    baseline_messages = baseline_context.model_messages(SYSTEM_PROMPT, [{"role": "user", "content": "continue"}])
    baseline_tokens = baseline_context.request_tokens(baseline_messages, Tool.resolved_schemas(baseline))
    original_limit = session.settings.max_context_tokens
    session.settings.max_context_tokens = baseline_tokens + 500 + session.config.provider.output_token_budget() + MIN_CONTEXT_SAFETY_TOKENS
    assert agent.run("continue") == "continued"
    session.settings.max_context_tokens = original_limit

    assert session.state.compaction_count == 1
    checkpoint = session.messages[0]["content"]
    assert checkpoint.startswith(COMPACTION_SUMMARY_TITLE)
    assert "Goal: set before compaction" in checkpoint
    key = "input" if api == "responses" else "messages"
    before_items = server.requests[-1][key]

    assert agent.run("update the goal") == "noted"

    # The Note landed, and the checkpoint did not move with it.
    assert session.state.goal == "changed after the checkpoint was written"
    assert session.messages[0]["content"] == checkpoint
    assert "Goal: set before compaction" in session.messages[0]["content"]
    # And the wire says the same thing: the prefix that request carried is still a prefix, so the
    # epoch the compaction opened is still being read back rather than replaced.
    assert server.requests[-1][key][: len(before_items)] == before_items
    assert server.cache_events[-1][1] > 0
