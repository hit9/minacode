"""builtin tools pause (split from tests/test_builtin_tools.py)."""
from agent_harness import session as agent_session
from model_harness import _AnthropicMockClientFactory, _session

from wizolt.base import (
    PAUSED_TURN_KEY,
    ToolCall,
)
from wizolt.config import (
    ProviderConfig,
)
from wizolt.context import ContextManager
from wizolt.engine import Agent
from wizolt.model import ModelClient
from wizolt.runner import ToolRunner
from wizolt.skill import SkillLibrary


def test_paused_turn_is_reported_and_replays_unchanged(tmp_path, monkeypatch):
    """A paused search must be resumed by sending the assistant message back exactly as received."""
    s = _session(tmp_path, model="claude-3", api="anthropic", stream=False, builtin_tools=({"type": "web_search_20250305", "name": "web_search"},))
    model = ModelClient(s)
    blocks = [
        {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "httpx timeout"}},
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srv_1",
            "content": [{"type": "web_search_result", "url": "https://e.example", "title": "E", "encrypted_content": "keep-me"}],
        },
    ]
    paused = {
        "id": "m1",
        "type": "message",
        "role": "assistant",
        "model": "claude-3",
        "content": blocks,
        "stop_reason": "pause_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    factory = _AnthropicMockClientFactory([(200, paused)])
    monkeypatch.setattr(model, "anthropic_client", factory)

    assistant, calls, _ = model.request_sync([{"role": "user", "content": "hi"}], [])

    assert assistant[PAUSED_TURN_KEY] is True
    assert calls == []
    # Replaying the paused message must preserve encrypted_content; the API rejects it otherwise.
    replayed = model.wire(model.session.config.provider).messages([{"role": "user", "content": "hi"}, assistant])
    assert replayed[-1]["content"] == blocks

def test_an_unpaused_response_carries_no_pause_marker(tmp_path, monkeypatch):
    s = _session(tmp_path, model="claude-3", api="anthropic", stream=False)
    model = ModelClient(s)
    factory = _AnthropicMockClientFactory(
        [
            (
                200,
                {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3",
                    "content": [{"type": "text", "text": "done"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        ]
    )
    monkeypatch.setattr(model, "anthropic_client", factory)

    assistant, _, _ = model.request_sync([{"role": "user", "content": "hi"}], [])

    assert PAUSED_TURN_KEY not in assistant

def test_agent_continues_a_paused_turn_instead_of_answering(tmp_path):
    """A pause carries no tool call of ours, so without this the turn would end early."""
    s = agent_session(tmp_path)
    s.skills = SkillLibrary({})
    agent = Agent(s, output_fn=lambda text: None)

    class PausingModel:
        def __init__(self):
            self.requests = []

        async def request(self, messages, tools=None):
            self.requests.append(messages)
            if len(self.requests) == 1:
                return {"role": "assistant", "content": None, PAUSED_TURN_KEY: True}, [], ""
            return {"role": "assistant", "content": "found it"}, [], "found it"

    agent.model = PausingModel()

    assert agent.run_sync("look it up") == "found it"
    assert len(agent.model.requests) == 2
    # The paused message is part of the conversation the second request sends back.
    assert agent.model.requests[1][-1].get(PAUSED_TURN_KEY) is True
    assert s.messages[-1]["content"] == "found it"

def test_a_paused_turn_is_bounded_by_max_steps(tmp_path):
    """A provider that never stops pausing must still end the turn."""
    s = agent_session(tmp_path)
    s.skills = SkillLibrary({})
    s.settings.max_steps = 3
    agent = Agent(s, output_fn=lambda text: None)

    class AlwaysPausing:
        def __init__(self):
            self.count = 0

        async def request(self, messages, tools=None):
            self.count += 1
            return {"role": "assistant", "content": None, PAUSED_TURN_KEY: True}, [], ""

    agent.model = AlwaysPausing()

    assert "Stopped after max_agent_steps=3" in agent.run_sync("look it up")
    assert agent.model.count == 3

def test_builtin_function_names_are_collected_from_config():
    provider = ProviderConfig.from_dict({"builtin_tools": [{"type": "web_search"}, {"type": "builtin_function", "function": {"name": "$web_search"}}]})

    assert provider.builtin_function_names() == ("$web_search",)

def test_a_declared_builtin_function_call_is_answered_with_its_arguments(tmp_path):
    """Kimi runs the search itself; the documented client side is to echo the arguments back."""
    s = agent_session(tmp_path)
    s.config.providers["default"].builtin_tools = ({"type": "builtin_function", "function": {"name": "$web_search"}},)
    logged = []
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda *a: "", output_fn=logged.append)

    messages = runner.run_sync([ToolCall("c1", "$web_search", [{"search_query": "httpx timeout"}])])

    assert messages == [{"role": "tool", "tool_call_id": "c1", "name": "$web_search", "content": '{"search_query": "httpx timeout"}'}]
    # No confirmation was asked for, and nothing was stored as a recallable result.
    assert s.tool_records == []
    assert logged and "Web Search" in str(logged[0])

def test_an_undeclared_builtin_function_call_is_still_an_unknown_tool(tmp_path):
    """The echo path is opened by config alone; it must not swallow arbitrary unknown names."""
    s = agent_session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda *a: "", output_fn=lambda text: None)

    messages = runner.run_sync([ToolCall("c1", "$web_search", [{"search_query": "x"}])])

    assert "unknown tool $web_search" in messages[0]["content"]

def test_a_batch_mixing_an_echo_and_a_real_tool_runs_both(tmp_path):
    s = agent_session(tmp_path)
    s.config.providers["default"].builtin_tools = ({"type": "builtin_function", "function": {"name": "$web_search"}},)
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda *a: "", output_fn=lambda text: None)

    messages = runner.run_sync([ToolCall("c1", "$web_search", [{"search_query": "q"}]), ToolCall("c2", "Read", [{"path": "a.txt", "ranges": [[0, 1]]}])])

    assert [message["tool_call_id"] for message in messages] == ["c1", "c2"]
    assert messages[0]["content"] == '{"search_query": "q"}'
    assert "<Read" in messages[1]["content"]
