"""Full-flow tests: the real agent loop driving the real ModelClient over a mocked wire.

Existing tests cover the two halves separately — `test_model_client.py` exercises ModelClient
against `httpx.MockTransport`, and `test_agent_logic.py` runs the agent loop against a hand-scripted
Python fake injected at `agent.model`. Neither crosses the seam between them: how the agent's
messages and tool schemas serialize onto the wire, and how a provider's response parses back into
tool calls that the runner then executes. These tests close that seam by pointing the real
ModelClient at a scripted in-process LLM (no sockets, no ports) and running `agent.run` end to end.
"""

import json

import httpx
from openai import OpenAI

import minacode as n


class ScriptedLLM:
    """A stand-in provider: each call to the client pops the next scripted chat-completion response,
    and every request body the SDK serializes is recorded so tests can assert on the wire format."""

    def __init__(self, responses: list[tuple[int, dict]]):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content.decode("utf-8")))
        status, body = self.responses.pop(0)
        return httpx.Response(status, json=body)

    def client(self) -> OpenAI:
        transport = httpx.MockTransport(self._handle)
        return OpenAI(api_key="sk-test", base_url="http://test", http_client=httpx.Client(transport=transport), max_retries=0)


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


def _session(tmp_path):
    config = n.Config()
    config.data_dir = str(tmp_path / "data")
    config.providers = {"default": n.ProviderConfig(url="http://test", key="sk-test", model="gpt-4")}
    session = n.Session(cwd=str(tmp_path), config=config)
    session.settings.yolo = True  # auto-approve mutating tools so the flow runs unattended
    session.skills = n.SkillLibrary({})  # no skills: keep the system frame deterministic
    return session


def test_full_flow_edit_then_answer(tmp_path, monkeypatch):
    """The model emits an Edit tool call, the runner applies it to disk, and the tool result rides
    back to the model on the next request before the final answer — the whole loop over the wire."""
    session = _session(tmp_path)
    edit_args = {"path": "hello.txt", "edits": [{"op": "create", "content": "hi\n"}]}
    llm = ScriptedLLM([_tool_call_response("call_1", "Edit", edit_args), _answer_response("Created hello.txt.")])
    monkeypatch.setattr(n.ModelClient, "client", lambda self: llm.client())

    answer = n.Agent(session, output_fn=lambda text: None).run("create hello.txt containing hi")

    # The tool really ran: the file exists on disk and the run returned the model's final answer.
    assert answer == "Created hello.txt."
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hi\n"
    assert [record.name for record in session.tool_records] == ["Edit"]

    # The wire round-trip: two model calls. The first carries the tool schemas; the second carries
    # the assistant's tool_calls and the tool result serialized back as a `tool` message.
    assert len(llm.requests) == 2
    assert llm.requests[0]["tools"]
    assert any(tool["function"]["name"] == "Edit" for tool in llm.requests[0]["tools"])

    second = llm.requests[1]["messages"]
    assistant_calls = [m for m in second if m["role"] == "assistant" and m.get("tool_calls")]
    assert assistant_calls and assistant_calls[0]["tool_calls"][0]["function"]["name"] == "Edit"
    tool_messages = [m for m in second if m["role"] == "tool"]
    assert tool_messages and tool_messages[0]["tool_call_id"] == "call_1"
    assert "<Edit" in tool_messages[0]["content"]


def test_full_flow_compacts_before_answering(tmp_path, monkeypatch):
    """An over-budget request first crosses the compactor wire, then the normal agent wire with
    the rebuilt context: history index, compacted conversation, task memory, and current turn."""
    session = _session(tmp_path)
    old_request = "archive request " + "x" * 200 + " OLD_BODY_SENTINEL " + "x" * 8000
    session.messages = [
        {"role": "user", "content": old_request},
        {"role": "assistant", "content": "archived answer " + "y" * 8000},
        {"role": "user", "content": "latest retained request"},
        {"role": "assistant", "content": "latest retained answer"},
    ]
    baseline = _session(tmp_path / "baseline")
    baseline_context = n.ContextManager(baseline)
    baseline_messages = baseline_context.model_messages(n.SYSTEM_PROMPT, [{"role": "user", "content": "continue"}])
    baseline_tokens = baseline_context.request_tokens(baseline_messages, n.Tool.resolved_schemas(baseline))
    session.settings.max_context_tokens = baseline_tokens + 500 + session.config.provider.output_token_budget() + n.MIN_CONTEXT_SAFETY_TOKENS

    compacted_state = json.dumps({"summary": "Archived work was completed.", "goal": "continue", "plan": [], "known": ["durable fact"], "check": "tests"})
    llm = ScriptedLLM([_answer_response(compacted_state), _answer_response("Continued successfully.")])
    monkeypatch.setattr(n.ModelClient, "client", lambda self: llm.client())

    answer = n.Agent(session, output_fn=lambda text: None).run("continue")

    assert answer == "Continued successfully."
    assert len(llm.requests) == 2

    compactor_request, agent_request = llm.requests
    assert "Compact the minacode working context." in compactor_request["messages"][0]["content"]
    assert "tools" not in compactor_request
    assert "OLD_BODY_SENTINEL" in compactor_request["messages"][1]["content"]

    active_messages = agent_request["messages"]
    contents = [str(message.get("content") or "") for message in active_messages]
    history_index = next(index for index, content in enumerate(contents) if content.startswith("--- History index ---"))
    conversation = next(index for index, content in enumerate(contents) if content.startswith(n.COMPACTION_SUMMARY_TITLE))
    memory = next(index for index, content in enumerate(contents) if content.startswith("--- Memory ---"))
    current_turn = max(index for index, content in enumerate(contents) if content == "continue")
    assert history_index < conversation < memory < current_turn
    assert "OLD_BODY_SENTINEL" not in "\n".join(contents)
    assert agent_request["tools"]

    assert session.state.summary == "Archived work was completed."
    assert session.state.compaction_count == 1
    assert [segment.key for segment in session.history] == ["seg.1"]
    assert "OLD_BODY_SENTINEL" in session.history[0].text
