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

import nanocode as n


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
