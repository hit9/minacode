"""agent replay (split from tests/test_agent_turn.py)."""
import json
import threading
import time
from types import SimpleNamespace

from agent_harness import call, queue, session
from test_agent_turn import _runner

from minacode.base import (
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
from minacode.runner import ToolRunner
from minacode.session import Session
from minacode.tools import BashTool, ReadTool


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
    failed_result = next(message for message in s.transcript_messages if message.get("role") == "tool")
    assert failed_result == {"role": "tool", "tool_call_id": "Bash-id", "result_key": "", "status": "failed"}

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
    assert "temperature" not in params
    assert params["extra_body"]["temperature"] == 0.2
    assert params["max_tokens"] == ANTHROPIC_DEFAULT_MAX_TOKENS
    # An unversioned gateway alias remains generic rather than guessing a thinking generation.
    assert "thinking" not in params
    assert params["messages"][0] == {"role": "user", "content": [{"type": "text", "text": "first\n\nsecond"}]}
    assert params["messages"][1]["content"][1]["type"] == "tool_use"
    # The last block of the conversation carries the rolling breakpoint, so the history itself --
    # not just tools+system -- is written to the cache and read back on the next turn.
    assert params["messages"][2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "tc.1",
        "content": "tool output",
        "cache_control": {"type": "ephemeral"},
    }
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
