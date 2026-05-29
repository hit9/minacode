import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

import nanocode as n


def session(tmp_path):
    return n.Session(cwd=str(tmp_path))


def call(name, args):
    return n.ToolCall(name + "-id", name, args)


def test_model_messages_are_two_message_context_snapshots(tmp_path):
    s = session(tmp_path)
    s.messages.extend([{"role": "user", "content": "old request"}, {"role": "assistant", "content": "old answer"}])
    turn = [
        {"role": "user", "content": "current request"},
        {"role": "user", "content": "extra one"},
        {"role": "user", "content": "extra two"},
    ]
    messages = n.ContextManager(s).model_messages(" system ", turn)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == "system"

    content = messages[1]["content"]
    assert content.startswith("--- Environment ---")
    assert "- cwd: " + str(tmp_path) in content
    sections = [
        "Environment",
        "State",
        "Summary",
        "Recent Conversation",
        "Tool Result Index",
        "File Context",
        "Discovery Context",
        "Error Feedback",
        "Latest Tool Results",
        "Current Turn Conversation",
    ]
    positions = [content.index(f"--- {section} ---") for section in sections]
    assert positions == sorted(positions)
    assert content.rfind("current request") > positions[-1]
    assert "user:\ncurrent request" in content
    assert "user:\nextra one" in content
    assert "user:\nextra two" in content


def test_environment_uses_cached_system_info(tmp_path, monkeypatch):
    calls = []

    def fake_which(name):
        calls.append(name)
        return "/bin/" + name if name in {"bash", "rg", "sed"} else None

    monkeypatch.setattr(n.platform, "system", lambda: "TestOS")
    monkeypatch.setattr(n.platform, "machine", lambda: "test-arch")
    monkeypatch.setattr(n.shutil, "which", fake_which)

    s = session(tmp_path)
    initial_calls = list(calls)
    context = n.ContextManager(s)
    first = context.environment()
    second = context.render([{"role": "user", "content": "request"}])

    assert calls == initial_calls
    assert "- cwd: " + str(tmp_path) in first
    assert "- os: TestOS" in first
    assert "- arch: test-arch" in first
    assert "- detected_commands: bash, rg, sed" in first
    assert "- detected_commands: bash, rg, sed" in second


def test_session_tool_result_store_prunes_and_forget_removes_records(tmp_path):
    s = session(tmp_path)
    for index in range(405):
        s.store_tool_result("Bash", [str(index)], f"output {index}")

    assert len(s.tool_results) == 400
    assert len(s.tool_records) == 400
    assert "tr.1" not in s.tool_results
    assert s.tool_records[0].key == "tr.6"

    removed = s.forget_tool_results(["tr.6", "tr.405", "tr.999"])
    assert removed == 2
    assert "tr.6" not in s.tool_results
    assert "tr.405" not in s.tool_results
    assert all(record.key not in {"tr.6", "tr.405"} for record in s.tool_records)


def test_latest_results_and_bounded_output_are_context_managed(tmp_path):
    s = session(tmp_path)
    context = n.ContextManager(s)
    previous = s.store_tool_result("Search", [], "previous result")

    context.start_tool_batch()
    current = context.store_tool_result(call("Read", [{"path": "a.txt", "ranges": [[0, 1]]}]), "current result")
    latest = context.latest_results()
    assert previous not in latest
    assert current in latest
    assert "current result" in latest

    large = "head\n" + "\n".join(f"line {index}" for index in range(20000)) + "\ntail\n"
    bounded = context.bound_output(large, "tr.large")
    assert "head" in bounded
    assert "tail" in bounded
    assert "<bounded_output" in bounded
    assert 'recall="tr.large"' in bounded


def test_file_context_tracks_edits_and_omits_stale_reads(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("old\nkeep\n", encoding="utf-8")
    s = session(tmp_path)
    context = n.ContextManager(s)

    read_output = n.ReadTool(s, [{"path": "a.txt", "ranges": [[0, 2]]}]).call()
    read_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 2]]}], read_output)
    assert "|old" in context.file_context()

    path.write_text("changed\nkeep\n", encoding="utf-8")
    stale = context.file_context()
    assert "|old" not in stale
    assert read_key in stale

    path.write_text("old\nkeep\n", encoding="utf-8")
    edit_output = n.EditTool(
        s,
        ["a.txt", [{"op": "replace", "start": "0:" + n.ReadTool.line_hash("old\n"), "end": "0:" + n.ReadTool.line_hash("old\n"), "content": "new\n"}]],
    ).call()
    edit_key = s.store_tool_result("Edit", ["a.txt"], edit_output)

    rendered = context.file_context()
    assert edit_key in rendered
    assert "|new" in rendered
    assert "|old" not in rendered


def test_discovery_and_error_context_are_separate_from_file_context(tmp_path):
    s = session(tmp_path)
    context = n.ContextManager(s)
    s.store_tool_result("List", ["."], "list output")
    search_key = s.store_tool_result("Search", [{"pattern": "x"}], "search output")
    inspect_key = s.store_tool_result("InspectCode", ["find", "X"], "inspect output")
    s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 1]]}], "read output")

    discovery = context.discovery_context()
    assert search_key in discovery
    assert inspect_key in discovery
    assert "search output" in discovery
    assert "inspect output" in discovery
    assert "list output" not in discovery
    assert "read output" not in discovery

    for index in range(7):
        s.record_tool_error(f"tr.{index}", "Bash", [str(index)], f"error {index}")
    assert [record.key for record in s.tool_errors] == ["tr.2", "tr.3", "tr.4", "tr.5", "tr.6"]
    feedback = context.error_feedback()
    assert "error 1" not in feedback
    assert "error 6" in feedback


def test_compaction_uses_configured_context_budget(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    s.messages = [{"role": "user", "content": str(index)} for index in range(10)]
    context = n.ContextManager(s)
    context.latest_keys = ["tr.1"]

    class FakeModel:
        def __init__(self):
            self.input = None

        def compact(self, text):
            self.input = text
            return {"summary": "compact summary", "plan": ["next"], "known": ["fact"]}

    model = FakeModel()
    context.maybe_compact(model, "system", [{"role": "user", "content": "request"}])
    assert model.input is not None
    assert s.state.summary == "compact summary"
    assert s.state.plan == ["next"]
    assert s.state.known == ["fact"]
    assert len(s.messages) == 6
    assert context.latest_keys == []


def test_tool_runner_refusal_stops_batch_and_invalid_args_are_not_stored(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: "skip it", output_fn=lambda text: None)
    runner.run([call("Bash", ["printf first"]), call("CreateFile", ["second.txt", "second"])])

    assert len(s.tool_records) == 1
    assert len(s.tool_errors) == 1
    assert "skip it" in s.tool_errors[0].error
    assert not (tmp_path / "second.txt").exists()

    outputs = []
    bad = session(tmp_path)
    n.ToolRunner(bad, n.ContextManager(bad), output_fn=outputs.append).run([call("Bash", [])])
    assert bad.tool_records == []
    assert len(bad.tool_errors) == 1
    assert outputs and "[failed]" in outputs[0]


def test_tool_runner_refuses_without_reason_on_n(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: "n", output_fn=lambda text: None)

    runner.run([call("Bash", ["printf first"])])

    assert s.tool_errors[0].error == "Cancelled: user refused tool call"


def test_tool_runner_refuses_with_direct_reason_input(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: "not now", output_fn=lambda text: None)

    runner.run([call("Bash", ["printf first"])])

    assert len(s.tool_records) == 1
    assert len(s.tool_errors) == 1
    assert "not now" in s.tool_errors[0].error


def test_recall_tool_runner_does_not_create_new_result_keys(tmp_path):
    s = session(tmp_path)
    key = s.store_tool_result("Read", ["a.txt"], "result")
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    runner.run([call("Recall", [key])])
    assert [record.key for record in s.tool_records] == [key]


def test_agent_runs_tool_loop_and_stops_at_max_steps(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    s = session(tmp_path)
    agent = n.Agent(s, output_fn=lambda text: None)

    class FakeModel:
        def __init__(self):
            self.messages = []

        def request(self, messages):
            self.messages.append(messages)
            if len(self.messages) == 1:
                return {}, [call("Read", [{"path": "a.txt", "ranges": [[0, 1]]}])], ""
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = FakeModel()
    assert agent.run("read file") == "done"
    assert len(agent.model.messages) == 2
    assert all(len(messages) == 2 for messages in agent.model.messages)
    assert len(s.tool_records) == 1
    assert s.messages[-1]["content"] == "done"

    limited = session(tmp_path)
    limited.settings.max_steps = 2
    limited_agent = n.Agent(limited, output_fn=lambda text: None)

    class LoopingModel:
        def request(self, messages):
            return {}, [call("LineCount", ["a.txt"])], ""

    limited_agent.model = LoopingModel()
    answer = limited_agent.run("keep going")
    assert limited.state.turn_step == 2
    assert len(limited.tool_records) == 2
    assert limited.messages[-1]["content"] == answer


def test_agent_rejects_empty_final_response(tmp_path):
    agent = n.Agent(session(tmp_path), output_fn=lambda text: None)

    class EmptyModel:
        def request(self, messages):
            return {"role": "assistant", "content": ""}, [], ""

    agent.model = EmptyModel()
    with pytest.raises(n.ModelError, match="empty final response"):
        agent.run("answer me")


def test_agent_injects_pending_user_input_once(tmp_path):
    s = session(tmp_path)
    s.pending_user_inputs.append("extra instruction")
    agent = n.Agent(s, output_fn=lambda text: None)

    class FakeModel:
        def __init__(self):
            self.messages = []

        def request(self, messages):
            self.messages.append(messages)
            if len(self.messages) == 1:
                s.pending_user_inputs.append("second instruction")
                return {}, [call("LineCount", ["missing.txt"])], "checking"
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = FakeModel()
    assert agent.run("initial request") == "done"

    first = agent.model.messages[0][1]["content"]
    second = agent.model.messages[1][1]["content"]
    assert "user:\nextra instruction" in first
    assert "user:\nextra instruction" in second
    assert "assistant:\n" in second
    assert "user:\nsecond instruction" in second
    assert s.messages[0]["content"] == "initial request"
    assert s.messages[1]["content"] == "extra instruction"
    assert s.messages[2]["content"] == "checking"
    assert s.messages[3]["content"] == "second instruction"
    assert s.messages[4]["role"] == "assistant"
    assert s.pending_user_inputs == []


def test_queued_input_pauses_before_reading_stdin(tmp_path, monkeypatch):
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, encoding="utf-8")
    writer = os.fdopen(write_fd, "w", encoding="utf-8")
    monkeypatch.setattr(n.sys, "stdin", reader)
    s = session(tmp_path)
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    stop = threading.Event()
    loop.queue_input_paused.set()
    thread = threading.Thread(target=loop.queue_input_until, args=(stop,), daemon=True)
    thread.start()
    try:
        writer.write("later\n")
        writer.flush()
        time.sleep(0.2)
        assert s.pending_user_inputs == []
        loop.queue_input_paused.clear()
        deadline = time.monotonic() + 1
        while not s.pending_user_inputs and time.monotonic() < deadline:
            time.sleep(0.02)
        assert s.pending_user_inputs == ["later"]
    finally:
        stop.set()
        writer.close()
        reader.close()


def test_tool_input_uses_multiline_approval(tmp_path, monkeypatch):
    s = session(tmp_path)
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), output_fn=lambda text: None)
    calls = []

    def fake_read(prompt, *, multiline=False, submit_on_enter=False, prompt_style="class:prompt"):
        calls.append((prompt, multiline, submit_on_enter, prompt_style))
        return ""

    loop.interactive_input = True
    monkeypatch.setattr(n.sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr(loop, "read_input", fake_read)

    loop.tool_input("[Y/n or reason] ")

    assert calls == [("[Y/n or reason] ", True, True, "class:approval")]


def test_bash_live_start_pauses_queue_before_app_is_active(tmp_path):
    loop = n.CommandLoop(n.Agent(session(tmp_path), output_fn=lambda text: None), output_fn=lambda text: None)
    loop.ui.color = True
    loop.interactive_input = True
    loop.live_preview.start = lambda: setattr(loop.live_preview, "active", True)

    loop.tool_live_start()
    assert loop.queue_input_paused.is_set()
    assert loop.live_queue_paused is True

    loop.tool_live_output("", "")
    assert not loop.queue_input_paused.is_set()
    assert loop.live_queue_paused is False


def test_agent_emits_and_records_intermediate_content_before_tools(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    s = session(tmp_path)
    output = []
    agent = n.Agent(s, output_fn=output.append)

    class TalkingModel:
        def __init__(self):
            self.messages = []

        def request(self, messages):
            self.messages.append(messages)
            if len(self.messages) == 1:
                return {}, [call("Read", [{"path": "a.txt", "ranges": [[0, 1]]}])], "I'll inspect that first."
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = TalkingModel()
    assert agent.run("read file") == "done"
    assert output[0] == "I'll inspect that first."
    assert any(line.startswith("tool Read") for line in output)
    assert s.messages == [
        {"role": "user", "content": "read file"},
        {"role": "assistant", "content": "I'll inspect that first."},
        {"role": "assistant", "content": "done"},
    ]
    assert "I'll inspect that first." in agent.model.messages[1][1]["content"]


def test_compaction_fallback_trims_when_model_compact_fails(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    s.state.summary = "existing"
    s.messages = [{"role": "user", "content": str(index)} for index in range(10)]
    context = n.ContextManager(s)
    context.latest_keys = ["tr.1"]

    class FailingModel:
        def compact(self, text):
            raise n.ModelError("failed")

    context.maybe_compact(FailingModel(), "system", [{"role": "user", "content": "request"}])
    assert s.state.summary != "existing"
    assert len(s.messages) == 6
    assert context.latest_keys == []


def test_manual_compact_clears_conversation_messages(tmp_path):
    s = session(tmp_path)
    s.messages = [{"role": "user", "content": str(index)} for index in range(5)]
    s.state.context_percent = 80
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), output_fn=lambda text: None)

    class FakeModel:
        def compact(self, text):
            return {"summary": "summary", "plan": ["next"], "known": ["fact"]}

    loop.agent.model = FakeModel()
    result = loop.compact("")

    assert s.messages == []
    assert s.state.summary == "summary"
    assert "messages 5 -> 0" in result
    assert "summary updated" in result


def test_agent_tool_error_feedback_is_visible_on_next_model_request(tmp_path):
    s = session(tmp_path)
    agent = n.Agent(s, output_fn=lambda text: None)

    class FeedbackModel:
        def __init__(self):
            self.messages = []

        def request(self, messages):
            self.messages.append(messages)
            if len(self.messages) == 1:
                return {}, [call("Bash", [])], ""
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = FeedbackModel()
    assert agent.run("run bad tool") == "done"
    assert len(s.tool_errors) == 1
    assert s.tool_records == []
    second_context = agent.model.messages[1][1]["content"]
    assert "--- Error Feedback ---" in second_context
    assert "Bash" in second_context


def test_provider_profiles_and_prompt_cache_key(tmp_path):
    opencode_claude = n.ProviderConfig(url="https://opencode.ai/zen/go/v1", key="k", model="claude-sonnet", api="auto")
    assert opencode_claude.resolved_api() == "anthropic"

    opencode_deepseek = n.ProviderConfig(url="https://opencode.ai/zen/go/v1", key="k", model="deepseek-v4-flash", api="auto")
    assert opencode_deepseek.resolved_api() == "chat"
    assert opencode_deepseek.resolved_chat_reasoning() == "reasoning"

    provider = n.ProviderConfig(url="https://api.openai.com/v1", key="k", model="gpt-5-mini", prompt_cache_key="auto")
    s = n.Session(cwd=str(tmp_path), config=n.Config(active_provider="p", providers={"p": provider}))
    client = n.ModelClient(s)
    first = client.prompt_cache_key(provider, [n.BashTool.schema(), n.ReadTool.schema()])
    second = client.prompt_cache_key(provider, [n.ReadTool.schema(), n.BashTool.schema()])
    assert first == second
    assert first.startswith("nanocode-")

    provider.prompt_cache_key = "fixed-key"
    assert client.prompt_cache_key(provider, None) == "fixed-key"
    provider.prompt_cache_key = "off"
    assert client.prompt_cache_key(provider, None) == ""


def test_anthropic_message_conversion_and_tool_result_parsing(tmp_path):
    provider = n.ProviderConfig(url="https://api.anthropic.com/v1/messages", key="k", model="claude-sonnet", api="anthropic", reasoning="off", temperature=0.2)
    s = n.Session(cwd=str(tmp_path), config=n.Config(active_provider="p", providers={"p": provider}))
    client = n.ModelClient(s)
    arguments = json.dumps({"files": [{"path": "a.txt", "ranges": [[0, 1]]}]})
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "thinking", "tool_calls": [{"id": "tc.1", "function": {"name": "Read", "arguments": arguments}}]},
        {"role": "tool", "tool_call_id": "tc.1", "content": "tool output"},
    ]

    params = client.anthropic_params(messages, [n.ReadTool.schema()])
    assert params["system"] == "system"
    assert params["temperature"] == 0.2
    assert "thinking" not in params
    assert params["messages"][0] == {"role": "user", "content": "first\n\nsecond"}
    assert params["messages"][1]["content"][1]["type"] == "tool_use"
    assert params["messages"][2]["content"][0]["type"] == "tool_result"
    assert params["tools"][0]["name"] == "Read"
    assert params["tools"][0]["input_schema"]["additionalProperties"] is False

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
    assert calls == [n.ToolCall(id="tc.2", name="Bash", args=["pwd"])]
