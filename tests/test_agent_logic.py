import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

import nanocode as n


def session(tmp_path):
    # Isolate the data dir so tests never read the developer's real ~/.nanocode (sessions, skills).
    config = n.Config()
    config.data_dir = str(tmp_path / "data")
    return n.Session(cwd=str(tmp_path), config=config)


def call(name, args):
    return n.ToolCall(name + "-id", name, args)


def queue(s, *texts):
    for text in texts:
        s.enqueue_user_input(text)


def queued_texts(s):
    return [item.text for item in s.pending_user_inputs]


def test_model_messages_are_ordered_context_messages(tmp_path):
    s = session(tmp_path)
    s.skills = n.SkillLibrary({})  # no skills: assert the base frame ordering
    s.messages.extend([{"role": "user", "content": "old request"}, {"role": "assistant", "content": "old answer"}])
    turn = [
        {"role": "user", "content": "current request"},
        {"role": "user", "content": "extra one"},
        {"role": "user", "content": "extra two"},
    ]
    messages = n.ContextManager(s).model_messages(" system ", turn)

    assert [message["role"] for message in messages] == ["system", "user", "user", "assistant", "user", "user", "user", "user"]
    assert messages[0]["content"] == "system"
    assert messages[1]["content"].startswith("--- Environment ---")
    assert "- cwd: " + str(tmp_path) in messages[1]["content"]
    assert [message["content"] for message in messages[2:7]] == ["old request", "old answer", "current request", "extra one", "extra two"]
    assert messages[-1]["content"].startswith("--- Memory ---")
    assert "Date:" in messages[-1]["content"]
    assert not any("FILE STATE" in message["content"] for message in messages)


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
    second = context.model_messages("sys", [{"role": "user", "content": "request"}])[1]["content"]

    assert calls == initial_calls
    assert "- cwd: " + str(tmp_path) in first
    assert "- os: TestOS" in first
    assert "- arch: test-arch" in first
    assert "- detected_commands (available via Bash): bash, rg, sed" in first
    assert "- detected_commands (available via Bash): bash, rg, sed" in second


def test_session_tool_result_store_prunes_old_records(tmp_path):
    s = session(tmp_path)
    for index in range(405):
        s.store_tool_result("Bash", [str(index)], f"output {index}")

    assert len(s.tool_results) == 400
    assert len(s.tool_records) == 400
    assert "tr.1" not in s.tool_results
    assert s.tool_records[0].key == "tr.6"
    assert "tr.405" in s.tool_results


def test_bounded_output_marks_recall_key(tmp_path):
    s = session(tmp_path)
    context = n.ContextManager(s)
    large = "head\n" + "\n".join(f"line {index}" for index in range(20000)) + "\ntail\n"
    bounded = context.bound_output(large, "tr.large")

    assert "head" in bounded
    assert "tail" in bounded
    assert "<bounded_output" in bounded
    assert 'recall="tr.large"' in bounded


def test_read_tool_message_inlines_bounded_output(tmp_path):
    path = tmp_path / "large.txt"
    path.write_text("first\n" + "\n".join(f"middle-{index}" for index in range(20000)) + "\nlast\n", encoding="utf-8")
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)
    call_obj = call("Read", [{"path": "large.txt", "ranges": [[0, 0]]}])
    output = n.ReadTool(s, call_obj.args).call()
    key = s.store_tool_result("Read", call_obj.args, output)

    message = runner.tool_message(call_obj, key, output)

    assert message.startswith("tool tr.1 Read large.txt 0:0\noutput:\n")
    assert "<Read" in message
    assert "<bounded_output" in message
    assert 'recall="tr.1"' in message
    assert "-> FILE STATE" not in message


def test_tool_error_records_keep_recent_failures(tmp_path):
    s = session(tmp_path)
    for index in range(7):
        s.record_tool_error(f"tr.{index}", "Bash", [str(index)], f"error {index}")

    assert [record.key for record in s.tool_errors] == ["tr.2", "tr.3", "tr.4", "tr.5", "tr.6"]


def test_working_context_includes_recent_tool_errors(tmp_path):
    s = session(tmp_path)
    for index in range(6):
        s.record_tool_error(f"tr.{index}", "Bash", [f"cmd {index}"], f"error {index}")

    context = n.ContextManager(s).model_messages("sys")[-1]["content"]

    assert "Recent tool errors:" in context
    assert "tr.0" not in context
    assert "tr.5 Bash cmd 5" in context
    assert "error 5" in context


def test_compaction_uses_configured_context_budget(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    s.messages = [{"role": "user", "content": "old user"}, {"role": "assistant", "content": "old answer"}, *({"role": "assistant", "content": f"recent {index}"} for index in range(8)), {"role": "user", "content": "latest"}, {"role": "tool", "content": "tool kept"}]
    context = n.ContextManager(s)

    class FakeModel:
        def __init__(self):
            self.input = None

        def compact(self, text):
            self.input = text
            return {"summary": "compact summary", "plan": ["next"], "known": ["fact"]}

    model = FakeModel()
    context.prepare_messages(model, "system", [{"role": "user", "content": "request"}])
    assert model.input is not None
    assert "Older Messages:" in model.input
    assert "old answer" in model.input
    assert "Recent Messages (rewrite briefly inside summary):" in model.input
    assert "recent 7" in model.input
    assert "latest" not in model.input
    assert "request" not in model.input
    assert s.state.summary == "compact summary"
    assert [item.to_json() for item in s.state.plan] == [{"status": "todo", "text": "next"}]
    assert s.state.known == ["fact"]
    assert [message["role"] for message in s.messages] == ["user", "user", "tool"]
    assert s.messages[0]["content"].startswith(n.ContextManager.COMPACT_TITLE)
    assert "compact summary" in s.messages[0]["content"]
    assert s.messages[1]["content"] == "latest"
    assert s.messages[2]["content"] == "tool kept"
    assert all("recent 7" not in str(message.get("content") or "") for message in s.messages)


def test_compaction_parts_keep_latest_user_turn_after_prior_summary(tmp_path):
    s = session(tmp_path)
    summary = n.ContextManager.COMPACT_TITLE + "\nold summary"
    s.messages = [
        {"role": "user", "content": summary},
        {"role": "assistant", "content": "before"},
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "latest request"},
        {"role": "assistant", "content": "working"},
        {"role": "tool", "content": "tool tr.1"},
    ]

    compacted, keep = n.ContextManager(s).compaction_parts()

    assert [message["content"] for message in compacted] == [summary, "before", "old request", "old answer"]
    assert [message["content"] for message in keep] == ["latest request", "working", "tool tr.1"]


def test_compaction_parts_compact_all_without_plain_user_message(tmp_path):
    s = session(tmp_path)
    s.messages = [
        {"role": "user", "content": n.ContextManager.COMPACT_TITLE + "\nold summary"},
        {"role": "assistant", "content": "answer"},
        {"role": "tool", "content": "tool tr.1"},
    ]

    compacted, keep = n.ContextManager(s).compaction_parts()

    assert compacted == s.messages
    assert keep == []


def test_compaction_parts_for_uses_last_fixed_window(tmp_path):
    messages = [{"role": "assistant", "content": f"m{index}"} for index in range(10)]

    older, recent = n.ContextManager(session(tmp_path)).compaction_parts_for(messages)

    assert [message["content"] for message in older] == ["m0", "m1"]
    assert [message["content"] for message in recent] == [f"m{index}" for index in range(2, 10)]


def test_prepare_messages_skips_compaction_when_context_under_budget(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 999_999
    s.messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "answer"}]

    class ExplodingModel:
        def compact(self, text):
            raise AssertionError(text)

    n.ContextManager(s).prepare_messages(ExplodingModel(), "system", [{"role": "user", "content": "request"}])

    assert s.messages == [{"role": "user", "content": "old"}, {"role": "assistant", "content": "answer"}]


def test_prepare_messages_builds_under_budget_context_once(tmp_path, monkeypatch):
    context = n.ContextManager(session(tmp_path))
    calls = 0
    original = context.model_messages

    def model_messages(base_system, turn_messages=None):
        nonlocal calls
        calls += 1
        return original(base_system, turn_messages)

    monkeypatch.setattr(context, "model_messages", model_messages)
    context.prepare_messages(object(), "system", [{"role": "user", "content": "request"}])

    assert calls == 1


def test_compaction_keeps_assistant_with_tool_results(tmp_path):
    context = n.ContextManager(session(tmp_path))
    messages = [
        *({"role": "user", "content": f"old {index}"} for index in range(3)),
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc.1", "type": "function", "function": {"name": "Read", "arguments": "{}"}},
                {"id": "tc.2", "type": "function", "function": {"name": "Read", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "tc.1", "content": "one"},
        {"role": "tool", "tool_call_id": "tc.2", "content": "two"},
        *({"role": "user", "content": f"recent {index}"} for index in range(6)),
    ]

    compacted, keep = context.compaction_parts_for(messages)

    assert compacted == messages[:3]
    assert keep == messages[3:]
    assert keep[0]["role"] == "assistant"
    assert [message["role"] for message in keep[1:3]] == ["tool", "tool"]


def test_compaction_keeps_tool_records_referenced_from_summary(tmp_path):
    s = session(tmp_path)
    context = n.ContextManager(s)
    kept = s.store_tool_result("Bash", ["kept"], "kept output")
    dropped = s.store_tool_result("Bash", ["dropped"], "dropped output")

    context.apply_compaction({"summary": f"Continue from {kept}."}, [])

    assert kept in s.tool_results
    assert dropped not in s.tool_results
    assert [record.key for record in s.tool_records] == [kept]


def test_compaction_prunes_unreferenced_tool_records(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("one\n", encoding="utf-8")
    s = session(tmp_path)
    context = n.ContextManager(s)
    old_key = s.store_tool_result("Bash", ["old"], "old output")
    read_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 0]]}], n.ReadTool(s, [{"path": "a.txt", "ranges": [[0, 0]]}]).call())
    current_key = s.store_tool_result("Bash", ["current"], "current output")

    context.apply_compaction({"summary": "summary"}, [{"role": "tool", "content": f"tool {current_key} Bash current"}])

    assert old_key not in s.tool_results
    assert read_key not in s.tool_results
    assert {record.key for record in s.tool_records} == {current_key}
    assert set(s.tool_results) == {current_key}


def test_compaction_keeps_current_turn_tool_records(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    s.messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "old answer"}, {"role": "user", "content": "latest"}]
    old_key = s.store_tool_result("Bash", ["old"], "old output")
    current_key = s.store_tool_result("Bash", ["current"], "current output")

    class FakeModel:
        def compact(self, text):
            return {"summary": "summary"}

    n.ContextManager(s).prepare_messages(FakeModel(), "system", [{"role": "tool", "content": f"tool {current_key} Bash current"}])

    assert old_key not in s.tool_results
    assert current_key in s.tool_results
    assert [record.key for record in s.tool_records] == [current_key]


def test_compaction_drops_unreferenced_read_edit_records(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    s = session(tmp_path)
    context = n.ContextManager(s)
    read_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 0]]}], n.ReadTool(s, [{"path": "a.txt", "ranges": [[0, 0]]}]).call())
    edit_key = s.store_tool_result(
        "Edit",
        ["a.txt"],
        n.EditTool(s, ["a.txt", [{"op": "delete", "start": "1:" + n.ReadTool.line_hash("b\n"), "end": "1:" + n.ReadTool.line_hash("b\n")}]]).call(),
    )

    context.apply_compaction({"summary": "summary"}, [])

    assert read_key not in s.tool_results
    assert edit_key not in s.tool_results
    assert s.tool_records == []


def test_tool_runner_refusal_stops_batch_and_invalid_args_are_not_stored(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: "skip it", output_fn=lambda text: None)
    runner.run([call("Bash", [":"]), call("Edit", ["second.txt", [{"op": "create", "content": "second"}]])])

    assert s.tool_records == []
    assert len(s.tool_errors) == 1
    assert "skip it" in s.tool_errors[0].error
    assert not (tmp_path / "second.txt").exists()

    outputs = []
    bad = session(tmp_path)
    n.ToolRunner(bad, n.ContextManager(bad), output_fn=lambda text: outputs.append(str(text))).run([call("Bash", [])])
    assert bad.tool_records == []
    assert len(bad.tool_errors) == 1
    assert outputs and "· rejected:" in outputs[0]  # argument errors collapse to a quiet line


def test_tool_runner_refuses_without_reason_on_n(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: "n", output_fn=lambda text: None)

    runner.run([call("Bash", [":"])])

    assert s.tool_errors[0].error == "Cancelled: user refused tool call"


def test_tool_runner_refuses_with_direct_reason_input(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: "not now", output_fn=lambda text: None)

    runner.run([call("Bash", [":"])])

    assert s.tool_records == []
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
    s.skills = n.SkillLibrary({})  # no skills: assert the base frame layout
    agent = n.Agent(s, output_fn=lambda text: None)

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
    assert [len(messages) for messages in agent.model.messages] == [4, 6]
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
    limited.skills = n.SkillLibrary({})
    limited.settings.max_steps = 2
    limited_agent = n.Agent(limited, output_fn=lambda text: None)

    class LoopingModel:
        def request(self, messages, tools=None):
            return {}, [call("Read", [{"path": "a.txt", "ranges": [[0, 0]]}])], ""

    limited_agent.model = LoopingModel()
    answer = limited_agent.run("keep going")
    assert limited.state.turn_step == 2
    assert len(limited.tool_records) == 2
    assert limited.messages[-1]["content"] == answer


def test_interrupted_turn_persists_completed_tool_batches_for_resume(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    s = session(tmp_path)
    s.skills = n.SkillLibrary({})
    agent = n.Agent(s, output_fn=lambda text: None)

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

    assert [message["role"] for message in s.messages] == ["user", "assistant", "tool"]
    assert s._active_turn_messages == []
    restored = n.Session.load_snapshot(s.uid, config=s.config, settings=s.settings)
    messages = [message for message in restored.messages if not n.SessionSnapshotCodec.is_internal_message(message)]
    assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
    assert messages[1]["tool_calls"][0]["id"] == "Read-id"
    assert messages[2]["tool_call_id"] == "Read-id"
    assert "<Read" in messages[2]["content"]
    assert [record.name for record in restored.tool_records] == ["Read"]


def test_agent_cancel_stops_after_active_tool_batch(tmp_path):
    agent = n.Agent(session(tmp_path), output_fn=lambda text: None)

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


def test_agent_rejects_empty_final_response(tmp_path):
    agent = n.Agent(session(tmp_path), output_fn=lambda text: None)

    class EmptyModel:
        def request(self, messages, tools=None):
            return {"role": "assistant", "content": ""}, [], ""

    agent.model = EmptyModel()
    with pytest.raises(n.ModelError, match="empty final response"):
        agent.run("answer me")


def test_model_cancel_closes_active_client_and_interrupts_request(tmp_path):
    s = session(tmp_path)
    s.config.provider.url = "https://example.test/v1"
    s.config.provider.key = "test"
    s.config.provider.model = "model"
    started = threading.Event()
    closed = threading.Event()

    class Completions:
        def create(self, **_params):
            started.set()
            closed.wait(timeout=1)
            raise RuntimeError("connection closed")

    class Client:
        chat = SimpleNamespace(completions=Completions())

        def close(self):
            closed.set()

    model = n.ModelClient(s)
    model.client = Client
    errors = []

    def request():
        try:
            model.request([{"role": "user", "content": "hello"}], [])
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=request)
    thread.start()
    assert started.wait(timeout=1)
    model.cancel()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], KeyboardInterrupt)


def test_agent_injects_pending_user_input_once(tmp_path):
    s = session(tmp_path)
    queue(s, "extra instruction")
    output = []
    agent = n.Agent(s, output_fn=output.append)

    class FakeModel:
        def __init__(self):
            self.messages = []

        def request(self, messages, tools=None):
            self.messages.append(messages)
            if len(self.messages) == 1:
                s.enqueue_user_input("second instruction")
                return {}, [call("Bash", ["wc -l missing.txt"])], "checking"
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = FakeModel()
    assert agent.run("initial request") == "done"

    first = "\n\n".join(message.get("content") or "" for message in agent.model.messages[0])
    second = "\n\n".join(message.get("content") or "" for message in agent.model.messages[1])
    first_followup = next(message["content"] for message in agent.model.messages[0] if "extra instruction" in (message.get("content") or ""))
    second_followup = next(message["content"] for message in agent.model.messages[1] if "second instruction" in (message.get("content") or ""))
    assert "[Live follow-up received while you were working]" in n.Agent.LIVE_FOLLOWUP_PREFIX
    assert "[Live follow-up received while you were working]" in n.Agent.SYSTEM_PROMPT
    assert "must include a brief visible text response" in n.Agent.LIVE_FOLLOWUP_PREFIX
    assert "never respond with tool calls only" in n.Agent.SYSTEM_PROMPT
    assert first_followup == n.Agent.LIVE_FOLLOWUP_PREFIX + "extra instruction"
    assert second_followup == n.Agent.LIVE_FOLLOWUP_PREFIX + "second instruction"
    assert "extra instruction" in first
    assert "extra instruction" in second
    assert "checking" in second
    assert "second instruction" in second
    assert s.messages[0]["content"] == "initial request"
    assert s.messages[1]["content"] == "extra instruction"
    assert s.messages[2]["content"] == "checking"
    assert s.messages[3]["role"] == "tool"
    assert s.messages[3]["content"].startswith("tool tr.1 Bash wc -l missing.txt")
    assert s.messages[4]["content"] == "second instruction"
    assert "checking" in output
    assert s.messages[5]["role"] == "assistant"
    assert s.pending_user_inputs == []


def test_agent_shares_resolved_tools_with_model_request(tmp_path, monkeypatch):
    s = session(tmp_path)
    agent = n.Agent(s, output_fn=lambda text: None)
    tools = [{"type": "function", "function": {"name": "Test", "parameters": {}}}]
    resolved = []

    def resolve(session):
        resolved.append(session)
        return tools

    class FakeModel:
        received_tools = None

        def request(self, messages, request_tools=None):
            self.received_tools = request_tools
            return {"role": "assistant", "content": "done"}, [], "done"

    monkeypatch.setattr(n, "resolved_tool_schemas", resolve)
    agent.model = FakeModel()

    assert agent.run("hello") == "done"
    assert resolved == [s]
    assert agent.model.received_tools is tools


def test_startup_tip_respects_context(tmp_path, monkeypatch):
    s = session(tmp_path)
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)

    # Force startup_tip to always pick the last candidate so we can check what's in the pool.
    monkeypatch.setattr(n.random, "choice", lambda seq: seq[-1])

    # No MCP: MCP tips are absent from the pool.
    tip = loop.startup_tip()
    assert "@server.tool" not in tip

    # Enabling MCP appends the MCP tip; with the seeded random.choice the last MCP tip wins.
    s.config.mcp["example"] = {"url": "http://x"}
    assert "`/mcp`" in loop.startup_tip()


def test_ps_command_uses_markdown_renderer(tmp_path):
    s = session(tmp_path)
    s.jobs["job.1"] = SimpleNamespace(id="job.1", status="running", command="pytest -q", elapsed=lambda: 13.7, update_status=lambda: None)
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    rendered = []
    plain = []
    loop.ui.emit_answer = rendered.append
    loop.emit = plain.append

    assert loop.command("/ps") == (True, False)

    assert plain == []
    assert len(rendered) == 1
    assert rendered[0].startswith("### Active jobs")
    assert "| id | status | elapsed | command |" in rendered[0]


def test_tui_completion_applies_single_match():
    class OneCompletion(n.Completer):
        def get_completions(self, document, _complete_event):
            yield n.Completion("hello", start_position=-len(document.text))

    buffer = n.Buffer(document=n.Document("he"), completer=OneCompletion())
    n.TuiApp.complete_input(buffer)
    assert buffer.text == "hello"


def test_tui_completion_starts_and_cycles_multiple_matches():
    class MultipleCompletions(n.Completer):
        def get_completions(self, document, _complete_event):
            yield n.Completion("alpha", start_position=-len(document.text))
            yield n.Completion("alpine", start_position=-len(document.text))

    completer = MultipleCompletions()
    buffer = n.Buffer(document=n.Document("al"), completer=completer)
    started = []
    buffer.start_completion = lambda **kwargs: started.append(kwargs)

    n.TuiApp.complete_input(buffer)
    assert started == [{"select_first": False}]

    completions = list(completer.get_completions(buffer.document, n.CompleteEvent()))
    buffer._set_completions(completions)
    n.TuiApp.complete_input(buffer)
    assert buffer.text == "alpha"
    n.TuiApp.complete_input(buffer, reverse=True)
    assert buffer.text == "al"
    n.TuiApp.complete_input(buffer, reverse=True)
    assert buffer.text == "alpine"


def test_queue_acknowledges_only_claimed_duplicate_messages(tmp_path):
    s = session(tmp_path)
    queue(s, "same", "same")
    claimed = s.claim_user_inputs()
    s.enqueue_user_input("same")

    s.acknowledge_user_inputs(claimed)

    assert queued_texts(s) == ["same"]
    assert not s.pending_user_inputs[0].inflight


def test_queue_release_restores_interrupted_inputs(tmp_path):
    s = session(tmp_path)
    s.enqueue_user_input("ready")
    queued = s.pending_user_inputs[0]

    assert s.claim_user_inputs() == [queued]
    s.release_user_inputs()

    assert not queued.inflight


def test_recall_pending_input_can_revise_latest_inflight_message(tmp_path):
    s = session(tmp_path)
    queue(s, "first", "second")
    s.claim_user_inputs()
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    retried = []

    text = loop.recall_pending_input(lambda: retried.append(True))

    assert text == "second"
    assert queued_texts(s) == ["first"]
    assert s.pending_user_inputs[0].inflight is False
    assert retried == [True]


def test_clearing_recalled_message_leaves_it_deleted(tmp_path):
    s = session(tmp_path)
    queue(s, "first", "delete me")
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)

    assert loop.recall_pending_input(lambda: None) == "delete me"

    assert queued_texts(s) == ["first"]
    restored = n.Session.load_snapshot(s.uid, config=s.config)
    assert queued_texts(restored) == ["first"]


def test_pending_user_inputs_auto_submit_at_round_end(tmp_path):
    """Unconsumed pending_user_inputs are auto-submitted as next input."""
    s = session(tmp_path)
    queue(s, "leftover instruction")

    class FakeModel:
        def request(self, messages, tools=None):
            return {"role": "assistant", "content": "done"}, [], "done"

    agent = n.Agent(s, output_fn=lambda text: None)
    agent.model = FakeModel()

    def fake_read(prompt="", **kw):
        raise EOFError()

    loop = n.CommandLoop(agent, input_fn=fake_read, output_fn=lambda text: None)

    loop.run()

    assert s.pending_user_inputs == []
    assert any("leftover instruction" in msg.get("content", "") for msg in s.messages)


def test_queue_live_region_shows_divider_and_pending(tmp_path):
    s = session(tmp_path)
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    queue(s, "run tests", "then push")

    text = "".join(t for _, t in loop.queue_region_fragments())
    assert "2 queued" in text and "working" in text
    assert "+ run tests" in text and "+ then push" in text

    # The divider animates a comet head across the dashes while its label remains stable.
    with pytest.MonkeyPatch.context() as mp:
        seen_head = False
        for tick in range(200):
            mp.setattr(n.time, "monotonic", lambda tick=tick: tick * 0.1)
            fragments = loop.queue_divider_fragments()
            seen_head = seen_head or any(style == "class:divider.glow0" and text == "-" for style, text in fragments)
            assert any(style == "class:divider.working" and text.startswith("working") for style, text in fragments)
            assert all(not style.startswith("class:divider.glow") or text == "-" for style, text in fragments)
        assert seen_head

    s.pending_user_inputs = []
    empty = "".join(t for _, t in loop.queue_region_fragments())
    assert "working" in empty and "queued" not in empty and "run tests" not in empty


def test_queue_flush_moves_messages_into_log(tmp_path, monkeypatch):
    s = session(tmp_path)
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda _text: None)
    # The agent's flush hook is wired to move queued messages up into the scrollback log.
    assert loop.agent.on_queue_flush == loop.flush_queued_to_log

    echoed = []
    monkeypatch.setattr(n, "print_formatted_text", lambda value, **_kwargs: echoed.append("".join(text for _style, text in value)))

    loop.flush_queued_to_log(["do a thing", "then verify", "  "])

    assert echoed == ["\n• do a thing\n\n• then verify\n\n"]


def test_flush_sigint_ignores_stale_retry_signal(tmp_path):
    s = session(tmp_path)
    shortcut = n.ModelRetryShortcut(s)
    s.state.manual_model_retry_requested = True
    s.state.current_model_call_started_at = 0.0

    shortcut.handle_sigint(n.signal.SIGINT, None)

    assert s.state.manual_model_retry_requested is False


def test_flush_sigint_still_interrupts_active_retry_request(tmp_path):
    s = session(tmp_path)
    shortcut = n.ModelRetryShortcut(s)
    s.state.manual_model_retry_requested = True
    s.state.current_model_call_started_at = 123.0

    with pytest.raises(KeyboardInterrupt):
        shortcut.handle_sigint(n.signal.SIGINT, None)


def test_queue_command_runs_readonly(tmp_path):
    """A read-only slash command in the queue runs immediately and is not queued for the LLM."""
    s = session(tmp_path)
    out = []
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), input_fn=lambda *a, **k: "", output_fn=out.append)

    loop.run_queued_command("/status")

    assert s.pending_user_inputs == []
    assert out and not any("unavailable" in t for t in out)


def test_queue_command_runs_yolo_toggle(tmp_path):
    """/yolo flips the runtime flag from the queue while the agent works."""
    s = session(tmp_path)
    out = []
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), input_fn=lambda *a, **k: "", output_fn=out.append)

    before = s.settings.yolo
    loop.run_queued_command("/yolo")

    assert s.settings.yolo is (not before)
    assert s.pending_user_inputs == []


def test_queue_command_rejects_mutating(tmp_path):
    """A state-mutating slash command is refused while the agent works, not queued or run."""
    s = session(tmp_path)
    out = []
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), input_fn=lambda *a, **k: "", output_fn=out.append)

    loop.run_queued_command("/model")

    assert s.pending_user_inputs == []
    assert any("unavailable while the agent is working" in t for t in out)


def test_queue_command_rejects_mutating_mcp_subcommand(tmp_path):
    """Read-only /mcp is allowed; mutating subcommands like refresh are refused."""
    s = session(tmp_path)
    out = []
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), input_fn=lambda *a, **k: "", output_fn=out.append)

    loop.run_queued_command("/mcp refresh")

    assert any("read-only /mcp" in t for t in out)


def test_tool_input_without_tui_uses_injected_input(tmp_path):
    s = session(tmp_path)
    calls = []
    loop = n.CommandLoop(
        n.Agent(s, output_fn=lambda text: None),
        input_fn=lambda prompt: calls.append(prompt) or "y",
        output_fn=lambda text: None,
    )

    assert loop.tool_input("[Y/n or reason] ") == "y"

    assert calls == ["[Y/n or reason] "]


def test_tool_runner_edit_approval_prints_full_inline_preview(tmp_path, monkeypatch):
    s = session(tmp_path)
    outputs = []
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: "")
    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: "y", output_fn=lambda text: outputs.append(str(text)))
    content = "".join(f"line {index}\n" for index in range(50))

    runner.run([call("Edit", ["new.txt", [{"op": "create", "content": content}]])])

    assert outputs[0].startswith("  Edit  new.txt\n    ├ preview")
    assert "+line 49" in outputs[0]
    assert "preview truncated" not in outputs[0]
    assert any("[approved]" in output for output in outputs)

def test_exit_command_prints_resume_command(tmp_path):
    s = session(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    output = []
    loop = n.CommandLoop(n.Agent(s, output_fn=output.append), output_fn=output.append)

    handled, exit_now = loop.command("/exit")

    assert (handled, exit_now) == (True, True)
    assert output[-1] == f"Resume with:\nnanocode --resume {s.uid}"
    assert os.path.exists(s.data_path("sessions", f"{s.uid}.jsonl"))


def test_empty_exit_does_not_print_resume_command(tmp_path):
    s = session(tmp_path)
    output = []
    loop = n.CommandLoop(n.Agent(s, output_fn=output.append), output_fn=output.append)

    handled, exit_now = loop.command("/exit")

    assert (handled, exit_now) == (True, True)
    assert output == []
    assert not os.path.exists(s.data_path("sessions", f"{s.uid}.jsonl"))


def test_resumed_session_does_not_render_tool_results(tmp_path):
    s = session(tmp_path)
    s.resumed = True
    arguments = json.dumps({"files": [{"path": "a.py", "ranges": [[0, 1]]}]})
    s.messages.extend(
        [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "need tool",
                "tool_calls": [{"id": "tc.1", "type": "function", "function": {"name": "Read", "arguments": arguments}}],
            },
            {"role": "tool", "tool_call_id": "tc.1", "content": "raw tool result"},
            {"role": "system", "content": f"[Session resumed: uid={s.uid}]"},
        ]
    )
    s.tool_records.append(n.ToolResultRecord("tr.1", "Read", [{"path": "a.py", "ranges": [[0, 1]]}], "raw tool result", "a.py 0:1"))
    output = []
    loop = n.CommandLoop(n.Agent(s, output_fn=output.append), output_fn=output.append)

    loop.render_resumed_session()

    text = "\n".join(output)
    assert s.resumed is False
    assert f"Restored session: {s.uid}" in text
    assert "• hello" in text
    assert "  need tool" in text
    assert "user:" not in text and "assistant:" not in text
    assert "Read  a.py 0:1 → tr.1" in text
    assert "tool:" not in text
    assert "raw tool result" not in text


def test_resumed_session_renders_saved_tool_records_without_matching_tool_calls(tmp_path):
    s = session(tmp_path)
    s.resumed = True
    s.messages.extend(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "compacted answer\nfinal detail"},
        ]
    )
    s.tool_records.append(
        n.ToolResultRecord("tr.1", "Bash", ["wc -l nanocode.py"], "999 nanocode.py", "wc -l nanocode.py")
    )
    output = []
    loop = n.CommandLoop(n.Agent(s, output_fn=output.append), output_fn=output.append)

    loop.render_resumed_session()

    text = "\n".join(output)
    assert f"Restored session: {s.uid}" in text
    assert "compacted answer\nfinal detail" in text
    assert "user:" not in text and "assistant:" not in text
    assert "  Bash  wc -l nanocode.py\n    └ stored tr.1" in text
    assert "999 nanocode.py" not in text


def test_resumed_session_separates_turn_boxes(tmp_path):
    s = session(tmp_path)
    s.resumed = True
    s.messages.extend(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "one"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "two"},
        ]
    )
    output = []
    loop = n.CommandLoop(n.Agent(s, output_fn=output.append), output_fn=output.append)

    loop.render_resumed_session()

    assert output[1:] == ["\n• first", "one", "", "\n• second", "two"]


def test_turn_box_groups_followup_users_until_final_assistant():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "working", "tool_calls": [{"id": "one"}]},
        {"role": "user", "content": "follow-up"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "next"},
    ]

    boxes = n.TurnBox.group(messages)

    assert [len(box.messages) for box in boxes] == [4, 1]


def test_turn_box_groups_tool_results_with_calling_assistant():
    # Tool results (role="tool") are kept in the same TurnBox as the
    # assistant that issued the tool_calls, not split prematurely.
    messages = [
        {"role": "user", "content": "read a.py"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "tr.1", "function": {"name": "Read"}}]},
        {"role": "tool", "tool_call_id": "tr.1", "content": "# file content"},
        {"role": "assistant", "content": "done"},
    ]
    boxes = n.TurnBox.group(messages)
    assert len(boxes) == 1
    assert len(boxes[0].messages) == 4
    roles = [m["role"] for m in boxes[0].messages]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_eof_exit_prints_resume_command(tmp_path):
    s = session(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    output = []
    loop = n.CommandLoop(n.Agent(s, output_fn=output.append), input_fn=lambda prompt="": (_ for _ in ()).throw(EOFError()), output_fn=output.append)

    assert loop.run() == 0

    assert output[-1] == f"Resume with:\nnanocode --resume {s.uid}"
    assert os.path.exists(s.data_path("sessions", f"{s.uid}.jsonl"))


def test_select_choice_noninteractive_does_not_prompt(tmp_path):
    output = []
    loop = n.CommandLoop(n.Agent(session(tmp_path), output_fn=output.append), input_fn=lambda prompt="": "1", output_fn=output.append)

    assert loop.select_choice("Pick", ("a", "b"), labels={"a": "A"}, current="a") is None
    assert output == []


def test_choice_application_expands_escaped_preview_newlines(tmp_path):
    output = []
    loop = n.CommandLoop(n.Agent(session(tmp_path), output_fn=output.append), input_fn=lambda prompt="": "", output_fn=output.append)
    loop.interactive_input = True
    rendered = []

    class Modal:
        def show_modal(self, fragments_fn, key_fn):
            rendered.extend(fragments_fn())
            return key_fn("enter", "")

    loop.tui = Modal()

    result = loop.choice_application(
        "Select:",
        ("A", "B"),
        {},
        "",
        set(),
        preview_fn=lambda choice: "one\\ntwo" if choice == "A" else "",
        free_text=True,
    )

    assert result == "A"
    previews = [text for style, text in rendered if style == "class:choice.preview"]
    assert previews == ["  │ one\n", "  │ two\n"]
    assert all("\\n" not in text for _, text in rendered)


def test_ask_free_text_prompt_has_no_control_newline(tmp_path):
    output = []
    loop = n.CommandLoop(n.Agent(session(tmp_path), output_fn=output.append), input_fn=lambda prompt="": "", output_fn=output.append)
    loop.interactive_input = True
    emitted = []
    prompts = []
    loop.emit = emitted.append
    loop.choice_application = lambda *args, **kwargs: n.SELECTION_FREE_TEXT

    def fake_read_input(prompt_text="> ", **kwargs):
        prompts.append(prompt_text)
        return "typed answer"

    loop.read_input = fake_read_input

    assert loop.question_application(n.AskSpec("Pick?", choices=["A"], previews=["preview"])) == "typed answer"
    assert prompts == ["> "]
    assert all(not prompt.startswith("\n") for prompt in prompts)
    assert emitted[-1] == ""


def test_ask_without_choices_uses_shared_tui_input(tmp_path):
    loop = n.CommandLoop(n.Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt: "fallback", output_fn=lambda text: None)
    prompts = []
    loop.tui = SimpleNamespace(request_input=lambda prompt: prompts.append(prompt) or "typed answer")

    assert loop.question_application(n.AskSpec("Explain the issue")) == "typed answer"
    assert prompts == ["\nExplain the issue"]


def test_elapsed_since_uses_whole_seconds(monkeypatch):
    monkeypatch.setattr(n.time, "monotonic", lambda: 104.9)
    assert n.Text.elapsed_since(100.0) == "4s"

    monkeypatch.setattr(n.time, "monotonic", lambda: 162.9)
    assert n.Text.elapsed_since(100.0) == "1m02s"


def test_bash_live_start_pauses_standalone_status(tmp_path):
    loop = n.CommandLoop(n.Agent(session(tmp_path), output_fn=lambda text: None), output_fn=lambda text: None)
    loop.ui.color = True
    loop.live_preview.start = lambda: setattr(loop.live_preview, "active", True)
    loop.status_bar.thread = object()
    loop.status_bar.stop = lambda: setattr(loop.status_bar, "thread", None)
    loop.status_bar.start = lambda **_kwargs: setattr(loop.status_bar, "thread", object())

    loop.tool_live_start()
    assert loop.live_status_paused is True
    assert loop.status_bar.thread is None
    assert loop.agent.tools.bash_live_preview_shown is not None
    assert loop.agent.tools.bash_live_preview_shown() is True
    assert loop.agent.tools.bash_live_preview_shown() is False

    loop.tool_live_output("", "")
    assert loop.live_status_paused is False
    assert loop.status_bar.thread is not None


def test_agent_emits_and_records_intermediate_content_before_tools(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    s = session(tmp_path)
    output = []
    agent = n.Agent(s, output_fn=output.append)

    class TalkingModel:
        def __init__(self):
            self.messages = []

        def request(self, messages, tools=None):
            self.messages.append(messages)
            if len(self.messages) == 1:
                return {}, [call("Read", [{"path": "a.txt", "ranges": [[0, 1]]}])], "I'll inspect that first."
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = TalkingModel()
    assert agent.run("read file") == "done"
    assert output[0] == "I'll inspect that first."
    assert any(isinstance(line, n.LogBlock) and str(line).startswith("  Read  ") for line in output)
    assert [message["role"] for message in s.messages] == ["user", "assistant", "tool", "assistant"]
    assert s.messages[0]["content"] == "read file"
    assert s.messages[1]["content"] == "I'll inspect that first."
    assert s.messages[2]["content"].startswith("tool tr.1 Read a.txt 0:1")
    assert "<Read" in s.messages[2]["content"]
    assert "-> FILE STATE" not in s.messages[2]["content"]
    assert s.messages[3]["content"] == "done"
    assert any("I'll inspect that first." in (message.get("content") or "") for message in agent.model.messages[1])


def test_command_loop_indents_intermediate_and_final_messages(tmp_path):
    output = []
    loop = n.CommandLoop(n.Agent(session(tmp_path), output_fn=output.append), output_fn=output.append)

    loop.emit_agent_output("First line.\nSecond line.")
    loop.ui.emit_answer("Done.\nFinal detail.")

    assert output == ["  First line.\n  Second line.", "Done.\nFinal detail."]


def test_compaction_fallback_trims_when_model_compact_fails(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    s.state.summary = "existing"
    s.messages = [{"role": "user", "content": str(index)} for index in range(10)]
    context = n.ContextManager(s)

    class FailingModel:
        def compact(self, text):
            raise n.ModelError("failed")

    context.prepare_messages(FailingModel(), "system", [{"role": "user", "content": "request"}])
    assert s.state.summary != "existing"
    assert len(s.messages) == 2
    assert s.messages[0]["content"].startswith(n.ContextManager.COMPACT_TITLE)
    assert "deterministically trimmed" in s.messages[0]["content"]
    assert s.messages[1]["content"] == "9"


def test_manual_compact_inserts_summary_before_latest_user(tmp_path):
    s = session(tmp_path)
    s.messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "old answer"}, {"role": "user", "content": "latest"}, {"role": "tool", "content": "tool kept"}]
    s.state.context_percent = 80
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), output_fn=lambda text: None)
    transitions = []
    loop.tui = SimpleNamespace(set_running=transitions.append, set_dispatching=lambda: transitions.append("dispatch"))

    class FakeModel:
        def compact(self, text):
            assert transitions == ["compacting context"]
            return {"summary": "summary", "plan": ["next"], "known": ["fact"]}

    loop.agent.model = FakeModel()
    result = loop.compact("")

    assert [message["role"] for message in s.messages] == ["user", "user", "tool"]
    assert s.messages[0]["content"].startswith(n.ContextManager.COMPACT_TITLE)
    assert s.messages[1]["content"] == "latest"
    assert s.messages[2]["content"] == "tool kept"
    assert s.state.summary == "summary"
    assert transitions == ["compacting context", "dispatch"]
    assert "messages 4 -> 3" in result
    assert "prior summary inserted" in result


def test_agent_tool_error_feedback_is_visible_on_next_model_request(tmp_path):
    s = session(tmp_path)
    agent = n.Agent(s, output_fn=lambda text: None)

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
    # system is a cache_control-marked block so the tools+system prefix is cached across turns.
    assert params["system"] == [{"type": "text", "text": "system", "cache_control": {"type": "ephemeral"}}]
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


def test_malformed_tool_args_defer_to_execution_chat(tmp_path):
    """A live chat tool call whose args fail payload validation (Bash with empty command) must not
    raise out of parsing; the error is deferred onto the call so the turn is not aborted."""
    s = n.Session(cwd=str(tmp_path))
    client = n.ModelClient(s)
    raw = SimpleNamespace(id="x1", function=SimpleNamespace(name="Bash", arguments='{"command": ""}'))
    message = SimpleNamespace(tool_calls=[raw])
    calls = client.tool_calls(message)  # must not raise ToolError
    assert len(calls) == 1
    assert calls[0].args == []
    assert "non-empty" in calls[0].error


def test_malformed_tool_args_defer_to_execution_anthropic(tmp_path):
    """Same deferral on the anthropic path: a tool_use with invalid input is captured, not raised."""
    s = n.Session(cwd=str(tmp_path))
    client = n.ModelClient(s)
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
    s = n.Session(cwd=str(tmp_path))
    ctx = n.ContextManager(s)
    runner = n.ToolRunner(s, ctx, input_fn=lambda *a: "", output_fn=lambda *a: None)
    call = n.ToolCall(id="x1", name="Bash", args=[], error="Bash command must be non-empty")
    results = runner.run([call])
    assert len(results) == 1
    assert results[0]["role"] == "tool"
    assert "non-empty" in results[0]["content"]


def _runner(tmp_path, input_reply=""):
    s = n.Session(cwd=str(tmp_path))
    return s, n.ToolRunner(s, n.ContextManager(s), input_fn=lambda *a: input_reply, output_fn=lambda *a: None)


def test_parallel_safe_classification(tmp_path):
    _, runner = _runner(tmp_path)

    def safe(name, args):
        return runner.parallel_safe(n.ToolCall(id="x", name=name, args=args))

    assert safe("Read", [{"path": "f.txt"}])
    assert safe("Search", [{"pattern": "x"}])
    assert not safe("Bash", ["git status --short"])  # Bash streams live output, so it stays serial
    assert not safe("Bash", ["git commit -m x"])  # mutating command
    assert not safe("Bash", ["echo hi"])  # live-output command
    assert not safe("Edit", ["f.txt", [{"op": "insert_after", "start": "0:a", "content": "x"}]])
    assert not safe("Ask", [{"question": "q?"}])  # interactive
    assert not safe("Nope", [])  # unknown tool


def test_parallel_readonly_preserves_request_order(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text(f"content-{i}\n")
    s, runner = _runner(tmp_path)
    s.settings.max_parallel_tools = 4
    calls = [n.ToolCall(id=f"r{i}", name="Read", args=[{"path": f"f{i}.txt", "ranges": [[0, 0]]}]) for i in range(5)]

    # Force overlapping execution and record peak concurrency.
    active = {"cur": 0, "max": 0}
    guard = threading.Lock()
    original = n.ReadTool.call

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

    n.ReadTool.call = traced
    try:
        messages = runner.run(calls)
    finally:
        n.ReadTool.call = original

    assert [m["tool_call_id"] for m in messages] == [f"r{i}" for i in range(5)]
    assert active["max"] >= 2  # actually ran concurrently


def test_parallel_disabled_runs_serial(tmp_path):
    for i in range(3):
        (tmp_path / f"f{i}.txt").write_text(f"c{i}\n")
    s, runner = _runner(tmp_path)
    s.settings.max_parallel_tools = 1  # disabled -> identical to legacy serial behavior
    calls = [n.ToolCall(id=f"r{i}", name="Read", args=[{"path": f"f{i}.txt", "ranges": [[0, 0]]}]) for i in range(3)]

    active = {"cur": 0, "max": 0}
    guard = threading.Lock()
    original = n.ReadTool.call

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

    n.ReadTool.call = traced
    try:
        messages = runner.run(calls)
    finally:
        n.ReadTool.call = original

    assert [m["tool_call_id"] for m in messages] == ["r0", "r1", "r2"]
    assert active["max"] == 1  # never overlapped


def test_refusal_short_circuits_across_parallel_and_serial(tmp_path):
    for i in range(3):
        (tmp_path / f"f{i}.txt").write_text(f"c{i}\n")
    s, runner = _runner(tmp_path, input_reply="no")  # decline confirmation
    s.settings.max_parallel_tools = 4
    calls = [
        n.ToolCall(id="r0", name="Read", args=[{"path": "f0.txt", "ranges": [[0, 0]]}]),
        n.ToolCall(id="r1", name="Read", args=[{"path": "f1.txt", "ranges": [[0, 0]]}]),
        n.ToolCall(id="b0", name="Bash", args=[":"]),  # confirmation required, refused
        n.ToolCall(id="r2", name="Read", args=[{"path": "f2.txt", "ranges": [[0, 0]]}]),  # skipped
    ]
    messages = runner.run(calls)
    by_id = {m["tool_call_id"]: m["content"] for m in messages}
    assert [m["tool_call_id"] for m in messages] == ["r0", "r1", "b0", "r2"]
    assert "refused" in by_id["b0"].lower()
    assert "Skipped" in by_id["r2"]


def _write_skill(root, name, description, body, *, scripts=None):
    folder = os.path.join(root, ".nanocode", "skills", name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "SKILL.md"), "w", encoding="utf-8") as handle:
        handle.write(f"---\nname: {name}\ndescription: {description}\n---\n{body}\n")
    for script_name, script_body in (scripts or {}).items():
        script_dir = os.path.join(folder, "scripts")
        os.makedirs(script_dir, exist_ok=True)
        with open(os.path.join(script_dir, script_name), "w", encoding="utf-8") as handle:
            handle.write(script_body)
    return folder


def test_skill_library_index_and_lookup(tmp_path):
    _write_skill(tmp_path, "release-notes", "Draft a CHANGELOG entry.", "Do the thing.")
    s = session(tmp_path)

    index = s.skills.index()
    assert index.startswith("--- SKILLS ---")
    assert "- release-notes: Draft a CHANGELOG entry." in index
    assert s.skills.get("Release-Notes").name == "release-notes"  # case-insensitive
    assert s.skills.get("missing") is None


def test_skill_project_overrides_user(tmp_path, monkeypatch):
    user_home = tmp_path / "home"
    user_skill = user_home / ".nanocode" / "skills" / "shared"
    user_skill.mkdir(parents=True)
    (user_skill / "SKILL.md").write_text("---\nname: shared\ndescription: user version\n---\nuser body\n", encoding="utf-8")
    _write_skill(tmp_path, "shared", "project version", "project body")
    monkeypatch.setattr(n.os.path, "expanduser", lambda path: path.replace("~", str(user_home)))

    s = session(tmp_path)
    skill = s.skills.get("shared")
    assert skill.source == "project"
    assert skill.description == "project version"


def test_skill_tool_expands_skill_dir(tmp_path):
    folder = _write_skill(tmp_path, "build", "build it", 'Run python "{skill_dir}/scripts/go.py".', scripts={"go.py": "print(1)"})
    s = session(tmp_path)

    output = n.SkillTool(s, ["build"]).call()
    assert output.startswith('<Skill name="build">')
    assert f'python "{folder}/scripts/go.py"' in output
    assert "{skill_dir}" not in output


def test_skill_tool_unknown_lists_available(tmp_path):
    _write_skill(tmp_path, "known", "known skill", "body")
    s = session(tmp_path)
    with pytest.raises(n.ToolError) as excinfo:
        n.SkillTool(s, ["nope"]).call()
    assert "unknown skill 'nope'" in str(excinfo.value)
    assert "known" in str(excinfo.value)


def test_skill_mentions_inject_body(tmp_path):
    _write_skill(tmp_path, "triage", "triage a bug", "Reproduce first.")
    s = session(tmp_path)

    resolved = s.skills.resolve_mentions("please $triage this")
    assert "--- SKILL MENTIONS ---" in resolved
    assert "[triage] triage a bug" in resolved
    assert "Reproduce first." in resolved
    # a bare word without $ is not a mention; an unknown $token is ignored
    assert s.skills.resolve_mentions("triage this") == ""
    assert s.skills.resolve_mentions("$unknown") == ""


def test_skill_tool_absent_only_when_no_skills(tmp_path):
    # With the built-in nanocode-help present, the Skill tool and SKILLS section are offered.
    withskill = n.ContextManager(session(tmp_path))
    assert "--- SKILLS ---" in withskill.skills_context()
    assert any(t["function"]["name"] == "Skill" for t in n.resolved_tool_schemas(withskill.session))
    messages = withskill.model_messages("system", [{"role": "user", "content": "hi"}])
    assert any(m["content"].startswith("--- SKILLS ---") for m in messages)

    # When truly no skills exist, the tool and section drop out and the prefix stays clean.
    bare = n.ContextManager(session(tmp_path))
    bare.session.skills = n.SkillLibrary({})
    assert bare.skills_context() == ""
    tools = n.resolved_tool_schemas(bare.session)
    assert not any(t["function"]["name"] == "Skill" for t in tools)
    assert all("--- SKILLS ---" not in text for _name, text in bare.cache_prefix_regions(n.Agent.SYSTEM_PROMPT, tools))


def test_skills_command_lists_builtin_and_installed(tmp_path):
    # The built-in nanocode-help skill always ships; /skills lists it plus any installed ones.
    base = n.CommandLoop(n.Agent(session(tmp_path), output_fn=lambda t: None), output_fn=lambda t: None)
    base_out = base.skills_command("")
    assert "| skill | source | description |" in base_out  # rendered as a markdown table
    assert "| `nanocode-help` | builtin |" in base_out

    _write_skill(tmp_path, "release-notes", "Draft a CHANGELOG entry.", "body")
    loop = n.CommandLoop(n.Agent(session(tmp_path), output_fn=lambda t: None), output_fn=lambda t: None)
    output = loop.skills_command("")
    assert "| `release-notes` | project | Draft a CHANGELOG entry. |" in output
    assert "| `nanocode-help` | builtin |" in output


def test_skills_command_empty_when_builtins_absent(tmp_path):
    loop = n.CommandLoop(n.Agent(session(tmp_path), output_fn=lambda t: None), output_fn=lambda t: None)
    loop.session.skills = n.SkillLibrary({})
    assert "No skills installed" in loop.skills_command("")


def test_skill_loads_dedup_on_repeat(tmp_path):
    _write_skill(tmp_path, "guide", "a guide", "FULL GUIDE INSTRUCTIONS")
    s = session(tmp_path)
    body = n.SkillTool(s, ["guide"]).call()
    messages = [{"role": "tool", "content": "tr.1 " + body}, {"role": "tool", "content": "tr.7 " + body}]

    deduped = n.ContextManager(s).dedup_skill_loads(messages)
    assert "FULL GUIDE INSTRUCTIONS" in deduped[0]["content"]  # first copy kept
    assert "FULL GUIDE INSTRUCTIONS" not in deduped[1]["content"]  # repeat collapsed
    assert "repeat load of skill guide" in deduped[1]["content"]
    assert "tr.1" in deduped[1]["content"]


def test_status_and_bar_show_skill_count(tmp_path):
    _write_skill(tmp_path, "one", "d1", "b")
    _write_skill(tmp_path, "two", "d2", "b")
    s = session(tmp_path)
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda t: None), output_fn=lambda t: None)

    count = len(s.skills.skills)  # 2 project + built-in nanocode-help
    assert count == 3
    assert f"skills `{count}`" in loop.status("")
    bar_text = " | ".join(text for text, _ in n.StatusBar(s).entries(0.0, show_elapsed=False))
    assert f"skills {count}" in bar_text


def test_debug_command_shows_bounded_cache_prefix_records(tmp_path):
    s = session(tmp_path)
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), output_fn=lambda text: None)
    loop.agent.context.check_cache_prefix("first")
    loop.agent.context.check_cache_prefix("second")

    output = loop.debug("")

    assert "### Debug · 1 cache-prefix mismatch" in output
    assert "#### Latest · call 1 · round 0 · step 0" in output
    assert "| system |" in output
    assert "first" not in output and "second" not in output
    assert "prefix mismatches `1`; see `/debug`" in loop.status("")
    assert loop.debug("on") == "Usage: /debug"


def test_builtin_nanocode_help_skill_is_self_contained(tmp_path):
    s = session(tmp_path)
    skill = s.skills.get("nanocode-help")
    assert skill is not None and skill.source == "builtin"
    body = n.SkillTool(s, ["nanocode-help"]).call()
    # Authored manual prose so how-to / feature / troubleshooting questions need no source read.
    assert "## How it works" in body and "## Troubleshooting" in body
    assert "prefix-mismatch" in body  # a concept /help does not explain
    # Plus lists assembled from in-code constants (so they cannot drift).
    assert "/strict" in body and "/skills" in body  # command list (from /help)
    assert "InspectCode:" in body  # tool details (from DESCRIPTIONs)
    assert "provider.model" in body and "runtime.max_agent_steps" in body  # settable keys
    assert os.path.abspath(n.__file__) in body  # source named only as a last-resort fallback


def test_project_skill_overrides_builtin(tmp_path):
    _write_skill(tmp_path, "nanocode-help", "custom help", "my own instructions")
    s = session(tmp_path)
    skill = s.skills.get("nanocode-help")
    assert skill.source == "project"
    assert "my own instructions" in n.SkillTool(s, ["nanocode-help"]).call()


def test_session_from_config_file_theme_param(tmp_path):
    cfg = tmp_path / "nanocode.toml"
    cfg.write_text("[runtime]\ntheme = \"light\"\n")
    s = n.Session.from_config_file(path=str(cfg), theme="dark")
    assert s.settings.theme == "dark"

    s2 = n.Session.from_config_file(path=str(cfg))
    assert s2.settings.theme == "light"

    s3 = n.Session.from_config_file(path=str(cfg), theme="")
    assert s3.settings.theme == "light"


def test_agent_state_prefix_fingerprints_truncated_to_last_three():
    state = n.AgentState(prefix_fingerprints=["a", "b", "c", "d", "e"])
    assert state.prefix_fingerprints == ["c", "d", "e"]

    state2 = n.AgentState(prefix_fingerprints=["x"])
    assert state2.prefix_fingerprints == ["x"]

    state3 = n.AgentState(prefix_fingerprints=[])
    assert state3.prefix_fingerprints == []


def test_memory_context_includes_tool_errors_when_present(tmp_path):
    s = session(tmp_path)
    s.state.goal = "test goal"
    s.state.check = "all good"
    s.record_tool_error("tr.1", "Bash", ["bad"], "failed")

    ctx = n.ContextManager(s).memory_context()

    assert "Goal:" in ctx
    assert "test goal" in ctx
    assert "Check:" in ctx
    assert "all good" in ctx
    assert "Recent tool errors:" in ctx
    assert "tr.1 Bash bad: failed" in ctx
