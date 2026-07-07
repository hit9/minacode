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

    assert [message["role"] for message in messages] == ["system", "user", "user", "assistant", "user", "user", "user", "user", "user"]
    assert messages[0]["content"] == "system"
    assert messages[1]["content"].startswith("--- Environment ---")
    assert "- cwd: " + str(tmp_path) in messages[1]["content"]
    assert [message["content"] for message in messages[2:7]] == ["old request", "old answer", "current request", "extra one", "extra two"]
    assert messages[-2]["content"].startswith("--- Memory ---")
    assert "Date:" in messages[-2]["content"]
    assert messages[-1]["content"].startswith("--- FILE STATE ---")



def test_empty_file_context_is_empty(tmp_path):
    assert n.ContextManager(session(tmp_path)).file_context() == ""


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
    assert "- detected_commands: bash, rg, sed" in first
    assert "- detected_commands: bash, rg, sed" in second


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


def test_file_context_tracks_edits_and_omits_stale_reads(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("old\nkeep\n", encoding="utf-8")
    s = session(tmp_path)
    context = n.ContextManager(s)
    s.state.plan = ["inspect", "patch"]

    read_output = n.ReadTool(s, [{"path": "a.txt", "ranges": [[0, 2]]}]).call()
    read_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 2]]}], read_output)
    assert "| old" in context.file_context()

    path.write_text("changed\nkeep\n", encoding="utf-8")
    stale = context.file_context()
    assert "| old" not in stale
    assert read_key in stale

    path.write_text("old\nkeep\n", encoding="utf-8")
    edit_output = n.EditTool(
        s,
        ["a.txt", [{"op": "replace", "start": "0:" + n.ReadTool.line_hash("old\n"), "end": "0:" + n.ReadTool.line_hash("old\n"), "content": "new\n"}]],
    ).call()
    edit_key = s.store_tool_result("Edit", ["a.txt"], edit_output)

    rendered = context.file_context()
    assert edit_key in rendered
    assert "Current focus: inspect" in rendered
    assert f"source={edit_key} tool=Edit" in rendered
    assert "Files:\n- a.txt 0:2" in rendered
    assert "Read/Edit outputs update this section." in rendered
    assert f"Recent file events:\n- {read_key} Read" in rendered
    assert "Format: anchor=line:hash | text, where hash = hash(line_content). Use the full line:hash value as Edit anchors." in rendered
    assert f"@@ a.txt 0:1 current source={edit_key} tool=Edit" in rendered
    assert "| new" in rendered
    assert "| old" not in rendered


def test_empty_files_overview(tmp_path):
    assert n.ContextManager(session(tmp_path)).files_overview() == "#### File State\n(no files in context)"


def test_files_overview_and_detail(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("old\nkeep\n", encoding="utf-8")
    s = session(tmp_path)
    context = n.ContextManager(s)
    s.state.plan = ["inspect", "patch"]

    read_output = n.ReadTool(s, [{"path": "a.txt", "ranges": [[0, 2]]}]).call()
    read_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 2]]}], read_output)

    overview = context.files_overview()
    assert "#### File State  ·  focus: inspect" in overview
    # markdown table row for the file, with a source column
    assert f"| `a.txt` | 0:2 | 2 | {read_key} Read |" in overview
    assert f"**Recent events**\n- {read_key} Read" in overview
    # overview omits the full anchored content dump
    assert "| old" not in overview
    assert "```" not in overview

    detail = context.file_detail("a.txt")
    assert detail.startswith("**a.txt** — current, 2 lines")
    assert "```" in detail
    assert f"@@ 0:2  {read_key} Read" in detail
    assert "| old" in detail

    # basename resolves the same file; unknown path lists what is available
    assert context.file_detail("a.txt") == detail
    missing = context.file_detail("nope.txt")
    assert "No in-context content for `nope.txt`" in missing
    assert "- `a.txt`" in missing


def test_file_context_marks_full_file_reads(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("one\ntwo\n", encoding="utf-8")
    s = session(tmp_path)
    output = n.ReadTool(s, [{"path": "a.txt", "ranges": [[0, 0]]}]).call()
    s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 0]]}], output, n.ToolRunner(s, n.ContextManager(s)).tool_note(call("Read", []), output))

    rendered = n.ContextManager(s).file_context()
    assert "- a.txt 0:2 current" in rendered
    assert "| one" in rendered
    assert "| two" in rendered


def test_file_context_keeps_current_lines_without_local_budget(tmp_path):
    old_path = tmp_path / "old.txt"
    new_path = tmp_path / "new.txt"
    old_path.write_text("old-0\n" + "".join(f"old-{index}\n" for index in range(1, 80)), encoding="utf-8")
    new_path.write_text("new-0\n" + "".join(f"new-{index}\n" for index in range(1, 80)), encoding="utf-8")
    s = session(tmp_path)
    context = n.ContextManager(s)

    s.store_tool_result("Read", [{"path": "old.txt", "ranges": [[0, 0]]}], n.ReadTool(s, [{"path": "old.txt", "ranges": [[0, 0]]}]).call())
    new_key = s.store_tool_result(
        "Read", [{"path": "new.txt", "ranges": [[0, 0]]}], n.ReadTool(s, [{"path": "new.txt", "ranges": [[0, 0]]}]).call()
    )

    rendered = context.file_context()
    assert f"source={new_key} tool=Read" in rendered
    assert "| old-70" in rendered
    assert "| new-" in rendered


def test_file_context_edit_invalidate_replaces_only_changed_range(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    s = session(tmp_path)
    context = n.ContextManager(s)

    read_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 0]]}], n.ReadTool(s, [{"path": "a.txt", "ranges": [[0, 0]]}]).call())
    edit_output = n.EditTool(
        s,
        ["a.txt", [{"op": "replace", "start": "1:" + n.ReadTool.line_hash("b\n"), "end": "1:" + n.ReadTool.line_hash("b\n"), "content": "B\n"}]],
    ).call()
    edit_key = s.store_tool_result("Edit", ["a.txt"], edit_output)

    rendered = context.file_context()
    assert "| a" in rendered
    assert "| B" in rendered
    assert "| c" in rendered
    assert "| b" not in rendered
    assert f"@@ a.txt 1:2 current source={edit_key} tool=Edit" in rendered
    assert f"source={read_key} tool=Read" in rendered


def test_file_context_drops_drifted_old_lines_instead_of_guessing(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    s = session(tmp_path)
    context = n.ContextManager(s)

    read_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 0]]}], n.ReadTool(s, [{"path": "a.txt", "ranges": [[0, 0]]}]).call())
    n.EditTool(s, ["a.txt", [{"op": "insert_before", "start": "0:" + n.ReadTool.line_hash("a\n"), "content": "x\n"}]]).call()

    rendered = context.file_context()
    assert "| a" not in rendered
    assert "| b" not in rendered
    assert "| c" not in rendered
    assert f"{read_key}" in rendered
    assert "Omitted content:" in rendered


def test_file_context_uses_raw_current_lines_not_bounded_middle(tmp_path):
    path = tmp_path / "large.txt"
    path.write_text("first\n" + "".join(f"middle-{index}\n" for index in range(80)) + "last\n", encoding="utf-8")
    s = session(tmp_path)
    context = n.ContextManager(s)

    key = s.store_tool_result("Read", [{"path": "large.txt", "ranges": [[0, 0]]}], n.ReadTool(s, [{"path": "large.txt", "ranges": [[0, 0]]}]).call())

    rendered = context.file_context()
    assert f"source={key} tool=Read" in rendered
    assert "| first" in rendered
    assert "| middle-40" in rendered
    assert "| last" in rendered
    assert "<bounded_output" not in rendered


def test_file_context_merges_current_ranges_within_same_file(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    s = session(tmp_path)
    context = n.ContextManager(s)

    old_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 1]]}], n.ReadTool(s, [{"path": "a.txt", "ranges": [[0, 1]]}]).call())
    new_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[2, 3]]}], n.ReadTool(s, [{"path": "a.txt", "ranges": [[2, 3]]}]).call())

    rendered = context.file_context()
    assert f"source={old_key} tool=Read" in rendered
    assert f"source={new_key} tool=Read" in rendered
    assert "| a" in rendered
    assert "| c" in rendered


def test_file_context_edit_read_edit_keeps_final_state(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    s = session(tmp_path)
    context = n.ContextManager(s)

    edit1 = n.EditTool(
        s,
        ["a.txt", [{"op": "replace", "start": "0:" + n.ReadTool.line_hash("a\n"), "end": "0:" + n.ReadTool.line_hash("a\n"), "content": "A\n"}]],
    ).call()
    s.store_tool_result("Edit", ["a.txt"], edit1)
    read_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 0]]}], n.ReadTool(s, [{"path": "a.txt", "ranges": [[0, 0]]}]).call())
    edit2 = n.EditTool(
        s,
        ["a.txt", [{"op": "replace", "start": "2:" + n.ReadTool.line_hash("c\n"), "end": "2:" + n.ReadTool.line_hash("c\n"), "content": "C\n"}]],
    ).call()
    edit2_key = s.store_tool_result("Edit", ["a.txt"], edit2)

    rendered = context.file_context()
    assert "| A" in rendered
    assert "| b" in rendered
    assert "| C" in rendered
    assert "| a" not in rendered
    assert "| c" not in rendered
    assert f"@@ a.txt 0:2 current source={read_key} tool=Read" in rendered
    assert f"@@ a.txt 2:3 current source={edit2_key} tool=Edit" in rendered


def test_file_context_read_edit_read_uses_latest_read(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    s = session(tmp_path)
    context = n.ContextManager(s)

    read1_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 0]]}], n.ReadTool(s, [{"path": "a.txt", "ranges": [[0, 0]]}]).call())
    edit = n.EditTool(
        s,
        ["a.txt", [{"op": "replace", "start": "1:" + n.ReadTool.line_hash("b\n"), "end": "1:" + n.ReadTool.line_hash("b\n"), "content": "B\n"}]],
    ).call()
    s.store_tool_result("Edit", ["a.txt"], edit)
    read2_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 0]]}], n.ReadTool(s, [{"path": "a.txt", "ranges": [[0, 0]]}]).call())

    rendered = context.file_context()
    assert "| a" in rendered
    assert "| B" in rendered
    assert "| c" in rendered
    assert "| b" not in rendered
    assert f"@@ a.txt 0:3 current source={read2_key} tool=Read" in rendered
    assert f"source={read1_key} tool=Read" not in rendered


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
    context.maybe_compact(model, "system", [{"role": "user", "content": "request"}])
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


def test_maybe_compact_skips_when_context_under_budget(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 999_999
    s.messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "answer"}]

    class ExplodingModel:
        def compact(self, text):
            raise AssertionError(text)

    n.ContextManager(s).maybe_compact(ExplodingModel(), "system", [{"role": "user", "content": "request"}])

    assert s.messages == [{"role": "user", "content": "old"}, {"role": "assistant", "content": "answer"}]


def test_compaction_keeps_tool_records_referenced_from_summary(tmp_path):
    s = session(tmp_path)
    context = n.ContextManager(s)
    kept = s.store_tool_result("Bash", ["kept"], "kept output")
    dropped = s.store_tool_result("Bash", ["dropped"], "dropped output")

    context.apply_compaction({"summary": f"Continue from {kept}."}, [])

    assert kept in s.tool_results
    assert dropped not in s.tool_results
    assert [record.key for record in s.tool_records] == [kept]


def test_compaction_prunes_old_non_file_tool_records(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("one\n", encoding="utf-8")
    s = session(tmp_path)
    context = n.ContextManager(s)
    old_key = s.store_tool_result("Bash", ["old"], "old output")
    read_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 0]]}], n.ReadTool(s, [{"path": "a.txt", "ranges": [[0, 0]]}]).call())
    current_key = s.store_tool_result("Bash", ["current"], "current output")

    context.apply_compaction({"summary": "summary"}, [{"role": "tool", "content": f"tool {current_key} Bash current"}])

    assert old_key not in s.tool_results
    assert {record.key for record in s.tool_records} == {read_key, current_key}
    assert set(s.tool_results) == {read_key, current_key}
    assert "| one" in context.file_context()


def test_compaction_keeps_current_turn_tool_records(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    s.messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "old answer"}, {"role": "user", "content": "latest"}]
    old_key = s.store_tool_result("Bash", ["old"], "old output")
    current_key = s.store_tool_result("Bash", ["current"], "current output")

    class FakeModel:
        def compact(self, text):
            return {"summary": "summary"}

    n.ContextManager(s).maybe_compact(FakeModel(), "system", [{"role": "tool", "content": f"tool {current_key} Bash current"}])

    assert old_key not in s.tool_results
    assert current_key in s.tool_results
    assert [record.key for record in s.tool_records] == [current_key]


def test_compaction_keeps_edit_invalidations_needed_for_file_state(tmp_path):
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

    rendered = context.file_context()
    assert {record.key for record in s.tool_records} == {read_key, edit_key}
    assert "| a" in rendered
    assert "| b" not in rendered
    assert "Omitted content:" in rendered


def test_tool_runner_refusal_stops_batch_and_invalid_args_are_not_stored(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: "skip it", output_fn=lambda text: None)
    runner.run([call("Bash", ["printf first"]), call("Edit", ["second.txt", [{"op": "create", "content": "second"}]])])

    assert s.tool_records == []
    assert len(s.tool_errors) == 1
    assert "skip it" in s.tool_errors[0].error
    assert not (tmp_path / "second.txt").exists()

    outputs = []
    bad = session(tmp_path)
    n.ToolRunner(bad, n.ContextManager(bad), output_fn=outputs.append).run([call("Bash", [])])
    assert bad.tool_records == []
    assert len(bad.tool_errors) == 1
    assert outputs and "· rejected:" in outputs[0]  # argument errors collapse to a quiet line in non-debug


def test_tool_runner_refuses_without_reason_on_n(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: "n", output_fn=lambda text: None)

    runner.run([call("Bash", ["printf first"])])

    assert s.tool_errors[0].error == "Cancelled: user refused tool call"


def test_tool_runner_refuses_with_direct_reason_input(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: "not now", output_fn=lambda text: None)

    runner.run([call("Bash", ["printf first"])])

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

        def request(self, messages):
            self.messages.append(messages)
            if len(self.messages) == 1:
                return {}, [call("Read", [{"path": "a.txt", "ranges": [[0, 1]]}])], ""
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = FakeModel()
    assert agent.run("read file") == "done"
    assert len(agent.model.messages) == 2
    assert [len(messages) for messages in agent.model.messages] == [5, 7]
    assert agent.model.messages[1][3]["role"] == "assistant"
    assert agent.model.messages[1][3]["tool_calls"][0]["id"] == "Read-id"
    assert agent.model.messages[1][4]["role"] == "tool"
    assert agent.model.messages[1][4]["tool_call_id"] == "Read-id"
    assert any("tool tr.1 Read a.txt 0:1" in (message.get("content") or "") for message in agent.model.messages[1])
    assert any(message["role"] == "tool" and "-> FILE STATE" in message["content"] for message in agent.model.messages[1])
    assert len(s.tool_records) == 1
    assert s.messages[-1]["content"] == "done"
    assert s.state.goal == ""

    limited = session(tmp_path)
    limited.skills = n.SkillLibrary({})
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

    first = "\n\n".join(message.get("content") or "" for message in agent.model.messages[0])
    second = "\n\n".join(message.get("content") or "" for message in agent.model.messages[1])
    assert "extra instruction" in first
    assert "extra instruction" in second
    assert "checking" in second
    assert "second instruction" in second
    assert s.messages[0]["content"] == "initial request"
    assert s.messages[1]["content"] == "extra instruction"
    assert s.messages[2]["content"] == "checking"
    assert s.messages[3]["role"] == "tool"
    assert s.messages[3]["content"].startswith("tool tr.1 LineCount")
    assert s.messages[4]["content"] == "second instruction"
    assert s.messages[5]["role"] == "assistant"
    assert s.pending_user_inputs == []


def test_startup_tip_respects_toggle_and_context(tmp_path):
    s = session(tmp_path)
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)

    s.settings.tips = False
    assert loop.startup_tip() == ""

    s.settings.tips = True
    all_tips = [tip for _, tip in n.CommandLoop.TIPS]
    eligible = [tip for predicate, tip in n.CommandLoop.TIPS if predicate(s)]
    strict_tip = next(tip for tip in all_tips if tip.startswith("`/strict`"))
    mcp_tip = next(tip for tip in all_tips if "@server.tool" in tip)

    # Unknown host + no MCP: strict and MCP tips are filtered out, but a tip is still shown.
    assert strict_tip not in eligible
    assert mcp_tip not in eligible
    for _ in range(20):
        assert loop.startup_tip() in eligible

    # DeepSeek provider unlocks the strict tip.
    s.config.providers["default"].url = "https://api.deepseek.com"
    assert strict_tip in [tip for predicate, tip in n.CommandLoop.TIPS if predicate(s)]


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



def test_queued_text_auto_submits_at_round_end(tmp_path):
    """queue_input_text set during agent run is auto-submitted as next input."""
    s = session(tmp_path)

    class FakeModel:
        def request(self, messages):
            return {"role": "assistant", "content": "done"}, [], "done"

    agent = n.Agent(s, output_fn=lambda text: None)
    agent.model = FakeModel()

    def fake_read(prompt="", **kw):
        raise EOFError()

    loop = n.CommandLoop(agent, input_fn=fake_read, output_fn=lambda text: None)
    loop.queue_input_text = "auto instruction"

    loop.run()

    assert loop.queue_input_text == ""
    assert any("auto instruction" in msg.get("content", "") for msg in s.messages)


def test_pending_user_inputs_auto_submit_at_round_end(tmp_path):
    """Unconsumed pending_user_inputs are auto-submitted as next input."""
    s = session(tmp_path)
    s.pending_user_inputs.append("leftover instruction")

    class FakeModel:
        def request(self, messages):
            return {"role": "assistant", "content": "done"}, [], "done"

    agent = n.Agent(s, output_fn=lambda text: None)
    agent.model = FakeModel()

    def fake_read(prompt="", **kw):
        raise EOFError()

    loop = n.CommandLoop(agent, input_fn=fake_read, output_fn=lambda text: None)

    loop.run()

    assert s.pending_user_inputs == []
    assert any("leftover instruction" in msg.get("content", "") for msg in s.messages)



def test_queued_combined_order_auto_submits_at_round_end(tmp_path):
    """pending_user_inputs comes first, then queue_input_text."""
    s = session(tmp_path)
    s.pending_user_inputs.append("first pending")

    class FakeModel:
        def request(self, messages):
            return {"role": "assistant", "content": "done"}, [], "done"

    agent = n.Agent(s, output_fn=lambda text: None)
    agent.model = FakeModel()

    def fake_read(prompt="", **kw):
        raise EOFError()

    loop = n.CommandLoop(agent, input_fn=fake_read, output_fn=lambda text: None)
    loop.queue_input_text = "second queued"

    loop.run()

    assert s.pending_user_inputs == []
    assert loop.queue_input_text == ""
    joined = "\n".join(msg.get("content", "") for msg in s.messages if msg.get("role") == "user")
    assert "first pending" in joined
    assert "second queued" in joined
    assert joined.index("first pending") < joined.index("second queued")


def test_queued_blank_text_is_cleared(tmp_path):
    """Blank queue_input_text is cleared but does not auto-submit."""
    s = session(tmp_path)

    class FakeModel:
        def request(self, messages):
            return {"role": "assistant", "content": "done"}, [], "done"

    agent = n.Agent(s, output_fn=lambda text: None)
    agent.model = FakeModel()

    def fake_read(prompt="", **kw):
        raise EOFError()

    loop = n.CommandLoop(agent, input_fn=fake_read, output_fn=lambda text: None)
    loop.queue_input_text = "   "

    loop.run()

    assert loop.queue_input_text == ""
    # blank text did not auto-submit (no user message with spaces-only content)
    assert not any(
        msg.get("content", "").strip() == "" and msg.get("role") == "user"
        for msg in s.messages
    )
def test_interactive_entered_input_auto_submits_without_reprompt(tmp_path):
    """In interactive mode, Enter-committed queue input auto-submits as the next turn (no second
    Enter) and half-typed text is carried back to the box instead of blocking the submit."""
    s = session(tmp_path)
    s.pending_user_inputs.append("entered instruction")

    class FakeModel:
        def request(self, messages):
            return {"role": "assistant", "content": "done"}, [], "done"

    agent = n.Agent(s, output_fn=lambda text: None)
    agent.model = FakeModel()

    loop = n.CommandLoop(agent, input_fn=lambda *a, **k: "", output_fn=lambda text: None)
    loop.interactive_input = True
    loop.queue_input_text = "half typed"

    reads = []

    def fake_read_input(prompt_text="nano> ", *, initial_text="", **kw):
        reads.append(initial_text)
        raise EOFError()

    loop.read_input = fake_read_input
    loop.run()

    # entered input was auto-submitted without ever going through the editable read prompt
    assert any("entered instruction" in msg.get("content", "") for msg in s.messages)
    assert s.pending_user_inputs == []
    # the only read prompt was for the leftover half-typed text, pre-filled for review (not auto-sent)
    assert reads == ["half typed"]


def test_queue_command_runs_readonly(tmp_path):
    """A read-only slash command in the queue runs immediately and is not queued for the LLM."""
    s = session(tmp_path)
    out = []
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), input_fn=lambda *a, **k: "", output_fn=out.append)

    loop.run_queued_command("/context")

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


def test_approval_prompt_fragments_keep_text_and_spinner(tmp_path, monkeypatch):
    loop = n.CommandLoop(n.Agent(session(tmp_path), output_fn=lambda text: None), output_fn=lambda text: None)
    monkeypatch.setattr(n.time, "monotonic", lambda: 0.2)

    fragments = loop.input_prompt_fragments("[Y/n] ", "class:approval")

    assert fragments == [("class:approval", "[Y/n] "), ("class:approval.wait", "/ ")]
    assert loop.input_prompt_fragments("nano> ", "class:prompt") == [("class:prompt", "nano> ")]


def test_tool_preview_handles_only_interactive_edit_approval(tmp_path, monkeypatch):
    s = session(tmp_path)
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), output_fn=lambda text: None)
    shown = []
    loop.interactive_input = True
    monkeypatch.setattr(n.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(loop, "show_transient_tool_preview", shown.append)

    assert loop.tool_preview("approve Edit a.py\n  preview\n  diff")
    assert shown == ["approve Edit a.py\n  preview\n  diff"]
    assert not loop.tool_preview("approve Bash echo ok")


def test_tool_runner_edit_approval_can_use_preview_callback(tmp_path, monkeypatch):
    s = session(tmp_path)
    outputs = []
    previews = []
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: "")
    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: "y", output_fn=outputs.append)
    runner.preview_fn = lambda text: previews.append(text) or True

    runner.run([call("Edit", ["new.txt", [{"op": "create", "content": "x\n"}]])])

    assert previews and previews[0].startswith("approve Edit new.txt\n  preview")
    assert not any(output.startswith("approve Edit") for output in outputs)
    assert any("[approved]" in output for output in outputs)


def test_context_command_shows_context_frame(tmp_path):
    s = session(tmp_path)
    s.state.goal = "ship"
    s.state.plan = ["inspect"]
    s.state.known = ["pytest"]
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), output_fn=lambda text: None)

    output = loop.context_view("")

    # unified markdown frame: environment, memory, and file state sections
    assert "### Context" in output
    assert "#### Environment" in output
    assert "#### Memory" in output
    assert "#### File State" in output
    assert "| goal | ship |" in output
    assert "- [ ] inspect" in output
    assert "- pytest" in output
    # memory section is the model-facing one, so it never shows summary
    assert "summary" not in output


def test_context_command_renders_plan_item_objects(tmp_path):
    s = session(tmp_path)
    s.state.plan = [n.PlanItem("done", "设置 goal 和 plan")]
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), output_fn=lambda text: None)

    output = loop.context_view("")

    assert "设置 goal 和 plan" in output
    assert "PlanItem(" not in output


def test_context_view_opens_tabs_when_interactive(tmp_path):
    """At an interactive prompt the bare /context launches the tab viewer and emits nothing inline."""
    s = session(tmp_path)
    loop = n.CommandLoop(n.Agent(s, output_fn=lambda text: None), output_fn=lambda text: None)
    loop.interactive_input = True
    loop.ui.color = True
    loop.ui.capture_ansi = False
    opened = []
    loop.context_tabs = lambda context: opened.append(context)

    assert loop.context_view("") is None
    assert opened == [loop.agent.context]

    # while the agent works (queue path sets capture_ansi) it falls back to the static dump
    loop.ui.capture_ansi = True
    assert loop.context_view("").startswith("### Context")


def test_render_markdown_lines_splits_per_line(tmp_path):
    loop = n.CommandLoop(n.Agent(session(tmp_path), output_fn=lambda text: None), output_fn=lambda text: None)
    loop.ui.color = False
    assert loop.render_markdown_lines("alpha\nbeta", 60) == [[("", "alpha")], [("", "beta")]]


def _drive_context_tabs(tmp_path, keys, *, term=(80, 12)):
    import shutil

    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    loop = n.CommandLoop(n.Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda *a, **k: "", output_fn=lambda text: None)
    loop.ui.color = True
    # Force a tall page so scrolling has room, and a short viewport.
    loop.render_markdown_lines = lambda markdown, width: [[("", f"line{i}")] for i in range(100)]
    original_size = shutil.get_terminal_size
    shutil.get_terminal_size = lambda *a: os.terminal_size(term)
    with create_pipe_input() as pipe:
        real_app = n.Application
        n.Application = lambda **kw: real_app(**{**kw, "input": pipe, "output": DummyOutput()})
        loop.run_input_app = lambda app: app.run()
        try:
            pipe.send_text(keys)
            loop.context_tabs(loop.agent.context)
        finally:
            n.Application = real_app
            shutil.get_terminal_size = original_size
    return loop.context_tab_state


def test_context_tabs_scroll_and_switch_keys(tmp_path):
    # j/down scroll the body; k/up scroll back; h/l switch tabs. 'q' closes.
    assert _drive_context_tabs(tmp_path, "jjjq")["scroll"] == 3
    assert _drive_context_tabs(tmp_path, "jjjkq")["scroll"] == 2
    assert _drive_context_tabs(tmp_path, "llq")["tab"] == 2
    assert _drive_context_tabs(tmp_path, "lhq")["tab"] == 0


def test_exit_command_prints_resume_command(tmp_path):
    s = session(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    output = []
    loop = n.CommandLoop(n.Agent(s, output_fn=output.append), output_fn=output.append)

    handled, exit_now = loop.command("/exit")

    assert (handled, exit_now) == (True, True)
    assert output[-1] == f"Resume with: nanocode --resume {s.uid}"
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
    assert "hello" in text
    assert "need tool" in text
    assert "tool Read a.py 0:1 -> tr.1" in text
    assert "tool:" not in text
    assert "raw tool result" not in text


def test_eof_exit_prints_resume_command(tmp_path):
    s = session(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    output = []
    loop = n.CommandLoop(n.Agent(s, output_fn=output.append), input_fn=lambda prompt="": (_ for _ in ()).throw(EOFError()), output_fn=output.append)

    assert loop.run() == 0

    assert output[-1] == f"Resume with: nanocode --resume {s.uid}"
    assert os.path.exists(s.data_path("sessions", f"{s.uid}.jsonl"))


def test_select_choice_noninteractive_does_not_prompt(tmp_path):
    output = []
    loop = n.CommandLoop(n.Agent(session(tmp_path), output_fn=output.append), input_fn=lambda prompt="": "1", output_fn=output.append)

    assert loop.select_choice("Pick", ("a", "b"), labels={"a": "A"}, current="a") is None
    assert output == []


def test_bash_live_start_pauses_queue_before_app_is_active(tmp_path):
    loop = n.CommandLoop(n.Agent(session(tmp_path), output_fn=lambda text: None), output_fn=lambda text: None)
    loop.ui.color = True
    loop.interactive_input = True
    loop.live_preview.start = lambda command="": setattr(loop.live_preview, "active", True)

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
    assert [message["role"] for message in s.messages] == ["user", "assistant", "tool", "assistant"]
    assert s.messages[0]["content"] == "read file"
    assert s.messages[1]["content"] == "I'll inspect that first."
    assert s.messages[2]["content"].startswith("tool tr.1 Read a.txt 0:1")
    assert "-> FILE STATE" in s.messages[2]["content"]
    assert s.messages[3]["content"] == "done"
    assert any("I'll inspect that first." in (message.get("content") or "") for message in agent.model.messages[1])


def test_compaction_fallback_trims_when_model_compact_fails(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    s.state.summary = "existing"
    s.messages = [{"role": "user", "content": str(index)} for index in range(10)]
    context = n.ContextManager(s)

    class FailingModel:
        def compact(self, text):
            raise n.ModelError("failed")

    context.maybe_compact(FailingModel(), "system", [{"role": "user", "content": "request"}])
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

    class FakeModel:
        def compact(self, text):
            return {"summary": "summary", "plan": ["next"], "known": ["fact"]}

    loop.agent.model = FakeModel()
    result = loop.compact("")

    assert [message["role"] for message in s.messages] == ["user", "user", "tool"]
    assert s.messages[0]["content"].startswith(n.ContextManager.COMPACT_TITLE)
    assert s.messages[1]["content"] == "latest"
    assert s.messages[2]["content"] == "tool kept"
    assert s.state.summary == "summary"
    assert "messages 4 -> 3" in result
    assert "prior summary inserted" in result


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


def test_malformed_tool_args_defer_to_execution_chat(tmp_path):
    """A live chat tool call whose args fail payload validation (Git with empty argv) must not
    raise out of parsing; the error is deferred onto the call so the turn is not aborted."""
    s = n.Session(cwd=str(tmp_path))
    client = n.ModelClient(s)
    raw = SimpleNamespace(id="x1", function=SimpleNamespace(name="Git", arguments='{"argv": []}'))
    message = SimpleNamespace(tool_calls=[raw])
    calls = client.tool_calls(message)  # must not raise ToolError
    assert len(calls) == 1
    assert calls[0].args == []
    assert "non-empty 'argv'" in calls[0].error


def test_malformed_tool_args_defer_to_execution_anthropic(tmp_path):
    """Same deferral on the anthropic path: a tool_use with invalid input is captured, not raised."""
    s = n.Session(cwd=str(tmp_path))
    client = n.ModelClient(s)
    result = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id="a1", name="Git", input={"argv": []})],
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
    call = n.ToolCall(id="x1", name="Git", args=[], error="Git requires a non-empty 'argv' list")
    results = runner.run([call])
    assert len(results) == 1
    assert results[0]["role"] == "tool"
    assert "non-empty 'argv'" in results[0]["content"]


def _runner(tmp_path, input_reply=""):
    s = n.Session(cwd=str(tmp_path))
    return s, n.ToolRunner(s, n.ContextManager(s), input_fn=lambda *a: input_reply, output_fn=lambda *a: None)


def test_parallel_safe_classification(tmp_path):
    _, runner = _runner(tmp_path)

    def safe(name, args):
        return runner.parallel_safe(n.ToolCall(id="x", name=name, args=args))

    assert safe("Read", [{"path": "f.txt"}])
    assert safe("Search", [{"pattern": "x"}])
    assert safe("Git", ["status"])  # read-only subcommand
    assert not safe("Git", ["commit", "-m", "x"])  # mutating subcommand
    assert not safe("Bash", ["echo hi"])  # mutates + streams live output
    assert not safe("Edit", ["f.txt", [{"op": "insert_after", "start": "0:a", "content": "x"}]])
    assert not safe("Question", [{"question": "q?"}])  # interactive
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
        n.ToolCall(id="b0", name="Bash", args=["echo hi"]),  # mutating, refused
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
    assert any(t["function"]["name"] == "Skill" for t in withskill.tool_schemas())
    messages = withskill.model_messages("system", [{"role": "user", "content": "hi"}])
    assert any(m["content"].startswith("--- SKILLS ---") for m in messages)

    # When truly no skills exist, the tool and section drop out and the prefix stays clean.
    bare = n.ContextManager(session(tmp_path))
    bare.session.skills = n.SkillLibrary({})
    assert bare.skills_context() == ""
    assert not any(t["function"]["name"] == "Skill" for t in bare.tool_schemas())
    assert "--- SKILLS ---" not in bare.cache_prefix(n.Agent.SYSTEM_PROMPT, bare.tool_schemas())


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


def test_builtin_nanocode_help_skill_is_self_contained(tmp_path):
    s = session(tmp_path)
    skill = s.skills.get("nanocode-help")
    assert skill is not None and skill.source == "builtin"
    body = n.SkillTool(s, ["nanocode-help"]).call()
    # Authored manual prose so how-to / feature / troubleshooting questions need no source read.
    assert "## How the agent works" in body and "## Troubleshooting" in body
    assert "prefix churn" in body  # a concept /help does not explain
    # Plus lists assembled from in-code constants (so they cannot drift).
    assert "/context" in body and "/skills" in body  # command list (from /help)
    assert "InspectCode:" in body  # tool details (from DESCRIPTIONs)
    assert "provider.model" in body and "runtime.max_agent_steps" in body  # settable keys
    assert os.path.abspath(n.__file__) in body  # source named only as a last-resort fallback


def test_project_skill_overrides_builtin(tmp_path):
    _write_skill(tmp_path, "nanocode-help", "custom help", "my own instructions")
    s = session(tmp_path)
    skill = s.skills.get("nanocode-help")
    assert skill.source == "project"
    assert "my own instructions" in n.SkillTool(s, ["nanocode-help"]).call()
