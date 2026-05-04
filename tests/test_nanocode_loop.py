import os

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from nanocode import AgentLoop, ParsedToolCall, Session, StatusBar


def test_cleanup_old_logs_removes_only_logs_older_than_three_days(tmp_path):
    now = 10_000_000.0
    log_dir = tmp_path / ".nanocode" / "tool_results"
    nested = log_dir / "nested"
    nested.mkdir(parents=True)
    old_log = log_dir / "old.log"
    new_log = log_dir / "new.log"
    old_text = log_dir / "old.txt"
    nested_old_log = nested / "nested-old.log"
    for path in [old_log, new_log, old_text, nested_old_log]:
        path.write_text("x", encoding="utf-8")
    old_time = now - 4 * 86400
    new_time = now - 2 * 86400
    os.utime(old_log, (old_time, old_time))
    os.utime(nested_old_log, (old_time, old_time))
    os.utime(old_text, (old_time, old_time))
    os.utime(new_log, (new_time, new_time))

    session = Session(cwd=str(tmp_path))
    session.cleanup_old_logs(days=3, now=now)

    assert not old_log.exists()
    assert not nested_old_log.exists()
    assert new_log.exists()
    assert old_text.exists()


def test_session_reports_missing_required_envs(tmp_path):
    session = Session(cwd=str(tmp_path), api_url="", api_key="", model="")

    assert session.REQUIRED_ENVS == (
        ("NANOCODE_API_URL", "api_url"),
        ("NANOCODE_API_KEY", "api_key"),
        ("NANOCODE_MODEL", "model"),
    )
    assert session.missing_required_envs() == ["NANOCODE_API_URL", "NANOCODE_API_KEY", "NANOCODE_MODEL"]

    session.api_url = "url"
    session.api_key = "key"
    session.model = "model"

    assert session.missing_required_envs() == []


def test_status_bar_text_has_visible_sweep_marker(tmp_path):
    session = Session(cwd=str(tmp_path), model="provider/model", compact_at=9)
    session.session_total_tokens = 1200
    bar = StatusBar(session)

    text = bar._text(1.2, now=1.0)
    fragments = bar._fragments(1.2, now=1.0, show_sweep=True, show_elapsed=True)

    assert ">" not in text
    assert "model (medium)" in text
    assert "ctx:0/9" in text
    assert "tok:last:- session:1k" in text
    assert all(style.startswith("#") for style, _ in fragments)
    assert len({style for style, _ in fragments}) > 3
    snapshot = bar.snapshot()
    assert snapshot == "model (medium) | ctx:0/9 | tok:last:- session:1k"
    assert ">" not in snapshot


def test_agent_loop_highlights_only_diff_previews(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = Session(cwd=str(tmp_path), model="model")

    loop = AgentLoop(FakeAgent(), output_fn=lambda message: None)

    diff_segments = loop._preview_segments("--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new\n same\n")
    plain_segments = loop._preview_segments("-not a diff\n+still not a diff")
    false_positive_segments = loop._preview_segments("note\n--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new")

    assert ("ansibrightblack", "--- a\n") in diff_segments
    assert ("ansibrightblack", "+++ b\n") in diff_segments
    assert ("ansicyan", "@@ -1 +1 @@\n") in diff_segments
    assert ("ansired", "-old\n") in diff_segments
    assert ("ansigreen", "+new\n") in diff_segments
    assert ("ansicyan", "-not a diff\n") in plain_segments
    assert ("ansicyan", "+still not a diff\n") in plain_segments
    assert ("ansicyan", "--- a\n") in false_positive_segments
    assert ("ansicyan", "-old\n") in false_positive_segments


def test_agent_loop_prints_auto_approved_tool_calls(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = Session(cwd=str(tmp_path), model="model", yolo=True)

    class FakeTool:
        def display(self):
            return "preview"

    outputs = []
    loop = AgentLoop(FakeAgent(), output_fn=outputs.append)
    call = ParsedToolCall(name="Edit", intention="edit sample", args=["sample.txt", "old", "new"])

    loop._show_auto_tool_call(call, FakeTool())

    assert any("Auto Tool Call | auto approved" in output for output in outputs)
    assert any('Run     Edit("sample.txt", "old", "new")' in output for output in outputs)
    assert any("Why     edit sample" in output for output in outputs)
    assert any("Preview\npreview" in output for output in outputs)


def test_agent_loop_command_completer_matches_slash_commands(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = Session(cwd=str(tmp_path), model="model")

    loop = AgentLoop(FakeAgent(), output_fn=lambda message: None)
    completer = loop._command_completer()

    slash_completions = list(completer.get_completions(Document("/"), CompleteEvent(completion_requested=True)))
    compact_completions = list(completer.get_completions(Document("/compact-"), CompleteEvent(completion_requested=True)))

    assert "/help" in [completion.text for completion in slash_completions]
    assert "/compact-at" in [completion.text for completion in compact_completions]


def test_agent_loop_confirmation_accepts_refusal_reason(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = Session(cwd=str(tmp_path), model="model")

    outputs = []
    answers = iter(["do not edit generated files"])
    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: next(answers), output_fn=outputs.append)

    result = loop._wait_confirm("Proceed?", default=True)

    assert result == "do not edit generated files"
    assert outputs == ["Answer: no - do not edit generated files"]


def test_agent_loop_dispatches_commands_and_user_input(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = Session(cwd=str(tmp_path), model="model")
            self.runs = []

        def run(self, user_input, *, confirm=None, on_auto_approve=None, on_message=None):
            self.runs.append(user_input)
            if on_message is not None:
                on_message("assistant response")
            return {"message_to_user": "assistant response"}

    inputs = iter(["/status", "hello", "/exit"])
    outputs = []
    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: next(inputs), output_fn=outputs.append)

    result = loop.run()

    assert result == 0
    assert any("nanocode - AI coding assistant" in output for output in outputs)
    assert any("model (medium)" in output for output in outputs)
    assert any("model: model" in output for output in outputs)
    assert "assistant response" in outputs
    assert loop.agent.runs == ["hello"]


def test_agent_loop_uses_prompt_toolkit_session(tmp_path):
    calls = []

    class FakePromptSession:
        def __init__(self):
            self.inputs = iter(["/exit"])

        def prompt(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return next(self.inputs)
 
    class FakeAgent:
        def __init__(self):
            self.session = Session(cwd=str(tmp_path), model="model")

    loop = AgentLoop(FakeAgent(), prompt_session=FakePromptSession())

    assert loop.run() == 0

    assert calls[0][0] == "> "
    kwargs = calls[0][1]
    assert kwargs["multiline"] is False
    assert kwargs["enable_history_search"] is True
    assert kwargs["refresh_interval"] == StatusBar.INTERVAL
    assert "bottom_toolbar" not in kwargs
