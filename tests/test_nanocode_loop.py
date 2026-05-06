from prompt_toolkit.completion import CompleteEvent, WordCompleter
from prompt_toolkit.document import Document

from nanocode import AgentLoop, ParsedToolCall, ReferenceFileCompleter, Session, StatusBar


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
    session.last_total_tokens = 42
    session.last_cost_usd = 0.000008
    session.session_total_tokens = 1200
    session.session_cost_usd = 12.345678
    session.turn_tool_calls = 3
    bar = StatusBar(session)

    text = bar._text(1.2, now=1.0)
    fragments = bar._fragments(1.2, now=1.0, show_sweep=True, show_elapsed=True)

    assert ">" not in text
    assert "model (medium)" in text
    assert "ctx:0/9" in text
    assert "tools:3" in text
    assert "tok:last:42/$0.000008 session:1k/$12.345678" in text
    assert "usd:" not in text
    assert all(style.startswith("#") for style, _ in fragments)
    assert len({style for style, _ in fragments}) > 3
    snapshot = bar.snapshot()
    assert snapshot == "model (medium) | ctx:0/9 | tools:3 | tok:last:42/$0.000008 session:1k/$12.345678"
    assert ">" not in snapshot


def test_status_bar_shows_current_model_call_number(tmp_path):
    session = Session(cwd=str(tmp_path), model="provider/model")
    session.turn_model_calls = 2
    session.current_model_call_started_at = 0.4
    bar = StatusBar(session)

    text = bar._text(0.0, now=1.0)

    assert "calling(2):0.6s" in text


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


def test_agent_loop_styles_queued_tool_preview(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = Session(cwd=str(tmp_path), model="model")

    loop = AgentLoop(FakeAgent(), output_fn=lambda message: None)

    segments = loop._queued_segments("Queued: ReplaceRange sample.txt:1-2 - update sample")

    assert segments == [
        ("ansibrightblack", "Queued: "),
        ("ansicyan", "ReplaceRange sample.txt:1-2"),
        ("ansibrightblack", " - "),
        ("ansimagenta", "update sample"),
        ("", "\n"),
    ]


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


def test_reference_file_completer_completes_at_paths_and_keeps_command_fallback(tmp_path):
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')", encoding="utf-8")

    completer = ReferenceFileCompleter(str(tmp_path), WordCompleter(["/help"], WORD=True))
    event = CompleteEvent(completion_requested=True)

    file_completions = list(completer.get_completions(Document("see @READ"), event))
    dir_completions = list(completer.get_completions(Document("see @sr"), event))
    nested_completions = list(completer.get_completions(Document("see @src/ma"), event))
    command_completions = list(completer.get_completions(Document("/he"), event))

    assert "README.md" in [completion.text for completion in file_completions]
    assert "src/" in [completion.text for completion in dir_completions]
    assert "src/main.py" in [completion.text for completion in nested_completions]
    assert "/help" in [completion.text for completion in command_completions]


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
            return {"actions": [{"type": "message", "text": "assistant response"}]}

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
