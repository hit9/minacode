from prompt_toolkit.completion import CompleteEvent, WordCompleter
from prompt_toolkit.document import Document

import nanocode
from nanocode import AgentLoop, Config, ConfigFile, MainBlackboard, ParsedToolCall, ReferenceFileCompleter, RuntimeSettings, Session, StatusBar


def make_session(tmp_path, *, model: str = "", compact_at: int = 50, yolo: bool = False) -> Session:
    data = {"model": {"model": model}, "runtime": {"compact_at": compact_at}}
    return Session(cwd=str(tmp_path), config=Config.from_dict(data), settings=RuntimeSettings.from_dict(data, yolo=yolo))


def test_session_reports_missing_required_config(tmp_path):
    session = Session(cwd=str(tmp_path))

    assert session.missing_required_config() == ["api.url", "api.key", "model.model"]

    session.config.api.url = "url"
    session.config.api.key = "key"
    session.config.model.model = "model"

    assert session.missing_required_config() == []


def test_session_loads_user_rules_from_project_file(tmp_path, monkeypatch):
    rules_dir = tmp_path / ".nanocode"
    rules_dir.mkdir()
    (rules_dir / "user_rules.md").write_text("# User Rules\n\n- Prompt-only changes do not need tests.\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    session = Session.from_config_data({"api": {"url": "url", "key": "key"}, "model": {"model": "model"}})

    assert session.state.user_rules.format() == "# User Rules\n\n- Prompt-only changes do not need tests."


def test_init_config_file_writes_default_toml(tmp_path):
    config_path = tmp_path / "config.toml"

    created_path, created = ConfigFile.init(str(config_path))
    second_path, second_created = ConfigFile.init(str(config_path))
    config = ConfigFile.load(str(config_path))

    assert created_path == str(config_path)
    assert created is True
    assert second_path == str(config_path)
    assert second_created is False
    assert config["api"]["url"] == ""
    assert config["model"]["temperature"] == 0.7
    assert config["model"]["timeout"] == 90
    assert config["model"]["first_token_timeout"] == 60
    assert config["runtime"]["compact_at"] == 50


def test_main_init_config_uses_config_argument(tmp_path, capsys):
    config_path = tmp_path / "custom.toml"

    result = nanocode.main(["--config", str(config_path), "--init-config"])
    output = capsys.readouterr()

    assert result == 0
    assert config_path.exists()
    assert "Created config: " + str(config_path) in output.out


def test_main_loads_config_argument(tmp_path, monkeypatch):
    config_path = tmp_path / "custom.toml"
    config_path.write_text(
        """
[api]
url = "https://example.test/v1"
key = "key"

[model]
model = "custom-main"

[paths]
nanocode_dir = ".custom-nanocode"
""".strip(),
        encoding="utf-8",
    )
    sessions = []

    def fake_run(self):
        sessions.append(self.agent.session)
        return 0

    monkeypatch.setattr(nanocode.AgentLoop, "run", fake_run)

    result = nanocode.main(["--config", str(config_path)])

    assert result == 0
    assert sessions[0].config.api.url == "https://example.test/v1"
    assert sessions[0].config.api.key == "key"
    assert sessions[0].config.model.model == "custom-main"
    assert sessions[0].config.paths.nanocode_dir == ".custom-nanocode"


def test_status_bar_text_has_visible_sweep_marker(tmp_path):
    session = make_session(tmp_path, model="provider/model", compact_at=9)
    session.state.last_total_tokens = 42
    session.state.session_total_tokens = 1200
    session.state.turn_tool_calls = 3
    bar = StatusBar(session)

    text = bar._text(1.2, now=1.0)
    fragments = bar._fragments(1.2, now=1.0, show_sweep=True, show_elapsed=True)

    assert ">" not in text
    assert "model (medium)" in text
    assert "ctx:0/9" in text
    assert "tools:3" in text
    assert "tok(all):last:42 session:1k" in text
    assert all(style.startswith("#") for style, _ in fragments)
    assert len({style for style, _ in fragments}) > 3
    snapshot = bar.snapshot()
    assert snapshot == "model (medium) | ctx:0/9 | tools:3 | tok(all):last:42 session:1k"
    assert ">" not in snapshot


def test_status_bar_shows_current_model_call_number(tmp_path):
    session = make_session(tmp_path, model="provider/model")
    session.state.turn_model_calls = 2
    session.state.current_model_call_started_at = 0.4
    session.state.current_model_call_label = "provider/active-model"
    session.state.current_model_call_reasoning_label = "low"
    bar = StatusBar(session)

    text = bar._text(0.0, now=1.0)

    assert "active-model (low)" in text
    assert "calling(2):0.6s" in text


def test_agent_loop_highlights_only_diff_previews(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")

    loop = AgentLoop(FakeAgent(), output_fn=lambda message: None)

    diff_segments = loop._preview_segments("--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new\n same\n")
    plain_segments = loop._preview_segments("-not a diff\n+still not a diff")
    prefixed_diff_segments = loop._preview_segments("note\n--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new")
    false_positive_segments = loop._preview_segments("note\n--- a\n+still not a diff")

    assert ("ansibrightblack", "--- a\n") in diff_segments
    assert ("ansibrightblack", "+++ b\n") in diff_segments
    assert ("ansicyan", "@@ -1 +1 @@\n") in diff_segments
    assert ("ansired", "-old\n") in diff_segments
    assert ("ansigreen", "+new\n") in diff_segments
    assert ("ansicyan", "-not a diff\n") in plain_segments
    assert ("ansicyan", "+still not a diff\n") in plain_segments
    assert ("ansiyellow", "note\n") in prefixed_diff_segments
    assert ("ansibrightblack", "--- a\n") in prefixed_diff_segments
    assert ("ansired", "-old\n") in prefixed_diff_segments
    assert ("ansicyan", "--- a\n") in false_positive_segments


def test_agent_loop_styles_compact_tool_call_report(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")

    loop = AgentLoop(FakeAgent(), output_fn=lambda message: None)

    segments = loop._tool_segments("[success] Read sample.txt 0:1")

    assert ("ansigreen", "Read sample.txt 0:1\n") in segments
    assert all("ok " not in text for _, text in segments)


def test_agent_loop_indents_top_level_tool_report(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")

    captured = []
    loop = AgentLoop(FakeAgent(), output_fn=captured.append)

    loop._print_message("[success] Read sample.txt 0:1")

    assert captured == ["  Read sample.txt 0:1"]


def test_agent_loop_cancelled_message_mentions_context_is_kept(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")

    captured = []
    loop = AgentLoop(FakeAgent(), output_fn=captured.append)

    loop._print_message("Cancelled")

    assert captured == ["Cancelled\n  Context is kept; send a follow-up to continue."]


def test_agent_loop_styles_tool_arg_error_report(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")

    loop = AgentLoop(FakeAgent(), output_fn=lambda message: None)

    segments = loop._tool_segments("[failure] Read | tr.1 | error: Read args error: got 0 args")

    assert ("ansired", "Read | tr.1 | error: Read args error: got 0 args\n") in segments
    assert all("fail " not in text for _, text in segments)


def test_agent_loop_prints_auto_approved_tool_calls(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model", yolo=True)

    class FakeTool:
        def preview(self):
            return "preview"

        def is_editing(self):
            return True

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
            self.session = make_session(tmp_path, model="model")

    loop = AgentLoop(FakeAgent(), output_fn=lambda message: None)
    completer = loop._command_completer()

    slash_completions = list(completer.get_completions(Document("/"), CompleteEvent(completion_requested=True)))
    config_completions = list(completer.get_completions(Document("/con"), CompleteEvent(completion_requested=True)))
    set_key_completions = list(completer.get_completions(Document("/set model."), CompleteEvent(completion_requested=True)))
    set_bool_completions = list(completer.get_completions(Document("/set model.reasoning "), CompleteEvent(completion_requested=True)))
    set_effort_completions = list(completer.get_completions(Document("/set model.effort h"), CompleteEvent(completion_requested=True)))

    assert "/help" in [completion.text for completion in slash_completions]
    assert "/config" in [completion.text for completion in config_completions]
    assert "model.reasoning" in [completion.text for completion in set_key_completions]
    assert [completion.text for completion in set_bool_completions] == ["on", "off"]
    assert [completion.text for completion in set_effort_completions] == ["high"]


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
            self.session = make_session(tmp_path, model="model")

    outputs = []
    answers = iter(["do not edit generated files"])
    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: next(answers), output_fn=outputs.append)

    result = loop._wait_confirm("Proceed?", default=True)

    assert result == "do not edit generated files"
    assert outputs == ["Answer: no - do not edit generated files"]


def test_agent_loop_confirmation_discards_pending_tty_input(tmp_path, monkeypatch):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")

    calls = []

    class FakeStdin:
        def isatty(self):
            return True

        def fileno(self):
            return 42

    class FakeTermios:
        TCIFLUSH = 0
        error = OSError

        @staticmethod
        def tcflush(fd, queue):
            calls.append((fd, queue))

    outputs = []
    monkeypatch.setattr(nanocode.sys, "stdin", FakeStdin())
    monkeypatch.setitem(nanocode.sys.modules, "termios", FakeTermios)

    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: "", output_fn=outputs.append)
    result = loop._wait_confirm("Proceed?", default=True)

    assert result is True
    assert calls == [(42, FakeTermios.TCIFLUSH)]
    assert outputs == ["Answer: yes"]


def test_agent_loop_dispatches_commands_and_user_input(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")
            self.blackboard = MainBlackboard()
            self.runs = []

        def run(self, user_input, *, confirm=None, on_auto_approve=None, on_message=None):
            self.runs.append(user_input)
            if on_message is not None:
                on_message("assistant response")
            return {"actions": [{"type": "chat", "text": "assistant response"}]}

    inputs = iter(["/status", "hello", "/exit"])
    outputs = []
    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: next(inputs), output_fn=outputs.append)

    result = loop.run()

    assert result == 0
    assert any("nanocode - AI coding assistant" in output for output in outputs)
    assert any("model (medium)" in output for output in outputs)
    assert any("model: model reasoning=medium stream=on" in output for output in outputs)
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
            self.session = make_session(tmp_path, model="model")

    loop = AgentLoop(FakeAgent(), prompt_session=FakePromptSession())

    assert loop.run() == 0

    assert calls[0][0] == "> "
    kwargs = calls[0][1]
    assert kwargs["multiline"] is False
    assert kwargs["enable_history_search"] is True
    assert kwargs["refresh_interval"] == StatusBar.INTERVAL
    assert "bottom_toolbar" not in kwargs
