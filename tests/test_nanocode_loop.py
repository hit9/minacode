from prompt_toolkit.completion import CompleteEvent, WordCompleter
from prompt_toolkit.document import Document

import nanocode
from nanocode import AgentLoop, Blackboard, ConfigFile, EXPLORE_MESSAGE_PREFIX, ParsedToolCall, ReferenceFileCompleter, Session, StatusBar, VERIFY_MESSAGE_PREFIX


def test_session_reports_missing_required_config(tmp_path):
    session = Session(cwd=str(tmp_path), api_url="", api_key="", model="")

    assert session.missing_required_config() == ["api.url", "api.key", "main_model.model"]

    session.api_url = "url"
    session.api_key = "key"
    session.model = "model"

    assert session.missing_required_config() == []


def test_session_loads_project_knowledge_from_project_file(tmp_path, monkeypatch):
    knowledge_dir = tmp_path / ".nanocode"
    knowledge_dir.mkdir()
    (knowledge_dir / "project_knowledge.json").write_text(
        '{"version": 1, "summary": "Project summary.", "structure": ["single file"], "architecture": [], "workflows": [], "conventions": []}\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    session = Session.from_config_data({"api": {"url": "url", "key": "key"}, "main_model": {"model": "model"}})

    assert session.project_knowledge.summary == "Project summary."
    assert session.project_knowledge.structure == ["single file"]


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
    assert config["main_model"]["temperature"] == 0.7
    assert config["main_model"]["timeout"] == 90
    assert config["main_model"]["first_token_timeout"] == 60
    assert config["explore_agent"]["max_turns"] == 12
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

[main_model]
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
    assert sessions[0].api_url == "https://example.test/v1"
    assert sessions[0].api_key == "key"
    assert sessions[0].model == "custom-main"
    assert sessions[0].nanocode_dir == ".custom-nanocode"


def test_status_bar_text_has_visible_sweep_marker(tmp_path):
    session = Session(cwd=str(tmp_path), model="provider/model", compact_at=9)
    session.last_total_tokens = 42
    session.session_total_tokens = 1200
    session.turn_tool_calls = 3
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
    session = Session(cwd=str(tmp_path), model="provider/model")
    session.turn_model_calls = 2
    session.current_model_call_started_at = 0.4
    session.current_model_call_label = "provider/worker-model"
    session.current_model_call_reasoning_label = "low"
    bar = StatusBar(session)

    text = bar._text(0.0, now=1.0)

    assert "worker-model (low)" in text
    assert "calling(2):0.6s" in text


def test_agent_loop_highlights_only_diff_previews(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = Session(cwd=str(tmp_path), model="model")

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
            self.session = Session(cwd=str(tmp_path), model="model")

    loop = AgentLoop(FakeAgent(), output_fn=lambda message: None)

    segments = loop._tool_segments("[success] Read sample.txt 0:1")

    assert ("ansigreen", "Read sample.txt 0:1\n") in segments
    assert all("ok " not in text for _, text in segments)


def test_agent_loop_indents_top_level_tool_report(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = Session(cwd=str(tmp_path), model="model")

    captured = []
    loop = AgentLoop(FakeAgent(), output_fn=captured.append)

    loop._print_message("[success] Read sample.txt 0:1")

    assert captured == ["  Read sample.txt 0:1"]


def test_agent_loop_cancelled_message_mentions_context_is_kept(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = Session(cwd=str(tmp_path), model="model")

    captured = []
    loop = AgentLoop(FakeAgent(), output_fn=captured.append)

    loop._print_message("Cancelled")

    assert captured == ["Cancelled\n  Context is kept; send a follow-up to continue."]


def test_agent_loop_styles_tool_arg_error_report(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = Session(cwd=str(tmp_path), model="model")

    loop = AgentLoop(FakeAgent(), output_fn=lambda message: None)

    segments = loop._tool_segments("[failure] Read | tr.1 | error: Read args error: got 0 args")

    assert ("ansired", "Read | tr.1 | error: Read args error: got 0 args\n") in segments
    assert all("fail " not in text for _, text in segments)


def test_agent_loop_styles_explore_tool_report_with_scope_prefix(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = Session(cwd=str(tmp_path), model="model")

    captured = []
    loop = AgentLoop(FakeAgent(), output_fn=lambda message: captured.append(message))

    loop._print_message(EXPLORE_MESSAGE_PREFIX + '[success] Read("sample.txt", "0", "1")')

    assert captured == ['[explore]\n  Read("sample.txt", "0", "1")']


def test_agent_loop_styles_explore_tool_status_by_color(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = Session(cwd=str(tmp_path), model="model")

    captured = []
    loop = AgentLoop(FakeAgent(), output_fn=captured.append)

    loop._print_message(EXPLORE_MESSAGE_PREFIX + '[success] Search("producer")\n[failure] Read("missing.py")')

    assert captured == ['[explore]\n  Search("producer")\n  Read("missing.py")']


def test_agent_loop_merges_adjacent_scoped_sections(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = Session(cwd=str(tmp_path), model="model")

    captured = []
    loop = AgentLoop(FakeAgent(), output_fn=captured.append)

    loop._print_message(EXPLORE_MESSAGE_PREFIX + '[success] Search("producer")')
    loop._print_message(EXPLORE_MESSAGE_PREFIX + '[success] Read("producer.py")')
    loop._print_message(VERIFY_MESSAGE_PREFIX + '[success] Git("diff")')
    loop._print_message(VERIFY_MESSAGE_PREFIX + '[success] Read("sample.txt")')
    loop._print_message("done")
    loop._print_message(EXPLORE_MESSAGE_PREFIX + '[success] Search("again")')

    assert captured == [
        '[explore]\n  Search("producer")',
        '  Read("producer.py")',
        '[verify]\n  Git("diff")',
        '  Read("sample.txt")',
        "done",
        '[explore]\n  Search("again")',
    ]
    assert '"producer"' in captured[0]


def test_agent_loop_prints_auto_approved_tool_calls(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = Session(cwd=str(tmp_path), model="model", yolo=True)

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
            self.session = Session(cwd=str(tmp_path), model="model")

    loop = AgentLoop(FakeAgent(), output_fn=lambda message: None)
    completer = loop._command_completer()

    slash_completions = list(completer.get_completions(Document("/"), CompleteEvent(completion_requested=True)))
    config_completions = list(completer.get_completions(Document("/con"), CompleteEvent(completion_requested=True)))
    set_key_completions = list(completer.get_completions(Document("/set main."), CompleteEvent(completion_requested=True)))
    set_bool_completions = list(completer.get_completions(Document("/set main.reasoning "), CompleteEvent(completion_requested=True)))
    set_effort_completions = list(completer.get_completions(Document("/set main.effort h"), CompleteEvent(completion_requested=True)))

    assert "/help" in [completion.text for completion in slash_completions]
    assert "/config" in [completion.text for completion in config_completions]
    assert "main.reasoning" in [completion.text for completion in set_key_completions]
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
            self.session = Session(cwd=str(tmp_path), model="model")

    outputs = []
    answers = iter(["do not edit generated files"])
    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: next(answers), output_fn=outputs.append)

    result = loop._wait_confirm("Proceed?", default=True)

    assert result == "do not edit generated files"
    assert outputs == ["Answer: no - do not edit generated files"]


def test_agent_loop_confirmation_discards_pending_tty_input(tmp_path, monkeypatch):
    class FakeAgent:
        def __init__(self):
            self.session = Session(cwd=str(tmp_path), model="model")

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
            self.session = Session(cwd=str(tmp_path), model="model")
            self.blackboard = Blackboard()
            self.runs = []

        def run(self, user_input, *, confirm=None, on_auto_approve=None, on_message=None, stop_after_learn=False):
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
    assert any("main: model reasoning=medium stream=on" in output for output in outputs)
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
