from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
import time

import nanocode
from nanocode import AgentLoop, CommandLexer, Config, ConfigFile, Blackboard, ParsedToolCall, RuntimeSettings, Session, StatusBar, ToolCallDisplayFormatter


def make_session(tmp_path, *, model: str = "", compact_at: int = 80, yolo: bool = False) -> Session:
    data = {
        "provider": {"active": "default", "default": {"model": model}},
        "paths": {"data_dir": str(tmp_path / ".nanocode")},
        "runtime": {"compact_at": compact_at},
    }
    return Session(cwd=str(tmp_path), config=Config.from_dict(data), settings=RuntimeSettings.from_dict(data, yolo=yolo))


def _status_text(bar: StatusBar) -> str:
    return "".join(text for _, text in bar._fragments(0.0, now=time.monotonic(), show_sweep=False, show_elapsed=False))


def test_session_reports_missing_required_config(tmp_path):
    session = Session(cwd=str(tmp_path))

    assert session.missing_required_config() == ["provider.url", "provider.key", "provider.model"]

    session.config.provider.url = "url"
    session.config.provider.key = "key"
    session.config.provider.model = "model"

    assert session.missing_required_config() == []


def test_session_loads_user_rules_from_project_file(tmp_path, monkeypatch):
    data_dir = tmp_path / "nanocode-home"
    project_key = Session(cwd=str(tmp_path), config=Config(data_dir=str(data_dir))).project_key()
    rules_dir = data_dir / "projects" / project_key
    rules_dir.mkdir(parents=True)
    (rules_dir / "user_rules.md").write_text("# User Rules\n\n- Prompt-only changes do not need tests.\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    session = Session.from_config_data(
        {
            "provider": {"active": "default", "default": {"url": "url", "key": "key", "model": "model"}},
            "paths": {"data_dir": str(data_dir)},
        }
    )

    assert session.state.user_rules.format() == "# User Rules\n\n- Prompt-only changes do not need tests."


def test_runtime_settings_loads_yolo_from_config():
    data = {"runtime": {"yolo": True}}

    settings = RuntimeSettings.from_dict(data)

    assert settings.yolo is True
    assert not hasattr(settings, "plan_mode")


def test_runtime_settings_loads_auto_clean_recent():
    settings = RuntimeSettings.from_dict({"runtime": {"auto_clean_recent": "12h"}})
    disabled = RuntimeSettings.from_dict({"runtime": {"auto_clean_recent": "off"}})

    assert settings.auto_clean_recent == "12h"
    assert RuntimeSettings.clean_retention_seconds(settings.auto_clean_recent) == 12 * 3600
    assert disabled.auto_clean_recent == "off"
    assert RuntimeSettings.clean_retention_seconds(disabled.auto_clean_recent) == 0


def test_init_config_file_writes_default_toml(tmp_path):
    config_path = tmp_path / "config.toml"

    created_path, created = ConfigFile.init(str(config_path))
    second_path, second_created = ConfigFile.init(str(config_path))
    config = ConfigFile.load(str(config_path))

    assert created_path == str(config_path)
    assert created is True
    assert second_path == str(config_path)
    assert second_created is False
    assert config["provider"]["active"] == "default"
    assert config["provider"]["default"]["url"] == ""
    assert "available_models" not in config["provider"]["default"]
    assert "temperature" not in config["provider"]["default"]
    assert config["provider"]["default"]["reasoning"] == "medium"
    assert "chat_reasoning" not in config["provider"]["default"]
    assert config["provider"]["default"]["timeout"] == 180
    assert config["provider"]["default"]["first_token_timeout"] == 90
    assert config["runtime"]["compact_at"] == 80
    assert config["runtime"]["context_budget"] == "medium"
    assert config["runtime"]["auto_clean_recent"] == "1d"
    assert config["runtime"]["yolo"] is False
    assert "plan_timeout" not in config["runtime"]
    assert "plan_first_token_timeout" not in config["runtime"]
    assert "plan_mode" not in config["runtime"]


def test_main_init_config_uses_config_argument(tmp_path, capsys):
    config_path = tmp_path / "custom.toml"

    result = nanocode.main(["--config", str(config_path), "--init-config"])
    output = capsys.readouterr()

    assert result == 0
    assert config_path.exists()
    assert "Created config: " + str(config_path) in output.out


def test_main_rejects_plan_argument(capsys):
    try:
        nanocode.main(["--plan"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("--plan should be rejected by argparse")

    output = capsys.readouterr()
    assert "unrecognized arguments: --plan" in output.err


def test_main_loads_config_argument(tmp_path, monkeypatch):
    config_path = tmp_path / "custom.toml"
    config_path.write_text(
        """
[provider]
active = "custom"

[provider.custom]
url = "https://example.test/v1"
key = "key"
model = "custom-model"
available_models = ["custom-model", "other-model"]

[paths]
data_dir = ".custom-nanocode"
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
    assert sessions[0].config.provider.url == "https://example.test/v1"
    assert sessions[0].config.provider.key == "key"
    assert sessions[0].config.provider.model == "custom-model"
    assert sessions[0].config.provider.available_models == ("custom-model", "other-model")
    assert sessions[0].config.data_dir == ".custom-nanocode"
    assert not hasattr(sessions[0].settings, "plan_mode")


def test_status_bar_text_has_visible_sweep_marker(tmp_path):
    session = make_session(tmp_path, model="provider/model", compact_at=9)
    session.state.last_total_tokens = 42
    session.state.session_total_tokens = 1200
    session.state.turn_tool_calls = 3
    bar = StatusBar(session)

    fragments = bar._fragments(1.2, now=1.0, show_sweep=True, show_elapsed=True)
    text = "".join(text for _, text in fragments)

    assert ">" not in text
    assert "model (medium)" in text
    assert "ctx:0%" in text
    assert "tool:3" in text
    assert "tok:last:42 sess:1k" in text
    assert "turn:1s" in text
    assert all(style.startswith("#") for style, _ in fragments)
    assert len({style for style, _ in fragments}) > 3
    snapshot = _status_text(bar)
    assert snapshot == "model (medium) | ctx:0% | tool:3 | tok:last:42 sess:1k"
    assert ">" not in snapshot


def test_status_bar_hides_current_model_call_timer(tmp_path):
    session = make_session(tmp_path, model="provider/model")
    session.state.turn_model_calls = 2
    session.state.current_model_call_started_at = 0.4
    session.state.current_model_call_label = "provider/active-model"
    session.state.current_model_call_reasoning_label = "low"
    session.state.current_model_call_activity = "agent"
    bar = StatusBar(session)

    text = "".join(text for _, text in bar._fragments(0.0, now=1.0, show_sweep=True, show_elapsed=True))

    assert "active-model (low)" in text
    assert "working(" not in text

    session.state.current_model_call_has_content = True
    session.state.current_model_call_streaming_chars = 24
    streamed = "".join(text for _, text in bar._fragments(74.2, now=1.0, show_sweep=True, show_elapsed=True))
    assert "working" not in streamed
    assert "10t/s" in streamed
    assert "turn:1m14s" in streamed
    session.state.current_model_call_has_content = False

    session.state.current_model_call_activity = "compact"
    assert "compacting(" not in "".join(text for _, text in bar._fragments(0.0, now=1.0, show_sweep=True, show_elapsed=True))


def test_status_bar_shows_active_modes(tmp_path):
    session = make_session(tmp_path, model="provider/model", yolo=True)
    bar = StatusBar(session)

    assert _status_text(bar) == "model (medium) | yolo | ctx:0% | tool:0 | tok:last:- sess:-"


def test_status_bar_shows_recent_status_notice(tmp_path):
    session = make_session(tmp_path, model="provider/model")
    session.state.status_notice = "err:format"
    session.state.status_notice_until = time.monotonic() + 5
    bar = StatusBar(session)

    assert "model (medium) | err:format | ctx:" in _status_text(bar)

    session.state.status_notice_until = 0

    assert "err:format" not in _status_text(bar)


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
    keyed_segments = loop._tool_segments('[success] Search "sse|feed|history" glob=*.py path=. -> tr.2 | excerpt')

    assert ("ansigreen", "Read sample.txt 0:1\n") in segments
    assert all("ok " not in text for _, text in segments)
    assert ("ansigreen", 'Search "sse|feed|history" glob=*.py path=.') in keyed_segments
    assert ("ansibrightblack", " -> tr.2") in keyed_segments
    assert ("ansibrightblack", " | excerpt") in keyed_segments


def test_tool_call_display_formats_structured_args_for_humans():
    read = ParsedToolCall(
        name="Read",
        intention="",
        args=[{"files": [{"path": "one.py", "range": [0, 10]}, {"path": "two.py", "ranges": [[20, 30], [40, 50]]}]}],
    )
    search = ParsedToolCall(
        name="Search",
        intention="",
        args=[{"pattern": "class Foo", "path": ".", "glob": "*.py", "context": 2}],
    )
    inspect_code = ParsedToolCall(
        name="InspectCode",
        intention="",
        args=["find", "Tool", {"kind": "class", "limit": 20, "exact_only": True}],
    )

    assert ToolCallDisplayFormatter.format_call(read) == "Read one.py 0:10 two.py 20:30 40:50"
    assert ToolCallDisplayFormatter.format_call(search) == 'Search "class Foo" path=. glob=*.py context=2'
    assert ToolCallDisplayFormatter.format_call(inspect_code) == "InspectCode find Tool kind=class limit=20 exact_only=true"


def test_tool_call_report_compacts_interrupted_bash_result():
    output = "<BashToolResult>\n* exit_code: -1\n* interrupted: true\n* reason: user_ctrl_c\n</BashToolResult>"

    assert ToolCallDisplayFormatter._compact_tool_error(output) == "interrupted by user"


def test_agent_loop_indents_top_level_tool_report(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")

    captured = []
    loop = AgentLoop(FakeAgent(), output_fn=captured.append)

    loop._print_message("[success] Read sample.txt 0:1")

    assert captured == ["  Read sample.txt 0:1"]


def test_agent_loop_renders_tool_result_context_as_weak_status(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")

    captured = []
    loop = AgentLoop(FakeAgent(), output_fn=captured.append)

    loop._print_message("Tool Result Context: +tr.12 +tr.15")

    assert captured == ["  ctx: +tr.12 +tr.15"]


def test_agent_loop_styles_compact_state_section_labels(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")

    loop = AgentLoop(FakeAgent(), output_fn=lambda message: None)

    segments = loop._compact_state_segments("Leads + Facts Updated\nLeads\n  1. h1\nFacts\n  1. fact")

    assert ("bold ansicyan", "Leads + Facts Updated\n") in segments
    assert ("ansicyan", "Leads\n") in segments
    assert ("ansicyan", "Facts\n") in segments


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
        EFFECT = nanocode.ToolEffect.EDIT

        def preview(self):
            return "preview"

    outputs = []
    loop = AgentLoop(FakeAgent(), output_fn=outputs.append)
    call = ParsedToolCall(name="Edit", intention="edit sample", args=["sample.txt", [{"op": "replace", "start": "0:abcdef", "end": "0:abcdef", "content": "new\n"}]])

    loop._show_auto_tool_call(call, FakeTool())

    assert any("Auto Tool Call | auto approved" in output for output in outputs)
    assert any('Run     Edit("sample.txt", ' in output for output in outputs)
    assert any("Why     edit sample" in output for output in outputs)
    assert any("Preview\npreview" in output for output in outputs)


def test_agent_loop_command_completer_matches_slash_commands():
    completer = nanocode.CommandCompleter([])

    slash_completions = list(completer.get_completions(Document("/"), CompleteEvent(completion_requested=True)))
    config_completions = list(completer.get_completions(Document("/con"), CompleteEvent(completion_requested=True)))
    set_key_completions = list(completer.get_completions(Document("/set provider."), CompleteEvent(completion_requested=True)))
    set_reasoning_completions = list(completer.get_completions(Document("/set provider.reasoning h"), CompleteEvent(completion_requested=True)))
    set_chat_reasoning_completions = list(completer.get_completions(Document("/set provider.chat_reasoning rea"), CompleteEvent(completion_requested=True)))
    model_completions = list(nanocode.CommandCompleter(models=["qwen3", "deepseek"]).get_completions(Document("/model q"), CompleteEvent(completion_requested=True)))
    api_completions = list(completer.get_completions(Document("/api r"), CompleteEvent(completion_requested=True)))
    reason_payload_completions = list(completer.get_completions(Document("/reason-payload rea"), CompleteEvent(completion_requested=True)))

    assert "/help" in [completion.text for completion in slash_completions]
    assert "/api" in [completion.text for completion in slash_completions]
    assert "/reason-payload" in [completion.text for completion in slash_completions]
    assert "/plan" not in [completion.text for completion in slash_completions]
    assert "/config" in [completion.text for completion in config_completions]
    assert "provider.reasoning" in [completion.text for completion in set_key_completions]
    assert [completion.text for completion in set_reasoning_completions] == ["high"]
    assert [completion.text for completion in set_chat_reasoning_completions] == ["reasoning", "reasoning_effort"]
    assert [completion.text for completion in model_completions] == ["qwen3"]
    assert [completion.text for completion in api_completions] == ["responses"]
    assert [completion.text for completion in reason_payload_completions] == ["reasoning", "reasoning_effort"]


def test_command_lexer_highlights_known_command_prefix_only():
    lexer = CommandLexer()

    removed = lexer.lex_document(Document("/plan how?"))(0)
    unknown = lexer.lex_document(Document("/somecommand"))(0)
    spaced = lexer.lex_document(Document(" /plan how?"))(0)

    assert removed == [("", "/plan how?")]
    assert unknown == [("", "/somecommand")]
    assert spaced == [("", " /plan how?")]


def test_agent_loop_command_completer_completes_provider_names():
    completer = nanocode.CommandCompleter(["qwen", "openai"])
    event = CompleteEvent(completion_requested=True)

    q_completions = list(completer.get_completions(Document("/provider q"), event))
    o_completions = list(completer.get_completions(Document("/provider o"), event))
    all_completions = list(completer.get_completions(Document("/provider "), event))

    assert [c.text for c in q_completions] == ["qwen"]
    assert [c.text for c in o_completions] == ["openai"]
    assert {c.text for c in all_completions} == {"qwen", "openai"}


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


def test_model_retry_shortcut_signal_only_retries_active_model_request(tmp_path):
    session = make_session(tmp_path, model="model")
    shortcut = nanocode.ModelRetryShortcut(session)

    shortcut._handle_signal(0, None)

    assert session.state.manual_model_retry_requested is False

    session.state.current_model_call_started_at = 1.0
    try:
        shortcut._handle_signal(0, None)
    except KeyboardInterrupt:
        interrupted = True
    else:
        interrupted = False

    assert interrupted is True
    assert session.state.manual_model_retry_requested is True


def test_agent_loop_dispatches_commands_and_user_input(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")
            self.blackboard = Blackboard()
            self.runs = []

        def run(self, user_input, *, confirm=None, on_auto_approve=None, on_message=None, poll_user_input=None):
            self.runs.append(user_input)
            if on_message is not None:
                on_message("assistant response")
            return {"actions": [], "_assistant_text": "assistant response"}

    inputs = iter(["/status", "hello", "/exit"])
    outputs = []
    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: next(inputs), output_fn=outputs.append)

    result = loop.run()

    assert result == 0
    assert any("nanocode - AI coding assistant" in output for output in outputs)
    assert any("model: model api=chat(auto) reasoning=medium(off) stream=on" in output for output in outputs)
    assert "assistant response" in outputs
    assert loop.agent.runs == ["hello"]


def test_agent_loop_welcome_suggests_index_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode.CodeIndex, "status", lambda self: ("missing", ""))

    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")

    outputs = []
    AgentLoop(FakeAgent(), input_fn=lambda prompt: "", output_fn=outputs.append)._print_welcome()

    assert any("tip: /index initializes indexed code tools" in output for output in outputs)


def test_agent_loop_starts_existing_index_refresh_async(tmp_path, monkeypatch):
    refreshed = []

    def refresh_existing(self, *, progress=None):
        refreshed.append(progress is not None)
        if progress is not None:
            progress("file", done=1, total=2)
        return True

    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")
            self.blackboard = Blackboard()

    monkeypatch.setattr(nanocode.CodeIndex, "refresh_existing_async", refresh_existing)
    outputs = []
    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: "/exit", output_fn=outputs.append)

    assert loop.run() == 0
    assert refreshed == [True]
    assert loop.agent.session.state.status_notice == "index:parse 1/2"


def test_agent_loop_consumes_queued_input_before_prompt(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")
            self.blackboard = Blackboard()
            self.runs = []

        def run(self, user_input, **kwargs):
            self.runs.append(user_input)

    inputs = iter(["/exit"])
    output = []
    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: next(inputs), output_fn=output.append)

    loop._append_queued_input(" queued message ")

    assert loop.run() == 0
    assert loop.agent.runs == ["queued message"]
    assert "sent: queued message" in output


def test_agent_loop_run_agent_uses_runtime_ui_without_status_thread(tmp_path, monkeypatch):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")
            self.blackboard = Blackboard()
            self.runs = []
            self.poll_user_input = None

        def run(self, user_input, **kwargs):
            self.runs.append(user_input)
            self.poll_user_input = kwargs["poll_user_input"]

    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: "", output_fn=lambda message: None)
    calls = []
    monkeypatch.setattr(loop, "_start_runtime_ui", lambda: calls.append("start-ui") or True)
    monkeypatch.setattr(loop, "_stop_runtime_ui", lambda: calls.append("stop-ui") or True)
    monkeypatch.setattr(loop.status_bar, "reset_timer", lambda: calls.append("reset"))
    monkeypatch.setattr(loop.status_bar, "resume", lambda: calls.append("resume"))
    monkeypatch.setattr(loop.status_bar, "pause", lambda: calls.append("pause"))
    monkeypatch.setattr(nanocode.CodeIndex, "update_pending", lambda self: calls.append("index"))

    loop._run_agent("hello")

    assert loop.agent.runs == ["hello"]
    assert loop.agent.poll_user_input.__self__ is loop
    assert loop.agent.poll_user_input.__func__ is AgentLoop._pop_queued_input
    assert calls == ["reset", "start-ui", "stop-ui", "index", "pause"]


def test_agent_loop_clears_queued_input_on_cancel(tmp_path, monkeypatch):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")
            self.blackboard = Blackboard()

        def run(self, user_input, **kwargs):
            raise KeyboardInterrupt

        def cancel_current_goal(self):
            pass

    output = []
    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: "", output_fn=output.append)
    monkeypatch.setattr(loop, "_start_runtime_ui", lambda: False)
    loop._append_queued_input("queued message")

    loop._run_agent("hello")

    assert loop._pop_queued_input() is None
    assert "queued cleared: 1" in output


def test_agent_loop_runtime_ui_empty_enter_only_refreshes(tmp_path, monkeypatch):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")

    class FakePromptApp:
        def __init__(self):
            self.invalidated = 0
            self.background_tasks = []

        def invalidate(self):
            self.invalidated += 1

        def create_background_task(self, task):
            self.background_tasks.append(task)

    class FakeEvent:
        def __init__(self, app):
            self.app = app

    def handler(bindings, key):
        return next(binding.handler for binding in bindings.bindings if binding.keys == (key,))

    prompt_app = FakePromptApp()

    class FakeApplication:
        def __init__(self, **kwargs):
            self.bindings = kwargs["key_bindings"]

        def run(self, handle_sigint=False):
            handler(self.bindings, nanocode.Keys.ControlM)(FakeEvent(prompt_app))

    terminal_calls = []
    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: "", output_fn=lambda message: None)
    monkeypatch.setattr(nanocode, "Application", FakeApplication)
    monkeypatch.setattr(nanocode, "run_in_terminal", lambda *args, **kwargs: terminal_calls.append((args, kwargs)))

    loop._run_runtime_ui()

    assert loop._pop_queued_input() is None
    assert prompt_app.invalidated == 1
    assert prompt_app.background_tasks == []
    assert terminal_calls == []


def test_agent_loop_runtime_ui_pause_restarts_for_confirm(tmp_path, monkeypatch):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")

    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: "", output_fn=lambda message: None)
    calls = []
    monkeypatch.setattr(loop, "_stop_runtime_ui", lambda: calls.append("stop-ui") or True)
    monkeypatch.setattr(loop, "_start_runtime_ui", lambda: calls.append("start-ui") or True)
    monkeypatch.setattr(loop, "_with_status_paused", lambda action: action())
    monkeypatch.setattr(loop, "_print_tool_call_display", lambda *args, **kwargs: calls.append("display"))
    monkeypatch.setattr(loop, "_wait_confirm", lambda *args, **kwargs: True)

    result = loop._confirm_tool_call(ParsedToolCall("Edit", "edit", ["a", "b", "c"]), object())

    assert result is True
    assert calls == ["stop-ui", "display", "start-ui"]


def test_agent_loop_bash_live_preview_keeps_latest_lines(tmp_path, monkeypatch):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")

    class FakeApp:
        def __init__(self):
            self.invalidated = 0

        def invalidate(self):
            self.invalidated += 1

    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: "")
    app = FakeApp()
    loop._runtime_ui_app = app
    printed = []
    monkeypatch.setattr(nanocode, "print_formatted_text", lambda formatted, **kwargs: printed.append(list(formatted)))

    loop._show_tool_live_output("stdout", "\n".join("line" + str(index) for index in range(8)))

    assert app.invalidated == 1
    assert loop._has_tool_live_preview() is True
    assert loop._tool_live_preview_fragments() == [("class:bash-preview", "line2\nline3\nline4\nline5\nline6\nline7")]

    loop._show_tool_live_output("", "")

    assert app.invalidated == 2
    assert loop._has_tool_live_preview() is False
    assert printed == [[("ansibrightblack", "line2\nline3\nline4\nline5\nline6\nline7\n")]]


def test_agent_loop_runtime_interrupt_requests_sigint(tmp_path, monkeypatch):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")

    class FakeApp:
        def __init__(self):
            self.exited = False

        def exit(self):
            self.exited = True

    app = FakeApp()
    calls = []
    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: "", output_fn=lambda message: None)
    loop._runtime_ui_app = app
    monkeypatch.setattr(nanocode.os, "kill", lambda pid, sig: calls.append((pid, sig)))

    loop._interrupt_current_turn(exit_after=True)

    assert loop._exit_after_current_turn is True
    assert app.exited is True
    assert calls == [(nanocode.os.getpid(), nanocode.signal.SIGINT)]


def test_agent_loop_runtime_retry_requests_model_retry(tmp_path, monkeypatch):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="model")

    class FakeApp:
        def __init__(self):
            self.exited = False

        def exit(self):
            self.exited = True

    app = FakeApp()
    calls = []
    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: "", output_fn=lambda message: None)
    loop._runtime_ui_app = app
    loop.agent.session.state.current_model_call_started_at = 1.0
    monkeypatch.setattr(nanocode.os, "kill", lambda pid, sig: calls.append((pid, sig)))

    loop._retry_current_model_call()

    assert loop.agent.session.state.manual_model_retry_requested is True
    assert app.exited is False
    assert calls == [(nanocode.os.getpid(), nanocode.signal.SIGINT)]


def test_agent_loop_model_command_prompts_for_reasoning_effort(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="old")

    inputs = iter(["/model new-model", "5", "/exit"])
    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: next(inputs), output_fn=lambda message: None)

    assert loop.run() == 0
    assert loop.agent.session.config.provider.model == "new-model"
    assert loop.agent.session.config.provider.reasoning == "high"


def test_agent_loop_model_command_prompts_for_model_when_available(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="old")
            self.session.config.provider.available_models = ("old", "new-model")

    inputs = iter(["/model", "2", "", "/exit"])
    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: next(inputs), output_fn=lambda message: None)

    assert loop.run() == 0
    assert loop.agent.session.config.provider.model == "new-model"


def test_agent_loop_provider_command_prompts_for_provider(tmp_path):
    class FakeAgent:
        def __init__(self):
            data = {
                "provider": {
                    "active": "one",
                    "one": {"model": "model-one"},
                    "two": {"model": "model-two"},
                }
            }
            self.session = Session(cwd=str(tmp_path), config=Config.from_dict(data), settings=RuntimeSettings.from_dict(data))

    inputs = iter(["/provider", "2", "/exit"])
    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: next(inputs), output_fn=lambda message: None)

    assert loop.run() == 0
    assert loop.agent.session.config.active_provider == "two"


def test_agent_loop_model_command_can_keep_reasoning_effort(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="old")
            self.session.config.provider.reasoning = "xhigh"

    inputs = iter(["/model new-model", "", "/exit"])
    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: next(inputs), output_fn=lambda message: None)

    assert loop.run() == 0
    assert loop.agent.session.config.provider.model == "new-model"
    assert loop.agent.session.config.provider.reasoning == "xhigh"


def test_agent_loop_choice_prompt_styles_selected_effort_and_erases_when_done(tmp_path, monkeypatch):
    class FakeStdin:
        @staticmethod
        def isatty():
            return True

    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="old")

    captured = {}

    class FakeApplication:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return "low"

    monkeypatch.setattr(nanocode.sys, "stdin", FakeStdin())
    monkeypatch.setattr(nanocode, "Application", FakeApplication)

    loop = AgentLoop(FakeAgent(), prompt_session=object())

    assert loop._select_reasoning() == "low"
    attrs = captured["style"].get_attrs_for_style_str("class:selected-option")
    assert attrs.bgcolor == "e6f2f3"
    assert attrs.color == "0f4c5c"
    assert attrs.bold is True
    assert captured["erase_when_done"] is True
    assert captured["layout"] is not None


def test_agent_loop_choice_prompt_filters_with_slash_search(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="old")

    inputs = iter(["/remote", "1"])
    outputs = []
    loop = AgentLoop(FakeAgent(), input_fn=lambda prompt: next(inputs), output_fn=outputs.append)

    selected = loop._select_choice(
        "Model",
        (
            nanocode.CommandDispatcher.MODEL_CONFIGURED_LABEL,
            "old",
            "manual",
            nanocode.CommandDispatcher.MODEL_DISCOVERED_LABEL,
            "remote-a",
            "remote-b",
        ),
        disabled=set(nanocode.CommandDispatcher.MODEL_LABELS),
    )

    assert selected == "remote-a"
    assert "Model /remote:" in outputs[-1]
    assert "remote-a" in outputs[-1]
    assert "old" not in outputs[-1]


def test_agent_loop_choice_prompt_enter_confirms_search_before_select(tmp_path, monkeypatch):
    class FakeStdin:
        @staticmethod
        def isatty():
            return True

    class FakeAgent:
        def __init__(self):
            self.session = make_session(tmp_path, model="old")

    class FakePromptApp:
        result = None

        def invalidate(self):
            pass

        def exit(self, result=None, exception=None):
            if exception is not None:
                raise exception
            self.result = result

    def handler(bindings, key):
        return next(binding.handler for binding in bindings.bindings if binding.keys == (key,))

    class FakeEvent:
        def __init__(self, app, data=""):
            self.app = app
            self.data = data

    class FakeApplication:
        def __init__(self, **kwargs):
            self.bindings = kwargs["key_bindings"]

        def run(self):
            app = FakePromptApp()
            handler(self.bindings, "/")(FakeEvent(app, "/"))
            any_key = handler(self.bindings, nanocode.Keys.Any)
            for char in "remote":
                any_key(FakeEvent(app, char))
            enter = handler(self.bindings, nanocode.Keys.ControlM)
            enter(FakeEvent(app, "\r"))
            assert app.result is None
            enter(FakeEvent(app, "\r"))
            return app.result

    monkeypatch.setattr(nanocode.sys, "stdin", FakeStdin())
    monkeypatch.setattr(nanocode, "Application", FakeApplication)

    loop = AgentLoop(FakeAgent(), prompt_session=object())

    assert loop._select_choice("Model", ("old", "remote-a", "remote-b"), current="old") == "remote-a"


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
    assert callable(kwargs["bottom_toolbar"])
    assert "".join(text for _, text in kwargs["bottom_toolbar"]()) == (
        "model (medium) | ctx:0% | tool:0 | tok:last:- sess:-"
    )
