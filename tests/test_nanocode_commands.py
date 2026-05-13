import os

import shutil

from nanocode import Config, MainAgent, CommandDispatcher, CommandStatus, ModelUsage, RuntimeSettings, Session, UserMessage


class FakeModelClient:
    def __init__(self, summary="LLM compact summary"):
        self.summary = summary
        self.requests = []

    def request(self, system_prompt, user_prompt, *, activity="main"):
        self.requests.append((system_prompt, user_prompt, activity))
        return {"summary": self.summary}


def make_session(tmp_path, *, model: str = "", stream: bool | None = None, compact_at: int = 50) -> Session:
    data = {"main_model": {"model": model}, "runtime": {"compact_at": compact_at}}
    if stream is not None:
        data["main_model"]["stream"] = stream
    return Session(cwd=str(tmp_path), config=Config.from_dict(data), settings=RuntimeSettings.from_dict(data))


def test_command_dispatcher_updates_config_and_auto_compacts(tmp_path):
    session = make_session(tmp_path, model="old", compact_at=100)
    agent = MainAgent(session)
    fake_client = FakeModelClient()
    agent.compactor.model_client = fake_client
    dispatcher = CommandDispatcher(agent)
    session.state.conversation = [UserMessage(content="one"), UserMessage(content="two"), UserMessage(content="three")]

    model_result = dispatcher.dispatch("/set main.model new-model")
    worker_model_result = dispatcher.dispatch("/set worker.model worker-model")
    effort_result = dispatcher.dispatch("/set main.effort high")
    reason_result = dispatcher.dispatch("/set main.reasoning off")
    stream_result = dispatcher.dispatch("/set main.stream off")
    first_token_result = dispatcher.dispatch("/set main.first_token_timeout 6")
    yolo_result = dispatcher.dispatch("/set runtime.yolo on")
    compact_result = dispatcher.dispatch("/set runtime.compact_at 2")
    exit_result = dispatcher.dispatch("/exit")

    assert model_result.status == CommandStatus.HANDLED
    assert session.config.main_model.model == "new-model"
    assert worker_model_result.message == "Set worker.model = worker-model"
    assert session.config.worker_model.model == "worker-model"
    assert effort_result.message == "Set main.effort = high"
    assert session.config.main_model.reasoning_effort == "high"
    assert reason_result.message == "Set main.reasoning = off"
    assert session.config.main_model.reasoning is False
    assert stream_result.message == "Set main.stream = off"
    assert session.config.main_model.stream is False
    assert first_token_result.message == "Set main.first_token_timeout = 6"
    assert session.config.main_model.first_token_timeout == 6
    assert yolo_result.message == "Set runtime.yolo = on"
    assert session.settings.yolo is True
    assert compact_result.message == "Set runtime.compact_at = 2"
    assert session.settings.compact_at == 2
    assert len(session.state.conversation) == 3
    assert fake_client.requests == []
    assert exit_result.status == CommandStatus.EXIT


def test_status_reports_tokens_in_human_readable_format(tmp_path):
    session = make_session(tmp_path, model="model")
    session.state.last_total_tokens = 1200
    session.state.session_total_tokens = 2_345_678
    session.state.model_usage["model"] = ModelUsage(calls=2, total_tokens=2_345_678)
    dispatcher = CommandDispatcher(MainAgent(session))

    result = dispatcher.dispatch("/status")

    assert result.status == CommandStatus.HANDLED
    assert "tokens: last=1k session=2m" in result.message
    assert "main: model reasoning=medium stream=on" in result.message
    assert "explore: turns=12" in result.message
    assert "runtime: yolo=off compact_at=50" in result.message
    assert "models:" in result.message
    assert "model: calls=2 tokens=2m" in result.message
    assert "tool_calls: turn=0 session=0" in result.message
    assert "blackboard" not in result.message


def test_set_command_shows_and_validates_runtime_config(tmp_path):
    session = make_session(tmp_path, stream=True)
    dispatcher = CommandDispatcher(MainAgent(session))

    status_result = dispatcher.dispatch("/set main.stream")
    off_result = dispatcher.dispatch("/set main.stream off")
    off_status_result = dispatcher.dispatch("/set main.stream")
    on_result = dispatcher.dispatch("/set main.stream on")
    invalid_result = dispatcher.dispatch("/set main.stream maybe")

    assert status_result.message == "Current main.stream is on"
    assert off_result.message == "Set main.stream = off"
    assert off_status_result.message == "Current main.stream is off"
    assert on_result.message == "Set main.stream = on"
    assert invalid_result.message == "Usage: /set main.stream [on|off]"
    assert session.config.main_model.stream is True


def test_config_command_reports_resolved_model_config(tmp_path):
    session = make_session(tmp_path, model="main-model")
    session.config.worker_model.model = "worker-model"
    dispatcher = CommandDispatcher(MainAgent(session))

    result = dispatcher.dispatch("/config")

    assert result.status == CommandStatus.HANDLED
    assert "config: " in result.message
    assert "main.model: main-model" in result.message
    assert "main.first_token_timeout: 60" in result.message
    assert "worker.model: worker-model" in result.message
    assert "worker.first_token_timeout: 60" in result.message
    assert "explore.max_turns: 12" in result.message


def test_blackboard_command_is_not_registered(tmp_path):
    dispatcher = CommandDispatcher(MainAgent(Session(cwd=str(tmp_path))))

    result = dispatcher.dispatch("/blackboard")

    assert result.status == CommandStatus.UNHANDLED
    assert result.message == ""


def test_rules_command_shows_rules_content(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.state.user_rules.add("Prompt-only changes do not need tests.")
    dispatcher = CommandDispatcher(MainAgent(session))

    result = dispatcher.dispatch("/rules")

    assert result.status == CommandStatus.HANDLED
    assert result.message == "# User Rules\n\n- Prompt-only changes do not need tests."


def test_knowledge_command_shows_stable_knowledge(tmp_path):
    agent = MainAgent(Session(cwd=str(tmp_path)))
    dispatcher = CommandDispatcher(agent)

    empty_result = dispatcher.dispatch("/knowledge")
    usage_result = dispatcher.dispatch("/knowledge extra")
    agent.blackboard.stable_knowledge = {
        "workflow": ["Project test command is make test."],
        "structure": ["Main runtime lives in nanocode.py."],
    }
    result = dispatcher.dispatch("/knowledge")

    assert empty_result.message == "No stable knowledge stored."
    assert usage_result.message == "Usage: /knowledge"
    assert result.status == CommandStatus.HANDLED
    assert result.message == "\n".join(
        [
            "Stable knowledge:",
            "structure:",
            "- Main runtime lives in nanocode.py.",
            "workflow:",
            "- Project test command is make test.",
        ]
    )


def test_command_dispatcher_auto_compacts_only_when_history_exceeds_keep_recent(tmp_path):
    session = make_session(tmp_path, compact_at=2)
    agent = MainAgent(session)
    agent.compactor.model_client = FakeModelClient()
    dispatcher = CommandDispatcher(agent)
    session.state.conversation = [
        UserMessage(content="old"),
        UserMessage(content="keep 1"),
        UserMessage(content="keep 2"),
        UserMessage(content="keep 3"),
        UserMessage(content="keep 4"),
        UserMessage(content="keep 5"),
    ]

    result = dispatcher.dispatch("/set runtime.compact_at 2")

    assert result.message == "Set runtime.compact_at = 2 and compacted history"
    assert len(session.state.conversation) == 6
    assert session.state.conversation[0].content == "Conversation compact summary:\nLLM compact summary"
    assert session.state.conversation[1].content == "keep 1"


def test_command_dispatcher_runs_compact_with_status_runner(tmp_path):
    session = make_session(tmp_path, compact_at=2)
    agent = MainAgent(session)
    agent.compactor.model_client = FakeModelClient()
    session.state.conversation = [
        UserMessage(content="old"),
        UserMessage(content="keep 1"),
        UserMessage(content="keep 2"),
        UserMessage(content="keep 3"),
        UserMessage(content="keep 4"),
        UserMessage(content="keep 5"),
    ]
    status_calls = []

    def run_with_status(action):
        status_calls.append("run")
        return action()

    dispatcher = CommandDispatcher(agent, run_with_status=run_with_status)

    result = dispatcher.dispatch("/compact")

    assert result.status == CommandStatus.HANDLED
    assert result.message == "Compacted conversation history: 6 item(s) -> 6 item(s)"
    assert status_calls == ["run"]
    assert session.state.conversation[0].content == "Conversation compact summary:\nLLM compact summary"


def test_command_dispatcher_auto_compact_uses_status_runner(tmp_path):
    session = make_session(tmp_path, compact_at=100)
    agent = MainAgent(session)
    agent.compactor.model_client = FakeModelClient()
    session.state.conversation = [
        UserMessage(content="old"),
        UserMessage(content="keep 1"),
        UserMessage(content="keep 2"),
        UserMessage(content="keep 3"),
        UserMessage(content="keep 4"),
        UserMessage(content="keep 5"),
    ]
    status_calls = []
    dispatcher = CommandDispatcher(agent, run_with_status=lambda action: status_calls.append("run") or action())

    result = dispatcher.dispatch("/set runtime.compact_at 2")

    assert result.message == "Set runtime.compact_at = 2 and compacted history"
    assert status_calls == ["run"]
    assert session.state.conversation[0].content == "Conversation compact summary:\nLLM compact summary"


def test_command_dispatcher_reports_unhandled_input(tmp_path):
    dispatcher = CommandDispatcher(MainAgent(Session(cwd=str(tmp_path))))

    result = dispatcher.dispatch("regular user request")

    assert result.status == CommandStatus.UNHANDLED
    assert result.message == ""


def test_help_question_runs_agent_with_source_aware_prompt(tmp_path):
    prompts = []
    dispatcher = CommandDispatcher(MainAgent(Session(cwd=str(tmp_path))), run_agent=prompts.append)

    result = dispatcher.dispatch("/help how does compact work?")

    assert result.status == CommandStatus.HANDLED
    assert result.message == ""
    assert len(prompts) == 1


def test_clean_logs_command_removes_log_files(tmp_path):
    session = Session(cwd=str(tmp_path))
    tool_results_dir = session.tool_results_dir()
    os.makedirs(tool_results_dir, exist_ok=True)

    # Create some log files and a non-log file
    log1 = os.path.join(tool_results_dir, "test1.log")
    log2 = os.path.join(tool_results_dir, "test2.log")
    other = os.path.join(tool_results_dir, "other.txt")
    with open(log1, "w"):
        pass
    with open(log2, "w"):
        pass
    with open(other, "w"):
        pass

    dispatcher = CommandDispatcher(MainAgent(session))
    result = dispatcher.dispatch("/clean-logs")

    assert result.status == CommandStatus.HANDLED
    assert "Cleaned 2 log file(s)" in result.message
    assert not os.path.exists(log1)
    assert not os.path.exists(log2)
    assert os.path.exists(other)


def test_clean_logs_command_no_directory(tmp_path):
    session = Session(cwd=str(tmp_path))
    # Ensure tool_results_dir does not exist
    tool_results_dir = session.tool_results_dir()
    if os.path.exists(tool_results_dir):
        shutil.rmtree(tool_results_dir)

    dispatcher = CommandDispatcher(MainAgent(session))
    result = dispatcher.dispatch("/clean-logs")

    assert result.status == CommandStatus.HANDLED
    assert "No tool_results directory found" in result.message


def test_clean_logs_command_empty_directory(tmp_path):
    session = Session(cwd=str(tmp_path))
    tool_results_dir = session.tool_results_dir()
    os.makedirs(tool_results_dir, exist_ok=True)

    dispatcher = CommandDispatcher(MainAgent(session))
    result = dispatcher.dispatch("/clean-logs")

    assert result.status == CommandStatus.HANDLED
    assert "Cleaned 0 log file(s)" in result.message


def test_clean_logs_command_with_args_returns_usage(tmp_path):
    session = Session(cwd=str(tmp_path))
    tool_results_dir = session.tool_results_dir()
    os.makedirs(tool_results_dir, exist_ok=True)

    dispatcher = CommandDispatcher(MainAgent(session))
    result = dispatcher.dispatch("/clean-logs extra-arg")

    assert result.status == CommandStatus.HANDLED
    assert result.message == "Usage: /clean-logs"


def test_clean_logs_command_reports_failed_deletions(tmp_path):
    session = Session(cwd=str(tmp_path))
    tool_results_dir = session.tool_results_dir()
    os.makedirs(tool_results_dir, exist_ok=True)

    # Create two log files
    log1 = os.path.join(tool_results_dir, "good.log")
    log2 = os.path.join(tool_results_dir, "fail.log")
    with open(log1, "w"):
        pass
    with open(log2, "w"):
        pass

    # Mock os.remove to fail on the second file
    original_remove = os.remove
    call_count = [0]

    def mock_remove(path):
        call_count[0] += 1
        if call_count[0] == 2:
            raise OSError("Permission denied")
        original_remove(path)

    import unittest.mock
    with unittest.mock.patch("os.remove", side_effect=mock_remove):
        dispatcher = CommandDispatcher(MainAgent(session))
        result = dispatcher.dispatch("/clean-logs")

    assert result.status == CommandStatus.HANDLED
    assert "Cleaned 1 log file(s)" in result.message
    assert "1 failed" in result.message
