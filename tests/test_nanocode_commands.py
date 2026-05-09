import os

import shutil

from nanocode import MainAgent, CommandDispatcher, CommandStatus, ModelUsage, Session, UserMessage


class FakeModelClient:
    def __init__(self, summary="LLM compact summary"):
        self.summary = summary
        self.requests = []

    def request(self, system_prompt, user_prompt, *, activity="main"):
        self.requests.append((system_prompt, user_prompt, activity))
        return {"summary": self.summary}


def test_command_dispatcher_updates_config_and_auto_compacts(tmp_path):
    session = Session(cwd=str(tmp_path), model="old", compact_at=100)
    agent = MainAgent(session)
    fake_client = FakeModelClient()
    agent.compactor.model_client = fake_client
    dispatcher = CommandDispatcher(agent)
    session.conversation = [UserMessage(content="one"), UserMessage(content="two"), UserMessage(content="three")]

    model_result = dispatcher.dispatch("/set main.model new-model")
    worker_model_result = dispatcher.dispatch("/set worker.model worker-model")
    effort_result = dispatcher.dispatch("/set main.effort high")
    reason_result = dispatcher.dispatch("/set main.reasoning off")
    stream_result = dispatcher.dispatch("/set main.stream off")
    yolo_result = dispatcher.dispatch("/set runtime.yolo on")
    compact_result = dispatcher.dispatch("/set runtime.compact_at 2")
    exit_result = dispatcher.dispatch("/exit")

    assert model_result.status == CommandStatus.HANDLED
    assert session.model == "new-model"
    assert worker_model_result.message == "Set worker.model = worker-model"
    assert session.worker_model_config.model == "worker-model"
    assert effort_result.message == "Set main.effort = high"
    assert session.reasoning_effort == "high"
    assert reason_result.message == "Set main.reasoning = off"
    assert session.reasoning is False
    assert stream_result.message == "Set main.stream = off"
    assert session.stream is False
    assert yolo_result.message == "Set runtime.yolo = on"
    assert session.yolo is True
    assert compact_result.message == "Set runtime.compact_at = 2"
    assert session.compact_at == 2
    assert len(session.conversation) == 3
    assert fake_client.requests == []
    assert exit_result.status == CommandStatus.EXIT


def test_status_reports_tokens_in_human_readable_format(tmp_path):
    session = Session(cwd=str(tmp_path), model="model")
    session.last_total_tokens = 1200
    session.session_total_tokens = 2_345_678
    session.last_cost = 0.000008
    session.session_cost = 12.345678
    session.model_usage["model"] = ModelUsage(calls=2, total_tokens=2_345_678, cost=12.345678)
    dispatcher = CommandDispatcher(MainAgent(session))

    result = dispatcher.dispatch("/status")

    assert result.status == CommandStatus.HANDLED
    assert "tokens: last=1k session=2m" in result.message
    assert "cost: last=$0.000008 session=$12.345678" in result.message
    assert "main: model reasoning=medium stream=on" in result.message
    assert "explore: turns=50" in result.message
    assert "runtime: yolo=off compact_at=50" in result.message
    assert "models:" in result.message
    assert "model: calls=2 tokens=2m" in result.message
    assert "tool_calls: 0" in result.message
    assert "blackboard" not in result.message


def test_set_command_shows_and_validates_runtime_config(tmp_path):
    session = Session(cwd=str(tmp_path), stream=True)
    dispatcher = CommandDispatcher(MainAgent(session))

    status_result = dispatcher.dispatch("/set main.stream")
    off_result = dispatcher.dispatch("/set main.stream off")
    off_status_result = dispatcher.dispatch("/set main.stream")
    on_result = dispatcher.dispatch("/set main.stream on")
    invalid_result = dispatcher.dispatch("/set main.stream maybe")

    assert status_result.message == "main.stream = on"
    assert off_result.message == "Set main.stream = off"
    assert off_status_result.message == "main.stream = off"
    assert on_result.message == "Set main.stream = on"
    assert invalid_result.message == "Usage: /set main.stream [on|off]"
    assert session.stream is True


def test_config_command_reports_resolved_model_config(tmp_path):
    session = Session(cwd=str(tmp_path), model="main-model")
    session.worker_model_config.model = "worker-model"
    dispatcher = CommandDispatcher(MainAgent(session))

    result = dispatcher.dispatch("/config")

    assert result.status == CommandStatus.HANDLED
    assert "config: " in result.message
    assert "main.model: main-model" in result.message
    assert "worker.model: worker-model" in result.message
    assert "explore.max_turns: 50" in result.message


def test_blackboard_command_is_not_registered(tmp_path):
    dispatcher = CommandDispatcher(MainAgent(Session(cwd=str(tmp_path))))

    result = dispatcher.dispatch("/blackboard")

    assert result.status == CommandStatus.UNHANDLED
    assert result.message == ""


def test_learn_command_dispatches_default_learning_task(tmp_path):
    calls = []
    dispatcher = CommandDispatcher(MainAgent(Session(cwd=str(tmp_path))), run_agent=calls.append)

    result = dispatcher.dispatch("/learn")

    assert result.status == CommandStatus.HANDLED
    assert result.message == ""
    assert calls == [
        "Learn stable project knowledge for this codebase. Focus on structure, architecture, workflows, and conventions; "
        "workflows include durable test/lint/build/release/verification commands; "
        "use explore as needed; update Project_Knowledge with durable high-level facts only; correct stale facts by exact text; "
        "do not store temporary task details, line numbers, or large code."
    ]


def test_learn_command_dispatches_scoped_learning_task(tmp_path):
    calls = []
    dispatcher = CommandDispatcher(MainAgent(Session(cwd=str(tmp_path))), run_agent=calls.append)

    result = dispatcher.dispatch("/learn test layout")

    assert result.status == CommandStatus.HANDLED
    assert result.message == ""
    assert calls == [
        "Learn stable project knowledge about: test layout. Focus on structure, architecture, workflows, and conventions; "
        "workflows include durable test/lint/build/release/verification commands; "
        "use explore as needed; update Project_Knowledge with durable high-level facts only; correct stale facts by exact text; "
        "do not store temporary task details, line numbers, or large code."
    ]


def test_command_dispatcher_auto_compacts_only_when_history_exceeds_keep_recent(tmp_path):
    session = Session(cwd=str(tmp_path), compact_at=2)
    agent = MainAgent(session)
    agent.compactor.model_client = FakeModelClient()
    dispatcher = CommandDispatcher(agent)
    session.conversation = [
        UserMessage(content="old"),
        UserMessage(content="keep 1"),
        UserMessage(content="keep 2"),
        UserMessage(content="keep 3"),
        UserMessage(content="keep 4"),
        UserMessage(content="keep 5"),
    ]

    result = dispatcher.dispatch("/set runtime.compact_at 2")

    assert result.message == "Set runtime.compact_at = 2 and compacted history"
    assert len(session.conversation) == 6
    assert session.conversation[0].content == "Conversation compact summary:\nLLM compact summary"
    assert session.conversation[1].content == "keep 1"


def test_command_dispatcher_runs_compact_with_status_runner(tmp_path):
    session = Session(cwd=str(tmp_path), compact_at=2)
    agent = MainAgent(session)
    agent.compactor.model_client = FakeModelClient()
    session.conversation = [
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
    assert session.conversation[0].content == "Conversation compact summary:\nLLM compact summary"


def test_command_dispatcher_auto_compact_uses_status_runner(tmp_path):
    session = Session(cwd=str(tmp_path), compact_at=100)
    agent = MainAgent(session)
    agent.compactor.model_client = FakeModelClient()
    session.conversation = [
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
    assert session.conversation[0].content == "Conversation compact summary:\nLLM compact summary"


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
    assert "Answer this question about nanocode itself." in prompts[0]
    assert "nanocode.py" in prompts[0]
    assert "pyproject.toml" in prompts[0]
    assert "how does compact work?" in prompts[0]


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
