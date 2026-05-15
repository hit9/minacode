import os

import shutil

from nanocode import Config, Agent, CommandDispatcher, CommandStatus, ModelUsage, RuntimeSettings, Session, UserMessage


class FakeModelClient:
    def __init__(self, summary="LLM compact summary"):
        self.summary = summary
        self.requests = []

    def request(self, system_prompt, user_prompt, *, activity="agent"):
        self.requests.append((system_prompt, user_prompt, activity))
        return {"summary": self.summary}


def make_session(tmp_path, *, model: str = "", stream: bool | None = None, compact_at: int = 50) -> Session:
    provider: dict[str, object] = {"model": model}
    if stream is not None:
        provider["stream"] = stream
    data = {
        "provider": {"active": "default", "default": provider},
        "paths": {"data_dir": str(tmp_path / ".nanocode")},
        "runtime": {"compact_at": compact_at},
    }
    return Session(cwd=str(tmp_path), config=Config.from_dict(data), settings=RuntimeSettings.from_dict(data))


def test_command_dispatcher_updates_config_and_auto_compacts(tmp_path):
    session = make_session(tmp_path, model="old", compact_at=100)
    agent = Agent(session)
    fake_client = FakeModelClient()
    agent.compactor.model_client = fake_client
    dispatcher = CommandDispatcher(agent)
    session.state.conversation = [UserMessage(content="one"), UserMessage(content="two"), UserMessage(content="three")]

    model_result = dispatcher.dispatch("/set provider.model new-model")
    url_result = dispatcher.dispatch("/set provider.url https://example.test/v1")
    key_result = dispatcher.dispatch("/set provider.key secret")
    effort_result = dispatcher.dispatch("/set provider.effort high")
    reason_result = dispatcher.dispatch("/set provider.reasoning off")
    stream_result = dispatcher.dispatch("/set provider.stream off")
    first_token_result = dispatcher.dispatch("/set provider.first_token_timeout 6")
    yolo_result = dispatcher.dispatch("/set runtime.yolo on")
    compact_result = dispatcher.dispatch("/set runtime.compact_at 2")
    exit_result = dispatcher.dispatch("/exit")

    assert model_result.status == CommandStatus.HANDLED
    assert session.config.provider.model == "new-model"
    assert url_result.message == "Set provider.url = https://example.test/v1"
    assert session.config.provider.url == "https://example.test/v1"
    assert key_result.message == "Set provider.key = (set)"
    assert session.config.provider.key == "secret"
    assert effort_result.message == "Set provider.effort = high"
    assert session.config.provider.reasoning_effort == "high"
    assert reason_result.message == "Set provider.reasoning = off"
    assert session.config.provider.reasoning is False
    assert stream_result.message == "Set provider.stream = off"
    assert session.config.provider.stream is False
    assert first_token_result.message == "Set provider.first_token_timeout = 6"
    assert session.config.provider.first_token_timeout == 6
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
    dispatcher = CommandDispatcher(Agent(session))

    result = dispatcher.dispatch("/status")

    assert result.status == CommandStatus.HANDLED
    assert "tokens: last=1k session=2m" in result.message
    assert "model: model reasoning=medium stream=on" in result.message
    assert "session: " + session.session_id in result.message
    assert "runtime: yolo=off plan=off compact_at=50" in result.message
    assert "models:" in result.message
    assert "model: calls=2 tokens=2m" in result.message
    assert "tool_calls: turn=0 session=0" in result.message
    assert "task: done" in result.message
    assert "blackboard" not in result.message


def test_set_command_shows_and_validates_runtime_config(tmp_path):
    session = make_session(tmp_path, stream=True)
    dispatcher = CommandDispatcher(Agent(session))

    url_status_result = dispatcher.dispatch("/set provider.url")
    key_status_result = dispatcher.dispatch("/set provider.key")
    status_result = dispatcher.dispatch("/set provider.stream")
    off_result = dispatcher.dispatch("/set provider.stream off")
    off_status_result = dispatcher.dispatch("/set provider.stream")
    on_result = dispatcher.dispatch("/set provider.stream on")
    invalid_result = dispatcher.dispatch("/set provider.stream maybe")

    assert url_status_result.message == "Usage: /set provider.url <value>"
    assert key_status_result.message == "Usage: /set provider.key <value>"
    assert status_result.message == "Current provider.stream is on"
    assert off_result.message == "Set provider.stream = off"
    assert off_status_result.message == "Current provider.stream is off"
    assert on_result.message == "Set provider.stream = on"
    assert invalid_result.message == "Usage: /set provider.stream [on|off]"
    assert session.config.provider.stream is True


def test_config_command_reports_resolved_provider_config(tmp_path):
    session = make_session(tmp_path, model="config-model")
    session.config.provider.available_models = ("config-model", "other-model")
    dispatcher = CommandDispatcher(Agent(session))

    result = dispatcher.dispatch("/config")

    assert result.status == CommandStatus.HANDLED
    assert "config: " in result.message
    assert "provider.active: default" in result.message
    assert "provider.model: config-model" in result.message
    assert "provider.available_models: config-model, other-model" in result.message
    assert "provider.first_token_timeout: 60" in result.message
    assert "paths.data_dir: " + str(tmp_path / ".nanocode") in result.message
    assert "paths.project_dir: " in result.message
    assert "paths.session_dir: " in result.message
    assert "paths.history: " + str(tmp_path / ".nanocode" / "history") in result.message
    assert "runtime.max_agent_steps: 100" in result.message
    assert "runtime.plan_timeout: 180" in result.message
    assert "runtime.plan_first_token_timeout: 120" in result.message
    assert "runtime.plan_mode: off" in result.message


def test_set_command_updates_plan_timeouts(tmp_path):
    session = make_session(tmp_path)
    dispatcher = CommandDispatcher(Agent(session))

    timeout_result = dispatcher.dispatch("/set runtime.plan_timeout 240")
    first_token_result = dispatcher.dispatch("/set runtime.plan_first_token_timeout 80")

    assert timeout_result.message == "Set runtime.plan_timeout = 240"
    assert first_token_result.message == "Set runtime.plan_first_token_timeout = 80"
    assert session.settings.plan_timeout == 240
    assert session.settings.plan_first_token_timeout == 80


def test_plan_command_toggles_plan_mode(tmp_path):
    session = make_session(tmp_path)
    dispatcher = CommandDispatcher(Agent(session))

    on_result = dispatcher.dispatch("/plan")
    off_result = dispatcher.dispatch("/plan off")
    unknown_set_result = dispatcher.dispatch("/set runtime.plan_mode on")

    assert on_result.message == "Set plan mode = on"
    assert off_result.message == "Set plan mode = off"
    assert unknown_set_result.message == "Unknown config key: runtime.plan_mode"
    assert session.settings.plan_mode is False


def test_plan_command_runs_one_shot_plan_question(tmp_path):
    prompts = []
    session = make_session(tmp_path)

    def run_agent(prompt):
        prompts.append((prompt, session.settings.plan_mode))

    dispatcher = CommandDispatcher(Agent(session), run_agent=run_agent)

    result = dispatcher.dispatch("/plan how should lsp tools work?")

    assert result.message == ""
    assert prompts == [("how should lsp tools work?", True)]
    assert session.settings.plan_mode is False


def test_provider_command_switches_current_provider(tmp_path):
    data = {
        "provider": {
            "active": "one",
            "one": {"model": "model-one"},
            "two": {"model": "model-two", "stream": False},
        }
    }
    session = Session(cwd=str(tmp_path), config=Config.from_dict(data), settings=RuntimeSettings.from_dict(data))
    dispatcher = CommandDispatcher(Agent(session))

    show_result = dispatcher.dispatch("/provider")
    switch_result = dispatcher.dispatch("/provider two")
    model_result = dispatcher.dispatch("/model")
    set_model_result = dispatcher.dispatch("/model model-two-new")
    bad_result = dispatcher.dispatch("/provider missing")

    assert show_result.message == "provider: one\nproviders: one, two"
    assert switch_result.message == "Set provider = two"
    assert session.config.active_provider == "two"
    assert model_result.message == "Current provider.model is model-two"
    assert set_model_result.message == "Set provider.model = model-two-new"
    assert session.config.providers["one"].model == "model-one"
    assert session.config.providers["two"].model == "model-two-new"
    assert bad_result.message == "Unknown provider: missing\nproviders: one, two"


def test_provider_command_selects_provider(tmp_path):
    data = {
        "provider": {
            "active": "one",
            "one": {"model": "model-one"},
            "two": {"model": "model-two"},
        }
    }
    session = Session(cwd=str(tmp_path), config=Config.from_dict(data), settings=RuntimeSettings.from_dict(data))
    dispatcher = CommandDispatcher(Agent(session), select_provider=lambda providers, current: "two")

    result = dispatcher.dispatch("/provider")

    assert result.message == "Set provider = two"
    assert session.config.active_provider == "two"


def test_model_command_can_select_reasoning_effort(tmp_path):
    session = make_session(tmp_path, model="old")
    dispatcher = CommandDispatcher(Agent(session), select_reasoning=lambda: "high")

    result = dispatcher.dispatch("/model new-model")

    assert result.message == "Set provider.model = new-model\nSet provider.reasoning = on\nSet provider.effort = high"
    assert session.config.provider.model == "new-model"
    assert session.config.provider.reasoning is True
    assert session.config.provider.reasoning_effort == "high"


def test_model_command_can_disable_reasoning(tmp_path):
    session = make_session(tmp_path, model="old")
    dispatcher = CommandDispatcher(Agent(session), select_reasoning=lambda: "off")

    result = dispatcher.dispatch("/model new-model")

    assert result.message == "Set provider.model = new-model\nSet provider.reasoning = off"
    assert session.config.provider.model == "new-model"
    assert session.config.provider.reasoning is False


def test_model_command_selects_from_available_models(tmp_path):
    session = make_session(tmp_path, model="old")
    session.config.provider.available_models = ("old", "new-model")
    dispatcher = CommandDispatcher(Agent(session), select_model=lambda models, current: "new-model")

    result = dispatcher.dispatch("/model")

    assert result.message == "Set provider.model = new-model"
    assert session.config.provider.model == "new-model"


def test_blackboard_command_is_not_registered(tmp_path):
    dispatcher = CommandDispatcher(Agent(Session(cwd=str(tmp_path))))

    result = dispatcher.dispatch("/blackboard")

    assert result.status == CommandStatus.UNHANDLED
    assert result.message == ""


def test_rules_command_shows_rules_content(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.state.user_rules.add("Prompt-only changes do not need tests.")
    dispatcher = CommandDispatcher(Agent(session))

    result = dispatcher.dispatch("/rules")

    assert result.status == CommandStatus.HANDLED
    assert result.message == "# User Rules\n\n- Prompt-only changes do not need tests."


def test_knowledge_command_shows_stable_knowledge(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    dispatcher = CommandDispatcher(agent)

    empty_result = dispatcher.dispatch("/knowledge")
    usage_result = dispatcher.dispatch("/knowledge extra")
    agent.blackboard.stable_knowledge = {
        "workflow": ["Project test command is make test."],
        "structure": ["Main runtime lives in nanocode.py."],
    }
    result = dispatcher.dispatch("/knowledge")

    assert empty_result.message == "No stable knowledge. Use /knowledge update to record some."
    assert usage_result.message == "Usage: /knowledge [update]"
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


def test_knowledge_update_command_runs_agent(tmp_path):
    prompts = []
    agent = Agent(Session(cwd=str(tmp_path)))
    dispatcher = CommandDispatcher(agent, run_agent=prompts.append)

    result = dispatcher.dispatch("/knowledge update")

    assert result.status == CommandStatus.HANDLED
    assert result.message == ""
    assert prompts == ["Please perform a knowledge update: record stable knowledge about this project."]


def test_command_dispatcher_auto_compacts_only_when_history_exceeds_keep_recent(tmp_path):
    session = make_session(tmp_path, compact_at=2)
    agent = Agent(session)
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
    agent = Agent(session)
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


def test_compact_command_reports_short_history(tmp_path):
    session = make_session(tmp_path)
    agent = Agent(session)
    session.state.conversation = [UserMessage(content="one"), UserMessage(content="two")]
    dispatcher = CommandDispatcher(agent)

    result = dispatcher.dispatch("/compact")

    assert result.status == CommandStatus.HANDLED
    assert result.message == "Nothing to compact: 2 item(s), keeping recent 5."
    assert len(session.state.conversation) == 2


def test_command_dispatcher_auto_compact_uses_status_runner(tmp_path):
    session = make_session(tmp_path, compact_at=100)
    agent = Agent(session)
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
    dispatcher = CommandDispatcher(Agent(Session(cwd=str(tmp_path))))

    result = dispatcher.dispatch("regular user request")

    assert result.status == CommandStatus.UNHANDLED
    assert result.message == ""


def test_help_question_runs_agent_with_source_aware_prompt(tmp_path):
    prompts = []
    dispatcher = CommandDispatcher(Agent(Session(cwd=str(tmp_path))), run_agent=prompts.append)

    result = dispatcher.dispatch("/help how does compact work?")

    assert result.status == CommandStatus.HANDLED
    assert result.message == ""
    assert len(prompts) == 1


def test_clean_command_removes_all_session_log_files(tmp_path):
    session = Session(cwd=str(tmp_path))
    tool_results_dir = session.tool_results_dir()
    other_tool_results_dir = session.data_path("sessions", "other-session", "tool_results")
    os.makedirs(tool_results_dir, exist_ok=True)
    os.makedirs(other_tool_results_dir, exist_ok=True)

    # Create some log files and a non-log file
    log1 = os.path.join(tool_results_dir, "test1.log")
    log2 = os.path.join(tool_results_dir, "test2.log")
    log3 = os.path.join(other_tool_results_dir, "test3.log")
    other = os.path.join(tool_results_dir, "other.txt")
    with open(log1, "w"):
        pass
    with open(log2, "w"):
        pass
    with open(log3, "w"):
        pass
    with open(other, "w"):
        pass

    dispatcher = CommandDispatcher(Agent(session))
    result = dispatcher.dispatch("/clean")

    assert result.status == CommandStatus.HANDLED
    assert "Cleaned 3 log file(s)" in result.message
    assert not os.path.exists(log1)
    assert not os.path.exists(log2)
    assert not os.path.exists(log3)
    assert os.path.exists(other)


def test_clean_command_no_directory(tmp_path):
    session = Session(cwd=str(tmp_path))
    sessions_dir = session.data_path("sessions")
    if os.path.exists(sessions_dir):
        shutil.rmtree(sessions_dir)

    dispatcher = CommandDispatcher(Agent(session))
    result = dispatcher.dispatch("/clean")

    assert result.status == CommandStatus.HANDLED
    assert "No session logs directory found" in result.message


def test_clean_command_empty_directory(tmp_path):
    session = Session(cwd=str(tmp_path))
    tool_results_dir = session.tool_results_dir()
    os.makedirs(tool_results_dir, exist_ok=True)

    dispatcher = CommandDispatcher(Agent(session))
    result = dispatcher.dispatch("/clean")

    assert result.status == CommandStatus.HANDLED
    assert "Cleaned 0 log file(s)" in result.message


def test_clean_command_with_args_returns_usage(tmp_path):
    session = Session(cwd=str(tmp_path))
    tool_results_dir = session.tool_results_dir()
    os.makedirs(tool_results_dir, exist_ok=True)

    dispatcher = CommandDispatcher(Agent(session))
    result = dispatcher.dispatch("/clean extra-arg")

    assert result.status == CommandStatus.HANDLED
    assert result.message == "Usage: /clean"


def test_clean_command_reports_failed_deletions(tmp_path):
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
        dispatcher = CommandDispatcher(Agent(session))
        result = dispatcher.dispatch("/clean")

    assert result.status == CommandStatus.HANDLED
    assert "Cleaned 1 log file(s)" in result.message
    assert "1 failed" in result.message
