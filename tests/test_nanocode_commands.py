from nanocode import Agent, CommandDispatcher, CommandStatus, Session, UserMessage


class FakeModelClient:
    def __init__(self, summary="LLM compact summary"):
        self.summary = summary
        self.requests = []

    def request(self, system_prompt, user_prompt, *, activity="main"):
        self.requests.append((system_prompt, user_prompt, activity))
        return {"summary": self.summary}


def test_command_dispatcher_updates_config_and_auto_compacts(tmp_path):
    session = Session(cwd=str(tmp_path), model="old", compact_at=100)
    agent = Agent(session)
    fake_client = FakeModelClient()
    agent.compactor.model_client = fake_client
    dispatcher = CommandDispatcher(agent)
    session.conversation = [UserMessage(content="one"), UserMessage(content="two"), UserMessage(content="three")]

    model_result = dispatcher.dispatch("/model new-model")
    effort_result = dispatcher.dispatch("/reason_effort high")
    reason_result = dispatcher.dispatch("/reason off")
    stream_result = dispatcher.dispatch("/stream off")
    yolo_result = dispatcher.dispatch("/yolo on")
    compact_result = dispatcher.dispatch("/compact-at 2")
    exit_result = dispatcher.dispatch("/exit")

    assert model_result.status == CommandStatus.HANDLED
    assert session.model == "new-model"
    assert effort_result.message == "Reasoning effort set to: high"
    assert session.reasoning_effort == "high"
    assert reason_result.message == "Reasoning disabled"
    assert session.reasoning is False
    assert stream_result.message == "Streaming disabled"
    assert session.stream is False
    assert yolo_result.message == "YOLO enabled"
    assert session.yolo is True
    assert compact_result.message == "Auto-compact threshold set to: 2"
    assert session.compact_at == 2
    assert len(session.conversation) == 3
    assert fake_client.requests == []
    assert exit_result.status == CommandStatus.EXIT


def test_status_reports_tokens_in_human_readable_format(tmp_path):
    session = Session(cwd=str(tmp_path), model="model")
    session.last_total_tokens = 1200
    session.session_total_tokens = 2_345_678
    session.last_cost_usd = 0.000008
    session.session_cost_usd = 12.345678
    dispatcher = CommandDispatcher(Agent(session))

    result = dispatcher.dispatch("/status")

    assert result.status == CommandStatus.HANDLED
    assert "tokens: last=1k session=2m" in result.message
    assert "cost(usd): last=$0.000008 session=$12.345678" in result.message
    assert "stream: on" in result.message
    assert "blackboard: 0 items" in result.message


def test_stream_command_shows_and_updates_streaming_mode(tmp_path):
    session = Session(cwd=str(tmp_path), stream=True)
    dispatcher = CommandDispatcher(Agent(session))

    status_result = dispatcher.dispatch("/stream")
    off_result = dispatcher.dispatch("/stream off")
    off_status_result = dispatcher.dispatch("/stream status")
    on_result = dispatcher.dispatch("/stream on")
    invalid_result = dispatcher.dispatch("/stream maybe")

    assert status_result.message == "Streaming is on"
    assert off_result.message == "Streaming disabled"
    assert off_status_result.message == "Streaming is off"
    assert on_result.message == "Streaming enabled"
    assert invalid_result.message == "Usage: /stream [on|off|status]"
    assert session.stream is True


def test_blackboard_command_reports_empty_board(tmp_path):
    dispatcher = CommandDispatcher(Agent(Session(cwd=str(tmp_path))))

    result = dispatcher.dispatch("/blackboard")

    assert result.status == CommandStatus.HANDLED
    assert result.message == "Blackboard is empty"


def test_blackboard_command_reads_board_with_default_and_status_args(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.blackboard = ["note 1", "note 2"]
    dispatcher = CommandDispatcher(Agent(session))

    default_result = dispatcher.dispatch("/blackboard")
    status_result = dispatcher.dispatch("/blackboard status")

    assert default_result.status == CommandStatus.HANDLED
    assert default_result.message == "Blackboard:\nnote 1\nnote 2"
    assert status_result.status == CommandStatus.HANDLED
    assert status_result.message == default_result.message


def test_blackboard_command_clear_mutates_session_board(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.blackboard = ["note"]
    dispatcher = CommandDispatcher(Agent(session))

    result = dispatcher.dispatch("/blackboard   clear  ")

    assert result.status == CommandStatus.HANDLED
    assert result.message == "Blackboard cleared"
    assert session.blackboard == []


def test_blackboard_command_reports_usage_for_unknown_args(tmp_path):
    dispatcher = CommandDispatcher(Agent(Session(cwd=str(tmp_path))))

    result = dispatcher.dispatch("/blackboard append note")

    assert result.status == CommandStatus.HANDLED
    assert result.message == "Usage: /blackboard [status|clear]"


def test_command_dispatcher_auto_compacts_only_when_history_exceeds_keep_recent(tmp_path):
    session = Session(cwd=str(tmp_path), compact_at=2)
    agent = Agent(session)
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

    result = dispatcher.dispatch("/compact-at 2")

    assert result.message == "Auto-compact threshold set to: 2 and compacted history"
    assert len(session.conversation) == 6
    assert session.conversation[0].content == "Conversation compact summary:\nLLM compact summary"
    assert session.conversation[1].content == "keep 1"


def test_command_dispatcher_runs_compact_with_status_runner(tmp_path):
    session = Session(cwd=str(tmp_path), compact_at=2)
    agent = Agent(session)
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
    agent = Agent(session)
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

    result = dispatcher.dispatch("/compact-at 2")

    assert result.message == "Auto-compact threshold set to: 2 and compacted history"
    assert status_calls == ["run"]
    assert session.conversation[0].content == "Conversation compact summary:\nLLM compact summary"


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
    assert "Answer this question about nanocode itself." in prompts[0]
    assert "nanocode.py" in prompts[0]
    assert "pyproject.toml" in prompts[0]
    assert "how does compact work?" in prompts[0]
