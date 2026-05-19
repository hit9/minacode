import os
import shutil
import time

import nanocode
from nanocode import Config, Agent, CommandDispatcher, CommandStatus, ModelUsage, RuntimeSettings, Session, SessionLock, SessionCleaner, UserMessage


class FakeModelClient:
    def __init__(self, summary="LLM compact summary"):
        self.summary = summary
        self.requests = []

    def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
        self.requests.append((system_prompt, user_prompt, activity))
        return {"summary": self.summary}


def patch_openai_models(monkeypatch, models=None, error: Exception | None = None):
    seen = {}

    class FakeModels:
        def list(self, **kwargs):
            seen["list_kwargs"] = kwargs
            if error is not None:
                raise error
            return type("ModelList", (), {"data": [type("Model", (), {"id": model})() for model in (models or ())]})()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            seen["client_kwargs"] = kwargs
            self.models = FakeModels()

    monkeypatch.setattr(nanocode, "OpenAI", FakeOpenAI)
    return seen


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
    reason_result = dispatcher.dispatch("/set provider.reasoning high")
    chat_reasoning_result = dispatcher.dispatch("/set provider.chat_reasoning reasoning")
    stream_result = dispatcher.dispatch("/set provider.stream off")
    first_token_result = dispatcher.dispatch("/set provider.first_token_timeout 6")
    yolo_result = dispatcher.dispatch("/set runtime.yolo on")
    compact_result = dispatcher.dispatch("/set runtime.compact_at 2")
    context_result = dispatcher.dispatch("/set runtime.context_budget low")
    exit_result = dispatcher.dispatch("/exit")

    assert model_result.status == CommandStatus.HANDLED
    assert session.config.provider.model == "new-model"
    assert reason_result.message == "Set provider.reasoning = high"
    assert session.config.provider.reasoning == "high"
    assert chat_reasoning_result.message == "Set provider.chat_reasoning = reasoning"
    assert session.config.provider.chat_reasoning == "reasoning"
    assert stream_result.message == "Set provider.stream = off"
    assert session.config.provider.stream is False
    assert first_token_result.message == "Set provider.first_token_timeout = 6"
    assert session.config.provider.first_token_timeout == 6
    assert yolo_result.message == "Set runtime.yolo = on"
    assert session.settings.yolo is True
    assert compact_result.message == "Set runtime.compact_at = 2"
    assert session.settings.compact_at == 2
    assert context_result.message == "Set runtime.context_budget = low"
    assert session.settings.context_budget == "low"
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
    assert "model: model api=chat(auto) reasoning=medium(off) stream=on" in result.message
    assert "session: " + session.session_id in result.message
    assert "runtime: yolo=off plan=off compact_at=50" in result.message
    assert "models:" in result.message
    assert "model: calls=2 tokens=2m" in result.message
    assert "tool_calls: turn=0 session=0" in result.message
    assert "tools: cymbal=" in result.message
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
    temperature_result = dispatcher.dispatch("/set provider.temperature 0.2")
    temperature_off_result = dispatcher.dispatch("/set provider.temperature off")
    invalid_temperature_result = dispatcher.dispatch("/set provider.temperature nope")

    assert url_status_result.message == "Unknown config key: provider.url"
    assert key_status_result.message == "Unknown config key: provider.key"
    assert status_result.message == "Current provider.stream is on"
    assert off_result.message == "Set provider.stream = off"
    assert off_status_result.message == "Current provider.stream is off"
    assert on_result.message == "Set provider.stream = on"
    assert invalid_result.message == "Usage: /set provider.stream [on|off]"
    assert session.config.provider.stream is True
    assert temperature_result.message == "Set provider.temperature = 0.2"
    assert temperature_off_result.message == "Set provider.temperature = (fallback)"
    assert invalid_temperature_result.message == "Usage: /set provider.temperature <number|off>"
    assert session.config.provider.temperature is None


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
    assert "provider.first_token_timeout: 90" in result.message
    assert "paths.data_dir: " + str(tmp_path / ".nanocode") in result.message
    assert "paths.project_dir: " in result.message
    assert "paths.session_dir: " in result.message
    assert "paths.history: " + str(tmp_path / ".nanocode" / "history") in result.message
    assert "runtime.max_agent_steps: 100" in result.message
    assert "runtime.plan_timeout: 360" in result.message
    assert "runtime.plan_first_token_timeout: 180" in result.message
    assert "runtime.context_budget: medium" in result.message
    assert "runtime.auto_clean_recent: 1d" in result.message
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


def test_context_command_shows_and_sets_budget(tmp_path):
    session = make_session(tmp_path)
    agent = Agent(session)
    agent.tool_context.kept_results = ['- ok tool=Read args=["large.py"] key=tr.1\n  output:\n' + ("x" * 10_000)]
    dispatcher = CommandDispatcher(agent)

    show_result = dispatcher.dispatch("/context")
    set_result = dispatcher.dispatch("/context low")
    alias_result = dispatcher.dispatch("/context_budget high")
    invalid_result = dispatcher.dispatch("/context tiny")

    assert "context_budget: medium" in show_result.message
    assert "observe_after_results: 10" in show_result.message
    assert set_result.message.startswith("Set runtime.context_budget = low\ncontext_budget: low")
    assert session.settings.context_budget == "high"
    assert len(agent.tool_context.kept_results[0]) <= agent.context_budget().kept_block_chars
    assert alias_result.message.startswith("Set runtime.context_budget = high\ncontext_budget: high")
    assert invalid_result.message == "Usage: /context [low|medium|high]"


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

    assert result.message == "Set provider.model = new-model\nSet provider.reasoning = high"
    assert session.config.provider.model == "new-model"
    assert session.config.provider.reasoning == "high"


def test_api_command_shows_and_sets_provider_api(tmp_path):
    session = make_session(tmp_path, model="model")
    dispatcher = CommandDispatcher(Agent(session))

    show_result = dispatcher.dispatch("/api")
    responses_result = dispatcher.dispatch("/api responses")
    chat_result = dispatcher.dispatch("/api chat")
    auto_result = dispatcher.dispatch("/api auto")
    bad_result = dispatcher.dispatch("/api invalid")

    assert show_result.message == "provider.api: auto (chat)\nUsage: /api [auto|chat|responses]"
    assert responses_result.message == "Set provider.api = responses"
    assert chat_result.message == "Set provider.api = chat"
    assert auto_result.message == "Set provider.api = auto"
    assert bad_result.message == "Usage: /api [auto|chat|responses]"
    assert session.config.provider.api == "auto"


def test_model_command_can_disable_reasoning(tmp_path):
    session = make_session(tmp_path, model="old")
    dispatcher = CommandDispatcher(Agent(session), select_reasoning=lambda: "off")

    result = dispatcher.dispatch("/model new-model")

    assert result.message == "Set provider.model = new-model\nSet provider.reasoning = off"
    assert session.config.provider.model == "new-model"
    assert session.config.provider.reasoning == "off"


def test_model_command_reasoning_back_cancels_direct_model_change(tmp_path):
    session = make_session(tmp_path, model="old")
    dispatcher = CommandDispatcher(Agent(session), select_reasoning=lambda: nanocode.SELECTION_BACK)

    result = dispatcher.dispatch("/model new-model")

    assert result.message == "No change"
    assert session.config.provider.model == "old"


def test_model_command_reasoning_back_returns_to_model_selection(tmp_path):
    session = make_session(tmp_path, model="old")
    session.config.provider.available_models = ("first", "second")
    selected_models = iter(["first", "second"])
    selected_reasoning = iter([nanocode.SELECTION_BACK, "high"])
    dispatcher = CommandDispatcher(
        Agent(session),
        select_model=lambda models, current: next(selected_models),
        select_reasoning=lambda: next(selected_reasoning),
    )

    result = dispatcher.dispatch("/model")

    assert result.message == "Set provider.model = second\nSet provider.reasoning = high"
    assert session.config.provider.model == "second"
    assert session.config.provider.reasoning == "high"


def test_reason_command_selects_reasoning_effort(tmp_path):
    session = make_session(tmp_path, model="old")
    dispatcher = CommandDispatcher(Agent(session), select_reasoning=lambda: "high")

    result = dispatcher.dispatch("/reason")
    usage_result = dispatcher.dispatch("/reason high")

    assert result.message == "Set provider.reasoning = high"
    assert usage_result.message == "Usage: /reason"
    assert session.config.provider.reasoning == "high"


def test_reason_command_back_keeps_current_reasoning(tmp_path):
    session = make_session(tmp_path, model="old")
    dispatcher = CommandDispatcher(Agent(session), select_reasoning=lambda: nanocode.SELECTION_BACK)

    result = dispatcher.dispatch("/reason")

    assert result.message == "No change"
    assert session.config.provider.reasoning == "medium"


def test_reason_payload_command_shows_and_sets_chat_payload(tmp_path):
    session = make_session(tmp_path, model="old")
    dispatcher = CommandDispatcher(Agent(session))

    show_result = dispatcher.dispatch("/reason-payload")
    off_result = dispatcher.dispatch("/reason-payload off")
    reasoning_result = dispatcher.dispatch("/reason-payload reasoning")
    auto_result = dispatcher.dispatch("/reason-payload auto")
    bad_result = dispatcher.dispatch("/reason-payload bad")

    assert show_result.message == "\n".join(
        [
            "provider.chat_reasoning: auto",
            "provider.resolved_chat_reasoning: off",
            "Usage: /reason-payload [auto|off|reasoning|reasoning_effort|thinking|enable_thinking]",
        ]
    )
    assert off_result.message == "Set provider.chat_reasoning = off"
    assert reasoning_result.message == "Set provider.chat_reasoning = reasoning"
    assert auto_result.message == "Set provider.chat_reasoning = auto"
    assert bad_result.message == "Usage: /reason-payload [auto|off|reasoning|reasoning_effort|thinking|enable_thinking]"
    assert session.config.provider.chat_reasoning == "auto"


def test_model_command_selects_from_available_models(tmp_path):
    session = make_session(tmp_path, model="old")
    session.config.provider.available_models = ("old", "new-model")
    dispatcher = CommandDispatcher(Agent(session), select_model=lambda models, current: "new-model")

    result = dispatcher.dispatch("/model")

    assert result.message == "Set provider.model = new-model"
    assert session.config.provider.model == "new-model"


def test_model_command_lists_configured_models_before_remote_models(tmp_path, monkeypatch):
    session = make_session(tmp_path, model="old")
    session.config.provider.url = "https://provider.example/v1"
    session.config.provider.key = "key"
    session.config.provider.available_models = ("old", "manual")
    seen = patch_openai_models(monkeypatch, ("remote-b", "manual", "remote-a"))

    def select_model(models, current):
        seen["models"] = models
        seen["current"] = current
        return "remote-a"

    dispatcher = CommandDispatcher(Agent(session), select_model=select_model)

    result = dispatcher.dispatch("/model")

    assert seen == {
        "client_kwargs": {
            "api_key": "key",
            "base_url": "https://provider.example/v1",
            "timeout": 3,
            "max_retries": 0,
            "default_headers": {"User-Agent": "nanocode/" + nanocode.__version__},
        },
        "list_kwargs": {"timeout": 3},
        "models": (
            CommandDispatcher.MODEL_CONFIGURED_LABEL,
            "old",
            "manual",
            CommandDispatcher.MODEL_DISCOVERED_LABEL,
            "remote-a",
            "remote-b",
        ),
        "current": "old",
    }
    assert result.message == "Set provider.model = remote-a"
    assert session.config.provider.model == "remote-a"


def test_model_command_ignores_remote_model_failure(tmp_path, monkeypatch):
    session = make_session(tmp_path, model="old")
    session.config.provider.url = "https://provider.example/v1"
    session.config.provider.key = "key"
    session.config.provider.available_models = ("old", "manual")
    seen = {}

    def select_model(models, current):
        seen["models"] = models
        return "manual"

    patch_openai_models(monkeypatch, error=OSError("offline"))
    dispatcher = CommandDispatcher(Agent(session), select_model=select_model)

    result = dispatcher.dispatch("/model")

    assert seen["models"] == (CommandDispatcher.MODEL_CONFIGURED_LABEL, "old", "manual")
    assert result.message == "Set provider.model = manual"


def test_blackboard_command_is_not_registered(tmp_path):
    dispatcher = CommandDispatcher(Agent(Session(cwd=str(tmp_path))))

    result = dispatcher.dispatch("/blackboard")

    assert result.status == CommandStatus.HANDLED
    assert result.message == "Unknown command: /blackboard"


def test_rules_command_shows_rules_content(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.state.user_rules.add("Prompt-only changes do not need tests.")
    dispatcher = CommandDispatcher(Agent(session))

    result = dispatcher.dispatch("/rules")

    assert result.status == CommandStatus.HANDLED
    assert result.message == "# User Rules\n\n- Prompt-only changes do not need tests."


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
    spaced_command_result = dispatcher.dispatch(" /status")

    assert result.status == CommandStatus.UNHANDLED
    assert result.message == ""
    assert spaced_command_result.status == CommandStatus.UNHANDLED
    assert spaced_command_result.message == ""


def test_command_dispatcher_reports_unknown_slash_commands(tmp_path):
    dispatcher = CommandDispatcher(Agent(Session(cwd=str(tmp_path))))

    slash_result = dispatcher.dispatch("/")
    unknown_result = dispatcher.dispatch("/somecommand")

    assert slash_result.status == CommandStatus.HANDLED
    assert slash_result.message == "Unknown command: /"
    assert unknown_result.status == CommandStatus.HANDLED
    assert unknown_result.message == "Unknown command: /somecommand"


def test_help_question_runs_agent_with_source_aware_prompt(tmp_path):
    prompts = []
    dispatcher = CommandDispatcher(Agent(Session(cwd=str(tmp_path))), run_agent=prompts.append)

    result = dispatcher.dispatch("/help how does compact work?")

    assert result.status == CommandStatus.HANDLED
    assert result.message == ""
    assert len(prompts) == 1


def test_clean_command_removes_inactive_session_directories(tmp_path):
    session = Session(cwd=str(tmp_path))
    current_dir = session.session_dir()
    old_dir = session.data_path("sessions", "old-session")
    recent_dir = session.data_path("sessions", "recent-session")
    for path in (current_dir, old_dir, recent_dir):
        os.makedirs(path, exist_ok=True)
    dispatcher = CommandDispatcher(Agent(session))
    result = dispatcher.dispatch("/clean")

    assert result.status == CommandStatus.HANDLED
    assert "Cleaned 2 session(s)" in result.message
    assert os.path.exists(current_dir)
    assert not os.path.exists(old_dir)
    assert not os.path.exists(recent_dir)


def test_clean_command_skips_active_sessions(tmp_path):
    session = Session(cwd=str(tmp_path))
    active_dir = session.data_path("sessions", "active-session")
    stale_dir = session.data_path("sessions", "stale-session")
    os.makedirs(active_dir, exist_ok=True)
    os.makedirs(stale_dir, exist_ok=True)
    old_time = time.time() - 2 * 86400

    with SessionLock(os.path.join(active_dir, "session.lock")):
        os.utime(active_dir, (old_time, old_time))
        os.utime(stale_dir, (old_time, old_time))
        dispatcher = CommandDispatcher(Agent(session))
        result = dispatcher.dispatch("/clean")

    assert result.status == CommandStatus.HANDLED
    assert "Cleaned 1 session(s)" in result.message
    assert "1 active session(s) skipped" in result.message
    assert os.path.exists(active_dir)
    assert not os.path.exists(stale_dir)


def test_session_cleaner_removes_only_old_inactive_sessions(tmp_path):
    session = Session(cwd=str(tmp_path))
    old_dir = session.data_path("sessions", "old-session")
    recent_dir = session.data_path("sessions", "recent-session")
    current_dir = session.session_dir()
    for path in (old_dir, recent_dir, current_dir):
        os.makedirs(path, exist_ok=True)
    old_time = time.time() - 10 * 86400
    os.utime(old_dir, (old_time, old_time))
    os.utime(current_dir, (old_time, old_time))

    with SessionLock(session.lock_path()):
        result = SessionCleaner(session).clean(older_than_seconds=3 * 86400)

    assert result.cleaned == 1
    assert not os.path.exists(old_dir)
    assert os.path.exists(recent_dir)
    assert os.path.exists(current_dir)


def test_session_lock_removes_lock_file_on_release(tmp_path):
    session = Session(cwd=str(tmp_path))
    with SessionLock(session.lock_path()):
        assert os.path.exists(session.lock_path())
    assert not os.path.exists(session.lock_path())


def test_clean_command_no_directory(tmp_path):
    session = Session(cwd=str(tmp_path))
    sessions_dir = session.data_path("sessions")
    if os.path.exists(sessions_dir):
        shutil.rmtree(sessions_dir)

    dispatcher = CommandDispatcher(Agent(session))
    result = dispatcher.dispatch("/clean")

    assert result.status == CommandStatus.HANDLED
    assert "No sessions directory found" in result.message


def test_clean_command_empty_directory(tmp_path):
    session = Session(cwd=str(tmp_path))
    os.makedirs(session.session_dir(), exist_ok=True)

    dispatcher = CommandDispatcher(Agent(session))
    result = dispatcher.dispatch("/clean")

    assert result.status == CommandStatus.HANDLED
    assert "Cleaned 0 session(s)" in result.message


def test_clean_command_with_args_returns_usage(tmp_path):
    session = Session(cwd=str(tmp_path))
    os.makedirs(session.session_dir(), exist_ok=True)

    dispatcher = CommandDispatcher(Agent(session))
    result = dispatcher.dispatch("/clean extra-arg")

    assert result.status == CommandStatus.HANDLED
    assert result.message == "Usage: /clean"


def test_clean_command_reports_failed_deletions(tmp_path):
    session = Session(cwd=str(tmp_path))
    good_dir = session.data_path("sessions", "good-session")
    fail_dir = session.data_path("sessions", "fail-session")
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)
    original_rmtree = shutil.rmtree
    call_count = [0]

    def mock_rmtree(path):
        call_count[0] += 1
        if call_count[0] == 2:
            raise OSError("Permission denied")
        original_rmtree(path)

    import unittest.mock
    with unittest.mock.patch("shutil.rmtree", side_effect=mock_rmtree):
        dispatcher = CommandDispatcher(Agent(session))
        result = dispatcher.dispatch("/clean")

    assert result.status == CommandStatus.HANDLED
    assert "Cleaned 1 session(s)" in result.message
    assert "1 failed" in result.message
