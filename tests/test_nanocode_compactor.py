import nanocode
from nanocode import Agent, AssistantMessage, Session, UserMessage


class FakeModelClient:
    def __init__(self, snapshot="LLM working snapshot", known=None, response=None):
        self.snapshot = snapshot
        self.known = known
        self.response = response
        self.requests = []

    def request(self, system_prompt, user_prompt, *, activity="agent", **kwargs):
        self.requests.append((system_prompt, user_prompt, activity, kwargs))
        if self.response is not None:
            return self.response
        response = {"snapshot": self.snapshot}
        if self.known is not None:
            response["known"] = self.known
        return response


def test_agent_compact_history_builds_working_snapshot_and_keeps_recent(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    fake_client = FakeModelClient("LLM kept the old user request and assistant note.")
    agent.compactor.model_client = fake_client
    agent.blackboard.known = ["old known", "keep known"]
    session.state.conversation = [
        UserMessage(content="old user\nmessage"),
        AssistantMessage(content='old assistant note: inspected Read("a.py", "0", "1")'),
        AssistantMessage(content="old answer"),
        UserMessage(content="keep 1"),
        AssistantMessage(content="keep 2"),
        UserMessage(content="keep 3"),
        AssistantMessage(content="keep 4"),
        UserMessage(content="keep 5"),
    ]

    count = agent.compact_history()

    assert count == 8
    assert len(session.state.conversation) == 6
    assert isinstance(session.state.conversation[0], AssistantMessage)
    assert session.state.conversation[1].content == "keep 1"
    summary = session.state.conversation[0].content
    assert summary == "Working Context Snapshot:\nLLM kept the old user request and assistant note."
    assert agent.blackboard.known == ["old known", "keep known"]
    assert len(fake_client.requests) == 1
    _system_prompt, user_prompt, activity, kwargs = fake_client.requests[0]
    assert activity == "compact"
    assert kwargs == {}
    assert "Current_Blackboard" in user_prompt
    assert "Existing_Facts" in user_prompt
    assert "<raw_result>" not in summary


def test_agent_compact_history_replaces_known_with_compacted_known(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    fake_client = FakeModelClient("summary", known=["known " + str(index) for index in range(35)])
    agent.compactor.model_client = fake_client
    agent.blackboard.known = ["old " + str(index) for index in range(40)]
    session.state.conversation = [
        UserMessage(content="old 1"),
        UserMessage(content="old 2"),
        UserMessage(content="old 3"),
        UserMessage(content="keep 1"),
        UserMessage(content="keep 2"),
        UserMessage(content="keep 3"),
        UserMessage(content="keep 4"),
        UserMessage(content="keep 5"),
    ]

    count = agent.compact_history()

    assert count == 8
    assert len(agent.blackboard.known) == 35
    assert agent.blackboard.known[0] == "known 0"
    assert agent.blackboard.known[-1] == "known 34"


def test_agent_compact_history_preserves_known_sources(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    fake_client = FakeModelClient("summary", known=[{"text": "router lives in app.py", "source": ["tr.1"]}])
    agent.compactor.model_client = fake_client
    agent.blackboard.known = [nanocode.KnownItem(text="old fact", source=("tr.9",))]
    session.state.conversation = [
        UserMessage(content="old 1"),
        UserMessage(content="old 2"),
        UserMessage(content="old 3"),
        UserMessage(content="keep 1"),
        UserMessage(content="keep 2"),
        UserMessage(content="keep 3"),
        UserMessage(content="keep 4"),
        UserMessage(content="keep 5"),
    ]

    agent.compact_history()

    assert agent.blackboard.known == ["router lives in app.py"]
    assert agent.blackboard.known[0].source == ("tr.1",)


def test_agent_compact_history_applies_snapshot_state(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    fake_client = FakeModelClient(
        response={
            "snapshot": "Continue by editing app.py. Evidence: tr.1.",
            "goal": "fix route",
            "plan": [{"id": "p1", "text": "Patch route", "status": "doing", "context": "tr.1"}],
            "leads": [{"id": "h1", "text": "route is stale", "source": ["tr.1"]}],
            "checks": {"status": "blocked", "method": "pytest", "context": "missing dependency", "blocker": "environment"},
            "known": [{"text": "router lives in app.py", "source": ["tr.1"]}],
            "user_rules": ["Keep tests targeted."],
        }
    )
    agent.compactor.model_client = fake_client
    session.state.conversation = [
        UserMessage(content="old 1"),
        UserMessage(content="old 2"),
        UserMessage(content="old 3"),
        UserMessage(content="keep 1"),
        UserMessage(content="keep 2"),
        UserMessage(content="keep 3"),
        UserMessage(content="keep 4"),
        UserMessage(content="keep 5"),
    ]

    agent.compact_history()

    assert session.state.conversation[0].content == "Working Context Snapshot:\nContinue by editing app.py. Evidence: tr.1."
    assert agent.blackboard.goal == "fix route"
    assert agent.blackboard.plan == [nanocode.PlanItem(id="p1", text="Patch route", status=nanocode.PlanStatus.DOING, context="tr.1")]
    assert agent.blackboard.leads == [nanocode.Lead(id="h1", text="route is stale", source=("tr.1",))]
    assert agent.blackboard.checks.status == nanocode.CheckStatus.BLOCKED
    assert agent.blackboard.checks.blocker == nanocode.CheckBlocker.ENVIRONMENT
    assert agent.blackboard.known[0].source == ("tr.1",)
    assert "Keep tests targeted." in agent.session.state.user_rules.content


def test_agent_compact_history_skips_when_not_over_keep_recent(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    fake_client = FakeModelClient()
    agent.compactor.model_client = fake_client
    session.state.conversation = [
        UserMessage(content="one"),
        UserMessage(content="two"),
        UserMessage(content="three"),
        UserMessage(content="four"),
        UserMessage(content="five"),
    ]

    count = agent.compact_history()

    assert count == 0
    assert [item.content for item in session.state.conversation] == ["one", "two", "three", "four", "five"]
    assert fake_client.requests == []


def test_agent_compact_history_keeps_full_conversation_log(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.compactor.model_client = FakeModelClient()
    for content in ["old", "keep 1", "keep 2", "keep 3", "keep 4", "keep 5"]:
        session.append_conversation(UserMessage(content=content))

    agent.compact_history()

    assert [item.content for item in session.state.conversation_log] == ["old", "keep 1", "keep 2", "keep 3", "keep 4", "keep 5"]
    assert session.state.conversation[0].content == "Working Context Snapshot:\nLLM working snapshot"
