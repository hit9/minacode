from nanocode import Agent, AssistantMessage, Session, UserMessage


class FakeModelClient:
    def __init__(self, summary="LLM compact summary", known=None):
        self.summary = summary
        self.known = known
        self.requests = []

    def request(self, system_prompt, user_prompt, *, activity="agent"):
        self.requests.append((system_prompt, user_prompt, activity))
        response = {"summary": self.summary}
        if self.known is not None:
            response["known"] = self.known
        return response


def test_agent_compact_history_uses_llm_and_keeps_recent(tmp_path):
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
    assert summary == "Conversation compact summary:\nLLM kept the old user request and assistant note."
    assert agent.blackboard.known == ["old known", "keep known"]
    assert len(fake_client.requests) == 1
    _system_prompt, _user_prompt, activity = fake_client.requests[0]
    assert activity == "compact"
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
