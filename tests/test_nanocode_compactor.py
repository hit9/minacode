from nanocode import Agent, AssistantMessage, Session, UserMessage


class FakeModelClient:
    def __init__(self, summary="LLM compact summary"):
        self.summary = summary
        self.requests = []

    def request(self, system_prompt, user_prompt, *, activity="main"):
        self.requests.append((system_prompt, user_prompt, activity))
        return {"summary": self.summary}


def test_agent_compact_history_uses_llm_and_keeps_recent(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    fake_client = FakeModelClient("LLM kept the old user request and assistant note.")
    agent.compactor.model_client = fake_client
    session.conversation = [
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
    assert len(session.conversation) == 6
    assert isinstance(session.conversation[0], AssistantMessage)
    assert session.conversation[1].content == "keep 1"
    summary = session.conversation[0].content
    assert summary == "Conversation compact summary:\nLLM kept the old user request and assistant note."
    assert len(fake_client.requests) == 1
    system_prompt, user_prompt, activity = fake_client.requests[0]
    assert activity == "compact"
    assert "conversation-history compactor" in system_prompt
    assert "Preserve continuity-critical facts" in system_prompt
    assert "old user\nmessage" in user_prompt
    assert 'Read("a.py", "0", "1")' in user_prompt
    assert "keep 1" not in user_prompt
    assert "<raw_result>" not in summary


def test_agent_compact_history_skips_when_not_over_keep_recent(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    fake_client = FakeModelClient()
    agent.compactor.model_client = fake_client
    session.conversation = [
        UserMessage(content="one"),
        UserMessage(content="two"),
        UserMessage(content="three"),
        UserMessage(content="four"),
        UserMessage(content="five"),
    ]

    count = agent.compact_history()

    assert count == 0
    assert [item.content for item in session.conversation] == ["one", "two", "three", "four", "five"]
    assert fake_client.requests == []
