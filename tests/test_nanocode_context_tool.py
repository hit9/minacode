import pytest

from nanocode import Agent, ContextItem, ContextTool, KnownItem, Session, ToolCallError


def test_context_tool_gets_multiple_keys(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.context = {"parser.notes": "line 1\nline 2"}

    result = ContextTool.make(session, ["parser.notes", "missing"]).call()
    prefixed_result = ContextTool.make(session, ["get", "parser.notes"]).call()

    assert '<Context key="parser.notes">\nline 1\nline 2\n  </Context>' in result
    assert '<Missing key="missing"/>' in result
    assert '<Context key="parser.notes">\nline 1\nline 2\n  </Context>' in prefixed_result


def test_prompt_shows_context_descriptions_and_active_values(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.current.known = [KnownItem(fact="Parser notes were captured.", context_keys=["parser.notes"])]
    session.context = {"parser.notes": ContextItem(description="Parser notes from sample output.", value="line 1\nline 2")}

    prompt = Agent(session).build_user_prompt()

    assert "<Details_Keys>" not in prompt
    assert "<Context>" in prompt
    assert "<Active_Context>" in prompt
    assert "Parser notes were captured." in prompt
    assert "parser.notes" in prompt
    assert "Parser notes from sample output." in prompt
    assert "line 1" in prompt


def test_known_action_stores_context_descriptions_and_values(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "known",
                    "items": [
                        {
                            "fact": "Parser notes were captured.",
                            "context": [
                                {
                                    "key": "parser.notes",
                                    "description": "Parser notes from sample output.",
                                    "value": "line 1\nline 2",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    )

    assert session.current.known == [KnownItem(fact="Parser notes were captured.", context_keys=["parser.notes"])]
    assert session.context_store["parser.notes"] == ContextItem(description="Parser notes from sample output.", value="line 1\nline 2")
    assert "parser.notes" in agent.state_updater.latest_report
    assert "Parser notes from sample output." in agent.state_updater.latest_report
    assert "line 1" not in agent.state_updater.latest_report


def test_context_invalid_args(tmp_path):
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="Context requires"):
        ContextTool.make(session, []).call()
