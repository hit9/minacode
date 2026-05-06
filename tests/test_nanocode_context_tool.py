import pytest

from nanocode import Agent, ContextItem, ContextTool, Session, ToolCallError


def test_context_tool_gets_multiple_keys(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.context = {"parser.notes": "line 1\nline 2"}

    result = ContextTool.make(session, ["parser.notes", "missing"]).call()
    prefixed_result = ContextTool.make(session, ["get", "parser.notes"]).call()

    assert '<ContextItem key="parser.notes">\nline 1\nline 2\n  </ContextItem>' in result
    assert '<Missing key="missing"/>' in result
    assert '<ContextItem key="parser.notes">\nline 1\nline 2\n  </ContextItem>' in prefixed_result


def test_prompt_shows_context_descriptions_without_values(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.current.known = ["Parser notes were captured."]
    session.context = {"parser.notes": ContextItem(description="Parser notes from sample output.", value="line 1\nline 2")}

    prompt = Agent(session).build_user_prompt()

    assert "<Details_Keys>" not in prompt
    assert "<Context_Store>" in prompt
    assert "<ContextItem" in prompt
    assert "<Active_Context>" not in prompt
    assert "Parser notes were captured." in prompt
    assert "parser.notes" in prompt
    assert "Parser notes from sample output." in prompt
    assert "line 1" not in prompt
    known_section = prompt.split("<Known>", 1)[1].split("</Known>", 1)[0]
    context_section = prompt.split("<Context_Store>", 1)[1].split("</Context_Store>", 1)[0]
    assert "Parser notes were captured." in known_section
    assert "parser.notes" not in known_section
    assert "parser.notes" in context_section


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

    assert session.current.known == ["Parser notes were captured."]
    assert session.context_store["parser.notes"] == ContextItem(description="Parser notes from sample output.", value="line 1\nline 2")
    assert "parser.notes" in agent.state_updater.latest_report
    assert "Parser notes from sample output." in agent.state_updater.latest_report
    assert "line 1" not in agent.state_updater.latest_report


def test_context_invalid_args(tmp_path):
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="Context requires"):
        ContextTool.make(session, []).call()
