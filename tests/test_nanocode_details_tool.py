import pytest

from nanocode import Agent, DetailItem, DetailsTool, KnownItem, Session, ToolCallError


def test_details_action_stores_key_values(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "details",
                    "items": [
                        {"key": "parser.notes", "value": "line 1\nline 2"},
                        {"key": "empty", "value": ""},
                        {"key": "", "value": "ignored"},
                    ],
                }
            ]
        }
    )

    assert session.details["parser.notes"] == DetailItem(description="parser.notes", value="line 1\nline 2")
    assert "  Details 1\n" in agent.state_updater.latest_report
    assert "parser.notes" in agent.state_updater.latest_report
    assert "line 1" not in agent.state_updater.latest_report


def test_details_action_accepts_single_key_value(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response({"actions": [{"type": "details", "key": "parser.notes", "value": "line 1\nline 2"}]})

    assert session.details["parser.notes"] == DetailItem(description="parser.notes", value="line 1\nline 2")
    assert "  Details 1\n" in agent.state_updater.latest_report
    assert "parser.notes" in agent.state_updater.latest_report
    assert "line 1" not in agent.state_updater.latest_report


def test_details_tool_gets_multiple_keys(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.details = {"parser.notes": "line 1\nline 2"}

    result = DetailsTool.make(session, ["parser.notes", "missing"]).call()
    prefixed_result = DetailsTool.make(session, ["get", "parser.notes"]).call()

    assert '<Detail key="parser.notes">\nline 1\nline 2\n  </Detail>' in result
    assert '<Missing key="missing"/>' in result
    assert '<Detail key="parser.notes">\nline 1\nline 2\n  </Detail>' in prefixed_result


def test_prompt_shows_known_detail_keys_without_values(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.current.known = [KnownItem(fact="Parser notes were captured.", details=["parser.notes"])]
    session.details = {"parser.notes": DetailItem(description="Parser notes from sample output.", value="line 1\nline 2")}

    prompt = Agent(session).build_user_prompt()

    assert "<Details_Keys>" not in prompt
    assert "Parser notes were captured." in prompt
    assert "parser.notes" in prompt
    assert "Parser notes from sample output." in prompt
    assert "line 1" not in prompt


def test_known_action_stores_detail_descriptions_and_hidden_values(tmp_path):
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
                            "details": [
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

    assert session.current.known == [KnownItem(fact="Parser notes were captured.", detail_keys=["parser.notes"])]
    assert session.details["parser.notes"] == DetailItem(description="Parser notes from sample output.", value="line 1\nline 2")
    assert "parser.notes" in agent.state_updater.latest_report
    assert "Parser notes from sample output." in agent.state_updater.latest_report
    assert "line 1" not in agent.state_updater.latest_report


def test_details_invalid_args(tmp_path):
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="Details requires"):
        DetailsTool.make(session, []).call()
