import pytest

from nanocode import Agent, DetailsTool, Session, ToolCallError


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

    assert session.details == {"parser.notes": "line 1\nline 2"}
    assert "  Details\n" in agent.state_updater.latest_report
    assert "parser.notes" in agent.state_updater.latest_report
    assert "line 1" not in agent.state_updater.latest_report


def test_details_tool_gets_multiple_keys(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.details = {"parser.notes": "line 1\nline 2"}

    result = DetailsTool.make(session, ["parser.notes", "missing"]).call()
    prefixed_result = DetailsTool.make(session, ["get", "parser.notes"]).call()

    assert '<Detail key="parser.notes">\nline 1\nline 2\n  </Detail>' in result
    assert '<Detail key="missing">\n\n  </Detail>' in result
    assert '<Detail key="parser.notes">\nline 1\nline 2\n  </Detail>' in prefixed_result


def test_details_invalid_args(tmp_path):
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="Details requires"):
        DetailsTool.make(session, []).call()
