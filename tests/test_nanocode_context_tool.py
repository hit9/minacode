import pytest

from nanocode import Agent, Session, ToolCallError, ToolResultItem, ToolResultTool


def test_tool_result_tool_gets_multiple_keys(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.state.tool_result_store = {
        "tr.1": ToolResultItem(
            description="Read sample.",
            value="line 1\nline 2",
            log_path=".nanocode/tool_results/sample.log",
            original_lines=2,
            original_chars=13,
        )
    }

    assert ToolResultTool.name() == "Recall"
    result = ToolResultTool.make(session, ["tr.1", "missing"]).call()

    assert result.startswith("RecallToolResult:")
    assert "- result_key: tr.1" in result
    assert "description: Read sample." in result
    assert "log: .nanocode/tool_results/sample.log" in result
    assert "size: 2 lines, 13 chars" in result
    assert "<content>" in result
    assert "line 1\nline 2" in result
    assert "- result_key: missing" in result
    assert "status: missing" in result


def test_known_action_accepts_string_items(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "plan",
                    "mode": "patch",
                    "items": [],
                    "known": [
                        "Parser notes were captured.",
                    ],
                }
            ]
        }
    )

    assert agent.blackboard.known == ["Parser notes were captured."]
    assert session.state.tool_result_store == {}


def test_tool_result_invalid_args(tmp_path):
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="Recall requires"):
        ToolResultTool.make(session, []).call()
