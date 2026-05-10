import pytest

from nanocode import MainAgent, Session, ToolCallError, ToolResultItem, ToolResultTool


def test_tool_result_tool_gets_multiple_keys(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.tool_result_store = {
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

    assert result.startswith("<RecallToolResult>")
    assert '<Result key="tr.1">' in result
    assert "<log_path>.nanocode/tool_results/sample.log</log_path>" in result
    assert "<original_lines>2</original_lines>" in result
    assert "<original_chars>13</original_chars>" in result
    assert "line 1\nline 2" in result
    assert '<Missing key="missing"/>' in result


def test_known_action_accepts_string_items(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "known",
                    "items": [
                        "Parser notes were captured.",
                    ],
                }
            ]
        }
    )

    assert agent.blackboard.known == ["Parser notes were captured."]
    assert session.tool_result_store == {}


def test_tool_result_invalid_args(tmp_path):
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="Recall requires"):
        ToolResultTool.make(session, []).call()
