import pytest

from nanocode import Agent, Session, ToolCallError, ToolResultItem, ToolResultTool


def test_tool_result_tool_gets_multiple_keys(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.state.tool_result_store = {
        "tr.1": ToolResultItem(
            description="Read sample.",
            value="line 1\nline 2",
            log_path=".nanocode/sessions/test-session/tool_results/sample.log",
            original_lines=2,
            original_chars=13,
        )
    }

    assert ToolResultTool.name() == "Recall"
    result = ToolResultTool.make(session, ["tr.1", "missing"]).call()

    assert result.startswith("RecallToolResult:")
    assert "- result_key: tr.1" in result
    assert "description: Read sample." in result
    assert "log: .nanocode/sessions/test-session/tool_results/sample.log" in result
    assert "size: 2 lines, 13 chars" in result
    assert "<content>" in result
    assert "line 1\nline 2" in result
    assert "- result_key: missing" in result
    assert "status: missing" in result


def test_tool_result_tool_bounds_large_recall_result(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.state.tool_result_store["tr.1"] = ToolResultItem(description="Read large.", value="x" * 20_000)

    result = ToolResultTool.make(session, ["tr.1"]).call()

    assert len(result) <= 12_000
    assert "[tool result excerpt]" in result
    assert "original_chars:" in result


def test_tool_result_item_details_hint_avoids_recall_call_syntax():
    item = ToolResultItem(description="Read sample.", value="line", excerpted=True)

    result = item.format(result_key="tr.1", details_hint=True)

    assert "details: full=tr.1 if excerpt insufficient" in result
    assert "Recall(" not in result


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
