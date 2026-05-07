import pytest

from nanocode import Agent, Session, ToolCallError, ToolResultItem, ToolResultTool


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

    result = ToolResultTool.make(session, ["tr.1", "missing"]).call()

    assert '<ToolResult key="tr.1">' in result
    assert "<log_path>.nanocode/tool_results/sample.log</log_path>" in result
    assert "<original_lines>2</original_lines>" in result
    assert "<original_chars>13</original_chars>" in result
    assert "line 1\nline 2" in result
    assert '<Missing key="missing"/>' in result


def test_prompt_shows_tool_result_descriptions_without_values(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.current.known = ["Parser notes were captured."]
    session.tool_result_store = {
        "tr.1": ToolResultItem(
            description='success Read("sample.txt")',
            value="line 1\nline 2",
            log_path=".nanocode/tool_results/sample.log",
            original_lines=2,
            original_chars=13,
        )
    }

    prompt = Agent(session).build_user_prompt()

    assert "<Details_Keys>" not in prompt
    assert "<Tool_Result_Store>" in prompt
    assert "<ToolResult" in prompt
    assert "<Active_Context>" not in prompt
    assert "Parser notes were captured." in prompt
    assert "tr.1" in prompt
    assert 'success Read("sample.txt")' in prompt
    assert ".nanocode/tool_results/sample.log" in prompt
    assert "line 1" not in prompt
    known_section = prompt.split("<Known>", 1)[1].split("</Known>", 1)[0]
    store_section = prompt.split("<Tool_Result_Store>", 1)[1].split("</Tool_Result_Store>", 1)[0]
    assert "Parser notes were captured." in known_section
    assert "tr.1" not in known_section
    assert "tr.1" in store_section


def test_known_action_accepts_string_items(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

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

    assert session.current.known == ["Parser notes were captured."]
    assert session.tool_result_store == {}


def test_tool_result_invalid_args(tmp_path):
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="ToolResult requires"):
        ToolResultTool.make(session, []).call()
