import pytest

from nanocode import BlackboardTool, Session, ToolCallError


def test_blackboard_append():
    session = Session(cwd=".")

    tool = BlackboardTool.make(session, ["append", "first note"])
    result = tool.call()

    assert session.blackboard == ["first note"]
    assert result == "<BlackboardToolResult>appended</BlackboardToolResult>"


def test_blackboard_append_strips_trailing_whitespace():
    session = Session(cwd=".")

    BlackboardTool.make(session, ["append", "first note\n\t  "]).call()

    assert session.blackboard == ["first note"]


def test_blackboard_append_preserves_multiline_content_until_trailing_whitespace():
    session = Session(cwd=".")

    BlackboardTool.make(session, ["append", "line 1\nline 2\n"]).call()
    result = BlackboardTool.make(session, ["read"]).call()

    assert session.blackboard == ["line 1\nline 2"]
    assert "line 1\nline 2" in result


def test_blackboard_append_empty_content_does_not_mutate_board():
    session = Session(cwd=".")
    session.blackboard = ["existing note"]

    result = BlackboardTool.make(session, ["append", ""]).call()

    assert session.blackboard == ["existing note"]
    assert result == "<BlackboardToolResult>appended</BlackboardToolResult>"


def test_blackboard_read():
    session = Session(cwd=".")
    session.blackboard = ["note 1", "note 2"]

    tool = BlackboardTool.make(session, ["read"])
    result = tool.call()

    assert result == "<BlackboardToolResult>\nnote 1\nnote 2\n</BlackboardToolResult>"


def test_blackboard_read_is_default_action_when_no_args_are_provided():
    session = Session(cwd=".")
    session.blackboard = ["default read note"]

    result = BlackboardTool.make(session, []).call()

    assert "default read note" in result


def test_blackboard_read_empty():
    session = Session(cwd=".")

    result = BlackboardTool.make(session, ["read"]).call()

    assert result == "<BlackboardToolResult>\n\n</BlackboardToolResult>"


def test_blackboard_clear():
    session = Session(cwd=".")
    session.blackboard = ["note 1"]

    tool = BlackboardTool.make(session, ["clear"])
    result = tool.call()

    assert session.blackboard == []
    assert result == "<BlackboardToolResult>cleared</BlackboardToolResult>"


def test_blackboard_is_session_local():
    first_session = Session(cwd=".")
    second_session = Session(cwd=".")

    BlackboardTool.make(first_session, ["append", "first session note"]).call()
    BlackboardTool.make(second_session, ["append", "second session note"]).call()

    assert first_session.blackboard == ["first session note"]
    assert second_session.blackboard == ["second session note"]


def test_blackboard_invalid_action():
    session = Session(cwd=".")

    with pytest.raises(ToolCallError, match="Blackboard action must be one of"):
        BlackboardTool.make(session, ["invalid_action"]).call()
