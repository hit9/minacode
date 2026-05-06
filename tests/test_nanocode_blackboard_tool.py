import pytest

from nanocode import BlackboardTool, Session, ToolCallError


def test_blackboard_set():
    session = Session(cwd=".")

    tool = BlackboardTool.make(session, ["set", "progress", "first note"])
    result = tool.call()

    assert session.blackboard == {"progress": "first note"}
    assert result == "<BlackboardToolResult>set</BlackboardToolResult>"


def test_blackboard_set_strips_trailing_value_whitespace():
    session = Session(cwd=".")

    BlackboardTool.make(session, ["set", "progress", "first note\n\t  "]).call()

    assert session.blackboard == {"progress": "first note"}


def test_blackboard_set_preserves_multiline_value_until_trailing_whitespace():
    session = Session(cwd=".")

    BlackboardTool.make(session, ["set", "progress", "line 1\nline 2\n"]).call()
    result = BlackboardTool.make(session, ["read"]).call()

    assert session.blackboard == {"progress": "line 1\nline 2"}
    assert "progress -> line 1\nline 2" in result


def test_blackboard_set_empty_value():
    session = Session(cwd=".")

    result = BlackboardTool.make(session, ["set", "empty", ""]).call()

    assert session.blackboard == {"empty": ""}
    assert result == "<BlackboardToolResult>set</BlackboardToolResult>"


def test_blackboard_read():
    session = Session(cwd=".")
    session.blackboard = {"note 1": "value 1", "note 2": "value 2"}

    tool = BlackboardTool.make(session, ["read"])
    result = tool.call()

    assert result == "<BlackboardToolResult>\nnote 1 -> value 1\nnote 2 -> value 2\n</BlackboardToolResult>"


def test_blackboard_read_key():
    session = Session(cwd=".")
    session.blackboard = {"note 1": "value 1", "note 2": "value 2"}

    result = BlackboardTool.make(session, ["read", "note 2"]).call()

    assert result == "<BlackboardToolResult>\nnote 2 -> value 2\n</BlackboardToolResult>"


def test_blackboard_read_is_default_action_when_no_args_are_provided():
    session = Session(cwd=".")
    session.blackboard = {"default": "default read note"}

    result = BlackboardTool.make(session, []).call()

    assert "default -> default read note" in result


def test_blackboard_read_empty():
    session = Session(cwd=".")

    result = BlackboardTool.make(session, ["read"]).call()

    assert result == "<BlackboardToolResult>\n\n</BlackboardToolResult>"


def test_blackboard_clear():
    session = Session(cwd=".")
    session.blackboard = {"note 1": "value 1"}

    tool = BlackboardTool.make(session, ["clear"])
    result = tool.call()

    assert session.blackboard == {}
    assert result == "<BlackboardToolResult>cleared</BlackboardToolResult>"


def test_blackboard_delete():
    session = Session(cwd=".")
    session.blackboard = {"note 1": "value 1", "note 2": "value 2"}

    result = BlackboardTool.make(session, ["delete", "note 1"]).call()

    assert session.blackboard == {"note 2": "value 2"}
    assert result == "<BlackboardToolResult>deleted</BlackboardToolResult>"


def test_blackboard_is_session_local():
    first_session = Session(cwd=".")
    second_session = Session(cwd=".")

    BlackboardTool.make(first_session, ["set", "note", "first session note"]).call()
    BlackboardTool.make(second_session, ["set", "note", "second session note"]).call()

    assert first_session.blackboard == {"note": "first session note"}
    assert second_session.blackboard == {"note": "second session note"}


def test_blackboard_invalid_action():
    session = Session(cwd=".")

    with pytest.raises(ToolCallError, match="Blackboard action must be one of"):
        BlackboardTool.make(session, ["invalid_action"]).call()


def test_blackboard_set_requires_key_and_value():
    session = Session(cwd=".")

    with pytest.raises(ToolCallError, match="Blackboard set requires key and value"):
        BlackboardTool.make(session, ["set", "key"]).call()


def test_blackboard_delete_requires_key():
    session = Session(cwd=".")

    with pytest.raises(ToolCallError, match="Blackboard delete requires key"):
        BlackboardTool.make(session, ["delete"]).call()
