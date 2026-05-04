from nanocode import LineCountTool, Session


def test_line_count_tool_counts_file_lines(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = LineCountTool.make(session, ["sample.txt"])

    assert tool.requires_confirmation(session) is False
    assert tool.call() == "<LineCountToolResult>3</LineCountToolResult>"


def test_line_count_tool_counts_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = LineCountTool.make(session, ["empty.txt"])

    assert tool.call() == "<LineCountToolResult>0</LineCountToolResult>"
