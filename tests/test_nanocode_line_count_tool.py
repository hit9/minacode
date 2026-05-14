import pytest

import nanocode
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


def test_line_count_tool_counts_multiple_files(tmp_path):
    p1 = tmp_path / "a.txt"
    p1.write_text("line1\nline2\n", encoding="utf-8")
    p2 = tmp_path / "b.txt"
    p2.write_text("x\ny\nz\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    tool = LineCountTool.make(session, ["a.txt", "b.txt"])
    assert tool.call() == "<LineCountToolResult>5</LineCountToolResult>"


def test_line_count_tool_falls_back_when_wc_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: None)
    path = tmp_path / "fallback.txt"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    tool = LineCountTool.make(session, ["fallback.txt"])

    assert tool.call() == "<LineCountToolResult>3</LineCountToolResult>"


def test_line_count_tool_reports_invalid_path_without_wc(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: None)
    session = Session(cwd=str(tmp_path))
    tool = LineCountTool.make(session, ["nonexistent.txt"])
    with pytest.raises(FileNotFoundError):
        tool.call()
