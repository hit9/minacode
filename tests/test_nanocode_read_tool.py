import pytest

import nanocode
from nanocode import ReadTool, Session, ToolCallError


def test_read_tool_reads_requested_line_range(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = ReadTool.make(session, ["sample.txt", "1,3"])
    result = tool.call()

    assert tool.requires_confirmation(session) is False
    assert result.startswith("<ReadToolResult>")
    assert "<range>1:3</range>" in result
    assert "<fingerprint>" in result
    assert "<content line-numbered>" in result
    assert "      1 | beta\n      2 | gamma\n" in result
    assert "      0 | alpha" not in result


def test_read_tool_rejects_empty_args_with_actionable_error(tmp_path):
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match=r'Read args error: got 0 args; expected \["filepath"\]'):
        ReadTool.make(session, [])


def test_read_tool_rejects_multiple_start_end_pairs(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("zero\none\ntwo\nthree\nfour\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="Read args error: for multiple ranges use comma tokens"):
        ReadTool.make(session, ["sample.txt", "1", "2", "3", "5"])


def test_read_tool_reads_multiple_line_range_tokens(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("zero\none\ntwo\nthree\nfour\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = ReadTool.make(session, ["sample.txt", "1-2", "3-5"])
    result = tool.call()

    assert tool.ranges == [(1, 2), (3, 5)]
    assert "1:2, 3:5" in tool.preview()
    assert "<range>1:2</range>" in result
    assert "<range>3:5</range>" in result
    assert "      1 | one\n" in result
    assert "      3 | three\n      4 | four\n" in result
    assert "      0 | zero" not in result
    assert "      2 | two" not in result


def test_read_tool_reads_colon_and_comma_range_tokens(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("zero\none\ntwo\nthree\nfour\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = ReadTool.make(session, ["sample.txt", "1:2", "3,5"])
    result = tool.call()

    assert tool.ranges == [(1, 2), (3, 5)]
    assert "1:2, 3:5" in tool.preview()
    assert "<range>1:2</range>" in result
    assert "<range>3:5</range>" in result
    assert "      1 | one\n" in result
    assert "      3 | three\n      4 | four\n" in result
    assert "      0 | zero" not in result
    assert "      2 | two" not in result


def test_read_tool_reads_to_eof_when_end_is_zero(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    result = ReadTool.make(session, ["sample.txt", "1,0"]).call()

    assert "      1 | beta\n      2 | gamma\n" in result
    assert "      0 | alpha" not in result


def test_read_tool_allows_omitted_range_for_full_file_read(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = ReadTool.make(session, ["sample.txt"])
    result = tool.call()

    assert tool.start == 0
    assert tool.end == 0
    assert "<range>0:0</range>" in result
    assert "      0 | alpha\n      1 | beta\n" in result


def test_read_tool_reads_range_token_when_numeric_filenames_exist(tmp_path):
    (tmp_path / "sample.txt").write_text("zero\none\ntwo\nthree\n", encoding="utf-8")
    (tmp_path / "1").write_text("numeric filename one\n", encoding="utf-8")
    (tmp_path / "3").write_text("numeric filename three\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = ReadTool.make(session, ["sample.txt", "1,3"])
    result = tool.call()

    assert tool.ranges == [(1, 3)]
    assert "<range>1:3</range>" in result
    assert "      1 | one\n      2 | two\n" in result
    assert "numeric filename" not in result


def test_read_tool_truncates_full_file_reads_after_600_lines(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("".join(f"line-{index:04d}\n" for index in range(605)), encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    result = ReadTool.make(session, ["sample.txt"]).call()

    assert "<range>0:600</range>" in result
    assert "<truncated>true</truncated>" in result
    assert "<total_lines>605</total_lines>" in result
    assert "Read returned 600 lines from 0:600 of 605 total lines" in result
    assert "Use Search to locate relevant text or Read smaller ranges in batches." in result
    assert "    599 | line-0599\n" in result
    assert "    600 | line-0600\n" not in result


def test_read_tool_truncates_large_bounded_ranges_after_600_lines(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("".join(f"line-{index:04d}\n" for index in range(700)), encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    result = ReadTool.make(session, ["sample.txt", "10,650"]).call()

    assert "<range>10:610</range>" in result
    assert "<truncated>true</truncated>" in result
    assert "<total_lines>700</total_lines>" in result
    assert "Read returned 600 lines from 10:610 of 700 total lines" in result
    assert "    609 | line-0609\n" in result
    assert "    610 | line-0610\n" not in result


def test_read_tool_bounded_read_stops_at_end(tmp_path, monkeypatch):
    path = tmp_path / "sample.txt"
    path.write_text("zero\none\ntwo\nthree\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    real_open = open
    lines_read = []

    class TrackingFile:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __enter__(self):
            self.wrapped.__enter__()
            return self

        def __exit__(self, *args):
            return self.wrapped.__exit__(*args)

        def __iter__(self):
            for line in self.wrapped:
                lines_read.append(line)
                yield line

    def tracking_open(*args, **kwargs):
        return TrackingFile(real_open(*args, **kwargs))

    monkeypatch.setattr(nanocode, "open", tracking_open, raising=False)

    result = ReadTool.make(session, ["sample.txt", "1,3"]).call()

    assert "      1 | one\n      2 | two\n" in result
    assert "three" not in result
    assert lines_read == ["zero\n", "one\n", "two\n"]


def test_read_tool_clamps_out_of_bounds_range(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    result = ReadTool.make(session, ["sample.txt", "10,20"]).call()

    assert "alpha" not in result
    assert "  <content line-numbered>\n\n  </content>" in result


def test_read_tool_rejects_non_integer_range(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="invalid range"):
        ReadTool.make(session, ["sample.txt", "bad,1"])


def test_read_tool_rejects_partial_range(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="Read args error: invalid range token"):
        ReadTool.make(session, ["sample.txt", "0"])
