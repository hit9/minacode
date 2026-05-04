import pytest

import nanocode
from nanocode import ReadTool, Session, ToolCallError


def test_read_tool_reads_requested_line_range(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = ReadTool.make(session, ["sample.txt", "1", "3"])
    result = tool.call()

    assert tool.requires_confirmation(session) is False
    assert result.startswith("<ReadToolResult>")
    assert "<range>1:3</range>" in result
    assert "<fingerprint>" in result
    assert "beta\ngamma\n" in result
    assert "alpha" not in result


def test_read_tool_reads_to_eof_when_end_is_zero(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    result = ReadTool.make(session, ["sample.txt", "1", "0"]).call()

    assert "beta\ngamma\n" in result
    assert "alpha" not in result


def test_read_tool_allows_omitted_range_for_full_file_read(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = ReadTool.make(session, ["sample.txt"])
    result = tool.call()

    assert tool.start == 0
    assert tool.end == 0
    assert "<range>0:0</range>" in result
    assert "alpha\nbeta\n" in result


def test_read_tool_truncates_full_file_reads_after_1000_lines(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("".join(f"line-{index:04d}\n" for index in range(1005)), encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    result = ReadTool.make(session, ["sample.txt"]).call()

    assert "<range>0:1000</range>" in result
    assert "<truncated>true</truncated>" in result
    assert "<total_lines>1005</total_lines>" in result
    assert "Read returned 1000 lines from 0:1000 of 1005 total lines" in result
    assert "Use Search to locate relevant text or Read smaller ranges in batches." in result
    assert "line-0999\n" in result
    assert "line-1000\n" not in result


def test_read_tool_truncates_large_bounded_ranges_after_1000_lines(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("".join(f"line-{index:04d}\n" for index in range(1100)), encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    result = ReadTool.make(session, ["sample.txt", "10", "1050"]).call()

    assert "<range>10:1010</range>" in result
    assert "<truncated>true</truncated>" in result
    assert "<total_lines>1100</total_lines>" in result
    assert "Read returned 1000 lines from 10:1010 of 1100 total lines" in result
    assert "line-1009\n" in result
    assert "line-1010\n" not in result


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

    result = ReadTool.make(session, ["sample.txt", "1", "3"]).call()

    assert "one\ntwo\n" in result
    assert "three" not in result
    assert lines_read == ["zero\n", "one\n", "two\n"]


def test_read_tool_clamps_out_of_bounds_range(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    result = ReadTool.make(session, ["sample.txt", "10", "20"]).call()

    assert "alpha" not in result
    assert "  <content no-indention>\n\n  </content>" in result


def test_read_tool_rejects_non_integer_range(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="invalid start"):
        ReadTool.make(session, ["sample.txt", "bad", "1"])


def test_read_tool_rejects_partial_range(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="requires 1 or 3 args"):
        ReadTool.make(session, ["sample.txt", "0"])
