"""edit failure recovery (split from tests/test_edit_tool.py)."""

import json

import pytest
from test_edit_tool import session, view

from wizolt.base import ToolCall, ToolError
from wizolt.context import ContextManager
from wizolt.runner import ToolRunner
from wizolt.source import ToolOutput
from wizolt.tools import CodeIndex, EditTool


def rendered(out, s):
    assert isinstance(out, ToolOutput)
    return out.render(s.register_source_drafts(list(out.drafts)))


def test_stale_target_error_guides_fresh_view(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")
    key = view(s, "note.txt")
    EditTool(s, ["note.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "changed\n"}]]).call()

    with pytest.raises(ToolError) as error:
        EditTool(s, ["note.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "x\nchanged\n"}]]).call()

    message = str(error.value)
    assert "cannot relocate" in message
    assert "use the fresh view below, or Read again" in message
    assert path.read_text(encoding="utf-8") == "changed\n"


def test_out_of_range_line_reports_view_bounds(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    key = view(s, "note.txt")

    with pytest.raises(ToolError, match="lines 10:10 are outside view"):
        EditTool(s, ["note.txt", key, [{"op": "replace", "start": 10, "end": 10, "content": "x\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "a\nb\n"


def test_failure_recovery_is_bounded_fresh_view(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("".join(f"line{i}\n" for i in range(20)), encoding="utf-8")
    key = view(s, "note.txt")
    EditTool(s, ["note.txt", key, [{"op": "replace", "start": 11, "end": 11, "content": "CHANGED\n"}]]).call()

    with pytest.raises(ToolError) as error:
        EditTool(s, ["note.txt", key, [{"op": "replace", "start": 11, "end": 11, "content": "x\n"}]]).call()

    recovery = error.value.recovery
    assert isinstance(recovery, ToolOutput)
    text = rendered(recovery, s)
    assert "<source" in text
    rows = [line for line in text.splitlines() if " | " in line]
    assert 7 >= len(rows) >= 5  # at most seven current lines around the requested coordinate


def test_no_model_facing_text_teaches_the_removed_anchor_protocol():
    """The protocol is only as simple as the text that teaches it. Nothing the model reads --
    tool descriptions, argument schemas, examples, or the system prompt -- may still describe
    line hashes or anchors, and Bash must say outright that its output cannot be edited from."""
    from wizolt.prompts import SYSTEM_PROMPT
    from wizolt.tools import TOOL_REGISTRY, BashTool

    assert "source view" in EditTool.DESCRIPTION
    for tool in TOOL_REGISTRY.values():
        text = json.dumps([tool.DESCRIPTION, tool.EXAMPLE, tool.params_schema()])
        assert "anchor" not in text.lower(), tool.NAME
        assert "hashline" not in text.lower(), tool.NAME
    assert "anchor" not in SYSTEM_PROMPT.lower()
    assert "never a source view" in BashTool.DESCRIPTION
    assert "not a source view" in SYSTEM_PROMPT


def test_success_fresh_block_is_immediately_editable(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")

    first = EditTool(s, ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "B\n"}]]).call()
    fresh_key = s.register_source_drafts(list(first.drafts))[0]
    EditTool(s, ["code.txt", fresh_key, [{"op": "replace", "start": 2, "end": 2, "content": "B\nx\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "a\nB\nx\nc\n"


def test_success_fresh_block_clamps_to_file_bounds(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    key = view(s, "code.txt")

    first = EditTool(s, ["code.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "A\n"}]]).call()
    text = rendered(first, s)

    assert "1 | A" in text
    assert "4 | d" in text
    assert "5 |" not in text


def test_failed_edit_error_text_keeps_fresh_view_for_the_model(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    key = view(s, "code.txt")
    EditTool(s, ["code.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "A\n"}]]).call()
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run_sync([ToolCall("bad", "Edit", ["code.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "x\n"}]])])

    assert len(s.tool_errors) == 1
    assert "source target changed" in s.tool_errors[0].error
    assert "Read again" in s.tool_errors[0].error


def test_batch_stale_range_does_not_guess_after_prior_shift(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run_sync(
        [
            ToolCall(
                "first",
                "Edit",
                [
                    "code.txt",
                    key,
                    [
                        {"op": "replace", "start": 1, "end": 1, "content": "x\na\n"},
                        {"op": "replace", "start": 2, "end": 2, "content": "B\n"},
                    ],
                ],
            ),
            ToolCall(
                "second",
                "Edit",
                ["code.txt", key, [{"op": "replace", "start": 2, "end": 3, "content": "Y\n"}]],
            ),
        ]
    )

    assert path.read_text(encoding="utf-8") == "x\na\nB\nc\n"
    assert len(s.tool_errors) == 1
    assert "were replaced or deleted by an earlier edit in this batch" in s.tool_errors[0].error


def test_deletion_fresh_view_shows_the_seam_it_left(tmp_path, monkeypatch):
    # A deletion leaves no changed line to report, so a hunk-only view would come back empty and
    # force a Read just to keep working on the file the edit only just changed. The fresh view
    # covers the seam instead, and is a real view: the next edit can name it.
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    key = view(s, "code.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run_sync([ToolCall("cut", "Edit", ["code.txt", key, [{"op": "delete", "start": 2, "end": 3}]])])

    fresh = s.get_source_view("view.2")
    assert fresh.total_lines == 2
    assert fresh.spans and [line for span in fresh.spans for line in span.lines] == ["a\n", "d\n"]

    runner.run_sync([ToolCall("next", "Edit", ["code.txt", "view.2", [{"op": "replace", "start": 2, "end": 2, "content": "D\n"}]])])
    assert path.read_text(encoding="utf-8") == "a\nD\n"
    assert s.tool_errors == []


def test_deleting_the_whole_file_leaves_an_empty_file_view(tmp_path, monkeypatch):
    # Deleting every line leaves a view with no spans; the emptied file is then written with create.
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    (tmp_path / "code.txt").write_text("a\nb\n", encoding="utf-8")
    key = view(s, "code.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run_sync([ToolCall("cut", "Edit", ["code.txt", key, [{"op": "delete", "start": 1, "end": 2}]])])

    fresh = s.get_source_view("view.2")
    assert (fresh.total_lines, fresh.spans) == (0, ())
    runner.run_sync([ToolCall("fill", "Edit", ["code.txt", "", [{"op": "create", "content": "new\n"}]])])
    assert (tmp_path / "code.txt").read_text(encoding="utf-8") == "new\n"
    assert s.tool_errors == []


def test_empty_file_create_rejects_once_another_writer_filled_it(tmp_path, monkeypatch):
    # create on a file that was empty (or absent) when the call was written must not overwrite one
    # that is no longer empty by the time it runs: the non-empty file is refused, not clobbered.
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    path.write_text("written elsewhere\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run_sync([ToolCall("fill", "Edit", ["empty.txt", "", [{"op": "create", "content": "mine\n"}]])])

    assert path.read_text(encoding="utf-8") == "written elsewhere\n"
    assert s.tool_errors and "file already exists" in s.tool_errors[0].error


def test_expired_view_is_answered_with_the_current_lines_it_asked_for(tmp_path, monkeypatch):
    """A view id dies when compaction drops the message that named it, and nothing about `view.1`
    tells the model what it held. The refusal therefore reads the path the call named and returns
    the requested lines as they are now, so recovering costs one retry instead of a Read and a
    retry. The returned view is a real one: the same edit against it applies."""
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "app.py"
    path.write_text("".join(f"line {index}\n" for index in range(1, 21)), encoding="utf-8")
    key = view(s, "app.py")
    s.prune_source_views(set())  # compaction expires it
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)
    edits = [{"op": "replace", "start": 10, "end": 12, "content": "X\n"}]

    runner.run_sync([ToolCall("stale", "Edit", ["app.py", key, edits])])

    error = s.tool_errors[0].error
    assert "source missing" in error and "use the fresh view below" in error
    fresh = s.get_source_view("view.2")
    assert [(span.start, span.end) for span in fresh.spans] == [(7, 15)]  # the range plus context
    assert path.read_text(encoding="utf-8").splitlines()[9] == "line 10"

    runner.run_sync([ToolCall("retry", "Edit", ["app.py", "view.2", edits])])

    assert path.read_text(encoding="utf-8").splitlines()[9] == "X"
    assert len(s.tool_errors) == 1  # the retry did not fail


def test_expired_view_recovery_covers_every_edit_in_the_call(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "app.py"
    path.write_text("".join(f"line {index}\n" for index in range(1, 41)), encoding="utf-8")
    key = view(s, "app.py")
    s.prune_source_views(set())

    with pytest.raises(ToolError) as error:
        EditTool(
            s,
            ["app.py", key, [{"op": "replace", "start": 5, "end": 5, "content": "x\n"}, {"op": "replace", "start": 30, "end": 30, "content": "line 30\ny\n"}]],
        ).call()

    spans = error.value.recovery.drafts[0].spans
    assert [(span.start, span.end) for span in spans] == [(2, 8), (27, 33)]


def test_expired_view_recovery_falls_back_to_a_window_for_a_huge_request(tmp_path):
    """Answering an edit that named hundreds of lines would page the file back through an error
    message. Past the limit the model is told where it is and left to Read the range itself."""
    s = session(tmp_path)
    path = tmp_path / "app.py"
    path.write_text("".join(f"line {index}\n" for index in range(1, 301)), encoding="utf-8")
    key = view(s, "app.py")
    s.prune_source_views(set())

    with pytest.raises(ToolError) as error:
        EditTool(s, ["app.py", key, [{"op": "replace", "start": 100, "end": 250, "content": "x\n"}]]).call()

    draft = error.value.recovery.drafts[0]
    assert sum(len(span.lines) for span in draft.spans) <= EditTool.RECOVERY_MAX_LINES
    assert [(span.start, span.end) for span in draft.spans] == [(94, 100)]
    # Narrowing to a window changes which lines the recovery shows and nothing else: it is the
    # same draft as the un-narrowed branch, so it still counts the whole file and still names the
    # relative path the rest of the session uses, not the absolute one.
    assert (draft.display_path, draft.total_lines) == ("app.py", 300)


@pytest.mark.parametrize("outside", [True, False], ids=("outside-the-workspace", "unreadable"))
def test_expired_view_recovery_never_opens_a_file_it_may_not_show(tmp_path, outside):
    """An expired id is not a reason to project a file the user has not approved reading, and a
    path that cannot be read has nothing factual to offer. Both fall back to the plain refusal."""
    s = session(tmp_path / "work")
    (tmp_path / "work").mkdir()
    if outside:
        target = tmp_path / "outside.py"
        target.write_text("secret = 1\n", encoding="utf-8")
    else:
        target = tmp_path / "work" / "gone.py"
        target.write_text("a\n", encoding="utf-8")
    key = view(s, str(target))
    s.prune_source_views(set())
    if not outside:
        target.unlink()

    with pytest.raises(ToolError) as error:
        EditTool(s, [str(target), key, [{"op": "replace", "start": 1, "end": 1, "content": "x\n"}]]).call()

    assert "Read or Search again to obtain a current view" in str(error.value)
    assert error.value.recovery is None
    assert "secret" not in str(error.value)
