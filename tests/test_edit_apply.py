"""edit apply (split from tests/test_edit_tool.py)."""

import pytest
from test_edit_tool import session, view

from wizolt.base import ToolCall, ToolError, split_lines
from wizolt.context import ContextManager
from wizolt.runner import ToolRunner
from wizolt.tools import CodeIndex, EditTool
from wizolt.tools.editplan import EditBatchPlan


@pytest.mark.parametrize(
    ("original", "raw_edits"),
    [
        ("", [{"op": "create", "content": "a\nb"}]),
        (
            "a\nb\nc\n",
            [
                {"op": "replace", "start": 1, "end": 1, "content": "A\n"},
                {"op": "replace", "start": 2, "end": 2, "content": "b\nx\n"},
                {"op": "delete", "start": 3, "end": 3},
            ],
        ),
        ("a\nb\n", [{"op": "replace", "start": 2, "end": 2, "content": "inserted\nb\n"}]),
        # Both ranges copy a preserved outside line into content: each replacement edge duplicates
        # the neighbour it touches, so both boundary-duplicate advisories must fire.
        (
            "p\nq\nr\ns\n",
            [
                {"op": "replace", "start": 1, "end": 1, "content": "x\nq\n"},
                {"op": "replace", "start": 4, "end": 4, "content": "r\ny\n"},
            ],
        ),
        ("a\nb", [{"op": "delete", "start": 2, "end": 2}]),
    ],
)
def test_single_and_batch_edit_application_are_equivalent(tmp_path, original, raw_edits):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    creating = raw_edits[0]["op"] == "create"
    if not creating:
        path.write_text(original, encoding="utf-8")
    key = "" if creating else view(s, "code.txt")
    tool = EditTool(s, ["code.txt", key, raw_edits])
    _, _, edits = tool.parse()
    view_obj = s.get_source_view(key) if key else None
    single = tool.apply(original, edits, view_obj)
    original_lines = split_lines(original)
    plan = EditBatchPlan(tool.session)
    state = plan.FileState(
        "code.txt",
        [] if creating else [plan.Line(line, (key, i + 1)) for i, line in enumerate(original_lines)],
        original_lines,
        not creating,
    )

    batch = plan.apply(tool, state, edits, view_obj)

    assert "".join(line.text for line in batch.lines) == single.content
    assert batch.changes == single.changes
    assert batch.replacements == single.replacements
    assert batch.seam_duplicates == single.seam_duplicates


def test_single_and_batch_edit_application_raise_the_same_error(tmp_path):
    original = "a\nb\nc\n"
    raw_edits = [
        {"op": "replace", "start": 1, "end": 2, "content": "x\n"},
        {"op": "delete", "start": 2, "end": 2},
    ]
    s = session(tmp_path)
    (tmp_path / "code.txt").write_text(original, encoding="utf-8")
    key = view(s, "code.txt")
    tool = EditTool(s, ["code.txt", key, raw_edits])
    _, _, edits = tool.parse()
    view_obj = s.get_source_view(key)
    original_lines = split_lines(original)
    plan = EditBatchPlan(tool.session)
    state = plan.FileState("code.txt", [plan.Line(line, (key, i + 1)) for i, line in enumerate(original_lines)], original_lines, True)

    with pytest.raises(ToolError) as single_error:
        tool.apply(original, edits, view_obj)
    with pytest.raises(ToolError) as batch_error:
        plan.apply(tool, state, edits, view_obj)

    assert str(batch_error.value) == str(single_error.value)


def test_split_lines_matches_readlines_only_on_newline():
    # Edit's line model must number lines exactly like Read (file.readlines), i.e. split on "\n"
    # only. str.splitlines(True) also breaks on \x0c and friends, which would desync line numbers.
    assert split_lines("a\nb\x0cc\nd\n") == ["a\n", "b\x0cc\n", "d\n"]
    assert split_lines("a\nb") == ["a\n", "b"]
    assert split_lines("") == []
    assert split_lines("a\nb\x0cc\nd\n") != "a\nb\x0cc\nd\n".splitlines(True)


def test_tool_runner_batch_edit_accepts_drifted_view(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run_sync(
        [
            ToolCall("insert", "Edit", ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "x\nb\n"}]]),
            ToolCall("replace", "Edit", ["code.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "C\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nx\nb\nC\n"
    assert s.tool_errors == []


def test_tool_runner_relocates_view_drifted_before_batch(tmp_path, monkeypatch):
    # An external rewrite between the read and the batch renumbers the file underneath the view.
    # Line origins only describe shifts this batch made, so they cannot resolve this; the planned
    # index is a starting point and the exact target, still unique nearby, relocates and says so.
    # This is the ordinary stale-read case, and it runs through the plan like every other Edit.
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("x\na\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")
    path.write_text("x\nINS\na\nb\nc\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run_sync([ToolCall("replace", "Edit", ["code.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "B\n"}]])])

    assert path.read_text(encoding="utf-8") == "x\nINS\na\nB\nc\n"
    assert s.tool_errors == []
    record = next(record for record in s.tool_records if record.name == "Edit")
    assert f"relocated {key} lines 3:3 -> current lines 4:4" in record.output


def test_tool_runner_refuses_changed_target_drifted_before_batch(tmp_path, monkeypatch):
    # Same shape, but the target line itself was rewritten: nothing in the window matches it
    # exactly, so the call is refused and the file is left alone.
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("x\na\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")
    path.write_text("x\nINS\na\nCHANGED\nc\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run_sync([ToolCall("replace", "Edit", ["code.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "B\n"}]])])

    assert path.read_text(encoding="utf-8") == "x\nINS\na\nCHANGED\nc\n"
    assert s.tool_errors and "source target changed" in s.tool_errors[0].error


def test_tool_runner_batch_edit_barrier_rejects_ambiguous_relocation(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nc\n", encoding="utf-8")
    key = view(s, "code.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run_sync(
        [
            ToolCall("insert", "Edit", ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "x\nb\n"}]]),
            ToolCall("barrier", "Bash", [":"]),
            ToolCall("replace", "Edit", ["code.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "C\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nx\nb\nc\nc\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 1
    assert s.tool_errors


def test_tool_runner_batch_edit_can_create_empty_then_patch_same_file(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    # create the empty file, then write into it with create: an existing zero-byte file has
    # nothing to preserve, so create may fill it.
    runner.run_sync([ToolCall("create", "Edit", ["empty.txt", "", [{"op": "create", "content": ""}]])])
    runner.run_sync([ToolCall("patch", "Edit", ["empty.txt", "", [{"op": "create", "content": "filled\n"}]])])

    assert (tmp_path / "empty.txt").read_text(encoding="utf-8") == "filled\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 2
    assert s.tool_errors == []


def test_tool_runner_batch_edit_rejects_inline_patch_without_read(tmp_path, monkeypatch):
    # In one batch the plan resolves every call against views registered before the batch runs, so
    # a second call that tries to use the just-created file's fresh view inline cannot: the view
    # does not exist yet and the call is refused instead of guessing at line numbers.
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run_sync(
        [
            ToolCall("create", "Edit", ["empty.txt", "", [{"op": "create", "content": ""}]]),
            ToolCall("patch", "Edit", ["empty.txt", "view.1", [{"op": "replace", "start": 1, "end": 1, "content": "filled\n"}]]),
        ]
    )

    assert (tmp_path / "empty.txt").read_text(encoding="utf-8") == ""
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 1
    assert s.tool_errors and "source missing" in s.tool_errors[0].error


def test_tool_runner_batch_edit_can_create_then_patch_same_file(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run_sync([ToolCall("create", "Edit", ["new.txt", "", [{"op": "create", "content": "a\nb\n"}]])])
    key = view(s, "new.txt")
    runner.run_sync([ToolCall("patch", "Edit", ["new.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "B\n"}]])])

    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "a\nB\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 2
    assert s.tool_errors == []


def test_tool_runner_batch_edit_create_and_existing_file_edit_are_independent(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    (tmp_path / "old.txt").write_text("a\nb\n", encoding="utf-8")
    key = view(s, "old.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run_sync(
        [
            ToolCall("create", "Edit", ["new.txt", "", [{"op": "create", "content": "n\n"}]]),
            ToolCall("edit", "Edit", ["old.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "B\n"}]]),
        ]
    )

    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "n\n"
    assert (tmp_path / "old.txt").read_text(encoding="utf-8") == "a\nB\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 2
    assert s.tool_errors == []


def test_tool_runner_batch_edit_maps_original_view_after_delete(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    key = view(s, "code.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run_sync(
        [
            ToolCall("delete", "Edit", ["code.txt", key, [{"op": "delete", "start": 2, "end": 2}]]),
            ToolCall("replace", "Edit", ["code.txt", key, [{"op": "replace", "start": 4, "end": 4, "content": "D\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nc\nD\n"
    assert s.tool_errors == []


def test_tool_runner_batch_edit_maps_original_view_after_insert(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run_sync(
        [
            ToolCall("insert", "Edit", ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "x\nb\n"}]]),
            ToolCall("replace", "Edit", ["code.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "C\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nx\nb\nC\n"
    assert s.tool_errors == []


def test_tool_runner_batch_edit_plans_files_independently(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    (tmp_path / "a.txt").write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("x\ny\n", encoding="utf-8")
    key_a = view(s, "a.txt")
    key_b = view(s, "b.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run_sync(
        [
            ToolCall("edit-a", "Edit", ["a.txt", key_a, [{"op": "replace", "start": 1, "end": 1, "content": "a\nA\n"}]]),
            ToolCall("edit-b", "Edit", ["b.txt", key_b, [{"op": "replace", "start": 2, "end": 2, "content": "Y\n"}]]),
        ]
    )

    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a\nA\nb\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "x\nY\n"
    assert s.tool_errors == []


def test_tool_runner_batch_edit_read_between_edits_sees_intermediate_file(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run_sync(
        [
            ToolCall("insert", "Edit", ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "x\nb\n"}]]),
            ToolCall("read", "Read", [{"path": "code.txt", "ranges": [[1, 0]]}]),
            ToolCall("replace", "Edit", ["code.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "C\n"}]]),
        ]
    )

    read_record = next(record for record in s.tool_records if record.name == "Read")
    assert "| x" in read_record.output
    assert "| c" in read_record.output
    assert "| C" not in read_record.output
    assert path.read_text(encoding="utf-8") == "a\nx\nb\nC\n"


def test_inserting_past_an_unterminated_last_line_does_not_join_it(tmp_path, monkeypatch):
    # A file whose last line has no newline: appending after it must not fuse the new text onto
    # that line. replace N:N states the range's full final text, so the newline between the old
    # last line and the added line is part of the content, not something a splice has to fix.
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb", encoding="utf-8")
    key = view(s, "code.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run_sync([ToolCall("append", "Edit", ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "b\nc\n"}]])])

    assert path.read_text(encoding="utf-8") == "a\nb\nc\n"
    assert s.tool_errors == []


def test_batch_separates_a_consumed_target_from_a_shifted_one(tmp_path, monkeypatch):
    # Two outcomes that look alike from the view's side and must not be conflated. The line an
    # earlier edit in this batch deleted is refused as consumed -- relocation must not go hunting
    # for a copy of it elsewhere. A later line the same edit merely shifted is found by its origin
    # and applied, with no relocation reported because nothing needed relocating.
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nb\n", encoding="utf-8")  # line 2 is duplicated at line 4
    key = view(s, "code.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run_sync(
        [
            ToolCall("cut", "Edit", ["code.txt", key, [{"op": "delete", "start": 2, "end": 2}]]),
            ToolCall("gone", "Edit", ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "B\n"}]]),
            ToolCall("shifted", "Edit", ["code.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "C\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nC\nb\n"
    assert len(s.tool_errors) == 1
    assert s.tool_errors[0].error.startswith("ToolError: source target consumed view.1 lines 2:2 were replaced or deleted by an earlier edit in this batch")
    assert all("relocated" not in record.output for record in s.tool_records if record.name == "Edit")


def test_boundary_duplicate_advisory_names_both_seams(tmp_path):
    # The corruption this polices: content whose edge line is a verbatim copy of the preserved line
    # just outside the range -- context copied into content. Both seams of one batch fire, with
    # line numbers that point at the duplicated pair in the file the call produced.
    s = session(tmp_path)
    (tmp_path / "code.txt").write_text("p\nq\nr\ns\n", encoding="utf-8")
    key = view(s, "code.txt")
    tool = EditTool(s, ["code.txt", key, [
        {"op": "replace", "start": 1, "end": 1, "content": "x\nq\n"},
        {"op": "replace", "start": 4, "end": 4, "content": "r\ny\n"},
    ]])
    _, _, edits = tool.parse()

    result = tool.apply("p\nq\nr\ns\n", edits, s.get_source_view(key))

    assert result.content == "x\nq\nq\nr\nr\ny\n"
    assert result.seam_duplicates == [
        "boundary-duplicate: new lines 2 and 3 are identical; content repeats the line below the range",
        "boundary-duplicate: new lines 4 and 5 are identical; content repeats the line above the range",
    ]


def test_boundary_duplicate_advisory_silent_on_legitimate_edits(tmp_path):
    # Insertion by rewriting the anchor keeps the seam it already had; interior repeats and blank
    # neighbours are the model's own content, not copied context, and stay unpoliced.
    s = session(tmp_path)
    (tmp_path / "code.txt").write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")
    tool = EditTool(s, ["code.txt", key, [
        {"op": "replace", "start": 2, "end": 2, "content": "b\nnew\n"},
        {"op": "replace", "start": 3, "end": 3, "content": "d\nd\ne\n"},
    ]])
    _, _, edits = tool.parse()

    result = tool.apply("a\nb\nc\n", edits, s.get_source_view(key))

    assert result.seam_duplicates == []


def test_boundary_duplicate_advisory_silent_when_the_seam_already_existed(tmp_path):
    # a,a already adjacent before the edit: repeating the neighbour adds no new seam, so the
    # pre-existing duplicate run is not re-reported. The blank neighbour is spacing, never a copy.
    s = session(tmp_path)
    (tmp_path / "code.txt").write_text("a\na\nb\n", encoding="utf-8")
    key = view(s, "code.txt")
    tool = EditTool(s, ["code.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "x\na\n"}]])
    _, _, edits = tool.parse()
    existed = tool.apply("a\na\nb\n", edits, s.get_source_view(key))

    (tmp_path / "blank.txt").write_text("a\n\nb\n", encoding="utf-8")
    blank_key = view(s, "blank.txt")
    blank = EditTool(s, ["blank.txt", blank_key, [{"op": "replace", "start": 1, "end": 1, "content": "x\n\n"}]])
    _, _, blank_edits = blank.parse()

    assert existed.seam_duplicates == []
    assert blank.apply("a\n\nb\n", blank_edits, s.get_source_view(blank_key)).seam_duplicates == []


def test_boundary_duplicate_advisory_warns_in_rendered_envelope(tmp_path, monkeypatch):
    # Advisory, not rejection: the file is written, the envelope carries a <warnings> block naming
    # the duplicated pair, and the fresh view below it already describes the corrupted result.
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")

    out = EditTool(s, ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "x\nc\n"}]]).call()
    text = out.render(s.register_source_drafts(list(out.drafts)))

    assert path.read_text(encoding="utf-8") == "a\nx\nc\nc\n"
    assert "<warnings>" in text
    assert "boundary-duplicate: new lines 3 and 4 are identical; content repeats the line below the range" in text
