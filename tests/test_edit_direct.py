"""Direct replace mode: parsing, mode selection, the exact matcher, and the Edit lifecycle."""


class _StubModel:
    """Compactor requires a model; planning-only tests never touch it."""


import asyncio
import json
import os
import pathlib
import random
import threading

import pytest
from test_edit_tool import session, view

from wizolt import compaction
from wizolt.base import ToolCall, ToolError, split_lines
from wizolt.config import Config
from wizolt.context import ContextManager
from wizolt.runner import ToolRunner
from wizolt.session import Session, SessionSnapshotStore
from wizolt.tools import CodeIndex
from wizolt.tools.editplan import EditBatchPlan
from wizolt.tools.files import (
    MODE_CREATE,
    MODE_DIRECT,
    MODE_SOURCE_VIEW,
    DirectMatchError,
    Edit,
    EditTool,
    TextReplacement,
    direct_line_replacements,
    direct_occurrences,
    edit_mode,
    resolve_direct,
)


def parse(tmp_path, source, edits, path="code.txt"):
    s = session(tmp_path)
    (tmp_path / "code.txt").write_text("a\nb\n", encoding="utf-8")
    return EditTool(s, [path, source, edits]).parse()


# --- parsing and mode selection -------------------------------------------------------------


def test_direct_replace_and_delete_parse_without_a_source(tmp_path):
    _, source, edits = parse(tmp_path, "", [{"op": "replace", "old": "a\n", "content": "A\n"}, {"op": "delete", "old": "b\n"}])

    assert source == ""
    assert edits == [Edit(op="replace", old="a\n", content="A\n"), Edit(op="delete", old="b\n")]
    assert edit_mode(source, edits) is MODE_DIRECT


def test_mode_selection_covers_the_three_accepted_call_shapes(tmp_path):
    create = parse(tmp_path, "", [{"op": "create", "content": "x\n"}], path="new.txt")
    assert edit_mode(create[1], create[2]) is MODE_CREATE

    s = session(tmp_path)
    (tmp_path / "code.txt").write_text("a\nb\n", encoding="utf-8")
    key = view(s, "code.txt")
    _, source, edits = EditTool(s, ["code.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "A\n"}]]).parse()
    assert edit_mode(source, edits) is MODE_SOURCE_VIEW


def test_direct_replace_accepts_an_empty_content_string_as_a_removal(tmp_path):
    _, _, edits = parse(tmp_path, "", [{"op": "replace", "old": "a\n", "content": ""}])

    # The declared operation stays replace: an empty string is what the model asked to write.
    assert edits == [Edit(op="replace", old="a\n", content="")]


def test_direct_delete_accepts_an_omitted_or_empty_content(tmp_path):
    omitted = parse(tmp_path, "", [{"op": "delete", "old": "a\n"}])[2]
    empty = parse(tmp_path, "", [{"op": "delete", "old": "a\n", "content": ""}])[2]

    assert omitted == empty == [Edit(op="delete", old="a\n")]


def test_direct_replace_normalizes_line_endings_in_old_and_content(tmp_path):
    _, _, edits = parse(tmp_path, "", [{"op": "replace", "old": "a\r\nb\r", "content": "A\r\n"}])

    assert edits == [Edit(op="replace", old="a\nb\n", content="A\n")]


def test_omitted_op_is_inferred_only_for_an_unambiguous_direct_replace(tmp_path):
    _, _, edits = parse(tmp_path, "", [{"old": "a\n", "content": "A\n"}])
    assert edits == [Edit(op="replace", old="a\n", content="A\n")]

    # Neither create nor delete is ever inferred, and a half-specified direct edit stays an error.
    with pytest.raises(ToolError, match="Edit op is required"):
        parse(tmp_path, "", [{"old": "a\n"}])
    with pytest.raises(ToolError, match="Edit op is required"):
        parse(tmp_path, "", [{"old": "a\n", "content": 7}])
    with pytest.raises(ToolError, match="Edit op is required"):
        parse(tmp_path, "", [{"old": "a\n", "start": 1, "content": "A\n"}])


@pytest.mark.parametrize(
    ("source", "edits", "message"),
    [
        # Evidence modes are mutually exclusive, and a call picks exactly one of them.
        ("$VIEW", [{"op": "replace", "old": "a\n", "content": "A\n"}], "mixed edit evidence modes"),
        ("$VIEW", [{"op": "delete", "old": "a\n"}], "mixed edit evidence modes"),
        (
            "$VIEW",
            [{"op": "replace", "start": 1, "end": 1, "content": "A\n"}, {"op": "replace", "old": "b\n", "content": "B\n"}],
            "mixed edit evidence modes",
        ),
        ("", [{"op": "replace", "old": "a\n", "start": 1, "content": "A\n"}], "replace with old forbids start"),
        ("", [{"op": "delete", "old": "a\n", "start": 1, "end": 1}], "delete with old forbids end, start"),
        ("", [{"op": "replace", "start": 1, "end": 1, "content": "A\n"}], "replace needs evidence"),
        ("", [{"op": "delete", "start": 1, "end": 1}], "delete needs evidence"),
        # A call with no source is direct or nothing; a range beside an old is not a second mode.
        ("", [{"op": "replace", "old": "a\n", "content": "A\n"}, {"op": "delete", "start": 2, "end": 2}], "delete needs evidence"),
        ("", [{"op": "create", "content": "x\n"}, {"op": "replace", "old": "a\n", "content": "A\n"}], "create cannot be mixed"),
        ("", [{"op": "replace", "old": "a\n", "content": "A\n"}, {"op": "create", "content": "x\n"}], "create cannot be mixed"),
        ("", [{"op": "create", "old": "a\n", "content": "x\n"}], "create forbids old"),
        ("", [{"op": "create", "start": 1, "end": 1, "content": "x\n"}], "create forbids end, start"),
        # `old` is exact text, so every shape that is not a non-empty string is refused.
        ("", [{"op": "replace", "old": "", "content": "A\n"}], "replace old must be non-empty"),
        ("", [{"op": "replace", "old": None, "content": "A\n"}], "replace old must be a string"),
        ("", [{"op": "replace", "old": 7, "content": "A\n"}], "replace old must be a string"),
        ("", [{"op": "replace", "old": True, "content": "A\n"}], "replace old must be a string"),
        ("", [{"op": "replace", "old": ["a"], "content": "A\n"}], "replace old must be a string"),
        ("", [{"op": "replace", "old": {"text": "a"}, "content": "A\n"}], "replace old must be a string"),
        # A dropped content field can never read as a deletion, and delete never discards text.
        ("", [{"op": "replace", "old": "a\n"}], "replace requires content as a string"),
        ("", [{"op": "replace", "old": "a\n", "content": None}], "replace requires content as a string"),
        ("", [{"op": "replace", "old": "a\n", "content": 7}], "replace requires content as a string"),
        ("", [{"op": "delete", "old": "a\n", "content": "A\n"}], "delete forbids content"),
        # Neither the new mode nor the old one invents operations.
        ("", [{"op": "view", "old": "a\n", "content": "A\n"}], "Edit op must be create, replace, or delete"),
        ("", [{"op": "insert_after", "old": "a\n", "content": "A\n"}], "Edit op must be create, replace, or delete"),
        ("", [{"op": "replace", "olde": "a\n", "content": "A\n"}], "Edit unexpected field: olde"),
        ("", [{"op": "replace", "old": "a\n", "content": "A\n", "source": "view.1"}], "Edit unexpected field: source"),
    ],
)
def test_edit_rejects_ambiguous_or_malformed_direct_calls(tmp_path, source, edits, message):
    s = session(tmp_path)
    (tmp_path / "code.txt").write_text("a\nb\n", encoding="utf-8")
    key = view(s, "code.txt") if source == "$VIEW" else source

    with pytest.raises(ToolError, match=message):
        EditTool(s, ["code.txt", key, edits]).parse()

    assert (tmp_path / "code.txt").read_text(encoding="utf-8") == "a\nb\n"


# --- the pure exact matcher ------------------------------------------------------------------


def replace(old, content=""):
    return Edit(op="replace", old=old, content=content)


def splice(original, edits):
    """The matcher's result applied to `original`, straight from character offsets."""
    text = original
    for item in sorted(resolve_direct(original, edits), key=lambda item: item.start, reverse=True):
        text = text[: item.start] + item.content + text[item.end :]
    return text


@pytest.mark.parametrize(
    ("original", "old", "content", "expected"),
    [
        ("first\nmiddle\nlast\n", "first\n", "FIRST\n", "FIRST\nmiddle\nlast\n"),
        ("first\nmiddle\nlast\n", "middle\n", "MIDDLE\n", "first\nMIDDLE\nlast\n"),
        ("first\nmiddle\nlast\n", "last\n", "LAST\n", "first\nmiddle\nLAST\n"),
        ("one\ntwo\nthree\n", "two\nthree\n", "TWO\n", "one\nTWO\n"),
        # Part of a line, with no line boundary anywhere in the target.
        ("value = compute(x)\n", "compute", "derive", "value = derive(x)\n"),
        ("value = compute(x)\n", "= compute(x)", "= 1", "value = 1\n"),
        # A target and a replacement at EOF with no final newline.
        ("a\nb", "b", "B", "a\nB"),
        ("a\nb\n", "b\n", "b\nc", "a\nb\nc"),
        # Indentation, tabs, and trailing spaces are part of the text, not decoration.
        ("    x = 1\n", "    x = 1\n", "        x = 1\n", "        x = 1\n"),
        ("\tx\n", "\tx\n", "    x\n", "    x\n"),
        ("x  \n", "x  \n", "x\n", "x\n"),
        # Regex metacharacters are literal.
        ("a.*b\nacb\n", "a.*b\n", "ok\n", "ok\nacb\n"),
        ("x = f(a)|g(b)\n", "f(a)|g(b)", "h(a)", "x = h(a)\n"),
        # CJK, emoji, and a byte-order mark are ordinary characters.
        ("名前 = 1\n", "名前", "名字", "名字 = 1\n"),
        ("s = '🙂'\n", "🙂", "🙃", "s = '🙃'\n"),
        ("﻿import os\n", "﻿import os\n", "﻿import sys\n", "﻿import sys\n"),
        ("﻿import os\n", "import os", "import sys", "﻿import sys\n"),
    ],
)
def test_a_unique_literal_target_resolves_and_splices(original, old, content, expected):
    assert splice(original, [replace(old, content)]) == expected


@pytest.mark.parametrize(
    ("original", "old"),
    [
        ("value = 1\n", "Value = 1\n"),  # case is significant
        ("value = 1\n", "value  = 1\n"),  # inner whitespace is significant
        ("value = 1\n", "value = 1 \n"),  # ... and so is a trailing space
        ("\tx\n", "    x\n"),  # a tab is not four spaces
        ("say 'hi'\n", "say ‘hi’\n"),  # smart quotes are not ASCII quotes
        ("café\n", "café\n"),  # decomposed text is not recomposed
        ("café\n", "café\n"),
        ("﻿import os\n", "﻿﻿import os\n"),
        ("a\nb\n", "a\nb\nc\n"),
    ],
)
def test_a_target_that_is_not_literally_present_is_missing(original, old):
    with pytest.raises(DirectMatchError, match="direct target missing") as error:
        resolve_direct(original, [replace(old, "x")])

    assert error.value.offsets == ()


def test_a_one_character_external_change_inside_the_target_makes_it_missing():
    with pytest.raises(DirectMatchError, match="direct target missing"):
        resolve_direct("def run(value):\n    return value\n", [replace("def run(value):\n    return valve\n", "x")])


def test_a_uniquely_relocated_target_still_resolves():
    original = "import os\nimport sys\n\n\ndef run():\n    return 1\n"

    assert splice(original, [replace("    return 1\n", "    return 2\n")]) == "import os\nimport sys\n\n\ndef run():\n    return 2\n"


def test_repeated_targets_are_ambiguous_and_report_where_they_occur():
    original = "x = 1\ny = 1\nz = 1\n"

    with pytest.raises(DirectMatchError, match="occurs 3 times") as error:
        resolve_direct(original, [replace(" = 1\n", " = 2\n")])

    assert error.value.category == "direct target ambiguous"
    assert error.value.offsets == (1, 7, 13)


def test_overlapping_occurrences_count_as_separate_matches():
    with pytest.raises(DirectMatchError, match="occurs 2 times"):
        resolve_direct("aaa", [replace("aa", "b")])

    assert direct_occurrences("aaa", "aa") == [0, 1]


def test_a_very_common_target_reports_a_capped_count_rather_than_scanning_on():
    with pytest.raises(DirectMatchError, match="occurs more than 50 times") as error:
        resolve_direct("a" * 500, [replace("a", "b")])

    assert len(error.value.offsets) == 50


# --- several operations in one call ------------------------------------------------------------


def test_disjoint_replacements_apply_in_any_listed_order():
    original = "alpha\nbeta\ngamma\n"
    forward = [replace("alpha\n", "ALPHA\n"), replace("gamma\n", "GAMMA\n")]
    backward = [replace("gamma\n", "GAMMA\n"), replace("alpha\n", "ALPHA\n")]

    assert splice(original, forward) == splice(original, backward) == "ALPHA\nbeta\nGAMMA\n"


def test_adjacent_targets_are_accepted():
    assert splice("abcd\n", [replace("ab", "AB"), replace("cd", "CD")]) == "ABCD\n"


def test_matching_never_sees_text_an_earlier_operation_wrote():
    # The second target exists only in the first operation's replacement; it must not match there.
    with pytest.raises(DirectMatchError, match="direct target missing"):
        resolve_direct("one\n", [replace("one\n", "two\n"), replace("two\n", "three\n")])

    # And a replacement may freely contain another operation's target text.
    assert splice("one\ntwo\n", [replace("one\n", "two\n"), replace("two\n", "three\n")]) == "two\nthree\n"


@pytest.mark.parametrize(
    ("original", "edits", "category"),
    [
        # Identical targets resolve to one range each, which is the same range twice.
        ("abcdef\n", [replace("bcd", "X"), replace("bcd", "Y")], "direct targets overlap"),
        ("abcdef\n", [replace("bcd", "X"), replace("cde", "Y")], "direct targets overlap"),
        ("abcdef\n", [replace("bcde", "X"), replace("cd", "Y")], "direct targets overlap"),  # nested
        ("abcdef\n", [replace("cd", "Y"), replace("bcde", "X")], "direct targets overlap"),
        ("a\nb\n", [replace("a\n", "A\n"), replace("missing\n", "X\n")], "direct target missing"),
        ("a\na\nb\n", [replace("b\n", "B\n"), replace("a\n", "A\n")], "direct target ambiguous"),
    ],
)
def test_one_bad_operation_rejects_the_whole_call(original, edits, category):
    with pytest.raises(DirectMatchError, match=category):
        resolve_direct(original, edits)


def test_a_mix_of_direct_replace_and_direct_delete_applies():
    original = "keep\ndrop\nchange\n"
    edits = [Edit(op="delete", old="drop\n"), Edit(op="replace", old="change\n", content="changed\n")]

    assert splice(original, edits) == "keep\nchanged\n"


# --- character ranges rendered as line splices --------------------------------------------------


@pytest.mark.parametrize(
    ("original", "replacements", "expected"),
    [
        # A whole line.
        ("a\nb\nc\n", [TextReplacement(2, 4, "B\n")], [(1, 2, ["B\n"])]),
        # Part of one line keeps the text on either side of the target verbatim.
        ("a\nbbb\nc\n", [TextReplacement(3, 4, "X")], [(1, 2, ["bXb\n"])]),
        # Two targets inside one line cannot be told apart by line ranges, so they are one splice.
        ("abcd\n", [TextReplacement(0, 1, "A"), TextReplacement(2, 3, "C")], [(0, 1, ["AbCd\n"])]),
        # Targets on separate lines stay separate splices, adjacent ones included.
        ("a\nb\nc\n", [TextReplacement(0, 2, "A\n"), TextReplacement(4, 6, "C\n")], [(0, 1, ["A\n"]), (2, 3, ["C\n"])]),
        ("a\nb\n", [TextReplacement(0, 2, "A\n"), TextReplacement(2, 4, "B\n")], [(0, 1, ["A\n"]), (1, 2, ["B\n"])]),
        # A target spanning lines collapses into the one splice that covers all of them.
        ("a\nb\nc\n", [TextReplacement(0, 4, "X\n")], [(0, 2, ["X\n"])]),
        # A deletion that removes a whole line leaves an empty replacement.
        ("a\nb\nc\n", [TextReplacement(2, 4, "")], [(1, 2, [])]),
        # Nothing repairs a newline the model did not write.
        ("a\nb\nc\n", [TextReplacement(2, 4, "B")], [(1, 2, ["B"])]),
    ],
)
def test_character_replacements_become_line_splices_with_identical_text(original, replacements, expected):
    lines = split_lines(original)
    spliced = direct_line_replacements(lines, replacements)

    assert spliced == expected
    rebuilt = list(lines)
    for start, end, replacement in sorted(spliced, reverse=True):
        rebuilt[start:end] = replacement
    reference = original
    for item in sorted(replacements, key=lambda item: item.start, reverse=True):
        reference = reference[: item.start] + item.content + reference[item.end :]
    assert "".join(rebuilt) == reference


# --- the Edit tool end to end -------------------------------------------------------------------


def render(s, out):
    return out.render(s.register_source_drafts(list(out.drafts)))


def edit(s, path, edits):
    return EditTool(s, [path, "", edits]).call()


async def test_text_observed_through_bash_can_be_edited_without_a_read(tmp_path):
    """The round this mode exists to save: Bash, then Edit, with no Read in between."""
    from wizolt.tools.shell import BashTool

    s = session(tmp_path)
    (tmp_path / "app.py").write_text("import os\n\n\ndef run(value):\n    return value\n", encoding="utf-8")
    shown = await BashTool(s, ["cat app.py"]).call()
    assert "def run(value):" in shown  # exact text the model can now copy into `old`
    assert not s.source_views  # and nothing in that output is a source view

    out = edit(s, "app.py", [{"op": "replace", "old": "def run(value):\n    return value\n", "content": "def run(value):\n    return value * 2\n"}])

    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "import os\n\n\ndef run(value):\n    return value * 2\n"
    assert "return value * 2" in render(s, out)


async def test_text_observed_through_search_can_be_edited_without_a_source_view(tmp_path):
    from wizolt.tools.search import SearchTool

    s = session(tmp_path)
    (tmp_path / "sample.py").write_text("alpha\nNeedle\nomega\n", encoding="utf-8")
    found = await SearchTool(s, [{"pattern": "Needle", "path": "."}]).call()
    assert "2 | Needle" in found.retained_text

    edit(s, "sample.py", [{"op": "replace", "old": "Needle\n", "content": "Thread\n"}])

    assert (tmp_path / "sample.py").read_text(encoding="utf-8") == "alpha\nThread\nomega\n"


def test_a_direct_edit_returns_a_fresh_view_the_next_source_edit_can_use(tmp_path):
    s = session(tmp_path)
    (tmp_path / "code.txt").write_text("a\nb\nc\n", encoding="utf-8")

    out = edit(s, "code.txt", [{"op": "replace", "old": "b\n", "content": "B\n"}])
    key = s.register_source_drafts(list(out.drafts))[0]
    EditTool(s, ["code.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "C\n"}]]).call()

    assert (tmp_path / "code.txt").read_text(encoding="utf-8") == "a\nB\nC\n"


def test_preview_and_execution_agree_on_the_diff_and_the_result(tmp_path):
    s = session(tmp_path)
    (tmp_path / "code.txt").write_text("a\nb\nc\n", encoding="utf-8")
    edits = [{"op": "replace", "old": "a\n", "content": "A\n"}, {"op": "delete", "old": "c\n"}]

    previewed = EditTool(s, ["code.txt", "", edits]).preview()
    assert (tmp_path / "code.txt").read_text(encoding="utf-8") == "a\nb\nc\n"  # preview writes nothing

    out = EditTool(s, ["code.txt", "", edits]).call()

    assert previewed == EditTool(s, ["code.txt", "", edits]).diff(str(tmp_path / "code.txt"), "a\nb\nc\n", "A\nb\n")
    assert previewed in render(s, out)
    assert (tmp_path / "code.txt").read_text(encoding="utf-8") == "A\nb\n"


def test_a_partial_line_target_replaces_only_its_characters(tmp_path):
    s = session(tmp_path)
    (tmp_path / "code.py").write_text("total = compute(a) + compute_all(b)\n", encoding="utf-8")

    edit(s, "code.py", [{"op": "replace", "old": "compute(a)", "content": "derive(a)"}])

    assert (tmp_path / "code.py").read_text(encoding="utf-8") == "total = derive(a) + compute_all(b)\n"


def test_two_targets_inside_one_line_both_apply(tmp_path):
    s = session(tmp_path)
    (tmp_path / "code.py").write_text("x = left + right\n", encoding="utf-8")

    edit(s, "code.py", [{"op": "replace", "old": "left", "content": "LEFT"}, {"op": "replace", "old": "right", "content": "RIGHT"}])

    assert (tmp_path / "code.py").read_text(encoding="utf-8") == "x = LEFT + RIGHT\n"


def test_a_crlf_file_matches_through_the_normalized_text(tmp_path):
    s = session(tmp_path)
    (tmp_path / "code.txt").write_bytes(b"a\r\nb\r\nc\r\n")

    edit(s, "code.txt", [{"op": "replace", "old": "b\n", "content": "B\n"}])

    assert (tmp_path / "code.txt").read_text(encoding="utf-8") == "a\nB\nc\n"


def test_a_target_at_the_end_of_a_file_without_a_final_newline(tmp_path):
    s = session(tmp_path)
    (tmp_path / "code.txt").write_text("a\nb", encoding="utf-8")

    edit(s, "code.txt", [{"op": "replace", "old": "b", "content": "b\nc"}])

    assert (tmp_path / "code.txt").read_text(encoding="utf-8") == "a\nb\nc"


def test_ambiguity_shows_where_the_target_occurs_and_writes_nothing(tmp_path):
    s = session(tmp_path)
    original = "".join(f"line {n}\nvalue = 1\n" for n in range(1, 4))
    (tmp_path / "code.txt").write_text(original, encoding="utf-8")

    with pytest.raises(ToolError, match="direct target ambiguous") as error:
        edit(s, "code.txt", [{"op": "replace", "old": "value = 1\n", "content": "value = 2\n"}])

    recovery = render(s, error.value.recovery)
    assert "2 | value = 1" in recovery and "4 | value = 1" in recovery and "6 | value = 1" in recovery
    assert (tmp_path / "code.txt").read_text(encoding="utf-8") == original


def test_ambiguity_recovery_stays_inside_the_existing_view_budget(tmp_path):
    s = session(tmp_path)
    original = "".join(f"pad {n}\nvalue = 1\n" for n in range(1, 40))
    (tmp_path / "code.txt").write_text(original, encoding="utf-8")

    with pytest.raises(ToolError, match="occurs 39 times") as error:
        edit(s, "code.txt", [{"op": "replace", "old": "value = 1\n", "content": "value = 2\n"}])

    recovery = render(s, error.value.recovery)
    body = [line for line in recovery.splitlines() if " | " in line]
    assert 0 < len(body) <= EditTool.RECOVERY_MAX_LINES


@pytest.mark.parametrize(
    ("original", "edits", "message"),
    [
        ("a\nb\n", [{"op": "replace", "old": "z\n", "content": "Z\n"}], "direct target missing"),
        ("a\na\n", [{"op": "replace", "old": "a\n", "content": "A\n"}], "direct target ambiguous"),
        ("abcdef\n", [{"op": "replace", "old": "bcd", "content": "X"}, {"op": "replace", "old": "cde", "content": "Y"}], "direct targets overlap"),
        ("a\nb\n", [{"op": "replace", "old": "a\n", "content": "a\n"}], "edit produced no changes"),
        # One bad operation refuses the whole call, leaving the good one unwritten too.
        ("a\nb\n", [{"op": "replace", "old": "a\n", "content": "A\n"}, {"op": "replace", "old": "z\n", "content": "Z\n"}], "direct target missing"),
    ],
)
def test_a_refused_direct_call_leaves_the_file_untouched(tmp_path, original, edits, message):
    s = session(tmp_path)
    (tmp_path / "code.txt").write_text(original, encoding="utf-8")

    with pytest.raises(ToolError, match=message):
        edit(s, "code.txt", edits)

    assert (tmp_path / "code.txt").read_text(encoding="utf-8") == original


def test_direct_mode_keeps_the_existing_target_failures(tmp_path):
    s = session(tmp_path)
    (tmp_path / "folder").mkdir()

    with pytest.raises(ToolError, match="path is a directory"):
        edit(s, "folder", [{"op": "replace", "old": "a", "content": "b"}])
    with pytest.raises(ToolError, match="file does not exist; use op=create"):
        edit(s, "absent.txt", [{"op": "replace", "old": "a", "content": "b"}])

    (tmp_path / "binary.bin").write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(UnicodeDecodeError):
        edit(s, "binary.bin", [{"op": "replace", "old": "a", "content": "b"}])


def test_direct_failures_do_not_echo_a_large_target_back(tmp_path):
    s = session(tmp_path)
    (tmp_path / "code.txt").write_text("a\n", encoding="utf-8")
    huge = "x" * 20000

    with pytest.raises(ToolError) as error:
        edit(s, "code.txt", [{"op": "replace", "old": huge, "content": huge}])

    assert huge not in str(error.value)
    assert len(str(error.value)) < 200


def test_the_endif_shape_is_ambiguous_bare_and_unique_when_expanded(tmp_path):
    s = session(tmp_path)
    guards = "#if A\nbody\n#endif\n#endif\n"
    (tmp_path / "guards.h").write_text(guards, encoding="utf-8")

    # A bare repeated guard cannot say which one it means.
    with pytest.raises(ToolError, match="direct target ambiguous"):
        edit(s, "guards.h", [{"op": "replace", "old": "#endif\n", "content": "#endif // A\n"}])
    assert (tmp_path / "guards.h").read_text(encoding="utf-8") == guards

    # Expanded with the line above it, the same target is unique.
    edit(s, "guards.h", [{"op": "replace", "old": "body\n#endif\n", "content": "body\n#endif // A\n"}])
    assert (tmp_path / "guards.h").read_text(encoding="utf-8") == "#if A\nbody\n#endif // A\n#endif\n"


def test_direct_mode_still_warns_when_content_re_adds_a_preserved_boundary(tmp_path):
    """Exact evidence does not excuse the seam a copied neighbour leaves behind."""
    s = session(tmp_path)
    (tmp_path / "guards.h").write_text("#if A\nbody\n#endif\ntail\n", encoding="utf-8")

    out = edit(s, "guards.h", [{"op": "replace", "old": "body\n", "content": "body\nmore\n#endif\n"}])

    assert "boundary-duplicate" in render(s, out)
    assert (tmp_path / "guards.h").read_text(encoding="utf-8") == "#if A\nbody\nmore\n#endif\n#endif\ntail\n"


def test_confirmation_policy_does_not_depend_on_the_evidence_mode(tmp_path):
    s = session(tmp_path)
    (tmp_path / "code.txt").write_text("a\n", encoding="utf-8")
    key = view(s, "code.txt")
    direct = EditTool(s, ["code.txt", "", [{"op": "replace", "old": "a\n", "content": "b\n"}]])
    sourced = EditTool(s, ["code.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "b\n"}]])

    assert direct.needs_confirmation() == sourced.needs_confirmation() is True
    assert direct.always_confirms() == sourced.always_confirms() is False


# --- batch planning and stale writes -------------------------------------------------------------


async def ignore_index_update(_index, _paths):
    return ""


def runner(s):
    return ToolRunner(s, ContextManager(s), output_fn=lambda text: None)


def yolo_session(cwd, monkeypatch):
    s = session(cwd)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", ignore_index_update)
    return s


def direct_call(call_id, path, edits):
    return ToolCall(call_id, "Edit", [path, "", edits])


async def test_a_planned_direct_call_matches_the_standalone_one(tmp_path, monkeypatch):
    edits = [{"op": "replace", "old": "b\n", "content": "B\n"}, {"op": "delete", "old": "d\n"}]
    original = "a\nb\nc\nd\ne\n"
    (tmp_path / "alone").mkdir()
    (tmp_path / "planned").mkdir()
    (tmp_path / "alone" / "code.txt").write_text(original, encoding="utf-8")
    (tmp_path / "planned" / "code.txt").write_text(original, encoding="utf-8")

    alone = session(tmp_path / "alone")
    standalone = EditTool(alone, ["code.txt", "", edits]).call().retained_text

    s = yolo_session(tmp_path / "planned", monkeypatch)
    await runner(s).run([direct_call("e0", "code.txt", edits)])

    planned = next(record for record in s.tool_records if record.name == "Edit").output
    assert planned == standalone
    assert (tmp_path / "planned" / "code.txt").read_text(encoding="utf-8") == (tmp_path / "alone" / "code.txt").read_text(encoding="utf-8")


async def test_a_later_direct_call_observes_the_earlier_planned_result(tmp_path, monkeypatch):
    s = yolo_session(tmp_path, monkeypatch)
    (tmp_path / "code.txt").write_text("a\nb\nc\n", encoding="utf-8")

    await runner(s).run(
        [
            direct_call("first", "code.txt", [{"op": "replace", "old": "b\n", "content": "beta\n"}]),
            # `beta` exists only in the first call's planned result.
            direct_call("second", "code.txt", [{"op": "replace", "old": "beta\n", "content": "BETA\n"}]),
        ]
    )

    assert (tmp_path / "code.txt").read_text(encoding="utf-8") == "a\nBETA\nc\n"
    assert s.tool_errors == []


async def test_a_direct_call_can_follow_a_create_on_the_same_path(tmp_path, monkeypatch):
    s = yolo_session(tmp_path, monkeypatch)

    await runner(s).run(
        [
            ToolCall("make", "Edit", ["new.txt", "", [{"op": "create", "content": "one\ntwo\n"}]]),
            direct_call("patch", "new.txt", [{"op": "replace", "old": "two\n", "content": "TWO\n"}]),
        ]
    )

    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "one\nTWO\n"
    assert s.tool_errors == []


async def test_direct_calls_on_different_paths_stay_independent(tmp_path, monkeypatch):
    s = yolo_session(tmp_path, monkeypatch)
    (tmp_path / "one.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / "two.txt").write_text("x\n", encoding="utf-8")

    await runner(s).run(
        [
            direct_call("a", "one.txt", [{"op": "replace", "old": "x\n", "content": "1\n"}]),
            direct_call("b", "two.txt", [{"op": "replace", "old": "x\n", "content": "2\n"}]),
        ]
    )

    assert (tmp_path / "one.txt").read_text(encoding="utf-8") == "1\n"
    assert (tmp_path / "two.txt").read_text(encoding="utf-8") == "2\n"


async def test_one_batch_may_use_a_source_view_on_one_path_and_direct_evidence_on_another(tmp_path, monkeypatch):
    s = yolo_session(tmp_path, monkeypatch)
    (tmp_path / "viewed.txt").write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "direct.txt").write_text("a\nb\n", encoding="utf-8")
    key = view(s, "viewed.txt")

    await runner(s).run(
        [
            ToolCall("v", "Edit", ["viewed.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "A\n"}]]),
            direct_call("d", "direct.txt", [{"op": "replace", "old": "b\n", "content": "B\n"}]),
        ]
    )

    assert s.tool_errors == []
    assert (tmp_path / "viewed.txt").read_text(encoding="utf-8") == "A\nb\n"
    assert (tmp_path / "direct.txt").read_text(encoding="utf-8") == "a\nB\n"


@pytest.mark.parametrize("direct_first", [True, False], ids=("direct-then-view", "view-then-direct"))
async def test_one_path_may_not_change_evidence_mode_inside_one_batch(tmp_path, monkeypatch, direct_first):
    s = yolo_session(tmp_path, monkeypatch)
    (tmp_path / "code.txt").write_text("a\nb\n", encoding="utf-8")
    key = view(s, "code.txt")
    sourced = ToolCall("v", "Edit", ["code.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "A\n"}]])
    direct = direct_call("d", "code.txt", [{"op": "replace", "old": "b\n", "content": "B\n"}])

    await runner(s).run([direct, sourced] if direct_first else [sourced, direct])

    # The first call still lands; the second is refused before any transaction, not reordered.
    assert (tmp_path / "code.txt").read_text(encoding="utf-8") == ("a\nB\n" if direct_first else "A\nb\n")
    assert s.tool_errors and "mixed edit evidence modes" in s.tool_errors[0].error
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 1


async def test_a_refused_direct_call_leaves_the_other_planned_calls_alone(tmp_path, monkeypatch):
    s = yolo_session(tmp_path, monkeypatch)
    (tmp_path / "good.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "bad.txt").write_text("a\na\n", encoding="utf-8")

    await runner(s).run(
        [
            direct_call("bad", "bad.txt", [{"op": "replace", "old": "a\n", "content": "A\n"}]),
            direct_call("good", "good.txt", [{"op": "replace", "old": "a\n", "content": "A\n"}]),
        ]
    )

    assert (tmp_path / "bad.txt").read_text(encoding="utf-8") == "a\na\n"
    assert (tmp_path / "good.txt").read_text(encoding="utf-8") == "A\n"
    assert len(s.tool_errors) == 1 and "direct target ambiguous" in s.tool_errors[0].error


async def test_every_direct_call_in_a_batch_gets_exactly_one_result(tmp_path, monkeypatch):
    s = yolo_session(tmp_path, monkeypatch)
    (tmp_path / "code.txt").write_text("a\nb\n", encoding="utf-8")
    calls = [
        direct_call("ok", "code.txt", [{"op": "replace", "old": "a\n", "content": "A\n"}]),
        direct_call("missing", "code.txt", [{"op": "replace", "old": "zzz\n", "content": "Z\n"}]),
        direct_call("malformed", "code.txt", [{"op": "replace", "old": ""}]),
    ]

    messages = await runner(s).run(calls)

    assert [message["tool_call_id"] for message in messages] == ["ok", "missing", "malformed"]
    assert (tmp_path / "code.txt").read_text(encoding="utf-8") == "A\nb\n"


@pytest.mark.parametrize(
    ("disturb", "leftover"),
    [
        (lambda path: path.write_text("external\n", encoding="utf-8"), "external\n"),
        (lambda path: path.unlink(), None),
    ],
    ids=("rewritten", "deleted"),
)
async def test_a_planned_direct_edit_never_overwrites_a_file_that_changed_after_planning(tmp_path, disturb, leftover):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    call = direct_call("edit", "code.txt", [{"op": "replace", "old": "b\n", "content": "B\n"}])
    plan = await EditBatchPlan(s).build([call])
    disturb(path)

    with pytest.raises(ToolError, match="planned edit is stale"):
        await plan.planned[call.id].apply(EditTool(s, call.args))

    assert (path.read_text(encoding="utf-8") if leftover else None) == leftover


async def test_cancelling_a_direct_edit_settles_its_write_rather_than_abandoning_it(tmp_path, monkeypatch):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    call = direct_call("edit", "code.txt", [{"op": "replace", "old": "b\n", "content": "B\n"}])
    planned = (await EditBatchPlan(s).build([call])).planned[call.id]
    written, release = threading.Event(), threading.Event()
    original = EditBatchPlan.PlannedEdit.transact

    def blocked(receipt):
        result = original(receipt)
        written.set()
        release.wait()
        return result

    monkeypatch.setattr(EditBatchPlan.PlannedEdit, "transact", blocked)
    task = asyncio.create_task(planned.apply(EditTool(s, call.args)))
    while not written.is_set():
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert path.read_text(encoding="utf-8") == "a\nB\n"


async def test_a_direct_edit_updates_the_symbol_index_like_any_other(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    (tmp_path / "code.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    updated = []

    async def record_update(_index, paths):
        updated.append(tuple(paths))
        return ""

    monkeypatch.setattr(CodeIndex, "update", record_update)
    await runner(s).run([direct_call("e", "code.py", [{"op": "replace", "old": "return 1\n", "content": "return 2\n"}])])

    assert updated and any("code.py" in paths for paths in updated)


# --- deterministic generated coverage against a reference splice ---------------------------------


def reference_splice(original, targets):
    """The obvious implementation: replace each disjoint target by scanning from the left."""
    result, cursor = "", 0
    for start, end, content in sorted(targets):
        result += original[cursor:start] + content
        cursor = end
    return result + original[cursor:]


def generated_case(seed, count=None):
    """A file and some unique targets in it, reproducible from `seed`."""
    rng = random.Random(seed)
    words = [f"tok{index}{'x' * rng.randrange(0, 4)}" for index in range(40)]
    rng.shuffle(words)
    lines, cursor = [], 0
    while cursor < len(words):
        width = rng.randrange(1, 4)
        lines.append(" ".join(words[cursor : cursor + width]) + "\n")
        cursor += width
    if rng.random() < 0.5 and lines:
        lines[-1] = lines[-1].rstrip("\n")  # a file with no final newline
    original = "".join(lines)
    unique = [word for word in words if original.count(word) == 1 and not any(word != other and word in other for other in words)]
    rng.shuffle(unique)
    chosen = unique[: count or rng.randrange(1, 4)]
    return original, [(original.index(word), original.index(word) + len(word), word) for word in chosen]


SEEDS = tuple(range(60))


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_unique_replacements_match_a_reference_splice(seed):
    original, targets = generated_case(seed)
    rng = random.Random(seed + 10_000)
    contents = {word: word.upper() + ("\n" if rng.random() < 0.3 else "") for _, _, word in targets}
    edits = [replace(word, contents[word]) for _, _, word in targets]
    rng.shuffle(edits)  # the order the operations are listed in must not matter

    expected = reference_splice(original, [(start, end, contents[word]) for start, end, word in targets])

    assert splice(original, edits) == expected, f"seed {seed}: {original!r}"


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_line_splices_reproduce_the_character_splice(seed):
    original, targets = generated_case(seed)
    edits = [replace(word, "Z" * len(word)) for _, _, word in targets]
    lines = split_lines(original)

    rebuilt = list(lines)
    for start, end, replacement in sorted(direct_line_replacements(lines, resolve_direct(original, edits)), reverse=True):
        rebuilt[start:end] = replacement

    assert "".join(rebuilt) == splice(original, edits), f"seed {seed}"


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_adjacent_replacements_apply_together(seed):
    """Two targets that touch at one offset: neither range may swallow or shift the other."""
    original, targets = generated_case(seed, count=2)
    first, second = sorted(targets)
    # The second target is extended backwards to meet the first, which keeps it unique because it
    # still contains the whole unique word it was built from.
    head = original[first[0] : first[1]]
    tail = original[first[1] : second[1]]

    assert splice(original, [replace(head, "<"), replace(tail, ">")]) == original[: first[0]] + "<>" + original[second[1] :], f"seed {seed}"


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_duplicate_overlap_and_stale_targets_never_write(tmp_path, seed):
    original, targets = generated_case(seed)
    _, _, word = targets[0]
    (tmp_path / "code.txt").write_text(original, encoding="utf-8")
    s = session(tmp_path)
    stale = word[:-1] + ("z" if word[-1] != "z" else "y")
    cases = [
        [{"op": "replace", "old": word, "content": "A"}, {"op": "replace", "old": word, "content": "B"}],  # one target twice
        [{"op": "replace", "old": stale, "content": "A"}, {"op": "replace", "old": word, "content": "B"}],  # one stale character
    ]
    if original.count(word[1:]) == 1:  # a nested target inside the outer one
        cases.append([{"op": "replace", "old": word, "content": "A"}, {"op": "replace", "old": word[1:], "content": "B"}])

    for edits in cases:
        with pytest.raises(ToolError, match="direct target"):
            edit(s, "code.txt", edits)
        assert (tmp_path / "code.txt").read_text(encoding="utf-8") == original, f"seed {seed}: {edits}"


# --- session compatibility ------------------------------------------------------------------------


FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "session_pre_direct_replace.json"


def legacy_session(tmp_path):
    """A session restored from a snapshot written before direct mode existed.

    The fixture was produced by the pre-feature code and holds a Read view, the fresh view a
    completed source-view Edit returned, and that Edit's stored call -- none of which carry an
    `old` field, because nothing could write one.
    """
    fixture = json.loads(FIXTURE.read_text())
    config = Config(data_dir=str(tmp_path))
    path = SessionSnapshotStore.session_path(config.data_dir, str(tmp_path), fixture["uid"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for line in fixture["lines"]:
            file.write(json.dumps(line).replace("{CWD}", str(tmp_path)) + "\n")
    (tmp_path / "code.txt").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
    return Session.load_snapshot(fixture["uid"], config=config, cwd=str(tmp_path))


def test_a_pre_feature_session_needs_no_migration_and_still_edits_through_its_views(tmp_path):
    s = legacy_session(tmp_path)

    assert sorted(s.source_views) == ["view.1", "view.2"]
    stored = next(record for record in s.tool_records if record.name == "Edit")
    assert stored.args == ["code.txt", "view.1", [{"op": "replace", "start": 2, "end": 2, "content": "BETA\n"}]]
    assert "old" not in json.dumps(stored.args)  # no default was written in on load

    EditTool(s, ["code.txt", "view.2", [{"op": "replace", "start": 3, "end": 3, "content": "GAMMA\n"}]]).call()

    assert (tmp_path / "code.txt").read_text(encoding="utf-8") == "alpha\nBETA\nGAMMA\n"


def test_a_pre_feature_session_can_continue_with_a_direct_edit(tmp_path):
    s = legacy_session(tmp_path)

    out = edit(s, "code.txt", [{"op": "replace", "old": "gamma\n", "content": "GAMMA\n"}])

    assert (tmp_path / "code.txt").read_text(encoding="utf-8") == "alpha\nBETA\nGAMMA\n"
    # The new view continues the same counter the restored session was already using.
    assert s.register_source_drafts(list(out.drafts)) == ["view.3"]


async def test_direct_edit_arguments_survive_a_snapshot_and_resume(tmp_path, monkeypatch):
    config = Config(data_dir=str(tmp_path))
    s = Session(cwd=str(tmp_path), config=config)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", ignore_index_update)
    (tmp_path / "code.txt").write_text("a\nb\n", encoding="utf-8")
    edits = [{"op": "replace", "old": "b\n", "content": "B\n"}]
    await runner(s).run([direct_call("e0", "code.txt", edits)])
    await s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=config, cwd=str(tmp_path))

    stored = next(record for record in restored.tool_records if record.name == "Edit")
    assert stored.args == ["code.txt", "", edits]
    # Resuming replays no write: the edit already landed and the file is left exactly as it was.
    assert (tmp_path / "code.txt").read_text(encoding="utf-8") == "a\nB\n"


def test_compaction_keeps_a_direct_call_paired_with_its_result(tmp_path):
    context = ContextManager(session(tmp_path))
    arguments = json.dumps({"path": "code.txt", "edits": [{"op": "replace", "old": "b\n", "content": "B\n"}]})
    messages = [
        *({"role": "user", "content": f"old {index}"} for index in range(3)),
        {"role": "assistant", "content": None, "tool_calls": [{"id": "tc.1", "type": "function", "function": {"name": "Edit", "arguments": arguments}}]},
        {"role": "tool", "tool_call_id": "tc.1", "content": "done"},
        *({"role": "user", "content": f"recent {index}"} for index in range(6)),
    ]

    _, keep = compaction.Compactor(context, _StubModel()).parts_for(messages)

    assert keep[0]["tool_calls"][0]["function"]["arguments"] == arguments
    assert keep[1]["tool_call_id"] == "tc.1"


def test_view_numbering_stays_stable_across_mixed_evidence_modes(tmp_path):
    s = session(tmp_path)
    (tmp_path / "code.txt").write_text("a\nb\nc\n", encoding="utf-8")

    first = view(s, "code.txt")  # view.1
    direct = edit(s, "code.txt", [{"op": "replace", "old": "a\n", "content": "A\n"}])
    fresh = s.register_source_drafts(list(direct.drafts))[0]  # view.2
    sourced = EditTool(s, ["code.txt", fresh, [{"op": "replace", "start": 2, "end": 2, "content": "B\n"}]]).call()

    assert [first, fresh, s.register_source_drafts(list(sourced.drafts))[0]] == ["view.1", "view.2", "view.3"]
    assert (tmp_path / "code.txt").read_text(encoding="utf-8") == "A\nB\nc\n"
