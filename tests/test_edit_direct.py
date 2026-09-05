"""Direct replace mode: parsing, mode selection, and the pure exact matcher."""

import pytest
from test_edit_tool import session, view

from wizolt.base import ToolError, split_lines
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
