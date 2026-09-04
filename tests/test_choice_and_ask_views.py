"""choice and ask views (split from tests/test_ui_render.py)."""

import wizolt.render as render_module
from wizolt.base import (
    SELECTION_BACK,
    SELECTION_FREE_TEXT,
)
from wizolt.render import UiPrinter
from wizolt.tools import AskSpec
from wizolt.tui import ASK_DONE, ASK_FREE_TEXT, TUI_MODAL_PENDING, AskViewState, ChoiceViewState


def test_choice_view_g_and_shift_g_jump_first_and_last():
    state = ChoiceViewState(choices=("one", "two", "three"), labels={}, disabled=set())

    state.handle_key("G")
    assert state.selected == 2
    state.handle_key("g")
    assert state.selected == 0

    # While searching, g/G are query text, not jumps.
    state.searching = True
    state.handle_key("g")
    assert state.query == "g"
    assert state.selected == 0


def test_choice_view_state_default_filtering():
    state = ChoiceViewState(
        choices=("alpha", "---", "beta", "---", "gamma"),
        labels={"alpha": "Alpha", "beta": "Beta", "gamma": "Gamma"},
        disabled={"---"},
    )
    assert state.visible() == ("alpha", "---", "beta", "---", "gamma")
    assert state.enabled() == ("alpha", "beta", "gamma")
    assert state.clamp() == ("alpha", "beta", "gamma")
    assert state.selected_choice() == "alpha"


def test_choice_view_state_search_filters_visible():
    state = ChoiceViewState(
        choices=("alpha", "---", "beta", "---", "gamma"),
        labels={"alpha": "Alpha", "beta": "Beta"},
        disabled={"---"},
    )
    state.set_query("beta")
    assert "beta" in state.visible()
    assert "alpha" not in state.visible()
    assert state.selected == 0


def test_choice_view_state_move_navigation():
    state = ChoiceViewState(
        choices=("a", "b", "c"),
        labels={},
        disabled=set(),
    )
    assert state.selected_choice() == "a"
    state.move(1)
    assert state.selected_choice() == "b"
    state.move(2)
    assert state.selected_choice() == "c"
    state.move(1)  # clamped at end
    assert state.selected_choice() == "c"
    state.move(-1)
    assert state.selected_choice() == "b"


def test_choice_view_state_no_enabled_choices_returns_none():
    state = ChoiceViewState(
        choices=("x",),
        labels={},
        disabled={"x"},
    )
    assert state.enabled() == ()
    assert state.selected_choice() is None


def test_choice_view_state_key_navigation_and_selection():
    state = ChoiceViewState(
        choices=("a", "---", "b", ChoiceViewState.FREE_TEXT),
        labels={ChoiceViewState.FREE_TEXT: "Type freely..."},
        disabled={"---"},
    )

    assert state.handle_key("j") is TUI_MODAL_PENDING
    assert state.selected_choice() == "b"
    assert state.handle_key("1") is TUI_MODAL_PENDING
    assert state.handle_key("enter") == "a"

    state.selected = 2
    assert state.handle_key("enter") is SELECTION_FREE_TEXT


def test_choice_view_state_search_and_escape_layers():
    state = ChoiceViewState(choices=("alpha", "beta"), labels={}, disabled=set())

    state.handle_key("/")
    state.handle_key("any", "b")
    assert state.searching
    assert state.query == "b"
    assert state.selected_choice() == "beta"
    assert state.handle_key("escape") is TUI_MODAL_PENDING
    assert not state.searching
    assert state.query == "b"
    assert state.handle_key("escape") is TUI_MODAL_PENDING
    assert state.query == ""
    assert state.handle_key("escape") is SELECTION_BACK


def test_choice_view_state_fragments_preserve_headers_and_preview():
    state = ChoiceViewState(
        choices=("--- Models ---", "alpha"),
        labels={"--- Models ---": "  ---- Models ----", "alpha": "Alpha"},
        disabled={"--- Models ---"},
    )

    fragments = state.fragments("Model", lambda _choice: "first\\nsecond")
    rendered = "".join(text for _, text in fragments)

    assert "  ---- Models ----" in rendered
    assert ">  1. Alpha" in rendered
    assert "  │ first\n  │ second\n" in rendered


def test_emit_answer_compact_drops_invisible_lines(monkeypatch):
    out = []
    monkeypatch.setattr(render_module, "print_formatted_text", lambda text, **kwargs: out.append(getattr(text, "value", str(text))))
    ui = UiPrinter(output_fn=lambda text: None)
    ui.color = True
    ui.emit_answer("### Parent\n| status | value |\n| --- | --- |\n| model | `x` |\n", rule=False, compact=True)
    rendered = out[0]
    visible = [line for line in rendered.split("\n") if UiPrinter.SGR_RE.sub("", line).strip()]
    assert rendered.split("\n") == visible + [""]  # no blank or box-padding lines survive
    assert "Parent" in rendered and "model" in rendered


def _ask_state():
    return AskViewState.build(
        [
            AskSpec("Which shape?", choices=["Flat", "Sections"], previews=["**bold** flat table\n| a | b |", "sections tree"], recommended=0),
            AskSpec("Name?", choices=["core", "lib"]),
        ]
    )


def _rows(fragments):
    return "".join(text for _, text in fragments).splitlines()


def test_ask_view_side_by_side_joins_option_and_preview_rows():
    state = _ask_state()
    rows = _rows(state.fragments(width=120, max_height=30))
    # A leading blank row lifts the modal off whatever the activity region printed above it.
    assert rows[0] == ""
    assert rows[1] == "(1/2) Which shape?"
    assert rows[2] == ""  # blank line under the title
    assert rows[-2] == ""  # blank line above the key legend
    # The selected option's label and its rich preview land on the same rendered row.
    pair = next(row for row in rows if "Flat" in row and "flat table" in row)
    assert "1. Flat (recommended)" in pair
    # The longest option row and the preview column keep a visible gutter of at least 3 cells.
    before_preview = pair.split("bold")[0]
    assert len(before_preview) - len(before_preview.rstrip()) >= 3
    assert any("↑/↓ or j/k move" in row for row in rows)
    assert len(rows) <= 30


def test_ask_view_stacks_preview_below_options_on_narrow_terminals():
    state = _ask_state()
    rows = _rows(state.fragments(width=80, max_height=30))
    option_index = next(index for index, row in enumerate(rows) if "Flat" in row)
    preview_index = next(index for index, row in enumerate(rows) if "flat table" in row)
    assert preview_index > option_index  # stacked, not side-by-side
    assert rows[preview_index].startswith("  │ ")


def test_ask_view_truncates_overflow_with_more_lines():
    preview = "\n".join(f"line {i}" for i in range(40))
    state = AskViewState.build([AskSpec("Q?", choices=["A"], previews=[preview])])
    rows = _rows(state.fragments(width=120, max_height=8))
    assert len(rows) <= 8
    assert any("more lines" in row for row in rows)


def test_ask_view_preview_renders_rich_styles():
    state = AskViewState.build([AskSpec("Q?", choices=["Bold"], previews=["**bold text**"])])
    fragments = state.fragments(width=120, max_height=30)
    assert any(style for style, text in fragments if "bold text" in text)  # markdown bold carried a style


def test_ask_view_keys_navigate_advance_and_submit():
    state = _ask_state()
    assert state.pages[0].selected_choice() == "Flat"  # recommended pre-selected
    assert state.handle_key("j") is TUI_MODAL_PENDING
    assert state.pages[0].selected_choice() == "Sections"
    assert state.handle_key("k") is TUI_MODAL_PENDING
    assert state.pages[0].selected_choice() == "Flat"
    assert state.handle_key("enter") is TUI_MODAL_PENDING  # first page: pick and advance
    assert state.active == 1 and state.picked[0] == "Flat"
    assert state.handle_key("tab") is TUI_MODAL_PENDING  # cycle back to page 1
    assert state.active == 0
    assert state.handle_key("s-tab") is TUI_MODAL_PENDING
    assert state.active == 1
    assert state.handle_key("enter") is ASK_DONE  # last page submits the batch
    assert state.picked[1] == "core"


def test_ask_view_tab_to_last_page_does_not_submit_unanswered():
    """Tabbing to the last page and Enter must not submit a half-answered batch: it records the
    pick and jumps back to the first unanswered page."""
    state = AskViewState.build([AskSpec("One?", choices=["A"]), AskSpec("Two?", choices=["B"]), AskSpec("Three?", choices=["C"])])
    assert state.handle_key("tab") is TUI_MODAL_PENDING
    assert state.handle_key("tab") is TUI_MODAL_PENDING
    assert state.active == 2
    assert state.handle_key("enter") is TUI_MODAL_PENDING  # picked on the last page, batch not done
    assert state.picked[2] == "C"
    assert state.active == 0  # first unanswered page


def test_ask_view_out_of_order_answers_submit_when_all_answered():
    """Answers may land in any order; the batch only submits once every page has a pick."""
    state = AskViewState.build([AskSpec("One?", choices=["A"]), AskSpec("Two?", choices=["B"]), AskSpec("Three?", choices=["C"])])
    state.handle_key("tab")
    state.handle_key("tab")
    assert state.handle_key("enter") is TUI_MODAL_PENDING  # last page answered, batch not done
    assert state.active == 0
    assert state.handle_key("enter") is TUI_MODAL_PENDING  # page 0 picked
    assert state.active == 1
    assert state.handle_key("enter") is ASK_DONE  # page 1 picked: all answered
    assert state.picked == ["A", "B", "C"]


def test_ask_view_free_text_page_reports_and_escape_cancels():
    state = AskViewState.build([AskSpec("No choices")])
    assert state.handle_key("enter") == (ASK_FREE_TEXT, 0)
    assert state.handle_key("escape") is SELECTION_BACK  # whole batch cancelled
    result = state.handle_key("c-c")
    assert isinstance(result, KeyboardInterrupt)


def test_ask_view_notes_mode_edits_and_saves():
    state = AskViewState.build([AskSpec("Q?", choices=["A"])])
    assert state.handle_key("n") is TUI_MODAL_PENDING
    assert state.notes_mode
    assert state.handle_key("any", "x") is TUI_MODAL_PENDING
    assert state.handle_key("any", "y") is TUI_MODAL_PENDING
    assert state.note_buffer == "xy"
    assert state.handle_key("backspace") is TUI_MODAL_PENDING
    assert state.note_buffer == "x"
    assert state.handle_key("enter") is TUI_MODAL_PENDING  # save
    assert state.notes == {0: "x"} and not state.notes_mode
    assert state.handle_key("n") is TUI_MODAL_PENDING
    assert state.handle_key("any", "z") is TUI_MODAL_PENDING
    assert state.handle_key("escape") is TUI_MODAL_PENDING  # discard
    assert state.notes == {0: "x"}
    # The saved note renders on the page.
    assert "notes: x" in _rows(state.fragments(width=120, max_height=30))


def test_ask_view_notes_mode_opens_via_any_key_routing():
    """The bindings dispatch printable keys outside MODAL_KEYS as ("any", data); `n` must open
    notes mode through that path too, not only as the named key."""
    state = AskViewState.build([AskSpec("Q?", choices=["A"])])
    assert state.handle_key("any", "n") is TUI_MODAL_PENDING
    assert state.notes_mode


def test_ask_view_shift_tab_cycles_backwards():
    from wizolt.tui import TuiApp

    assert "s-tab" in TuiApp.MODAL_KEYS  # the binding table must route it into the modal
    state = AskViewState.build([AskSpec("1?", choices=["A"]), AskSpec("2?", choices=["B"])])
    assert state.active == 0
    assert state.handle_key("s-tab") is TUI_MODAL_PENDING
    assert state.active == 1
