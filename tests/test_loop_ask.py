"""loop ask (split from tests/test_loop_commands.py)."""
import json
import time
from types import SimpleNamespace

from agent_harness import session

import wizolt.cli.modals as modals_mod
from wizolt.base import (
    DISMISSED,
    SELECTION_BACK,
    LogBlock,
    LogLine,
    Text,
    ToolCall,
)
from wizolt.cli import CommandLoop
from wizolt.cli.modals import choice_application, question_interaction
from wizolt.engine import Agent
from wizolt.tools import AskSpec
from wizolt.tui import ASK_DONE, ASK_FREE_TEXT


def _answers(answer):
    """A stand-in for TuiApp.request_input: the question is put on the loop and awaited."""

    async def request_input(prompt):
        return answer(prompt) if callable(answer) else answer

    return request_input


def _modals(results):
    """A stand-in for TuiApp.show_modal, handing back one scripted result per call."""

    async def show_modal(fragments_fn, key_fn, **_kwargs):
        return next(results) if hasattr(results, "__next__") else results(fragments_fn, key_fn)

    return show_modal


async def test_choice_application_expands_escaped_preview_newlines(tmp_path):
    output = []
    loop = CommandLoop(Agent(session(tmp_path), output_fn=output.append), input_fn=lambda prompt="": "", output_fn=output.append)
    loop.interactive_input = True
    rendered = []

    class Modal:
        def show_modal_sync(self, fragments_fn, key_fn, exclusive=False):
            rendered.extend(fragments_fn())
            return key_fn("enter", "")

        async def show_modal(self, fragments_fn, key_fn, exclusive=False):
            return self.show_modal_sync(fragments_fn, key_fn, exclusive)

    loop.tui = Modal()

    result = await choice_application(
        loop,
        "Select:",
        ("A", "B"),
        {},
        "",
        set(),
        preview_fn=lambda choice: "one\\ntwo" if choice == "A" else "",
    )

    assert result == "A"
    previews = [text for style, text in rendered if style == "class:choice.preview"]
    assert previews == ["  │ one\n", "  │ two\n"]
    assert all("\\n" not in text for _, text in rendered)

async def test_ask_free_text_prompt_has_no_control_newline(tmp_path):
    """A free-text page drops out of the modal to the shared input row; the answer flows into
    the batch and the modal reopens (ASK_DONE ends it)."""
    output = []
    loop = CommandLoop(Agent(session(tmp_path), output_fn=output.append), input_fn=lambda prompt="": "", output_fn=output.append)
    loop.interactive_input = True
    prompts = []
    results = iter([(ASK_FREE_TEXT, 0), ASK_DONE])
    loop.tui = SimpleNamespace(
        request_input=_answers(lambda prompt: prompts.append(prompt) or "typed answer"),
        show_modal=_modals(results),
    )

    assert await question_interaction(loop, [AskSpec("Pick?", choices=["A"], previews=["preview"])]) == ["typed answer"]
    assert prompts == ["\nPick?"]  # one shared-input prompt, the question spelled out again

async def test_ask_free_text_empty_answer_is_kept(tmp_path):
    """An explicitly empty free-text answer is a legal answer: the batch must return [""] and
    never fall back to the question text (which is only the placeholder for unanswered pages)."""
    output = []
    loop = CommandLoop(Agent(session(tmp_path), output_fn=output.append), input_fn=lambda prompt="": "", output_fn=output.append)
    loop.interactive_input = True
    results = iter([(ASK_FREE_TEXT, 0), ASK_DONE])
    loop.tui = SimpleNamespace(
        request_input=_answers(""),
        show_modal=_modals(results),
    )

    assert await question_interaction(loop, [AskSpec("Pick?")]) == [""]

async def test_ask_free_text_on_last_question_submits_without_reentering_modal(tmp_path):
    """A free-text answer to the final question completes the batch right after the shared input
    row; the modal must not reopen for it (a second show_modal would fail the call-count assert)."""
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    loop.interactive_input = True
    calls = []

    def show_modal(fragments_fn, key_fn):
        calls.append(1)
        key_fn("enter")  # page 1: accept the preselected option, advance to page 2
        key_fn("2")  # page 2: move onto "Type freely..." (digits only move the cursor)
        return key_fn("enter")  # ...and select it -> drops to the shared input row

    loop.tui = SimpleNamespace(request_input=_answers("typed"), show_modal=_modals(show_modal))

    assert await question_interaction(loop, [AskSpec("One?", choices=["A"]), AskSpec("Two?", choices=["B"])]) == ["A", "typed"]
    assert len(calls) == 1

async def test_ask_without_choices_uses_shared_tui_input(tmp_path):
    """A question without choices is a single Type-freely page; Enter drops to the shared row."""
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt: "fallback", output_fn=lambda text: None)
    loop.interactive_input = True
    prompts = []
    results = iter([(ASK_FREE_TEXT, 0), ASK_DONE])
    loop.tui = SimpleNamespace(
        request_input=_answers(lambda prompt: prompts.append(prompt) or "typed answer"),
        show_modal=_modals(results),
    )

    assert await question_interaction(loop, [AskSpec("Explain the issue")]) == ["typed answer"]
    assert prompts == ["\nExplain the issue"]

async def test_ask_headless_keeps_plain_per_question_prompts(tmp_path):
    """Without a TUI the batch falls back to one read_input per question, in order."""
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt: "fallback", output_fn=lambda text: None)
    prompts = []
    loop.read_input_sync = lambda prompt: prompts.append(prompt) or "answer"

    assert await question_interaction(loop, [AskSpec("One?"), AskSpec("Two?", choices=["A"])]) == ["answer", "answer"]
    assert prompts == ["\nOne?", "\nTwo?"]

async def test_ask_choice_is_not_echoed_before_final_tool_log(tmp_path, monkeypatch):
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), output_fn=lambda text: None)
    emitted = []
    loop.emit = lambda text="", indent=0: emitted.append(text)
    async def answered(_loop, _specs):
        return ["B"]

    monkeypatch.setattr(modals_mod, "question_interaction", answered)

    assert await modals_mod.question_interaction(loop, [AskSpec("Which?", choices=["A", "B"])]) == ["B"]
    assert emitted == []

async def test_ask_notes_flow_into_the_answer(tmp_path):
    """A note entered on a page (`n`, text, Enter) is appended to that question's answer."""
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt: "fallback", output_fn=lambda text: None)
    loop.interactive_input = True

    def show_modal(fragments_fn, key_fn):
        key_fn("n")
        for ch in "keep the header":
            key_fn("any", ch)
        key_fn("enter")  # save the note
        return key_fn("enter")  # pick the recommended "A" and submit the batch

    loop.tui = SimpleNamespace(show_modal=_modals(show_modal))

    assert await question_interaction(loop, [AskSpec("Q?", choices=["A"], recommended=0)]) == ["A\n\nUser notes: keep the header"]

async def test_ask_escape_cancels_the_whole_batch(tmp_path):
    """Esc on any page cancels every question with the DISMISSED marker."""
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt: "fallback", output_fn=lambda text: None)
    loop.interactive_input = True
    loop.tui = SimpleNamespace(show_modal=_modals(lambda fragments_fn, key_fn: SELECTION_BACK))

    result = await question_interaction(loop, [AskSpec("One?"), AskSpec("Two?")])
    assert result == [DISMISSED, DISMISSED]

async def test_elapsed_since_uses_whole_seconds(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 104.9)
    assert Text.elapsed_since(100.0) == "4s"

    monkeypatch.setattr(time, "monotonic", lambda: 162.9)
    assert Text.elapsed_since(100.0) == "1m02s"

async def test_bash_live_start_pauses_standalone_status(tmp_path):
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), output_fn=lambda text: None)
    loop.ui.color = True
    loop.live_preview.start = lambda: setattr(loop.live_preview, "active", True)
    loop.status_bar.thread = object()
    loop.status_bar.stop = lambda: setattr(loop.status_bar, "thread", None)
    loop.status_bar.start = lambda **_kwargs: setattr(loop.status_bar, "thread", object())

    loop.tool_live_start()
    assert loop.live_status_paused is True
    assert loop.status_bar.thread is None

    loop.tool_live_output("", "")
    assert loop.live_status_paused is False
    assert loop.status_bar.thread is not None

async def test_command_loop_indents_intermediate_and_final_messages(tmp_path):
    output = []
    loop = CommandLoop(Agent(session(tmp_path), output_fn=output.append), output_fn=output.append)

    loop.emit_agent_output("First line.\nSecond line.")
    loop.ui.emit_answer("Done.\nFinal detail.")

    assert output == ["  First line.\n  Second line.", "Done.\nFinal detail."]

async def test_colored_assistant_and_tool_blocks_each_start_with_one_blank_line(tmp_path):
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda _text: None), output_fn=lambda _text: None)
    loop.ui.color = True
    loop.ui.emit_phase_rule = lambda: None  # the narration's opening rule is this test's noise
    loop.ui.rows_since_rule = 1  # not sitting directly under a rule: the blocks need their blank line
    events = []
    loop.emit = lambda text="", indent=0: events.append(text)
    loop.ui.emit_answer = lambda text, **_kwargs: events.append(text)
    first = LogBlock.hierarchy(LogLine("Bash", "first"), [])
    first_result = LogBlock.hierarchy(None, [LogLine("stored", "tr.1")])
    second = LogBlock.hierarchy(LogLine("Bash", "second"), [])

    loop.emit_agent_output("Working on it.")
    loop.tool_output(first)
    loop.tool_output(first_result)
    loop.tool_output(second)

    assert events == ["", "Working on it.", "", first, first_result, "", second]


def _colored_loop(tmp_path):
    """A CommandLoop with color on, whose rendered output is collected instead of printed."""
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda _text: None), output_fn=lambda _text: None)
    loop.ui.color = True
    loop.ui._scrollback_print = lambda _fragment: None
    return loop


async def test_interim_narration_closes_with_a_phase_rule_when_far_from_last_rule(tmp_path):
    """A turn's interim narration is closed by the same full-width rule the turn ends with,
    minus the label, provided the rule would not land too close to the one above it. The blank
    line and the narration's own rows count toward the distance."""
    loop = _colored_loop(tmp_path)
    rules = []
    loop.ui.emit_phase_rule = lambda: rules.append(1)
    # A rule has already been drawn this turn, so distance applies; one row short of the
    # threshold before the blank line and the narration itself count.
    loop.ui.rows_since_rule = loop.MIN_ROWS_BETWEEN_RULES - 1

    loop.emit_agent_output("Working on it.")

    assert rules == [1]


async def test_user_turn_opens_with_a_phase_rule(tmp_path):
    """The turn's opening rule sits under the user's message and always draws: the user's
    message is the top boundary of the turn, so every later rule measures its distance from it
    rather than the first narration being special-cased."""
    loop = _colored_loop(tmp_path)
    rules = []
    loop.ui.emit_phase_rule = lambda: rules.append(1)
    loop.ui.rows_since_rule = 100

    loop.user_turn_rule()

    assert rules == [1]


async def test_user_turn_rule_restarts_the_silent_batch_count(tmp_path):
    loop = _colored_loop(tmp_path)
    loop._silent_batches = 3

    loop.user_turn_rule()

    assert loop._silent_batches == 0


async def test_interim_narration_skips_the_rule_when_too_close_to_the_last_one(tmp_path):
    """The agent saying two things in quick succession is one phase, not two: a rule that would
    land within MIN_ROWS_BETWEEN_RULES of the one above it is skipped, so the transcript does not
    collect a row of dashes for every sentence."""
    loop = _colored_loop(tmp_path)
    rules = []
    loop.ui.emit_phase_rule = lambda: rules.append(1)
    loop.ui.rows_since_rule = 0

    loop.emit_agent_output("Working on it.")

    assert rules == []


async def test_final_answer_takes_no_phase_rule(tmp_path):
    """The turn-end rule already closes the turn, so the answer must not add a second rule of its
    own -- two rules in a row would read as a box."""
    loop = _colored_loop(tmp_path)
    rules = []
    loop.ui.emit_phase_rule = lambda: rules.append(1)
    loop.ui.rows_since_rule = 100

    loop.agent_answer_output("Done.")

    assert rules == []


async def test_tool_batch_closes_a_long_silent_run_with_a_phase_rule(tmp_path):
    """While the agent works in silence its calls run together; a stretch of silent tool
    batches -- the model never saying anything back -- closes with the same seam, fired after
    the batch's output is out so a batch is never cut in half."""
    loop = _colored_loop(tmp_path)
    rules = []
    loop.ui.emit_phase_rule = lambda: rules.append(1)
    loop._silent_batches = loop.TOOL_RUN_RULE_BATCHES - 1

    loop.tool_batch_output(True)

    assert rules == [1]


async def test_tool_batch_keeps_a_short_silent_run_together(tmp_path):
    loop = _colored_loop(tmp_path)
    rules = []
    loop.ui.emit_phase_rule = lambda: rules.append(1)
    loop._silent_batches = loop.TOOL_RUN_RULE_BATCHES - 2

    loop.tool_batch_output(True)

    assert rules == []


async def test_a_voiced_batch_is_not_silent(tmp_path):
    """A batch that carried narration reports through the same hook but does not count
    toward the silent run: the agent said something, so the seam is not needed yet."""
    loop = _colored_loop(tmp_path)
    rules = []
    loop.ui.emit_phase_rule = lambda: rules.append(1)
    loop._silent_batches = loop.TOOL_RUN_RULE_BATCHES - 1

    loop.tool_batch_output(False)

    assert rules == []
    assert loop._silent_batches == loop.TOOL_RUN_RULE_BATCHES - 1


async def test_engine_routes_the_answer_and_batch_end_to_the_loop(tmp_path):
    """The engine's final answer and batch-end facts are wired to the loop's presentation hooks:
    the answer goes to the no-rule path, and each tool batch reports its end."""
    loop = _colored_loop(tmp_path)

    assert loop.agent.final_output_fn.__func__ is CommandLoop.agent_answer_output
    assert loop.agent.on_tool_batch.__func__ is CommandLoop.tool_batch_output


async def test_phase_rule_renders_as_an_unlabelled_full_width_solid_rule(tmp_path):
    """The phase rule is the turn-end rule's line with the label removed: the same gray, the same
    solid dash, edge to edge, and no text of its own -- the narration it closes is the label."""
    loop = _colored_loop(tmp_path)
    frags = []
    loop.ui._scrollback_print = lambda fragment: frags.append(fragment)

    loop.ui.emit_phase_rule()

    assert [style for style, _ in frags[0]] == ["ansibrightblack"]
    text = "".join(fragment for _, fragment in frags[0])
    assert text.endswith("\n")
    assert set(text) <= {"─", "\n"}
    assert loop.ui.rows_since_rule == 0


async def test_emit_counts_rendered_rows_toward_rule_distance(tmp_path):
    """The distance between rules is measured in rendered rows, not in blocks: one Bash call with
    its output goes further than four Reads, and a wrapped block counts what it actually took."""
    loop = _colored_loop(tmp_path)

    loop.ui.emit("a\nb")
    assert loop.ui.rows_since_rule == 2
    loop.ui.emit("")
    assert loop.ui.rows_since_rule == 3


async def test_rule_due_reports_whether_a_rule_would_land_far_enough(tmp_path):
    """The distance query is the one place a phase rule's spacing is judged: color off never
    draws (there are no rules to be close to), and the threshold is inclusive."""
    loop = _colored_loop(tmp_path)
    assert not loop.ui.rule_due(loop.MIN_ROWS_BETWEEN_RULES)
    loop.ui.rows_since_rule = loop.MIN_ROWS_BETWEEN_RULES
    assert loop.ui.rule_due(loop.MIN_ROWS_BETWEEN_RULES)
    loop.ui.color = False
    assert not loop.ui.rule_due(loop.MIN_ROWS_BETWEEN_RULES)


async def test_turn_end_rule_resets_rule_distance(tmp_path):
    """The turn-end rule is a solid rule like any other, so the distance counter starts over at
    the close of a turn rather than carrying the whole turn's length into the next one."""
    loop = _colored_loop(tmp_path)
    loop.ui.emit("a\nb")

    loop.ui.emit_turn_end(1.0)

    assert loop.ui.rows_since_rule == 0


async def test_worker_interim_output_gets_the_same_phase_rule(tmp_path):
    """A worker's interim text is an interim reply like any other; it closes with the same rule,
    through the same narration path the main agent uses."""
    loop = _colored_loop(tmp_path)
    rules = []
    loop.ui.emit_phase_rule = lambda: rules.append(1)
    loop.ui.rows_since_rule = loop.MIN_ROWS_BETWEEN_RULES - 1

    loop.worker_answer_output("working")

    assert rules == [1]


async def test_full_turn_parts_at_user_rule_narration_and_silent_batches(tmp_path):
    """End to end through the engine: the turn opens with the user's rule, every interim
    narration closes with one once it is far enough from the rule above, a run of silent tool
    batches closes with one too, and the final answer takes none. Every tool batch reports its
    end through the engine hook."""
    loop = _colored_loop(tmp_path)
    rules = []
    loop.ui.emit_phase_rule = lambda: rules.append(loop.ui.rows_since_rule)
    silences = []
    on_batch = loop.agent.on_tool_batch
    loop.agent.on_tool_batch = lambda silent: (on_batch(silent), silences.append(silent))

    class FakeModel:
        on_stream = None

        def __init__(self):
            self.calls = 0

        async def request(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                return {}, [ToolCall("c1", "Bash", ["printf nar1"])], "先看入口。"
            if self.calls == 2:
                return {}, [ToolCall("c2", "Bash", ["printf nar2"])], "这里接着读。"
            if self.calls <= 6:
                return {}, [ToolCall(f"c{self.calls}", "Bash", ["printf silent"])], ""
            return {"role": "assistant", "content": "改完了。"}, [], "改完了。"

        def estimated_request_tokens(self, messages, tools=None):
            return 10

    loop.agent.model = FakeModel()

    loop.user_turn_rule()  # the turn's opening rule, drawn under the user's message
    assert await loop.agent.run("x") == "改完了。"

    assert len(rules) == 3  # user rule, the second narration's rule, the silent run's rule
    assert rules[0] == 0  # the user's rule always draws; the first narration lands too close to it and is skipped
    assert rules[1] >= loop.MIN_ROWS_BETWEEN_RULES  # the second narration's rule
    assert rules[2] >= loop.MIN_ROWS_BETWEEN_RULES  # the silent run's rule
    assert silences == [False, False, True, True, True, True]  # narration batches voiced, the rest silent


async def test_resumed_session_draws_user_narration_and_silent_batch_rules(tmp_path):
    """A resumed session replays its turns with the same phase rules the live run drew: the
    user's message opens each turn with a rule, interim narration closes with one once it is
    far enough from the rule above, and a silent run of tool batches closes with the batch
    rule -- even though the engine never runs again."""
    from wizolt.session import ToolResultRecord

    def rules_for(messages, records):
        loop = _colored_loop(tmp_path)
        s = loop.session
        s.resumed = True
        s.transcript_messages = messages
        s.tool_records = records
        rules = []
        real_rule = loop.ui.emit_phase_rule
        loop.ui.emit_phase_rule = lambda: (rules.append(loop.ui.rows_since_rule), real_rule())
        loop.render_resumed_session()
        return rules

    tool_call = lambda i: {"id": f"c{i}", "type": "function", "function": {"name": "Bash", "arguments": json.dumps([f"printf {i}"])}}
    record = lambda: ToolResultRecord(key="tr.1", name="Bash", args=[["printf x"]], output="x")

    # The user's message opens the turn with a rule even when nothing else draws.
    rules = rules_for(
        [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "answer"}],
        [],
    )
    assert len(rules) == 1

    # Interim narration opens with a rule once it is far enough from the rule above: the first
    # narration lands too close to the user's rule to draw, the second one clears it. A list
    # keeps its per-item lines instead of being folded into one paragraph by markdown.
    rules = rules_for(
        [
            {"role": "user", "content": "q1"},
            {
                "role": "assistant",
                "content": "- " + "\n- ".join(f"point {i}" for i in range(5)),
                "tool_calls": [tool_call(1)],
            },
            {"role": "assistant", "content": "later narration", "tool_calls": [tool_call(2)]},
            {"role": "assistant", "content": "answer"},
        ],
        [record(), record()],
    )
    assert len(rules) == 2
    assert rules[1] >= CommandLoop.MIN_ROWS_BETWEEN_RULES  # the second narration's rule is far enough

    # A silent run of four tool batches closes with the batch rule.
    rules = rules_for(
        [
            {"role": "user", "content": "q1"},
            *[{"role": "assistant", "content": "", "tool_calls": [tool_call(i)]} for i in range(1, 5)],
            {"role": "assistant", "content": "answer"},
        ],
        [record()] * 4,
    )
    assert len(rules) == 2
    assert rules[1] >= CommandLoop.MIN_ROWS_BETWEEN_RULES  # the silent run's rule is far enough
