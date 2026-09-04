"""compaction segments (split from tests/test_context.py)."""


class _StubModel:
    """Compactor requires a model; planning-only tests never touch it."""


import json
import os
import threading
from types import SimpleNamespace

import pytest
from agent_harness import session, session_with_provider
from test_context import RUNTIME_GENERATED_EVENTS

import wizolt.context as context_module
from wizolt import compaction
from wizolt.base import (
    SESSION_EVENT_KEY,
    ModelError,
)
from wizolt.cli import CommandLoop
from wizolt.cli.commands import compact
from wizolt.config import (
    DEFAULT_OUTPUT_RESERVE_TOKENS,
)
from wizolt.context import ContextManager
from wizolt.engine import Agent
from wizolt.model import ModelClient
from wizolt.prompts import (
    COMPACTION_SUMMARY_TITLE,
)
from wizolt.session import HistorySegment
from wizolt.skill import SkillLibrary


def test_compaction_captures_a_history_segment(tmp_path):
    s = session(tmp_path)
    context = ContextManager(s)
    compacted = [
        {"role": "user", "content": "find the parser bug"},
        {"role": "assistant", "content": "looking into it"},
    ]

    context.apply_compaction({"summary": "summary"}, [], compacted=compacted)

    assert len(s.history) == 1
    segment = s.history[0]
    assert segment.key == "seg.1"
    assert segment.title == "find the parser bug"
    assert "find the parser bug" in segment.text
    assert "looking into it" in segment.text


def test_segment_takes_the_name_the_compactor_gave_it(tmp_path):
    # The compaction reply already describes the span, and the model is the only party that read
    # all of it. The deterministic name is the first user message of the window, which says little
    # once a span starts mid-work — here, "ok".
    s = session(tmp_path)
    context = ContextManager(s)
    compacted = [
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "Extracted tokenize() and updated the imports."},
    ]

    data = {"title": "Tokenizer extraction", "summary": "s"}
    context.apply_compaction(data, [], compacted=compacted, title=compaction.Compactor.title(data))

    assert s.history[0].title == "Tokenizer extraction"


def test_segment_title_is_flattened_and_bounded(tmp_path):
    # Free-form model text landing in a listing, a checkpoint line, and a viewer column.
    s = session(tmp_path)
    context = ContextManager(s)

    data = {"title": '  "Parser refactor\n  and its tests"  ', "summary": "s"}
    context.apply_compaction(
        data,
        [],
        compacted=[{"role": "user", "content": "x"}],
        title=compaction.Compactor.title(data),
    )

    assert s.history[0].title == "Parser refactor and its tests"


def test_segment_title_falls_back_when_the_compactor_names_nothing(tmp_path):
    s = session(tmp_path)
    context = ContextManager(s)
    compacted = [{"role": "user", "content": "find the parser bug"}]

    for data in ({"summary": "s"}, {"title": "", "summary": "s"}, {"title": ["nope"], "summary": "s"}):
        s.history.clear()
        context.apply_compaction(data, [], compacted=compacted, title=compaction.Compactor.title(data))
        assert s.history[0].title == "find the parser bug", data


def test_deterministic_trim_still_names_its_segment(tmp_path):
    # No model reply at all: the summarizer failed and the span was trimmed deterministically.
    s = session(tmp_path)
    context = ContextManager(s)

    context.apply_compaction(None, [], compacted=[{"role": "user", "content": "find the parser bug"}], fallback_note="trimmed")

    assert s.history[0].title == "find the parser bug"


def test_large_history_segment_has_no_self_referential_recall_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(context_module, "MAX_TOOL_OUTPUT_TOKENS", 10)
    s = session(tmp_path)
    context = ContextManager(s)

    context.apply_compaction({"summary": "summary"}, [], compacted=[{"role": "user", "content": "x" * 1000}])

    assert "<bounded_output" in s.history[0].text
    assert 'recall="seg.1"' not in s.history[0].text


def test_compaction_history_keys_increment(tmp_path):
    s = session(tmp_path)
    context = ContextManager(s)

    context.apply_compaction({"summary": "s"}, [], compacted=[{"role": "user", "content": "first task"}])
    context.apply_compaction({"summary": "s"}, [], compacted=[{"role": "user", "content": "second task"}])

    assert [segment.key for segment in s.history] == ["seg.1", "seg.2"]


def test_checkpoint_names_the_whole_retained_archive_not_just_the_newest_span(tmp_path):
    """Each rebuild discards the previous checkpoint, so a line naming only the span this
    compaction stored leaves every older segment with no trace in context — and a model does not
    go looking for what it cannot see exists. Range and count only: whether to fetch, and what,
    stays the model's decision through RecallContext."""
    s = session(tmp_path)
    context = ContextManager(s)

    context.apply_compaction({"summary": "s"}, [], compacted=[{"role": "user", "content": "first task"}])
    assert "Recallable history: seg.1 (1 segment)" in s.messages[0]["content"]

    context.apply_compaction({"summary": "s"}, [], compacted=[{"role": "user", "content": "second task"}])
    context.apply_compaction({"summary": "s"}, [], compacted=[{"role": "user", "content": "third task"}])

    assert "Recallable history: seg.1..seg.3 (3 segments)" in s.messages[0]["content"]


def test_compaction_without_compacted_messages_captures_nothing(tmp_path):
    s = session(tmp_path)
    context = ContextManager(s)

    context.apply_compaction({"summary": "summary"}, [])

    assert s.history == []


async def test_prepare_messages_captures_history_and_turn_segments_in_one_pass(tmp_path):
    """An over-budget request can cross both compaction stages in one prepare: the history before the
    latest request becomes seg.1, then the oversized current turn itself becomes seg.2."""
    s = session_with_provider(tmp_path)
    s.settings.max_context_tokens = 1
    s.messages = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "latest request"},
    ]
    context = ContextManager(s)
    turn = [{"role": "user", "content": "current request"}, *({"role": "assistant", "content": f"step {index}"} for index in range(20))]

    class FakeModel:
        last_compaction_model = ""

        def __init__(self, session):
            self.session = session
            self.calls = 0
            self.cancel_requested = threading.Event()

        async def api_request(self, _messages, _tools, **_kwargs):
            self.calls += 1
            return "", "", json.dumps({"summary": f"summary {self.calls}"})

        @staticmethod
        def parse_json_object(content):
            return json.loads(content)

    model = FakeModel(s)
    await context.prepare_messages(model, "system", turn)

    assert model.calls == 2
    assert [segment.key for segment in s.history] == ["seg.1", "seg.2"]
    assert "old request" in s.history[0].text
    assert "step 0" in s.history[1].text
    assert "step 11" in s.history[1].text
    # The turn keeps its request and recent window; the compacted prefix is replaced by the summary.
    assert turn[0]["content"] == "current request"
    assert turn[1]["content"].startswith(COMPACTION_SUMMARY_TITLE)


def test_history_title_skips_summary_blocks(tmp_path):
    context = ContextManager(session(tmp_path))
    messages = [
        {"role": "user", "content": COMPACTION_SUMMARY_TITLE + "\nold summary"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "the real request"},
    ]

    assert context.history_title(messages) == "the real request"


def test_history_index_and_memory_are_not_injected_into_each_request(tmp_path):
    s = session(tmp_path)
    s.skills = SkillLibrary({})  # this test isolates history projection from optional context
    s.messages.extend([{"role": "user", "content": "old request"}, {"role": "assistant", "content": "old answer"}])
    s.history.append(HistorySegment(key="seg.1", title="find the bug", text="..."))
    context = ContextManager(s)

    messages = context.model_messages("system", [{"role": "user", "content": "current request"}])
    contents = [str(message.get("content") or "") for message in messages]
    assert contents[2:] == ["old request", "old answer", "current request"]
    assert not any(content.startswith("--- History index ---") for content in contents)
    assert not any(content.startswith("--- Memory ---") for content in contents)


async def test_compaction_fallback_trims_when_model_compact_fails(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    s.state.summary = "existing"
    s.messages = [{"role": "user", "content": str(index)} for index in range(10)]
    context = ContextManager(s)
    compaction_phases = []
    context.on_compaction = lambda active, _error: compaction_phases.append(active)

    class FailingModel:
        last_compaction_model = ""

        def compact(self, text, *_args, **_kwargs):
            raise ModelError("failed")

    await context.prepare_messages(FailingModel(), "system", [{"role": "user", "content": "request"}])

    assert compaction_phases == [True, False]
    assert s.state.summary != "existing"
    assert len(s.messages) == 2
    assert s.messages[0]["content"].startswith(COMPACTION_SUMMARY_TITLE)
    assert "deterministically trimmed" in s.messages[0]["content"]
    assert s.messages[1]["content"] == "9"
    # Even though summarization failed, the evicted conversation is still captured as a recallable
    # segment: the fallback summary is only a trim note, so this is the only way to recover it.
    assert [segment.key for segment in s.history] == ["seg.1"]
    assert "user:\n0" in s.history[0].text
    assert "user:\n8" in s.history[0].text


async def test_manual_compact_inserts_summary_before_latest_user(tmp_path):
    s = session_with_provider(tmp_path)
    # Long enough that something is actually evicted: the rule under test is where the checkpoint
    # lands relative to the latest request, which only exists once there is a head to replace.
    s.messages = [
        *({"role": "assistant", "content": f"old {index}"} for index in range(10)),
        {"role": "user", "content": "latest"},
        {"role": "tool", "content": "tool kept"},
    ]
    s.state.context_percent = 80
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), output_fn=lambda text: None)
    transitions = []
    loop.tui = SimpleNamespace(set_running=transitions.append, set_dispatching=lambda: transitions.append("dispatch"))

    class FakeModel:
        last_compaction_model = ""

        def __init__(self, session):
            self.session = session
            self.cancel_requested = threading.Event()

        async def api_request(self, _messages, _tools, **_kwargs):
            assert transitions == ["compacting context"]
            return "", "", json.dumps({"summary": "summary", "plan": ["next"], "known": ["fact"]})

        @staticmethod
        def parse_json_object(content):
            return json.loads(content)

    loop.agent.model = FakeModel(s)
    result = await compact(loop, "")

    assert len(s.messages) < 12  # a head was evicted
    assert s.messages[0]["content"].startswith(COMPACTION_SUMMARY_TITLE)  # the checkpoint leads
    assert s.messages[-2]["content"] == "latest"  # and sits before the latest request, not after
    assert s.messages[-1]["content"] == "tool kept"
    assert s.state.summary == "summary"
    assert transitions == ["compacting context", "dispatch"]
    assert "messages 12 -> " in result
    assert "prior summary inserted" in result


async def test_manual_compact_names_the_segment_with_the_compactor_title(tmp_path):
    """`/compact` must name the span the way the automatic pass does.

    apply_compaction takes the title as a parameter rather than reading it off `data`, so every
    caller has to pass it; the manual path is the one that is easy to forget, and a miss is silent
    -- the segment quietly falls back to the deterministic first-user-message name.
    """
    s = session_with_provider(tmp_path)
    s.messages = [
        *({"role": "assistant", "content": f"old {index}"} for index in range(10)),
        {"role": "user", "content": "latest"},
        {"role": "tool", "content": "tool kept"},
    ]
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), output_fn=lambda text: None)

    class FakeModel:
        last_compaction_model = ""

        def __init__(self, session):
            self.session = session
            self.cancel_requested = threading.Event()

        async def api_request(self, _messages, _tools, **_kwargs):
            return "", "", json.dumps({"title": "Tokenizer extraction", "summary": "summary"})

        @staticmethod
        def parse_json_object(content):
            return json.loads(content)

    loop.agent.model = FakeModel(s)
    await compact(loop, "")

    assert s.history[0].title == "Tokenizer extraction"


def test_agent_state_format_is_available_for_explicit_checkpoints(tmp_path):
    s = session(tmp_path)
    s.state.goal = "test goal"
    s.state.check = "all good"
    s.record_tool_error("tr.1", "Bash", ["bad"], "failed")

    ctx = s.state.format()

    assert "Goal:" in ctx
    assert "test goal" in ctx
    assert "Check:" in ctx
    assert "all good" in ctx
    assert "Recent tool errors:" not in ctx


def test_estimated_text_tokens_stays_on_characters_for_output_trimming(tmp_path):
    """estimated_text_tokens drives tool-output trimming (head/tail excerpts and the omitted marker),
    so it stays chars/4: UTF-8 bytes there would shrink the head/tail slice for CJK, overlapping the
    head and tail or inflating the bounded output marker. The request-level estimate is what counts
    bytes (test_cjk_payload_compacts_where_character_estimate_would_not)."""
    context = ContextManager(session(tmp_path))
    assert context.estimated_text_tokens("hello world") == (len("hello world") + 3) // 4
    # CJK stays at chars/4 too: 4 chars -> 1 estimated token, not 3.
    assert context.estimated_text_tokens("你好世界") == 1


async def test_cjk_payload_compacts_where_character_estimate_would_not(tmp_path):
    """A CJK-heavy session that the chars/4 estimate kept under budget now compacts: the bytes/4
    estimate clears the same budget, closing the gap between the status-bar fill and the trigger."""
    import json

    s = session(tmp_path)
    s.settings.max_context_tokens = 23_000  # budget 2520: chars/4 estimate ~2017, bytes/4 5406
    s.messages = [
        {"role": "user", "content": "你好" * 300},
        {"role": "assistant", "content": "收到" * 300},
        *({"role": "assistant", "content": "继续" * 100} for _ in range(8)),
        {"role": "user", "content": "中文" * 2000},
    ]
    context = ContextManager(s)
    compaction_phases = []
    context.on_compaction = lambda active, _error: compaction_phases.append(active)

    class FakeModel:
        last_compaction_model = ""

        def compact(self, text, *_args, **_kwargs):
            return {"summary": "compact summary", "plan": ["next"], "known": ["fact"]}

    turn = [{"role": "user", "content": "请用中文回复"}]
    messages = context.model_messages("system", turn)
    raw = context.request_tokens(messages)
    budget = context.request_token_budget()
    # The chars/4 figure sits under the budget; the UTF-8 bytes/4 estimate clears it.
    assert len(json.dumps(messages, ensure_ascii=False)) // 4 < budget < raw

    await context.prepare_messages(FakeModel(), "system", turn)
    assert compaction_phases == [True, False]
    assert s.state.compaction_count == 1


async def test_overdue_usage_triggers_compaction_even_when_estimate_fits(tmp_path):
    """The last completed request filled >=99% of its budget, so the next one compacts even though the
    bytes/4 estimate still fits: a last line of defense when the estimate is off. Below 99% the
    estimate alone decides, so a small follow-up after an 80% request is not compacted."""
    s = session(tmp_path)
    s.settings.max_context_tokens = 21_000  # budget 520; the ASCII payload estimates ~326
    s.messages = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old answer"},
        *({"role": "assistant", "content": f"recent {index}"} for index in range(8)),
        {"role": "user", "content": "latest"},
    ]
    context = ContextManager(s)
    compaction_phases = []
    context.on_compaction = lambda active, _error: compaction_phases.append(active)

    class FakeModel:
        last_compaction_model = ""

        def compact(self, text, *_args, **_kwargs):
            return {"summary": "compact summary", "plan": ["next"], "known": ["fact"]}

    turn = [{"role": "user", "content": "request"}]
    assert context.request_tokens(context.model_messages("system", turn)) < context.request_token_budget()

    # 98%: estimate fits and nothing compacts.
    s.usage.last_prompt_budget = 520
    s.usage.last_prompt_tokens = 510
    await context.prepare_messages(FakeModel(), "system", turn)
    assert compaction_phases == []
    assert s.state.compaction_count == 0

    # 100%: the overdue flag forces compaction despite the fitting estimate.
    s.usage.last_prompt_tokens = 520
    await context.prepare_messages(FakeModel(), "system", turn)
    assert compaction_phases == [True, False]
    assert s.state.compaction_count == 1
    # Compaction cleared the last-* signals, so the next request is not double-compacted by the
    # guard (the compaction request's own usage was just wiped instead of being mistaken for an
    # ordinary 100%-full context).
    assert s.usage.last_prompt_tokens == 0
    assert s.usage.last_prompt_budget == 0
    await context.prepare_messages(FakeModel(), "system", turn)
    assert compaction_phases == [True, False]
    assert s.state.compaction_count == 1

    # A fresh session with no recorded usage never trips the flag.
    s2 = session(tmp_path)
    context2 = ContextManager(s2)
    assert context2._overdue_by_usage() is False


def test_apply_compaction_clears_last_usage_but_keeps_cumulative(tmp_path):
    """Compaction rewrites history, so the recorded last-* usage no longer describes the next
    request. Clearing them (not the cumulative totals) makes the overdue guard and the status bar
    fall back to the local estimate until the next ordinary request reports real usage."""
    s = session(tmp_path)
    s.usage.last_prompt_tokens = 1234
    s.usage.last_prompt_budget = 1200
    s.usage.last_cached_prompt_tokens = 300
    s.usage.last_cache_write_prompt_tokens = 50
    s.usage.prompt_tokens = 9999
    s.usage.calls = 7
    s.messages = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old answer"},
        *({"role": "assistant", "content": f"recent {index}"} for index in range(8)),
        {"role": "user", "content": "latest"},
    ]
    context = ContextManager(s)
    compacted, keep = compaction.Compactor(context, _StubModel()).parts()
    context.apply_compaction({"summary": "compact summary", "plan": ["next"], "known": ["fact"]}, keep, compacted=compacted)

    assert s.usage.last_prompt_tokens == 0
    assert s.usage.last_prompt_budget == 0
    assert s.usage.last_cached_prompt_tokens == 0
    assert s.usage.last_cache_write_prompt_tokens == 0
    assert s.usage.prompt_tokens == 9999
    assert s.usage.calls == 7
    assert context._overdue_by_usage() is False


def test_compaction_override_does_not_change_request_budget(tmp_path):
    """The [compaction] entry never feeds the context budget: even pointing at a small-window entry,
    requests are still prepared against the active provider's window."""
    from wizolt.config import Config, request_budget_for

    s = session(tmp_path)
    s.config = Config.from_dict(
        {
            "compaction": {"provider": "small"},
            "provider": {
                "active": "wide",
                "wide": {"model": "wide", "max_context_tokens": 1_048_576},
                "small": {"model": "small", "max_context_tokens": 16_384},
            },
        }
    )
    context = ContextManager(s)
    assert context.request_token_budget() == request_budget_for(1_048_576, DEFAULT_OUTPUT_RESERVE_TOKENS)
    assert context.request_token_budget() > request_budget_for(16_384, DEFAULT_OUTPUT_RESERVE_TOKENS)


@pytest.mark.parametrize("event", RUNTIME_GENERATED_EVENTS)
async def test_turn_compaction_keeps_the_request_a_runtime_message_follows(tmp_path, event):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    context = ContextManager(s)
    turn = [
        {"role": "user", "content": "the whole order"},
        {"role": "user", "content": "runtime expansion", SESSION_EVENT_KEY: event},
        *({"role": "assistant", "content": f"step {index}"} for index in range(20)),
    ]

    class FakeModel:
        last_compaction_model = ""

        def compact(self, text, *_args, **_kwargs):
            return {"summary": "summary"}

    messages = await context.prepare_messages(FakeModel(), "system", turn)
    assert turn[0]["content"] == "the whole order"
    assert any(message.get("content") == "the whole order" for message in messages)
    # Kept verbatim instead of summarized: the segment holds the steps, never the order itself.
    assert "the whole order" not in s.history[-1].text
    assert "step 0" in s.history[-1].text


def test_history_compaction_keeps_the_request_a_runtime_message_follows(tmp_path):
    s = session(tmp_path)
    # Long enough that the recent window leaves a compactable head: the rule under test is where
    # the split lands, and there is no split to inspect when everything fits inside the window.
    s.messages = [
        *({"role": "assistant", "content": f"old {index}"} for index in range(10)),
        {"role": "user", "content": "the whole order"},
        {"role": "user", "content": "runtime expansion", SESSION_EVENT_KEY: "skill_mentions"},
        {"role": "assistant", "content": "working"},
    ]

    compacted, keep = compaction.Compactor(ContextManager(s), _StubModel()).parts()

    assert compacted  # older history is summarized away
    assert "the whole order" not in [message["content"] for message in compacted]
    # The request and the runtime expansion it produced stay together on the kept side.
    assert [message["content"] for message in keep][-3:] == ["the whole order", "runtime expansion", "working"]


async def test_turn_compaction_leaves_the_request_inside_the_cached_prefix(tmp_path):
    """Compaction is a cache break, and where it breaks is what it costs. Keeping the request in
    place puts the break behind it rather than on it, so the whole stable head -- system,
    environment, the request itself -- is still reused by the request that follows a compaction.
    The summary lands immediately after the request, so the expansion is what moves."""
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    context = ContextManager(s)
    turn = [
        {"role": "user", "content": "the whole order"},
        {"role": "user", "content": "runtime expansion", SESSION_EVENT_KEY: "skill_mentions"},
        *({"role": "assistant", "content": f"step {index}"} for index in range(20)),
    ]

    class FakeModel:
        last_compaction_model = ""

        def compact(self, text, *_args, **_kwargs):
            return {"summary": "summary"}

    client = ModelClient(s)
    before = client.wire(client.session.config.provider).messages(context.model_messages("system", turn))
    after = client.wire(client.session.config.provider).messages(await context.prepare_messages(FakeModel(), "system", turn))
    shared = 0
    for old, new in zip(before, after):
        if old != new:
            break
        shared += 1

    # The break falls after the request: everything through it, the request included, is reused.
    assert before[shared - 1].get("content") == "the whole order"
    assert after[shared].get("content", "").startswith(COMPACTION_SUMMARY_TITLE)


async def test_engine_marks_its_own_user_messages_as_session_events(tmp_path):
    """The engine half of the same rule: what run() appends around the request is not a request."""
    folder = os.path.join(tmp_path, ".wizolt", "skills", "triage")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "SKILL.md"), "w", encoding="utf-8") as handle:
        handle.write("---\nname: triage\ndescription: triage a bug\n---\nReproduce first.\n")
    with open(os.path.join(tmp_path, "issue.py"), "w", encoding="utf-8") as handle:
        handle.write("raise RuntimeError\n")
    s = session(tmp_path)  # discovers the skill written above

    async def mcp_mentions(_text):
        return "--- MCP MENTIONS ---"

    s.mcp = SimpleNamespace(resolve_mentions=mcp_mentions, render_tools_index=lambda: "", tools={}, resources={})
    agent = Agent(s, output_fn=lambda text: None)

    class FakeModel:
        last_compaction_model = ""

        async def request(self, messages, tools=None):
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = FakeModel()
    assert await agent.run("please $triage @file:issue.py") == "done"

    assert [message.get(SESSION_EVENT_KEY) for message in s.messages] == [None, "mcp_mentions", "skill_mentions", "file_mentions", None]
    assert ContextManager(s).latest_user_index(s.messages) == 0


async def test_repeated_compaction_keeps_one_request_and_one_checkpoint(tmp_path):
    """Surviving one pass is not the same as surviving four. Each pass re-reads what the previous
    one left, so a request kept by accident (a second copy, a stale checkpoint it hides behind)
    would drift round by round: the invariant is one verbatim request, one checkpoint, and a
    message stream where every tool result still answers a call that is still there."""
    s = session_with_provider(tmp_path)
    s.settings.max_context_tokens = 12000
    context = ContextManager(s)

    def steps(tag, count=8):
        messages = []
        for index in range(count):
            key = f"{tag}{index}"
            call_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": key, "type": "function", "function": {"name": "Read", "arguments": "{}"}}],
            }
            messages.extend((call_message, {"role": "tool", "tool_call_id": key, "content": f"tr.{key} " + "filler " * 900}))
        return messages

    class FakeModel:
        last_compaction_model = ""

        def __init__(self, session):
            self.session = session
            self.calls = 0
            self.cancel_requested = threading.Event()

        async def api_request(self, _messages, _tools, **_kwargs):
            self.calls += 1
            return "", "", json.dumps({"summary": f"summary {self.calls}"})

        @staticmethod
        def parse_json_object(content):
            return json.loads(content)

    model = FakeModel(s)
    turn = [
        {"role": "user", "content": "the whole order"},
        {"role": "user", "content": "--- SKILL MENTIONS ---\nbody", SESSION_EVENT_KEY: "skill_mentions"},
        {"role": "user", "content": "[Runtime protocol correction] ...", SESSION_EVENT_KEY: "tool_call_correction"},
        *steps("a"),
    ]
    for round_index in range(4):
        messages = await context.prepare_messages(model, "system", turn)
        assert [message.get("content") for message in messages].count("the whole order") == 1
        assert turn[0]["content"] == "the whole order"
        assert sum(1 for message in turn if str(message.get("content") or "").startswith(COMPACTION_SUMMARY_TITLE)) == 1
        called = [raw["id"] for message in messages for raw in message.get("tool_calls") or []]
        answered = [message.get("tool_call_id") for message in messages if message.get("role") == "tool"]
        assert sorted(called) == sorted(answered), f"round {round_index} split a tool call from its result"
        turn.extend(steps(f"b{round_index}", 6))

    assert model.calls == 4  # one pass per round, never a compaction loop
    assert [segment.key for segment in s.history] == ["seg.1", "seg.2", "seg.3", "seg.4"]
