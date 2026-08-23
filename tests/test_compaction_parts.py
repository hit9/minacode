"""compaction parts (split from tests/test_context.py)."""
import os
import platform
import re
import shutil
from dataclasses import replace
from types import SimpleNamespace
import pytest
from agent_harness import call, session
import minacode.context as context_module
from minacode.base import (
    MAX_AGENTS_MD_TOKENS,
    MAX_TOOL_OUTPUT_TOKENS,
    SESSION_EVENT_KEY,
    ModelError,
)
from minacode.cli import CommandLoop
from minacode.cli.commands import compact
from minacode.config import (
    DEFAULT_OUTPUT_RESERVE_TOKENS,
    MIN_CONTEXT_SAFETY_TOKENS,
)
from minacode.context import ContextManager
from minacode.engine import Agent
from minacode.model import ModelClient
from minacode.prompts import (
    COMPACTION_SUMMARY_TITLE,
    CURRENT_TURN_CONTEXT_TRIMMED,
    LIVE_FOLLOWUP_PREFIX,
    PREVIOUS_CONTEXT_TRIMMED,
    SYSTEM_PROMPT,
)
from minacode.runner import ToolRunner
from minacode.session import HistorySegment, Session
from minacode.skill import SkillLibrary
from minacode.tools import EditTool, ReadTool

def test_compaction_parts_keep_latest_user_turn_after_prior_summary(tmp_path):
    s = session(tmp_path)
    summary = COMPACTION_SUMMARY_TITLE + "\nold summary"
    s.messages = [
        {"role": "user", "content": summary},
        {"role": "assistant", "content": "before"},
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
        # Padding so the recent window still leaves a head to compact; the rule under test is which
        # side of the split each kind of message lands on.
        *({"role": "assistant", "content": f"filler {index}"} for index in range(8)),
        {"role": "user", "content": "latest request"},
        {"role": "user", "content": summary},
        {"role": "assistant", "content": "working"},
        {"role": "tool", "content": "tool tr.1"},
    ]

    compacted, keep = ContextManager(s).compaction_parts()

    assert [message["content"] for message in compacted][:3] == ["before", "old request", "old answer"]
    # The latest request and everything after it stay together; the prior summary is dropped from
    # both sides rather than carried into either.
    assert [message["content"] for message in keep][-3:] == ["latest request", "working", "tool tr.1"]
    assert summary not in [message["content"] for message in (*compacted, *keep)]

def test_compaction_parts_compact_all_without_plain_user_message(tmp_path):
    s = session(tmp_path)
    s.messages = [
        {"role": "user", "content": COMPACTION_SUMMARY_TITLE + "\nold summary"},
        {"role": "assistant", "content": "answer"},
        {"role": "tool", "content": "tool tr.1"},
    ]

    compacted, keep = ContextManager(s).compaction_parts()

    assert compacted == s.messages[1:]
    assert keep == []

def test_compaction_selection_keeps_assistant_text_that_quotes_summary_marker(tmp_path):
    s = session(tmp_path)
    quoted = COMPACTION_SUMMARY_TITLE + "\nquoted by assistant"
    s.messages = [
        {"role": "user", "content": COMPACTION_SUMMARY_TITLE + "\nold summary"},
        {"role": "assistant", "content": quoted},
    ]

    compacted, keep = ContextManager(s).compaction_parts()

    assert compacted == [{"role": "assistant", "content": quoted}]
    assert keep == []

def test_prepare_messages_does_not_recompact_a_summary_by_itself(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    summary = COMPACTION_SUMMARY_TITLE + "\nold summary"
    s.state.summary = "old summary"
    s.messages = [{"role": "user", "content": summary}]

    class FakeModel:
        def compact(self, text, *_args, **_kwargs):
            raise AssertionError(f"synthetic summary was compacted again: {text}")

    ContextManager(s).prepare_messages(FakeModel(), "system")

    assert s.messages == [{"role": "user", "content": summary}]
    assert s.state.compaction_count == 0
    assert s.history == []

def test_turn_compaction_does_not_recompact_a_prior_summary(tmp_path):
    context = ContextManager(session(tmp_path))
    summary = COMPACTION_SUMMARY_TITLE + "\nold summary"
    messages = [
        {"role": "user", "content": "current request"},
        {"role": "user", "content": summary},
        *({"role": "assistant", "content": f"step {index}"} for index in range(10)),
    ]

    compacted, keep = context.turn_compaction_parts(messages)

    assert [message["content"] for message in compacted] == ["step 0", "step 1"]
    assert keep[0]["content"] == "current request"
    assert all(message.get("content") != summary for message in [*compacted, *keep])

def test_turn_compaction_evicts_the_prefix_before_a_late_followup(tmp_path):
    context = ContextManager(session(tmp_path))
    messages = [
        {"role": "user", "content": "original request"},
        *({"role": "assistant", "content": f"old step {index}"} for index in range(20)),
        {"role": "user", "content": "late follow-up"},
        *({"role": "assistant", "content": f"new step {index}"} for index in range(10)),
    ]

    compacted, keep = context.turn_compaction_parts(messages)

    assert compacted[0]["content"] == "original request"
    assert "old step 19" in [message["content"] for message in compacted]
    assert keep[0]["content"] == "late follow-up"
    assert [message["content"] for message in keep[1:]] == [f"new step {index}" for index in range(2, 10)]

def test_prepare_request_persists_current_turn_compaction_without_pending_input(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    agent = Agent(s, output_fn=lambda _text: None)
    turn = [
        {"role": "user", "content": "original request"},
        *({"role": "assistant", "content": f"old step {index}"} for index in range(20)),
        {"role": "user", "content": "continue"},
        *({"role": "assistant", "content": f"new step {index}"} for index in range(10)),
    ]
    agent.model.compact = lambda _text, *_args, **_kwargs: {"summary": "compact summary"}

    agent.prepare_request(turn)

    assert len(turn) < 21  # incidental: a 1-token budget collapses the kept tail by size
    assert turn[0]["content"] == "continue"
    assert turn[1]["content"].startswith(COMPACTION_SUMMARY_TITLE)
    assert "original request" in s.history[-1].text

def test_accepted_followup_commits_staged_current_turn_compaction(tmp_path):
    s = session(tmp_path)
    agent = Agent(s, output_fn=lambda _text: None)
    turn = [
        {"role": "user", "content": "original request"},
        *({"role": "assistant", "content": f"old step {index}"} for index in range(20)),
    ]
    transcript = [agent.transcript_message(turn[0])]
    s.enqueue_user_input("late follow-up")
    agent.model.compact = lambda _text, *_args, **_kwargs: {"summary": "compact summary"}
    agent.context.request_token_budget = lambda: 10
    agent.context.request_tokens = lambda messages, tools=None: 100 if any("old step" in str(message.get("content") or "") for message in messages) else 1

    request = agent.prepare_request(turn)

    assert any("old step" in str(message.get("content") or "") for message in turn)
    assert not any("old step" in str(message.get("content") or "") for message in request.turn_messages)
    agent.accept_pending_inputs(turn, transcript, request.pending, request.turn_messages)
    assert turn == request.turn_messages
    assert transcript[-1]["content"].endswith("late follow-up")
    assert s.state.compaction_count == 1

    agent.prepare_request(turn)

    assert s.state.compaction_count == 1

def test_interrupted_current_turn_compaction_falls_back_before_cancelling(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    context = ContextManager(s)
    turn = [{"role": "user", "content": "request"}, *({"role": "assistant", "content": f"step {index}"} for index in range(20))]
    phases = []
    context.on_compaction = lambda active, error: phases.append((active, error))

    class InterruptedModel:
        def compact(self, _text, *_args, **_kwargs):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        context.prepare_messages(InterruptedModel(), "system", turn)

    # The count is incidental here -- this budget is 1 token, so the size bound collapses the kept
    # tail. What matters is that the trim happened and left its marker before the interrupt flew.
    assert len(turn) < 21
    assert turn[0]["content"] == "request"
    assert turn[1]["content"].startswith(COMPACTION_SUMMARY_TITLE)
    assert CURRENT_TURN_CONTEXT_TRIMMED in turn[1]["content"]
    assert phases == [(True, ""), (False, "cancelled by user")]

def test_compaction_parts_bounds_the_work_after_the_last_request(tmp_path):
    """One request can drive dozens of tool calls. /compact must summarize that tail too, or a
    long turn leaves the context as large as it started."""
    s = session(tmp_path)
    s.messages = [{"role": "user", "content": "older"}, {"role": "assistant", "content": "older answer"}]
    s.messages.append({"role": "user", "content": "do the big thing"})
    for i in range(30):
        s.messages.append(
            {"role": "assistant", "content": f"step {i}", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "Read", "arguments": "{}"}}]}
        )
        s.messages.append({"role": "tool", "content": f"tool tr.{i}"})

    compacted, keep = ContextManager(s).compaction_parts()

    # The request that started the work is kept, plus a bounded window of what followed.
    assert keep[0] == {"role": "user", "content": "do the big thing"}
    assert len(keep) <= ContextManager.COMPACT_RECENT_MESSAGES + 1
    assert len(compacted) == len(s.messages) - len(keep)
    # A kept tool result never loses the call it answers.
    if keep[1].get("role") == "tool":
        raise AssertionError("kept tail starts with an orphaned tool result")

def test_compaction_parts_for_uses_last_fixed_window(tmp_path):
    messages = [{"role": "assistant", "content": f"m{index}"} for index in range(10)]

    older, recent = ContextManager(session(tmp_path)).compaction_parts_for(messages)

    assert [message["content"] for message in older] == ["m0", "m1"]
    assert [message["content"] for message in recent] == [f"m{index}" for index in range(2, 10)]

def test_prepare_messages_skips_compaction_when_context_under_budget(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 999_999
    s.messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "answer"}]
    context = ContextManager(s)
    compaction_phases = []
    context.on_compaction = lambda active, _error: compaction_phases.append(active)

    class ExplodingModel:
        def compact(self, text, *_args, **_kwargs):
            raise AssertionError(text)

    context.prepare_messages(ExplodingModel(), "system", [{"role": "user", "content": "request"}])

    assert compaction_phases == []
    assert s.messages == [{"role": "user", "content": "old"}, {"role": "assistant", "content": "answer"}]
