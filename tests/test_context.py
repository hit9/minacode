"""Context projection and compaction: what a request carries, what compaction keeps, and the
history index it leaves behind."""

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










# --- AGENTS.md / CLAUDE.md injection (runtime.agents_md, default on) ---


















































































def _huge_history(tmp_path, steps, *, budget_tokens=200_000, chars=160_000):
    """A session that has already been compacted once, still over budget, with `steps` large
    assistant messages after the latest user message."""
    s = session(tmp_path)
    s.settings.max_context_tokens = budget_tokens
    s.state.summary = "old summary"
    s.messages = [
        {"role": "user", "content": COMPACTION_SUMMARY_TITLE + "\nold summary"},
        {"role": "user", "content": "keep going"},
        *({"role": "assistant", "content": f"step {index} " + "y" * chars} for index in range(steps)),
    ]
    return s, ContextManager(s)


class _CountingModel:
    def __init__(self):
        self.calls = 0

    def compact(self, _text, *_args, **_kwargs):
        self.calls += 1
        return {"summary": "new summary"}




























































# The request that opened a turn survives compaction because latest_user_index protects the last
# plain user message. Every user message the runtime generates on its own -- a mention expansion, a
# protocol correction -- therefore has to be marked as a session event, or it takes that protection
# for itself and the request it was expanding gets summarized away mid-turn. The worker is where
# this bites hardest: that message is the entire order (docs/worker.md), the worker cannot see the
# parent's history, and nothing re-sends it.
RUNTIME_GENERATED_EVENTS = ("mcp_mentions", "skill_mentions", "file_mentions", "tool_call_correction")








































