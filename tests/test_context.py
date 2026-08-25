"""Context projection and compaction: what a request carries, what compaction keeps, and the
history index it leaves behind."""

import json
import threading

from agent_harness import session, session_with_provider

from minacode.base import Billing
from minacode.context import ContextManager
from minacode.prompts import (
    COMPACTION_SUMMARY_TITLE,
)

# --- AGENTS.md / CLAUDE.md injection (runtime.agents_md, default on) ---


def _huge_history(tmp_path, steps, *, budget_tokens=200_000, chars=160_000):
    """A session that has already been compacted once, still over budget, with `steps` large
    assistant messages after the latest user message."""
    s = session_with_provider(tmp_path)
    s.settings.max_context_tokens = budget_tokens
    s.state.summary = "old summary"
    s.messages = [
        {"role": "user", "content": COMPACTION_SUMMARY_TITLE + "\nold summary"},
        {"role": "user", "content": "keep going"},
        *({"role": "assistant", "content": f"step {index} " + "y" * chars} for index in range(steps)),
    ]
    return s, ContextManager(s)


class _CountingModel:
    def __init__(self, session):
        self.calls = 0
        self.session = session
        self.cancel_requested = threading.Event()
        self.last_compaction_model = ""

    def api_request(self, _messages, _tools, *, allow_stream, response_timeout, provider, json_object, billing=Billing.MAIN):
        self.calls += 1
        return None, [], '{"summary": "new summary"}'

    @staticmethod
    def parse_json_object(text):
        return json.loads(text)


# The request that opened a turn survives compaction because latest_user_index protects the last
# plain user message. Every user message the runtime generates on its own -- a mention expansion, a
# protocol correction -- therefore has to be marked as a session event, or it takes that protection
# for itself and the request it was expanding gets summarized away mid-turn. The worker is where
# this bites hardest: that message is the entire order (docs/worker.md), the worker cannot see the
# parent's history, and nothing re-sends it.
RUNTIME_GENERATED_EVENTS = ("mcp_mentions", "skill_mentions", "file_mentions", "tool_call_correction")
