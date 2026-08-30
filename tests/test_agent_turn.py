"""The agent turn: the tool loop, interrupts, textual tool-call correction, live follow-ups,
parallel execution, and provider message conversion."""



from wizolt.base import (
    SESSION_EVENT_KEY,
)
from wizolt.context import ContextManager
from wizolt.engine import Agent
from wizolt.runner import ToolRunner
from wizolt.session import Session


def _correction(name):
    """A protocol correction exactly as the engine commits it.

    Marked as a session event: it is runtime-generated, not a user turn, so compaction's
    latest-user-message protection keeps pointing at the request that started the turn."""
    return {"role": "user", "content": Agent.tool_call_correction(name), SESSION_EVENT_KEY: "tool_call_correction"}


def _runner(tmp_path, input_reply=""):
    s = Session(cwd=str(tmp_path))
    return s, ToolRunner(s, ContextManager(s), input_fn=lambda *a: input_reply, output_fn=lambda *a: None)














































































































