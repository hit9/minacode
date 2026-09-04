"""Whole-process probes for the one-loop runtime.

In-process tests observe behavior; these observe what the process itself does on the way out. A
loop that closed over a live client, a task nobody awaited, or a coroutine nobody ran shows up as
interpreter noise on stderr and nowhere else -- and pytest's own loop management would hide it.
"""
import os
import subprocess
import sys

PROBE_ENV = {**os.environ, "PYTHONWARNINGS": "error::RuntimeWarning", "PYTHONASYNCIODEBUG": "1"}


def run_probe(source: str) -> subprocess.CompletedProcess:
    # check=False: the exit code is one of the things being asserted, so a failing probe has to come
    # back as a result to compare, not as an exception that hides its stderr.
    return subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, cwd=os.getcwd(), env=PROBE_ENV, timeout=120, check=False)


CANCEL_DURING_REQUESTS = '''
import asyncio, sys, tempfile

from wizolt.config import Config
from wizolt.engine import Agent
from wizolt.session import Session, bootstrap_features


def build():
    tmp = tempfile.mkdtemp()
    raw = {
        "provider": {"active": "d", "d": {"url": "http://provider.invalid", "key": "k", "model": "m"}},
        "mcp": {"slow": {"url": "http://server.invalid/mcp"}},
        "paths": {"data_dir": tmp + "/data"},
    }
    session = Session(cwd=tmp, config=Config.from_dict(raw))
    bootstrap_features(session)
    return session


async def main():
    session = build()
    agent = Agent(session, output_fn=lambda text: None)

    async def never_answers(*args, **kwargs):
        await asyncio.Event().wait()

    agent.model.api_request = never_answers
    turn = asyncio.ensure_future(agent.run("go"))
    await asyncio.sleep(0.2)
    agent.cancel()
    try:
        await turn
    except asyncio.CancelledError:
        pass

    # And an MCP operation still in flight when the session is closed.
    session.mcp.tools["slow"] = []

    async def hangs(*args, **kwargs):
        await asyncio.Event().wait()

    session.mcp._call_tool = hangs
    call = asyncio.ensure_future(session.mcp.call_tool("slow", "echo", {}))
    await asyncio.sleep(0.1)
    await session.mcp.close()
    try:
        await call
    except BaseException:
        pass
    await agent.model.close()
    return 0


sys.exit(asyncio.run(main()))
'''


def test_cancelling_a_live_request_and_mcp_call_exits_cleanly():
    """Cancel a model turn and close MCP with a call in flight, then let the process end.

    Empty stderr is the assertion: "Task was destroyed but it is pending", "Event loop is closed",
    or a never-awaited coroutine would each print here and nowhere else."""
    result = run_probe(CANCEL_DURING_REQUESTS)

    assert result.stderr == ""
    assert result.returncode == 0


MUTATION_AFTER_CANCELLATION = '''
import asyncio, os, sys, tempfile, time

from wizolt.config import Config
from wizolt.context import ContextManager
from wizolt.base import ToolCall
from wizolt.runner import ToolRunner
from wizolt.session import Session, bootstrap_features
from wizolt.tools import TOOL_REGISTRY, Tool

MARKER = tempfile.mkdtemp() + "/marker.txt"


class SlowMutation(Tool):
    """A tool that mutates on a worker and cannot be interrupted.

    The worker is the risk: cancelling the task that awaits it does not stop it, so without the
    runner waiting, this write would land after its turn was already reported as cancelled."""

    NAME = "SlowMutation"

    def call(self):
        time.sleep(0.3)
        with open(MARKER, "w", encoding="utf-8") as handle:
            handle.write("first turn")
        return "done"


TOOL_REGISTRY["SlowMutation"] = SlowMutation


async def main():
    tmp = tempfile.mkdtemp()
    session = Session(cwd=tmp, config=Config.from_dict({"paths": {"data_dir": tmp + "/data"}}))
    bootstrap_features(session)
    session.settings.yolo = True
    runner = ToolRunner(session, ContextManager(session), output_fn=lambda text: None)

    batch = asyncio.ensure_future(runner.run([ToolCall("m1", "SlowMutation", [])]))
    await asyncio.sleep(0.05)
    batch.cancel()
    try:
        await batch
    except asyncio.CancelledError:
        pass

    # Cancellation was reported, so the tool is done touching anything.
    stamp = os.stat(MARKER).st_mtime_ns

    # A second turn, on the same runner: the first tool must not write again underneath it.
    second = ToolRunner(session, ContextManager(session), output_fn=lambda text: None)
    await second.run([ToolCall("m2", "Note", [{"replace_plan": ["next"]}])])
    await asyncio.sleep(0.4)

    if os.stat(MARKER).st_mtime_ns != stamp:
        print("the cancelled tool wrote again after the next turn", file=sys.stderr)
        return 1
    return 0


sys.exit(asyncio.run(main()))
'''


def test_a_cancelled_mutating_tool_cannot_write_after_the_next_turn():
    """Cancellation is only reported once the mutating tool has quiesced.

    The marker is the whole point: whatever the tool was going to write is written before the turn
    reports cancellation, and nothing appears under the turn that follows it."""
    result = run_probe(MUTATION_AFTER_CANCELLATION)

    assert result.stderr == ""
    assert result.returncode == 0
