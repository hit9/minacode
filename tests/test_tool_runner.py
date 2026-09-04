"""tool runner (split from tests/test_agent_turn.py)."""

import asyncio
import threading
import time

import pytest
from agent_harness import call, session

from wizolt.base import ToolCall
from wizolt.context import ContextManager
from wizolt.runner import ToolRunner
from wizolt.tools import Tool, toolblocks, tooloutput


def test_tool_runner_refusal_stops_batch_and_invalid_args_are_not_stored(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: "skip it", output_fn=lambda text: None)
    runner.run([call("Bash", [":"]), call("Edit", ["second.txt", [{"op": "create", "content": "second"}]])])

    assert s.tool_records == []
    assert len(s.tool_errors) == 1
    assert "skip it" in s.tool_errors[0].error
    assert not (tmp_path / "second.txt").exists()

    outputs = []
    bad = session(tmp_path)
    ToolRunner(bad, ContextManager(bad), output_fn=lambda text: outputs.append(str(text))).run([call("Bash", [])])
    assert bad.tool_records == []
    assert len(bad.tool_errors) == 1
    assert outputs and "· rejected:" in outputs[0]  # argument errors collapse to a quiet line


def test_rejected_and_failed_calls_collapse_a_multiline_display_to_one_line(tmp_path):
    # A tool's display is whatever its short_args produced, and Note's is the whole rendered note so
    # a successful call can print it. A rejection is meant to be a quiet one-liner and a failure
    # leads with a red tag, so neither may inherit those lines: a rejected Note used to dim its
    # entire body and bury the reason at the end of the last line.
    from wizolt.base import LogRole
    from wizolt.tools.toolblocks import ToolDisplay

    s = session(tmp_path)
    note = call("Note", [{"replace_plan": [f"Task {index}" for index in range(1, 11)]}])
    display = ToolDisplay()
    display.display = tooloutput.short_call(s, note)
    assert len(display.display.splitlines()) > 1, "precondition: Note renders a multi-line display"

    rejected = list(toolblocks.reject_display(s, note, "ToolError: Note fields is only valid for view", d=display).walk())
    assert [item.role for item, _ in rejected] == [LogRole.MUTED]
    assert len(rejected[0][0].text.splitlines()) == 1
    assert rejected[0][0].text.endswith("· rejected: Note fields is only valid for view")

    failed = list(toolblocks.finish_display(s, note, "", "Note: disk is full", failed=True, d=display).walk())
    assert all(len((item.text or "").splitlines()) == 1 for item, _ in failed)

    # A successful Note is the one case that should keep every line: printing the note is the point.
    assert len(toolblocks.finish_display(s, note, "", "ok", failed=False, d=display).splitlines()) > 1


def test_tool_runner_refuses_without_reason_on_n(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: "n", output_fn=lambda text: None)

    runner.run([call("Bash", [":"])])

    assert s.tool_errors[0].error == "Cancelled: user refused tool call"


def test_tool_runner_refuses_with_direct_reason_input(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: "not now", output_fn=lambda text: None)

    runner.run([call("Bash", [":"])])

    assert s.tool_records == []
    assert len(s.tool_errors) == 1
    assert "not now" in s.tool_errors[0].error


def test_recall_tool_runner_does_not_create_new_result_keys(tmp_path):
    s = session(tmp_path)
    key = s.store_tool_result("Read", ["a.txt"], "result")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run([call("Recall", [key])])
    assert [record.key for record in s.tool_records] == [key]


def test_replayed_delegate_keeps_its_call_line(tmp_path):
    # The finish block drops its own root when a worker_rule is wired, because live the closing
    # full-width rule carries the call instead. Transcript replay wires no rule and passes no
    # output, so the rule never fired and the saved call rendered as an orphaned `stored tr.N`
    # under nothing. Replay takes the fallback root; the live path still hands its rule over.
    from wizolt.base import ToolCall

    s = session(tmp_path)
    delegate = ToolCall(id="", name="Delegate", args=[{"action": "send", "order": "do it"}])

    replayed = str(toolblocks.finish_display(s, delegate, "tr.7", "", failed=False))
    assert replayed.splitlines()[0].strip().startswith("[worker]")
    assert "stored tr.7" in replayed

    # Live, a wired rule still takes the root away: the rule is the call line there.
    live = str(toolblocks.finish_display(s, delegate, "tr.7", "", failed=False, worker_rule=lambda text: None))
    assert "[worker]" not in live


def test_read_only_batch_keeps_model_order_and_honors_the_concurrency_cap(tmp_path):
    """Independent read-only calls overlap, but never more than max_parallel_tools at once, and
    their results are published in the order the model emitted them rather than completion order."""
    for name in ("a", "b", "c", "d"):
        (tmp_path / f"{name}.txt").write_text(name + "\n", encoding="utf-8")
    s = session(tmp_path)
    s.settings.max_parallel_tools = 2
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    lock = threading.Lock()
    live = peak = 0
    execute = runner.execute_readonly

    def traced(call):  # the runner's own parameter name; the module-level `call` helper is unused here
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        try:
            time.sleep(0.02)  # long enough for the cap to be observable, short enough for CI
            return execute(call)
        finally:
            with lock:
                live -= 1

    runner.execute_readonly = traced
    calls = [ToolCall(f"r{index}", "Read", [{"path": f"{name}.txt"}]) for index, name in enumerate("abcd")]
    messages = runner.run(calls)

    assert [message["tool_call_id"] for message in messages] == ["r0", "r1", "r2", "r3"]
    assert peak == 2


def test_one_failing_read_only_call_leaves_its_siblings_alone(tmp_path):
    """A failure is converted at that call's own result boundary: the batch still returns one
    matched result per call, and the healthy siblings keep their output."""
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8")
    s = session(tmp_path)
    s.settings.max_parallel_tools = 4
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    calls = [
        ToolCall("ok1", "Read", [{"path": "a.txt"}]),
        ToolCall("bad", "Read", [{"path": "missing.txt"}]),
        ToolCall("ok2", "Read", [{"path": "b.txt"}]),
    ]
    contents = {message["tool_call_id"]: str(message["content"]) for message in runner.run(calls)}

    assert list(contents) == ["ok1", "bad", "ok2"]
    assert "alpha" in contents["ok1"]
    assert "beta" in contents["ok2"]
    assert "status: failed" in contents["bad"]


def test_edit_barrier_splits_a_batch_and_serializes_its_mutations(tmp_path):
    """Edits plan together and run serially; a mutating non-Edit call is a barrier that ends the
    edit segment, so the file the later edit resolves against is the one the earlier edit left."""
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    s = session(tmp_path)
    s.settings.max_parallel_tools = 4
    s.settings.yolo = True
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    calls = [
        ToolCall("e1", "Edit", ["b.txt", "", [{"op": "create", "content": "two\n"}]]),
        ToolCall("e2", "Edit", ["c.txt", "", [{"op": "create", "content": "three\n"}]]),
    ]
    assert runner.edit_segment_end(calls, 0) == 2  # edits plan and run as one segment
    assert runner.parallel_segment_end(calls, 0) == 0  # Edit never joins a parallel segment

    messages = runner.run(calls)
    assert [message["tool_call_id"] for message in messages] == ["e1", "e2"]
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "two\n"
    assert (tmp_path / "c.txt").read_text(encoding="utf-8") == "three\n"

    barrier = [calls[0], ToolCall("b1", "Bash", [":"])]
    assert runner.edit_barrier(barrier[1])  # a mutating non-Edit call ends the edit segment
    assert runner.edit_segment_end(barrier, 0) == 1


class _SlowTool(Tool):
    """A tool that cannot be interrupted, and records exactly when it finished its work."""

    NAME = "Slow"
    marker = None
    delay = 0.15

    def call(self):
        time.sleep(self.delay)
        assert _SlowTool.marker is not None
        _SlowTool.marker.write_text("finished", encoding="utf-8")
        return "done"


async def test_cancellation_waits_for_a_thread_backed_tool_before_reporting(tmp_path, monkeypatch):
    """Cancelling the task that awaits a worker does not stop the worker. So the runner keeps
    waiting: the tool's own work is finished before cancellation is reported, which is what makes
    "cancelled" mean the turn is no longer touching anything."""
    s = session(tmp_path)
    marker = tmp_path / "marker.txt"
    monkeypatch.setattr(_SlowTool, "marker", marker)
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    call = asyncio.ensure_future(runner.call_tool_async(_SlowTool(s, [])))
    await asyncio.sleep(0.02)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    # Written before the cancellation surfaced, and nothing writes it afterwards.
    assert marker.read_text(encoding="utf-8") == "finished"
    before = marker.stat().st_mtime_ns
    await asyncio.sleep(0.25)
    assert marker.stat().st_mtime_ns == before


async def test_a_stop_hook_is_requested_once_per_cancellation_and_awaited(tmp_path):
    """The stop hook is a request the runner makes on the way to waiting, never a substitute for
    waiting: it may be asked repeatedly, and the invocation is still awaited to quiescence."""
    s = session(tmp_path)
    stops = []
    released = threading.Event()

    class Cooperative(Tool):
        NAME = "Cooperative"

        def call(self):
            released.wait(2)
            return "stopped"

        def request_stop(self):
            stops.append(True)
            released.set()

    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)
    call = asyncio.ensure_future(runner.call_tool_async(Cooperative(s, [])))
    await asyncio.sleep(0.02)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    assert stops  # asked to stop ...
    assert released.is_set()  # ... and the worker really unwound before cancellation surfaced


def test_repeated_synchronous_runs_retain_no_loop_bound_state(tmp_path):
    """`run()` is `asyncio.run` over `run_async`, so anything it left behind would belong to a
    closed loop. Two calls on the same runner must therefore both work."""
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    first = runner.run([ToolCall("r1", "Read", [{"path": "a.txt"}])])
    second = runner.run([ToolCall("r2", "Read", [{"path": "a.txt"}])])

    assert "alpha" in str(first[0]["content"])
    assert "alpha" in str(second[0]["content"])
    assert runner._capacity is None and runner._gateway is None


async def test_the_synchronous_facade_refuses_to_run_inside_a_loop(tmp_path):
    """asyncio.run cannot nest, and the alternative -- a second loop on a helper thread -- would
    own none of the resources it touched. So the facade names the async method instead."""
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    with pytest.raises(RuntimeError, match="ToolRunner.run_async"):
        runner.run([])
