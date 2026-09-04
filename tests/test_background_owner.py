"""The CommandLoop background owner and the cancellation-safe blocking bridge.

Two invariants are worth their own module: work admitted by a session is settled by that session,
and cancelling a blocking bridge does not mean the worker stopped.
"""

import asyncio
import threading

import pytest
from tui_harness import loop as build_loop

from wizolt.base import run_blocking


async def test_accepted_background_tasks_are_retained_until_done(tmp_path):
    command_loop = build_loop(tmp_path)
    command_loop.open_background()
    release = asyncio.Event()

    async def work() -> None:
        await release.wait()

    task = command_loop.spawn_background(work(), name="probe")
    assert task is not None
    assert task in command_loop._background
    release.set()
    await task
    assert task not in command_loop._background


async def test_closing_rejects_later_work_and_closes_its_coroutine(tmp_path):
    command_loop = build_loop(tmp_path)
    command_loop.open_background()
    await command_loop.close_background()
    started = False

    async def work() -> None:
        nonlocal started
        started = True

    coroutine = work()
    assert command_loop.spawn_background(coroutine, name="late") is None
    assert not started
    # A refused coroutine that was merely dropped would surface as a never-awaited RuntimeWarning;
    # closing it here is what keeps rejection quiet.
    with pytest.raises(RuntimeError):
        coroutine.send(None)


async def test_close_background_cancels_and_awaits_accepted_work(tmp_path):
    command_loop = build_loop(tmp_path)
    command_loop.open_background()
    entered = asyncio.Event()
    unwound = False

    async def work() -> None:
        nonlocal unwound
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            unwound = True
            raise

    task = command_loop.spawn_background(work(), name="long")
    assert task is not None
    await entered.wait()
    await command_loop.close_background()
    assert task.done()
    assert unwound
    assert not command_loop._background


async def test_unexpected_background_failure_is_reported_once(tmp_path):
    command_loop = build_loop(tmp_path)
    reported: list[str] = []
    command_loop.emit = lambda text="", indent=0: reported.append(str(text))
    command_loop.open_background()

    async def work() -> None:
        raise ValueError("broken")

    task = command_loop.spawn_background(work(), name="failing")
    assert task is not None
    await asyncio.gather(task, return_exceptions=True)
    assert [line for line in reported if "failing" in line and "broken" in line]


def test_background_owner_serves_a_later_fresh_invocation(tmp_path):
    """No loop-bound task may leak across runs; reopening starts from an empty set.

    Synchronous on purpose: the contract under test is behavior across two separate `asyncio.run`
    invocations, which is exactly the boundary a single test-owned loop cannot express."""

    command_loop = build_loop(tmp_path)

    async def one_run() -> asyncio.Task:
        command_loop.open_background()
        task = command_loop.spawn_background(asyncio.sleep(60), name="leftover")
        assert task is not None
        await command_loop.close_background()
        return task

    first = asyncio.run(one_run())
    assert first.done()
    second = asyncio.run(one_run())
    assert second.done()
    assert second is not first


async def test_run_blocking_returns_the_worker_result():
    assert await run_blocking(lambda: 21 * 2) == 42


async def test_run_blocking_raises_the_worker_exception():
    def boom() -> None:
        raise ValueError("worker failed")

    with pytest.raises(ValueError, match="worker failed"):
        await run_blocking(boom)


async def test_cancelling_run_blocking_waits_for_the_worker_to_finish():
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def work() -> str:
        entered.set()
        release.wait(5)
        finished.set()
        return "done"

    task = asyncio.ensure_future(run_blocking(work))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0.05)
    # The awaiter is cancelled, but the worker is still holding whatever it holds: the bridge must
    # not report cancellation while that is true.
    assert not task.done()
    assert not finished.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


async def test_a_worker_that_fails_while_cancelling_reports_the_cancellation():
    entered = threading.Event()
    release = threading.Event()

    def work() -> None:
        entered.set()
        release.wait(5)
        raise ValueError("late failure")

    task = asyncio.ensure_future(run_blocking(work))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0.05)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_job_kill_quiesces_before_cancellation_returns(tmp_path):
    """JobTool's kill is the foundation's proof integration: TERM/KILL must not be left mid-flight."""

    from wizolt.tools.shell import JobTool

    command_loop = build_loop(tmp_path)
    session = command_loop.session
    entered = threading.Event()
    release = threading.Event()
    killed = threading.Event()

    class FakeJob:
        id = "job-1"
        status = "running"
        exit_code = None

        def update_status(self) -> None:
            return

        def kill(self) -> None:
            entered.set()
            release.wait(5)
            killed.set()

    session.jobs["job-1"] = FakeJob()  # type: ignore[assignment]
    tool = JobTool(session, [])
    task = asyncio.ensure_future(tool._kill({"job": "job-1"}))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0.05)
    assert not killed.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert killed.is_set()
