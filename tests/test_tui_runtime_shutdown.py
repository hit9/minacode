"""The runtime's cancellation, ordered output, and shutdown contract on one owned loop.

Every test here drives `TuiRuntime` on the test's own event loop with a stand-in application, so
what is exercised is the runtime's ordering -- not prompt-toolkit's rendering, which the
interactive TUI tests cover.
"""
import asyncio

import pytest
from tui_harness import loop as command_loop_for
from tui_harness import wait_for

from wizolt.cli import TuiRuntime
from wizolt.cli.runtime import ScrollbackWriter
from wizolt.tools import Tool
from wizolt.tui import TuiApp


class FakeTui:
    """The application as the runtime sees it: a task it awaits, not a thread it joins."""

    def __init__(self):
        self.statuses: list[str] = []
        self.input_mode = "chat"
        self.exited = asyncio.Event()
        self.write_error: BaseException | None = None

    async def run(self, style=None):
        del style
        self.on_ready()
        await self.exited.wait()

    def exit(self):
        self.exited.set()

    def set_running(self, label):
        self.statuses.append(label)
        self.input_mode = "running"

    def set_idle(self):
        self.statuses.append("idle")
        self.input_mode = "chat"

    def invalidate(self):
        pass

    def invalidate_frame(self):
        pass

    async def write_to_scrollback(self, callback):
        if self.write_error is not None:
            raise self.write_error
        callback()


def runtime_for(tmp_path, monkeypatch, tui=None):
    """A runtime wired to a stand-in application, with the slow startup work stubbed out."""
    command_loop = command_loop_for(tmp_path)
    tui = tui or FakeTui()
    runtime = TuiRuntime(command_loop)
    # Attached for the tests that drive one turn directly; `run` installs the same object.
    command_loop.tui = tui
    monkeypatch.setattr(runtime, "build_tui", lambda: tui)
    monkeypatch.setattr(command_loop, "start_session", lambda: None)
    monkeypatch.setattr(command_loop, "refresh_mentions", lambda: None)
    return runtime, command_loop, tui


async def run_until(runtime, action):
    """Run one whole runtime session, calling `action` once it is live, and return its exit code."""
    session = asyncio.ensure_future(runtime.run())
    await wait_for(lambda: runtime.scrollback is not None)
    result = action()
    if asyncio.iscoroutine(result):
        await result
    return await session


def turn_that_unwinds(started, quiesced):
    """An `Agent.run` that owns its task the way the real one does, and unwinds slowly.

    Cancellation is where the interesting part is: the turn keeps running past its own
    CancelledError -- a tool still reaping a process, a snapshot still being written -- and only
    then settles, which is the moment the runtime is allowed to call itself idle."""

    async def run(_user_input, agent=None):
        agent._active_task = asyncio.current_task()
        agent._active_loop = asyncio.get_running_loop()
        try:
            started.set()
            await asyncio.Event().wait()  # never answers on its own
        except asyncio.CancelledError:
            await quiesced.wait()
            raise
        finally:
            agent._active_task = None
            agent._active_loop = None

    return run


async def test_ctrl_c_holds_cancelling_until_the_turn_has_settled(tmp_path, monkeypatch):
    """`running -> cancelling -> idle`, with the last step gated on settlement.

    Going idle at the moment Ctrl-C is pressed would be a lie: the turn is still unwinding, and the
    prompt would invite the next request on top of a tool that has not let go yet."""
    runtime, command_loop, tui = runtime_for(tmp_path, monkeypatch)
    runtime.runtime_loop = asyncio.get_running_loop()
    started, quiesced = asyncio.Event(), asyncio.Event()
    agent = command_loop.agent
    monkeypatch.setattr(agent, "run", lambda user_input: turn_that_unwinds(started, quiesced)(user_input, agent))
    emitted: list[str] = []
    monkeypatch.setattr(command_loop, "emit_turn", emitted.append)

    turn = asyncio.ensure_future(runtime.run_agent_turn("do it"))
    await started.wait()
    assert tui.statuses == ["working"]

    runtime.interrupt()
    await asyncio.sleep(0)  # let the cancellation reach the turn
    assert tui.statuses == ["working", "cancelling"]
    assert not turn.done()  # the turn is unwinding, so the runtime is not idle yet

    quiesced.set()
    await turn

    assert tui.statuses == ["working", "cancelling", "idle"]
    assert emitted == ["Cancelled"]
    assert not runtime.cancel_pending  # cleared for the next turn


async def test_a_second_ctrl_c_during_the_unwind_does_not_re_cancel(tmp_path, monkeypatch):
    """The status stays on `cancelling`: an impatient second press must not restart the sequence."""
    runtime, command_loop, tui = runtime_for(tmp_path, monkeypatch)
    runtime.runtime_loop = asyncio.get_running_loop()
    started, quiesced = asyncio.Event(), asyncio.Event()
    agent = command_loop.agent
    monkeypatch.setattr(agent, "run", lambda user_input: turn_that_unwinds(started, quiesced)(user_input, agent))
    cancels: list[int] = []
    real_cancel = agent.cancel
    monkeypatch.setattr(agent, "cancel", lambda: cancels.append(1) or real_cancel())

    turn = asyncio.ensure_future(runtime.run_agent_turn("do it"))
    await started.wait()
    runtime.interrupt()
    runtime.interrupt()
    runtime.interrupt()

    assert tui.statuses.count("cancelling") == 1
    assert len(cancels) == 1  # the turn is asked to stop once, however impatient the reader is

    quiesced.set()
    await turn
    assert tui.statuses == ["working", "cancelling", "idle"]


# --- ordered output ----------------------------------------------------------------------------


async def test_scrollback_writes_land_in_submit_order_and_the_barrier_awaits_them():
    """One FIFO queue, and a barrier that returns only once everything before it is on screen.

    This is what keeps a promoted answer above the tool batch that followed it: the turn awaits the
    barrier between the two, so the ordering is a wait rather than a hope."""
    written: list[str] = []

    async def write(callback):
        await asyncio.sleep(0)  # a real terminal write suspends the application first
        callback()

    writer = ScrollbackWriter(asyncio.get_running_loop(), write, lambda callback: written.append("direct"))
    for index in range(5):
        writer.submit(lambda index=index: written.append(f"write {index}"))

    assert written == []  # nothing runs on the submitting side
    await writer.barrier()

    assert written == [f"write {index}" for index in range(5)]
    await writer.close()


async def test_a_write_submitted_after_close_takes_the_direct_fallback():
    """A refused write is printed directly, never dropped: shutdown must not eat output."""
    written: list[str] = []
    writer = ScrollbackWriter(asyncio.get_running_loop(), _appending(written, "queued"), lambda callback: callback())
    await writer.close()

    writer.submit(lambda: written.append("late"))

    assert written == ["late"]


async def test_a_terminal_write_error_reaches_the_runtime():
    """A terminal that refused a write is the runtime's problem, not a background task's.

    The pump cannot raise where it happens -- nobody is awaiting it -- so it keeps the first failure
    and hands it to whoever next awaits a barrier."""
    tui = FakeTui()
    tui.write_error = OSError("terminal went away")
    writer = ScrollbackWriter(asyncio.get_running_loop(), tui.write_to_scrollback, lambda callback: None)
    writer.submit(lambda: None)

    with pytest.raises(OSError, match="terminal went away"):
        await writer.barrier()

    # Reported once: the next barrier is not haunted by the failure already raised.
    await writer.barrier()
    await writer.close()


def _appending(sink, label):
    async def write(callback):
        del callback
        sink.append(label)

    return write


# --- shutdown ----------------------------------------------------------------------------------


async def test_exit_during_a_model_request_shuts_down_gracefully(tmp_path, monkeypatch):
    """Ctrl-D while a turn is in flight: the turn is cancelled and awaited, then the app exits.

    `run` returning 0 is the whole assertion -- a shutdown that left the turn, the writer, or
    the application task behind would either hang here or surface as an unobserved task error."""
    runtime, command_loop, tui = runtime_for(tmp_path, monkeypatch)
    started, quiesced = asyncio.Event(), asyncio.Event()
    agent = command_loop.agent
    monkeypatch.setattr(agent, "run", lambda user_input: turn_that_unwinds(started, quiesced)(user_input, agent))
    monkeypatch.setattr(command_loop, "save_and_emit_resume", lambda: None)

    session = asyncio.ensure_future(runtime.run())
    await wait_for(lambda: runtime.runtime_loop is not None)
    runtime.submit_chat("do it")
    await started.wait()

    quiesced.set()  # the turn settles as soon as it is asked to
    runtime.request_exit()

    assert await session == 0
    assert tui.exited.is_set()
    assert runtime.scrollback is None
    assert command_loop.tui is None


async def test_shutdown_closes_output_after_the_turn_and_before_the_application(tmp_path, monkeypatch):
    """The fixed order: settle the turn, drain what was accepted, then take the terminal away.

    Exiting the application first would drop writes that had already been accepted, and closing
    resources under a live turn would pull the model client out from under it."""
    runtime, command_loop, tui = runtime_for(tmp_path, monkeypatch)
    order: list[str] = []

    async def close_resources():
        order.append("resources")

    monkeypatch.setattr(command_loop, "close_resources", close_resources)
    monkeypatch.setattr(command_loop, "close_background_output", lambda: order.append("gate"))
    monkeypatch.setattr(tui, "exit", lambda: order.append("exit") or tui.exited.set())
    turns: list[str] = []
    monkeypatch.setattr(command_loop.agent, "cancel", lambda: turns.append("cancel"))

    assert await run_until(runtime, runtime.request_shutdown) == 0
    assert order == ["resources", "gate", "exit"]
    assert turns == ["cancel"]


async def test_shutdown_cancels_and_awaits_a_background_task_it_started(tmp_path, monkeypatch):
    """Exit during discovery: a task the runtime spawned is cancelled and awaited, not abandoned."""
    runtime, command_loop, tui = runtime_for(tmp_path, monkeypatch)
    del command_loop, tui
    unwound = asyncio.Event()

    async def discovery():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            unwound.set()
            raise

    async def request_shutdown_after_spawn():
        runtime.spawn(discovery(), name="discovery")
        await wait_for(lambda: any(task.get_name() == "discovery" for task in runtime.tasks))
        runtime.request_shutdown()

    assert await run_until(runtime, request_shutdown_after_spawn) == 0
    assert unwound.is_set()
    assert runtime.tasks == set()


async def test_force_exit_arms_a_bounded_deadline_that_shutdown_disarms(tmp_path, monkeypatch):
    """Force exit asks for the graceful path first and only then arms its deadline.

    The timer is the escape hatch for a runtime that cannot unwind; a runtime that does unwind must
    cancel it, or a session that exited cleanly would still be SIGTERMed a second later."""
    runtime, command_loop, tui = runtime_for(tmp_path, monkeypatch)
    cancelled = []
    monkeypatch.setattr(command_loop.agent, "cancel", lambda: cancelled.append(1))

    assert await run_until(runtime, runtime.force_exit) == 0

    timer = runtime.force_exit_timer
    assert timer is not None
    # `finished`, not `is_alive`: cancel() sets that event, and the timer thread is only reaped
    # once it wakes -- asserting on liveness would be racing the scheduler, not the disarm.
    assert timer.finished.is_set()  # disarmed by the shutdown it asked for
    assert cancelled  # and the turn was asked to stop before the deadline was armed
    assert tui.exited.is_set()


# --- pending input requests --------------------------------------------------------------------


async def test_cancelling_a_pending_approval_restores_the_input_mode():
    """An approval or Ask prompt that is cancelled hands back None and gives the prompt back.

    None rather than a string: "" is a real submission that `confirm` reads as approve, so a
    cancelled request must be its own value. The mode is restored in `finally`, which is what keeps
    a cancelled tool from stranding the input row on `approval` for the rest of the session."""
    app = TuiApp()

    pending = asyncio.ensure_future(app.request_input("Approve? "))
    await wait_for(lambda: app.input_mode == "approval")

    app.cancel_input()

    assert await pending is None
    assert app.input_mode == "chat"


async def test_cancelling_the_waiter_itself_leaves_no_unobserved_task_error():
    """The turn awaiting an approval can be cancelled from under it -- Ctrl-C during a prompt.

    The request is not a task of its own here, so its cancellation belongs to the waiter: the
    CancelledError is delivered to whoever awaited it, and nothing is left for the loop's exception
    handler to complain about at shutdown."""
    app = TuiApp()
    unobserved: list[dict] = []
    asyncio.get_running_loop().set_exception_handler(lambda _loop, context: unobserved.append(context))

    pending = asyncio.ensure_future(app.request_input("Approve? "))
    await wait_for(lambda: app.input_mode == "approval")
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending

    assert app.input_mode == "chat"  # the prompt is back even though nobody answered
    assert unobserved == []


# --- the tool-output browser -------------------------------------------------------------------


class BrowsingTui(FakeTui):
    """A stand-in whose modal never answers, so an opened browser stays open for the test."""

    def __init__(self):
        super().__init__()
        self.opened = 0
        self.settled = 0

    async def show_modal(self, fragments_fn, key_fn, *, exclusive=False):
        del fragments_fn, key_fn, exclusive
        self.opened += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.settled += 1
            raise


async def browsing_runtime(tmp_path, monkeypatch):
    """A live runtime with one stored result, so Ctrl-O has something to show."""
    runtime, command_loop, tui = runtime_for(tmp_path, monkeypatch, tui=BrowsingTui())
    command_loop.session.store_tool_result("Bash", ["printf hi"], Tool.process_result("BashToolResult", 0, "hi", ""))
    return runtime, command_loop, tui


async def test_a_second_ctrl_o_does_not_open_a_competing_browser(tmp_path, monkeypatch):
    """Repeated Ctrl-O while the browser is open is ignored, not queued.

    Modals open one at a time, so a second browser would sit behind the first on the idle event and
    open itself the moment the reader closed the one they asked for."""
    runtime, _command_loop, tui = await browsing_runtime(tmp_path, monkeypatch)

    async def browse():
        runtime.expand_output()
        first = runtime.browser
        await wait_for(lambda: tui.opened == 1)
        runtime.expand_output()
        runtime.expand_output()
        await asyncio.sleep(0)
        assert runtime.browser is first
        assert tui.opened == 1
        runtime.request_shutdown()

    await run_until(runtime, browse)
    assert tui.opened == 1


async def test_shutdown_settles_an_open_browser(tmp_path, monkeypatch):
    """Closing the runtime cancels the browser and lets its modal finish unwinding.

    A modal future left pending is the failure this guards: the application would exit around a
    viewer that never resolved, and the next opener would wait on an idle event nobody sets."""
    runtime, _command_loop, tui = await browsing_runtime(tmp_path, monkeypatch)

    async def browse():
        runtime.expand_output()
        await wait_for(lambda: tui.opened == 1)
        runtime.request_shutdown()

    await run_until(runtime, browse)

    assert runtime.browser is not None
    assert runtime.browser.cancelled()
    assert tui.settled == 1


async def test_the_loop_keeps_running_while_the_browser_is_open(tmp_path, monkeypatch):
    """The browser is a task now, not a worker thread the loop waits on: a heartbeat still ticks."""
    runtime, _command_loop, tui = await browsing_runtime(tmp_path, monkeypatch)
    beats = 0

    async def heartbeat():
        nonlocal beats
        while True:
            beats += 1
            await asyncio.sleep(0)

    async def browse():
        pulse = asyncio.ensure_future(heartbeat())
        runtime.expand_output()
        await wait_for(lambda: tui.opened == 1)
        seen = beats
        await wait_for(lambda: beats > seen + 5)
        pulse.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pulse
        runtime.request_shutdown()

    await run_until(runtime, browse)
