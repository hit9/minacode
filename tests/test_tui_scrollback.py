"""TUI scrollback output through the owned renderer and stock fallback.

The TUI renderer writes above its live layout without erasing it when CPR established the layout's
terminal origin. Until then, UiPrinter falls back to prompt-toolkit's erase/write/repaint suspend.
Consecutive result lines are batched in both cases.

The tests cover both paths with recording outputs, then exercise CPR, animation, synchronous and
batched output, native history, and resize against a real tmux pane.
"""

import asyncio
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
import threading
import time

import pytest
from prompt_toolkit.application import Application, create_app_session
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.data_structures import Size
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.patch_stdout import patch_stdout
from tui_harness import session

from minacode.cli import CommandLoop, TuiRuntime
from minacode.engine import Agent
from minacode.render import UiPrinter
from minacode.tui.renderer import install_scrollback_renderer

# Every terminal-writing method of prompt_toolkit's Output interface. The recorder logs the
# method name and arguments, so the test can count the renderer.erase() fingerprint that marks
# each application suspend.
RECORDED_CALLS = [
    "write",
    "write_raw",
    "flush",
    "erase_screen",
    "erase_down",
    "erase_up",
    "erase_end_of_line",
    "reset_attributes",
    "set_attributes",
    "enable_autowrap",
    "disable_autowrap",
    "cursor_goto",
    "cursor_up",
    "cursor_down",
    "cursor_forward",
    "cursor_backward",
    "hide_cursor",
    "show_cursor",
    "ask_for_cpr",
    "set_title",
    "clear_title",
    "enter_alternate_screen",
    "quit_alternate_screen",
    "enable_mouse_support",
    "disable_mouse_support",
    "enable_bracketed_paste",
    "disable_bracketed_paste",
]


class RecordingOutput(DummyOutput):
    """A DummyOutput that records every terminal-writing call."""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[str, tuple]] = []
        self.lock = threading.Lock()

    def get_size(self):
        return Size(rows=24, columns=80)

    def _record(self, name, args):
        with self.lock:
            self.calls.append((name, args))

    def snapshot(self, baseline=0):
        with self.lock:
            return self.calls[baseline:]


for _name in RECORDED_CALLS:

    def _make(name):
        def method(self, *args):
            self._record(name, args)

        return method

    setattr(RecordingOutput, _name, _make(_name))


def suspend_count(calls):
    """Count application suspends by the renderer.erase() fingerprint.

    Renderer.erase writes cursor moves, then erase_down + reset_attributes + enable_autowrap
    + flush. That three-call run is unique to erase; ordinary diff repaints never call it.
    """
    count = 0
    for index in range(len(calls) - 2):
        if calls[index][0] == "erase_down" and calls[index + 1][0] == "reset_attributes" and calls[index + 2][0] == "enable_autowrap":
            count += 1
    return count


def written_text(calls):
    return "".join(args[0] for name, args in calls if name == "write" and args)


def fragment_text(fragments):
    return "".join(piece for _, piece in fragments)


def _run_application_with_emit(rec, emit, *, owned=False):
    """Run a minimal application; `emit` runs while it is live and must finish before exit."""
    layout = Layout(
        HSplit(
            [
                Window(
                    FormattedTextControl([("class:divider", "--- \u25cf responding ... ---")]),
                    dont_extend_height=True,
                ),
                Window(BufferControl(Buffer()), dont_extend_height=True),
            ]
        )
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=rec):
        app = Application(layout=layout, output=rec, input=pipe, full_screen=False)
        renderer = install_scrollback_renderer(app) if owned else app.renderer

        async def drive():
            await asyncio.sleep(0.05)  # let the first render land
            if owned:
                renderer.report_absolute_cursor_row(23)
            emit()
            await asyncio.sleep(0.3)  # batching window + run_in_terminal round trip
            app.exit()

        app.pre_run_callables.append(lambda: app.create_background_task(drive()))
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(asyncio.wait_for(app.run_async(), timeout=5))


def test_tool_output_burst_suspends_the_application_once():
    """Three consecutive tool-log emits erase the live application once, and the lines stay in order."""
    rec = RecordingOutput()
    ui = UiPrinter(print)
    ui.color = True  # the real TUI constructs the printer against a tty, so styling is on

    def emit():
        ui.emit("tool line one\n")
        ui.emit("tool line two\n")
        ui.emit("tool line three\n")

    _run_application_with_emit(rec, emit)

    burst = rec.snapshot()
    assert suspend_count(burst) == 1, "a burst of emits must suspend the application exactly once"
    text = written_text(burst)
    assert text.index("tool line one") < text.index("tool line two") < text.index("tool line three")


def test_owned_renderer_prints_without_suspending_and_restores_the_absolute_cursor():
    class BottomOutput(RecordingOutput):
        def get_rows_below_cursor_position(self):
            return 2

    rec = BottomOutput()
    ui = UiPrinter(print)
    ui.color = True

    _run_application_with_emit(rec, lambda: ui.emit("tool line\n"), owned=True)

    burst = rec.snapshot()
    assert suspend_count(burst) == 0
    raw = "".join(args[0] for name, args in burst if name == "write_raw" and args)
    assert "\x1b[1;22r" in raw
    region = next(index for index, call in enumerate(burst) if call == ("write_raw", ("\x1b[1;22r",)))
    assert burst[region + 1] == ("cursor_goto", (22, 1))
    reset = next(index for index, call in enumerate(burst) if call == ("write_raw", ("\x1b[r",)))
    assert burst[reset + 1] == ("cursor_goto", (24, 1))


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is required for the terminal integration boundary")
def test_real_tmux_scrollback_keeps_one_footer_and_orders_sync_and_batched_output():
    """Exercise CPR, the real tmux grid, synchronous promotion, and an ordinary queued emit."""
    socket = f"minacode-scrollback-{os.getpid()}-{time.monotonic_ns()}"
    script = textwrap.dedent(
        """
        import os
        import threading
        import time

        from minacode.render import UiPrinter
        from minacode.tui.app import TuiApp
        from minacode.tui.renderer import ScrollbackRenderer

        print("\\n".join(f"HISTORY_{index}" for index in range(12)), flush=True)
        ui = UiPrinter(print)
        tui = TuiApp(
            status_fragments_fn=lambda: [("", "STATUS_SENTINEL")],
            activity_fragments_fn=lambda: [("", "activity first row\\nACTIVITY_SENTINEL")],
            flush_scrollback=ui.drain_scrollback,
        )
        tui.set_running("working")

        def drive():
            if not tui.ready.wait(5):
                os._exit(2)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                app = tui.app
                if app is not None and isinstance(app.renderer, ScrollbackRenderer) and app.renderer.scrollback_ready:
                    break
                time.sleep(0.01)
            else:
                os._exit(3)
            tui.write_to_scrollback(lambda: ui.emit("PROMOTED_SENTINEL\\n"))
            ui.emit("TOOL_SENTINEL\\n")
            time.sleep(1)
            ui.emit("SMALL_SENTINEL\\n")
            time.sleep(1)
            ui.emit("GROWN_SENTINEL\\n")
            time.sleep(10)
            tui.exit()

        threading.Thread(target=drive, daemon=True).start()
        tui.run()
        """
    )
    command = shlex.join([sys.executable, "-c", script])
    try:
        subprocess.run(
            ["tmux", "-L", socket, "-f", "/dev/null", "new-session", "-d", "-x", "80", "-y", "20", command],
            cwd=os.getcwd(),
            check=True,
            timeout=5,
        )
        pane = subprocess.run(
            ["tmux", "-L", socket, "display-message", "-p", "#{pane_id}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout.strip()
        deadline = time.monotonic() + 5
        captured = ""
        while time.monotonic() < deadline:
            captured = subprocess.run(
                ["tmux", "-L", socket, "capture-pane", "-p", "-t", pane, "-S", "-50"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            ).stdout
            if "TOOL_SENTINEL" in captured:
                break
            time.sleep(0.05)
        assert "TOOL_SENTINEL" in captured
        sibling = subprocess.run(
            ["tmux", "-L", socket, "split-window", "-d", "-v", "-l", "14", "-P", "-F", "#{pane_id}", "sleep 10"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout.strip()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            captured = subprocess.run(
                ["tmux", "-L", socket, "capture-pane", "-p", "-t", pane, "-S", "-50"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            ).stdout
            if "SMALL_SENTINEL" in captured:
                break
            time.sleep(0.05)
        assert "SMALL_SENTINEL" in captured, captured
        assert "HISTORY_11" in captured, captured
        visible = subprocess.run(
            ["tmux", "-L", socket, "capture-pane", "-p", "-t", pane],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout
        assert "SMALL_SENTINEL" in visible, visible
        assert visible.count("ACTIVITY_SENTINEL") == 1
        assert visible.count("STATUS_SENTINEL") == 1
        subprocess.run(["tmux", "-L", socket, "kill-pane", "-t", sibling], check=True, timeout=2)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            captured = subprocess.run(
                ["tmux", "-L", socket, "capture-pane", "-p", "-t", pane, "-S", "-50"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            ).stdout
            if "GROWN_SENTINEL" in captured:
                break
            time.sleep(0.05)
        assert "GROWN_SENTINEL" in captured, captured
        visible = subprocess.run(
            ["tmux", "-L", socket, "capture-pane", "-p", "-t", pane],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout
        assert "GROWN_SENTINEL" in visible, visible
        assert "PROMOTED_SENTINEL" in captured, captured
        assert visible.count("STATUS_SENTINEL") == 1
        assert visible.count("ACTIVITY_SENTINEL") == 1
        assert captured.index("PROMOTED_SENTINEL") < captured.index("TOOL_SENTINEL") < captured.index("SMALL_SENTINEL") < captured.index("GROWN_SENTINEL")
    finally:
        subprocess.run(["tmux", "-L", socket, "kill-server"], capture_output=True, timeout=2, check=False)


def test_emit_without_running_application_prints_directly(monkeypatch):
    """Headless / pre-TUI emits keep printing immediately, one print per emit."""
    printed = []
    monkeypatch.setattr("minacode.render.print_formatted_text", lambda *args, **kwargs: printed.append(args))
    ui = UiPrinter(print)
    ui.color = True

    ui.emit("line one\n")
    ui.emit("line two\n")

    assert len(printed) == 2
    assert fragment_text(printed[0][0]) == "line one\n"
    assert fragment_text(printed[1][0]) == "line two\n"


def test_batched_replay_still_prints_in_one_call(monkeypatch):
    """The replay batch keeps its single-print contract even with the batching window present."""
    printed = []
    monkeypatch.setattr("minacode.render.print_formatted_text", lambda *args, **kwargs: printed.append(args))
    ui = UiPrinter(print)
    ui.color = True

    with ui.batched():
        ui.emit("a\n")
        ui.emit("b\n")
        ui.emit("c\n")

    assert len(printed) == 1


def test_direct_print_drains_queued_burst_first(monkeypatch):
    """A direct print (write_to_scrollback callback, post-TUI fallback) never lands ahead of queued emits.

    The queued burst is planted directly (its window timer is faked as never firing); the direct
    print must drain it first so terminal order matches emit order.
    """
    printed = []
    monkeypatch.setattr("minacode.render.print_formatted_text", lambda *args, **kwargs: printed.append(args))
    ui = UiPrinter(print)
    ui.color = True
    ui._scrollback_parts = [FormattedText([("", "queued line\n")])]

    ui._scrollback_print(FormattedText([("", "direct line\n")]))

    assert len(printed) == 2
    assert fragment_text(printed[0][0]) == "queued line\n"
    assert fragment_text(printed[1][0]) == "direct line\n"
    assert ui._scrollback_parts == []


def test_drain_scrollback_prints_queued_output_at_shutdown(monkeypatch):
    """Output queued when the application stops (timer never fires again) is not lost."""
    printed = []
    monkeypatch.setattr("minacode.render.print_formatted_text", lambda *args, **kwargs: printed.append(args))
    ui = UiPrinter(print)
    ui.color = True
    ui._scrollback_parts = [FormattedText([("", "last line\n")])]  # left behind by an un-fired timer

    ui.drain_scrollback()

    assert len(printed) == 1
    assert fragment_text(printed[0][0]) == "last line\n"
    assert ui._scrollback_parts == []


def test_tui_runtime_wires_scrollback_drain(tmp_path, monkeypatch):
    """TuiApp can synchronously drain UiPrinter's queued scrollback."""
    loop = CommandLoop(
        Agent(session(tmp_path), output_fn=lambda text: None),
        input_fn=lambda prompt="": "",
        output_fn=lambda text: None,
    )
    drained = []
    monkeypatch.setattr(loop.ui, "drain_scrollback", lambda: drained.append(True))

    tui = TuiRuntime(loop).build_tui()
    tui.flush_scrollback()

    assert drained == [True]


def test_agent_thread_emit_batches_into_one_suspend(monkeypatch):
    """The real calling shape: the application runs in its own thread under patch_stdout only.

    TuiApp.run never wraps create_app_session, so the app registers on the shared default
    AppSession and every thread's get_app_or_none() sees it -- the agent thread emits into the
    live application. The burst must still coalesce into one print_formatted_text call with
    every line present and the window drained, not fall back to per-emit direct prints.
    """
    printed = []
    monkeypatch.setattr("minacode.render.print_formatted_text", lambda *args, **kwargs: printed.append(args))
    rec = RecordingOutput()
    ui = UiPrinter(print)
    ui.color = True
    ready = threading.Event()
    done = threading.Event()
    errors = []

    def tui_main():
        try:
            with create_pipe_input() as pipe:
                layout = Layout(
                    HSplit(
                        [
                            Window(
                                FormattedTextControl([("class:divider", "--- \u25cf responding ... ---")]),
                                dont_extend_height=True,
                            ),
                            Window(BufferControl(Buffer()), dont_extend_height=True),
                        ]
                    )
                )
                app = Application(layout=layout, output=rec, input=pipe, full_screen=False)

                async def drive():
                    await asyncio.sleep(0.05)  # let the first render land
                    ready.set()
                    done.wait(5)
                    await asyncio.sleep(0.15)  # batching window + run_in_terminal round trip
                    app.exit()

                app.pre_run_callables.append(lambda: app.create_background_task(drive()))
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                with patch_stdout():
                    loop.run_until_complete(app.run_async())
        except BaseException as error:  # noqa: BLE001 - the thread must record every failure kind
            errors.append(error)

    t = threading.Thread(target=tui_main, name="tui", daemon=True)
    t.start()
    assert ready.wait(5)
    ui.emit("tool line one\n")
    ui.emit("tool line two\n")
    ui.emit("tool line three\n")
    done.set()
    t.join(10)

    assert errors == []
    assert len(printed) == 1, "a burst of agent-thread emits must coalesce into one print"
    text = "".join(fragment_text(part) for part in printed[0])
    assert text.index("tool line one") < text.index("tool line two") < text.index("tool line three")
    assert ui._scrollback_parts == []  # the window drained; nothing queued is left behind


def test_flush_exception_drains_the_batch_and_keeps_the_printer_usable(monkeypatch):
    """A throwing print during window flush must not double-print, deadlock, or wedge later emits.

    The batch is dequeued before printing (print happens outside the lock), so a terminal failure
    loses that batch but leaves the window consistent: nothing queued remains, no timer is pending,
    and the next emit still prints.
    """

    def throw_once(*args, **kwargs):
        raise RuntimeError("terminal failure")

    monkeypatch.setattr("minacode.render.print_formatted_text", throw_once)
    ui = UiPrinter(print)
    ui.color = True
    ui._scrollback_parts = [FormattedText([("", "queued line\n")])]

    with pytest.raises(RuntimeError):
        ui._flush_scrollback()

    assert ui._scrollback_parts == []  # the batch is gone; no retry, no double print
    assert ui._scrollback_timer is None

    # The printer keeps working: the next emit falls through to a direct print.
    printed = []
    monkeypatch.setattr("minacode.render.print_formatted_text", lambda *args, **kwargs: printed.append(args))
    ui.emit("after failure\n")
    assert len(printed) == 1
    assert fragment_text(printed[0][0]) == "after failure\n"
