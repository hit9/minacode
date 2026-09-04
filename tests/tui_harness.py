"""Shared harness for the TUI test modules: session/loop builders, recording prompt-toolkit
outputs, and the helpers that drive a real Application over a pipe input."""

import asyncio
import concurrent.futures
import threading
import time

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

import wizolt.tui.app as tui_module
from wizolt.cli import CommandLoop
from wizolt.config import (
    Config,
)
from wizolt.engine import Agent
from wizolt.session import Session, bootstrap_features


def session(tmp_path):
    config = Config()
    config.data_dir = str(tmp_path / "data")
    session = Session(cwd=str(tmp_path), config=config)
    bootstrap_features(session)
    return session


def loop(tmp_path):
    return CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt="": "", output_fn=lambda text: None)


class ResizableOutput(DummyOutput):
    def __init__(self, rows=24, columns=80):
        self.size = Size(rows=rows, columns=columns)

    def get_size(self):
        return self.size


# Generous on purpose: the TUI runs in a separate thread and these conditions are only
# reached after the event loop processes piped input. Under CI load (contended cores,
# GC pauses, slow /tmp) a 1s budget flakes, so allow several seconds. Success returns
# immediately, so this only lengthens genuinely-deadlocked failures.
def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("interactive TUI condition was not reached")


async def wait_for(predicate, timeout=5.0):
    """Yield to the loop until `predicate` holds. The timeout is a deadlock bound, not a pace."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not reached")
        await asyncio.sleep(0.001)


def request_input_from_driver(app, prompt="Approve? "):
    """Ask a live TuiApp for input from a driver thread, and hand back a future to read.

    The request itself belongs to the application's own loop -- that is where the prompt lives and
    where the answer resolves -- so the driver schedules it there rather than running it itself."""
    result: concurrent.futures.Future = concurrent.futures.Future()

    def start() -> None:
        async def request() -> None:
            try:
                result.set_result(await app.request_input(prompt))
            except BaseException as error:  # noqa: BLE001 - the driver reads every ending off the future
                result.set_exception(error)

        # A plain task, not one of the application's background tasks: in production the request is
        # awaited by the turn the runtime owns, so the app shutting down resolves it as cancelled
        # rather than tearing the waiter down with itself.
        app.app.loop.create_task(request())

    app.app.loop.call_soon_threadsafe(start)
    return result


def rendered_screen_text(application, output):
    screen = application.renderer.last_rendered_screen
    if screen is None:
        return ""
    return "\n".join("".join(screen.data_buffer[row][column].char for column in range(output.size.columns)).rstrip() for row in range(output.size.rows))


def run_interactive_tui(monkeypatch, tui, *, text="", drive=None, output=None, after_render=None, on_application=None):
    real_application = Application
    output = output or DummyOutput()
    driver_errors = []
    with create_pipe_input() as pipe_input:

        def application(**kwargs):
            app = real_application(input=pipe_input, after_render=after_render, **(kwargs | {"output": output}))
            if on_application is not None:
                on_application(app)
            return app

        monkeypatch.setattr(tui_module, "Application", application)
        if text:
            pipe_input.send_text(text)
        driver = None
        if drive is not None:

            def run_driver():
                try:
                    drive(pipe_input)
                except BaseException as error:  # noqa: BLE001 - harness collects every driver-thread failure
                    driver_errors.append(error)
                    if tui.app is not None:
                        tui.app.loop.call_soon_threadsafe(tui.app.exit)

            driver = threading.Thread(target=run_driver, daemon=True)
            driver.start()
        tui.run_sync()
        if driver is not None:
            driver.join(timeout=1)
            assert not driver.is_alive()
    if driver_errors:
        raise driver_errors[0]
