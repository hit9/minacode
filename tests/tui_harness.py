"""Shared harness for the TUI test modules: session/loop builders, recording prompt-toolkit
outputs, and the helpers that drive a real Application over a pipe input."""

import threading
import time

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

import minacode.tui.app as tui_module
from minacode.cli import CommandLoop
from minacode.config import (
    Config,
)
from minacode.engine import Agent
from minacode.session import Session, bootstrap_features


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


def rendered_screen_text(application, output):
    screen = application.renderer.last_rendered_screen
    if screen is None:
        return ""
    return "\n".join("".join(screen.data_buffer[row][column].char for column in range(output.size.columns)).rstrip() for row in range(output.size.rows))


def run_interactive_tui(monkeypatch, tui, *, text="", drive=None, output=None, after_render=None):
    real_application = Application
    output = output or DummyOutput()
    driver_errors = []
    with create_pipe_input() as pipe_input:

        def application(**kwargs):
            return real_application(input=pipe_input, after_render=after_render, **(kwargs | {"output": output}))

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
        tui.run()
        if driver is not None:
            driver.join(timeout=1)
            assert not driver.is_alive()
    if driver_errors:
        raise driver_errors[0]
