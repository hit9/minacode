from dataclasses import dataclass, field
from typing import Any, Callable

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

import nanocode as n


class SizedDummyOutput(DummyOutput):
    def __init__(self, columns: int, rows: int):
        self.size = Size(rows=rows, columns=columns)

    def get_size(self) -> Size:
        return self.size


@dataclass
class UIRun:
    result: Any
    app: Any
    fragments: list[tuple[str, str]]

    @property
    def text(self) -> str:
        return "".join(text for _style, text in self.fragments)


class UITestHarness:
    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch

    def run(self, loop: n.CommandLoop, action: Callable[[], Any], keys: str, *, size: tuple[int, int] = (80, 24)) -> UIRun:
        columns, rows = size
        captured = {}
        real_application = n.Application

        with create_pipe_input() as pipe:
            self.monkeypatch.setattr(n.shutil, "get_terminal_size", lambda *args: n.os.terminal_size((columns, rows)))
            self.monkeypatch.setattr(
                n,
                "Application",
                lambda **kwargs: real_application(**{**kwargs, "input": pipe, "output": SizedDummyOutput(columns, rows)}),
            )

            def run_input_app(app):
                captured["app"] = app
                return app.run()

            loop.run_input_app = run_input_app
            pipe.send_text(keys)
            result = action()

        app = captured["app"]
        fragments = list(app.layout.current_control.text())
        return UIRun(result, app, fragments)


@dataclass
class RecordingOutput:
    events: list[tuple[str, str]] = field(default_factory=list)

    def write_raw(self, text: str = "") -> None:
        self.events.append(("write", text))

    def erase_end_of_line(self) -> None:
        self.events.append(("erase", ""))

    def flush(self) -> None:
        self.events.append(("flush", ""))


@pytest.fixture
def ui_harness(monkeypatch):
    return UITestHarness(monkeypatch)


@pytest.fixture
def recording_output():
    return RecordingOutput()
