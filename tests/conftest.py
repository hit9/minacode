from dataclasses import dataclass, field

import pytest


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
def recording_output():
    return RecordingOutput()
