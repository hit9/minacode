from dataclasses import dataclass, field

import pytest
from rich.style import Style


@pytest.fixture(autouse=True)
def isolate_home(tmp_path_factory, monkeypatch):
    # `paths.data_dir` defaults to `~/.wizolt`, so any test that builds a config without setting
    # it and then saves a session writes into the developer's real home directory. Point HOME at a
    # per-test directory so `expanduser` resolves somewhere disposable.
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # expanduser prefers this on Windows
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    return home


@pytest.fixture(autouse=True)
def reset_rich_style_cache():
    # Rich caches parsed/combined Style objects globally (Style.parse, Style._add, ... are
    # @lru_cache'd) and each Style memoizes its rendered SGR in `_ansi` keyed on the first
    # color_system it was ever rendered with. A test that renders a color through a standard-color
    # console poisons that shared `_ansi`, so a later test asserting truecolor bytes for the same
    # color sees the downgraded code. Drop the caches before each test so styling is deterministic
    # regardless of order.
    for name in ("normalize", "parse", "get_html_style", "clear_meta_and_links", "_add"):
        cached = getattr(Style, name, None)
        if cached is not None and hasattr(cached, "cache_clear"):
            cached.cache_clear()
    yield


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
