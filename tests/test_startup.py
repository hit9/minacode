"""The startup spinner and the import chain that guard the blank wait before the first prompt.

The perceived startup cost is imports running on the main thread before any loop exists. Two
regression guards keep it bounded:

- on a terminal `wizolt` and a one-cell spinner appear while those imports run, and the finished
  line *is* the banner — these tests pin the spinner (tty only, one clean finished line, no frames
  after it) and the `start_session` seam that must not print a second banner;
- a fresh interpreter reaching the entry point must not pull in the heavy modules
  (prompt_toolkit, the provider SDKs, the code-symbol index), or the wait silently creeps back.
"""

import importlib
import subprocess
import sys

from wizolt import startup


def _sweep(monkeypatch) -> tuple[object, object]:
    """A freshly reloaded startup module writing to a fake terminal with a short tick."""
    startup.abort()  # a spinner left running by a failed earlier test must not outlive the reload
    module = importlib.reload(startup)
    monkeypatch.setattr(module, "TICK", 0.001)
    fake = _FakeTty()
    monkeypatch.setattr(module.sys, "stdout", fake)
    return module, fake


def _settle(module) -> None:
    """The sweep thread may still be in its tick sleep when a test ends; join it and, if the test
    left the sweep running, stop it."""
    module.abort()
    thread = module._thread
    if thread is not None:
        thread.join(timeout=0.1)
        assert not thread.is_alive()


class _FakeTty:
    """A stand-in stdout that reports a terminal and records what the sweep writes."""

    isatty_result = True

    def __init__(self):
        self.written = ""
        self.flush_count = 0

    def isatty(self):
        return self.isatty_result

    def write(self, text):
        self.written += text

    def flush(self):
        self.flush_count += 1


def test_sweep_is_a_noop_without_a_terminal(monkeypatch):
    module = importlib.reload(startup)
    fake = _FakeTty()
    fake.isatty_result = False
    monkeypatch.setattr(module.sys, "stdout", fake)
    try:
        assert module.start() is False
        assert fake.written == ""
        assert module.finish_banner("tail") is False
        assert module.preprinted() is False
    finally:
        _settle(module)


def test_sweep_finishes_as_one_clean_banner_line(monkeypatch):
    module, fake = _sweep(monkeypatch)
    try:
        assert module.start() is True
        assert "\x1b[1mwizolt\x1b[0m |" in fake.written
        assert module.finish_banner(" 0.37.1. /help for commands.") is True
        # The finished line is the banner, written once, with nothing (no further sweep frame, no
        # second banner) after it: only the finish write ends in a newline.
        assert fake.written.endswith("\r\x1b[2K\x1b[1mwizolt\x1b[0m 0.37.1. /help for commands.\n")
        assert fake.written.count("\n") == 1
        assert module.preprinted() is True
    finally:
        _settle(module)


def test_sweep_fast_forwards_no_matter_how_few_frames_ran(monkeypatch):
    """Finishing immediately still replaces the spinner with one clean banner."""
    module, fake = _sweep(monkeypatch)
    try:
        module.start()
        module.finish_banner(" tail")
        assert fake.written.endswith("\r\x1b[2K\x1b[1mwizolt\x1b[0m tail\n")
    finally:
        _settle(module)


def test_abort_takes_the_line_back_without_a_banner(monkeypatch):
    module, fake = _sweep(monkeypatch)
    try:
        assert module.start() is True
        module.abort()
        assert module.preprinted() is False
        assert module.finish_banner("tail") is False
        assert fake.written.endswith("\r\x1b[2K")  # the erase; no banner line was completed
        assert "\n" not in fake.written
    finally:
        _settle(module)


def test_abort_does_not_erase_a_completed_banner(monkeypatch):
    module, fake = _sweep(monkeypatch)
    module.start()
    module.finish_banner(" tail")
    written = fake.written

    module.abort()

    assert fake.written == written
    assert module.preprinted() is True


def test_fresh_interpreter_import_chain_stays_light():
    """The slow point under test: reaching the session-run entry point in a fresh interpreter must
    not import the heavy modules. prompt_toolkit (via base), the provider SDKs, and the optional
    code-symbol index belong to the first render or the first use, not to the entry point."""
    probe = (
        "import sys;"
        "import wizolt.base;"
        "import wizolt.__main__;"
        "import wizolt.tools.search;"
        "assert 'prompt_toolkit' not in sys.modules, 'base must not import prompt_toolkit';"
        "heavy = {'anthropic', 'openai', 'fastmcp', 'code_symbol_index'} & set(sys.modules);"
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True)


def test_start_session_prints_the_banner_once_per_run(tmp_path, monkeypatch):
    """When the sweep already wrote the banner (its finished line *is* the banner), start_session
    must not emit a second one; otherwise it prints it as always. Guard for the banner path in
    both frontends, which share start_session."""
    from agent_harness import session

    from wizolt.base import __version__
    from wizolt.cli import CommandLoop
    from wizolt.cli.update import UpdateChecker
    from wizolt.engine import Agent
    from wizolt.tools.search import CodeIndex

    command_loop = CommandLoop(
        Agent(session(tmp_path), output_fn=lambda text: None),
        input_fn=lambda prompt: "",
        output_fn=lambda text: None,
    )
    lines: list[str] = []
    monkeypatch.setattr(command_loop, "emit", lambda text="", indent=0: lines.append(str(text)))
    monkeypatch.setattr(UpdateChecker, "load_cached", lambda _checker: False)
    monkeypatch.setattr(CommandLoop, "render_resumed_session", lambda _loop: None)
    monkeypatch.setattr(CodeIndex, "status", lambda _index, **kwargs: ("ready", ""))

    banner = f"wizolt {__version__}. /help for commands."
    command_loop.start_session()
    assert lines[0] == banner

    # The sweep run wrote the banner itself and told the loop so: no second emit, nothing else
    # either (no update notice, no resume transcript, no index output on this session).
    lines.clear()
    command_loop._banner_preprinted = True
    command_loop.start_session()
    assert lines == []
