"""A small pre-loop spinner beside the startup banner while interactive imports finish.

The wait before the prompt is synchronous imports on the main thread, before an event loop exists.
Print ``wizolt`` immediately and animate one ASCII cell beside it on a short-lived daemon thread;
once imports finish, replace that line with the ordinary banner. Non-terminal paths are no-ops.

Keep this module stdlib-only and deliberately independent of wizolt's import graph: its whole job
is to be visible before heavier modules load.
"""

from __future__ import annotations

import sys
import threading

TEXT = "wizolt"
FRAMES = ("|", "/", "-", "\\")
TICK = 0.08

_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"
_CLEAR_LINE = "\r\x1b[2K"

_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop = threading.Event()
_started = False
_finished = False


def start() -> bool:
    """Show the banner prefix and start its one-cell spinner on a terminal.

    Called by `main` only once it is certain a session is about to run (argparse early exits and
    the config error path have already been ruled out), so no argv sniffing is needed here. A
    non-tty stdout — a pipe, a file, the test suite — means nobody is watching; skip.
    """
    global _started, _thread
    if _started or not sys.stdout.isatty():
        return False
    _started = True
    _stop.clear()
    with _lock:
        _write_frame(FRAMES[0])
    _thread = threading.Thread(target=_run, name="startup-spinner", daemon=True)
    _thread.start()
    return True


def finish_banner(remainder: str) -> bool:
    """Complete the banner line once the heavy imports are done, and report whether it was written.

    `remainder` follows the stable text and includes the version. Returns False when no spinner was
    started, so the caller prints the banner through its normal channel instead.
    """
    global _finished
    if not _started:
        return False
    _stop.set()
    thread = _thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=TICK + 0.05)
    with _lock:
        sys.stdout.write(_CLEAR_LINE + _BOLD + TEXT + _RESET + remainder + "\n")
        sys.stdout.flush()
    _finished = True
    return True


def abort() -> None:
    """Stop the spinner and erase its unfinished line before an error is printed."""
    global _started, _finished
    if not _started or _finished:
        return
    _stop.set()
    thread = _thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=TICK + 0.05)
    with _lock:
        sys.stdout.write(_CLEAR_LINE)
        sys.stdout.flush()
    _started = False
    _finished = False


def preprinted() -> bool:
    """Whether the finished banner line is already on screen (the caller should not print another)."""
    return _finished


def _write_frame(frame: str) -> None:
    sys.stdout.write(_CLEAR_LINE + _BOLD + TEXT + _RESET + " " + frame)
    sys.stdout.flush()


def _run() -> None:
    """Rotate one cell until finish/abort; all terminal writes share one lock."""
    index = 1
    while not _stop.wait(TICK):
        with _lock:
            _write_frame(FRAMES[index % len(FRAMES)])
        index += 1
