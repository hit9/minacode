"""The pre-loop startup sweep: `wizolt` revealed left to right while the interpreter loads.

The wait before the prompt is imports on the main thread, before any event loop exists, so there
is nothing to await and no loop to animate on. The sweep therefore runs on one daemon thread that
rewrites a single stdout line; `main` starts it the moment a session invocation is certain, and
`finish_banner` completes the line — fast-forwarding whatever the sweep has shown so far — once
imports are done.

The finished line *is* the banner: the command loop skips printing `wizolt <version>. /help for
commands.` again when this module already wrote it, so the swept text and the real banner are one
line, not two. Stdlib only, and deliberately free of the wizolt package's own import graph
(`wizolt/__init__.py` imports base, which is now light): the point is to appear before anything
heavy loads, so this module must not itself import anything heavy.

Everything here is a no-op when no human is watching a terminal — a piped run, an embedding, the
test suite — and those paths print the banner exactly as before.
"""

from __future__ import annotations

import sys
import threading

# The swept text. No version: printing it must not import the module that knows the version.
TEXT = "wizolt"
# Columns the light band covers at the reveal edge (bold reverse video reads as the shine).
EDGE = 2
# Seconds per frame; a full pass lasts len(TEXT) * TICK (~0.4s), which comfortably outlasts a warm
# import and reads as a deliberate reveal on a cold one. The line completes the moment imports are
# done, so the sweep never *adds* startup latency.
TICK = 0.07
# Frames the fully revealed text holds before the thread goes quiet and waits for finish.
HOLD_FRAMES = 3

_BOLD = "\x1b[1m"
_REVERSE = "\x1b[7m"
_RESET = "\x1b[0m"

_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop = threading.Event()
_started = False
_finished = False


def start() -> bool:
    """Begin the sweep on stdout. True when a sweep is now running.

    Called by `main` only once it is certain a session is about to run (argparse early exits and
    the config error path have already been ruled out), so no argv sniffing is needed here. A
    non-tty stdout — a pipe, a file, the test suite — means nobody is watching; skip.
    """
    global _started, _thread
    if _started or not sys.stdout.isatty():
        return False
    _started = True
    _stop.clear()
    _thread = threading.Thread(target=_run, name="startup-sweep", daemon=True)
    _thread.start()
    return True


def finish_banner(remainder: str) -> bool:
    """Complete the banner line once the heavy imports are done, and report whether it was written.

    `remainder` is what follows the swept text on the finished line (the caller supplies it with
    the version it can now import). Fast-forwards the sweep: whatever the animation had shown, the
    terminal ends with one clean, fully bright line plus the remainder. Returns False when no
    sweep is running, so the caller prints the banner through its normal channel instead.
    """
    global _finished
    if not _started:
        return False
    _stop.set()
    thread = _thread
    if thread is not None and thread.is_alive():
        # One frame at most; the thread checks the stop event between writes.
        thread.join(timeout=TICK + 0.05)
    with _lock:
        sys.stdout.write("\r" + _BOLD + TEXT + _RESET + remainder + "\n")
        sys.stdout.flush()
    _finished = True
    return True


def abort() -> None:
    """Stop the sweep and take its line back without completing a banner (an error path)."""
    global _started, _finished
    if not _started:
        return
    _stop.set()
    thread = _thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=TICK + 0.05)
    with _lock:
        sys.stdout.write("\r" + " " * len(TEXT) + "\r")
        sys.stdout.flush()
    _started = False
    _finished = False


def preprinted() -> bool:
    """Whether the finished banner line is already on screen (the caller should not print another)."""
    return _finished


def _frame(step: int) -> str:
    """One reveal frame: columns [0, step) bright, the next EDGE columns as the light band, the
    rest blank. Undrawn columns stay spaces so the line never shifts."""
    revealed = TEXT[:step]
    edge = TEXT[step : step + EDGE]
    return (_BOLD + revealed if revealed else "") + (_REVERSE + edge if edge else "") + _RESET + " " * (len(TEXT) - len(revealed) - len(edge))


def _run() -> None:
    """Tick the sweep until told to stop, then hold the finished state without further writes.

    Every write happens under the lock `finish_banner`/`abort` take, so the line can never be
    rewritten after it has been completed or erased."""
    total = len(TEXT) + HOLD_FRAMES
    for step in range(total + 1):
        if _stop.is_set():
            return
        with _lock:
            sys.stdout.write("\r" + _frame(min(step, len(TEXT))))
            sys.stdout.flush()
        if _stop.wait(TICK):
            return
    _stop.wait()
