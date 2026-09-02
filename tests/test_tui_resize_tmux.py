"""Real-tmux resize regression: a running wizolt under repeated tmux narrow/widen cycles must
not leave fragments of the live region in the visible pane nor stale copies stacked beside the
fresh draw. These tests drive the real TuiApp inside a tmux pane (like test_diff_command.py's
probes) and read the pane back with capture-pane; they need a working tmux and are skipped when
it is missing.
"""

import re
import shlex
import shutil
import subprocess
import sys
import time

PROBE = """
import tempfile
import threading
import time

from wizolt.config import Config
from wizolt.engine import Agent
from wizolt.cli import CommandLoop
from wizolt.session import Session
from wizolt.tui import TuiApp

session = Session(cwd="/tmp", config=Config(data_dir=tempfile.mkdtemp()))
loop = CommandLoop(Agent(session))
app = TuiApp(activity_fragments_fn=loop.view.tui_activity_fragments)
loop.tui = app


def drive():
    while app.app is None or not app.app.is_running:
        time.sleep(0.005)
    # Transcript lines print through the live application's patch_stdout path, as in a real turn.
    for i in range(4):
        loop.emit("user says: question number %d about the project and design" % i)
        time.sleep(0.05)
    loop.emit_agent_answer("assistant replies: here is the answer for that question, with some detail.")
    time.sleep(0.3)
    with loop.model_stream_lock:
        loop.model_stream_kind = "thinking"
        loop.model_stream_text = "\\n".join("LIVE line %d of the running region text" % i for i in range(4))
    app.set_running("thinking")
    app.invalidate_frame()
    print("READY", flush=True)
    time.sleep(300)


threading.Thread(target=drive, daemon=True).start()
app.run()
"""

LIVE_RE = re.compile(r"LIVE line (\d)")
PROMPT_RE = re.compile(r"^\+>")


def test_running_tui_resize_leaves_no_fragments_in_real_tmux(tmp_path):
    executable = shutil.which("tmux")
    if executable is None:
        return
    socket = "wizolt-tmux-" + tmp_path.name
    command = [executable, "-L", socket]
    probe = tmp_path / "resize_tmux_probe.py"
    probe.write_text(PROBE)
    pane_command = f"{shlex.quote(sys.executable)} {shlex.quote(str(probe))}"

    def capture(history=False):
        args = [*command, "capture-pane", "-p", "-t", "probe"] + (["-S", "-"] if history else [])
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout

    def resize(columns):
        subprocess.run([*command, "resize-window", "-t", "probe", "-x", str(columns), "-y", "40"], check=True)

    try:
        subprocess.run([*command, "kill-server"], check=False, capture_output=True)
        subprocess.run([*command, "new-session", "-d", "-s", "probe", "-x", "100", "-y", "40", "sleep 30"], check=True)
        subprocess.run([*command, "set-option", "-t", "probe", "window-size", "manual"], check=True)
        subprocess.run([*command, "respawn-pane", "-k", "-t", "probe", pane_command], check=True)
        screen = ""
        deadline = time.monotonic() + 30
        while "READY" not in screen and time.monotonic() < deadline:
            time.sleep(0.1)
            screen = capture()
        assert "READY" in screen, "probe never started"
        deadline = time.monotonic() + 10
        while len(LIVE_RE.findall(capture())) < 4 and time.monotonic() < deadline:
            time.sleep(0.2)
        assert len(LIVE_RE.findall(capture())) == 4, "live region never rendered"
        for cycle in range(1, 5):
            for columns in (40, 100):
                resize(columns)
                time.sleep(0.7)
                visible = capture()
                # The visible pane must show one copy of each live row: the resize erases the
                # reflowed copy instead of stacking a stale one beside the fresh draw.
                markers = LIVE_RE.findall(visible)
                assert len(markers) == len(set(markers)) == 4, f"cycle {cycle} at {columns}c: {markers}"
                # The input prompt stays on screen, bottom-anchored, in exactly one copy.
                prompts = [line for line in visible.splitlines() if PROMPT_RE.match(line)]
                assert len(prompts) == 1, f"cycle {cycle} at {columns}c: prompt rows {prompts}"
    finally:
        subprocess.run([*command, "kill-server"], check=False, capture_output=True)
