"""minacode entry point: command-line argument parsing and dispatch.

Invoked through the ``minacode`` console script or ``python -m minacode``.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import threading

from minacode.base import ConfigError, MinacodeError, UpdateStatus, __version__
from minacode.cli import CommandLoop
from minacode.config import (
    Config,
    ConfigFile,
    RuntimeSettings,
)
from minacode.engine import Agent
from minacode.render import Theme
from minacode.session import Session
from minacode.update import UpdateChecker


def run_update() -> int:
    """Check PyPI for a newer minacode and upgrade it via the detected package manager."""
    print(f"minacode {__version__}")
    try:
        latest = UpdateChecker.fetch_latest()
    except Exception as error:  # noqa: BLE001 - update failures from any network/backend layer are reported uniformly.
        print(f"Error: could not check the latest version: {error}", file=sys.stderr)
        return 1
    if not UpdateStatus(latest=latest).newer_than(__version__):
        print(f"already up to date ({__version__})")
        return 0
    command = UpdateChecker.upgrade_command()
    print(f"updating {__version__} -> {latest}: {' '.join(command)}")
    try:
        return subprocess.call(command)
    except OSError as error:
        print(f"Error: could not run the upgrade command: {error}", file=sys.stderr)
        return 1


def warm_provider_sdks() -> None:
    """Import the provider SDKs off the main thread so the prompt accepts input immediately.

    ModelClient imports them lazily because they cost ~0.8s, which was the whole of the delay
    before a fresh prompt echoed keystrokes. Loading them here in the background keeps the prompt
    instant without moving that cost onto the first request: the user's first message takes far
    longer to type than the import takes to finish.

    Racing this thread against the request path is safe, and deliberately so:

    - CPython locks imports per module (`importlib._bootstrap._ModuleLock`), so a request-path
      `from openai import OpenAI` that lands mid-warm-up blocks on that module's lock and then
      reads the finished module from `sys.modules`. It cannot observe a half-initialized module,
      and both threads therefore bind the same class object.
    - Per-module locks can deadlock only on an import cycle entered from two threads at once.
      `anthropic` and `openai` do not import each other, and their shared dependencies form a DAG,
      so the lock-wait graph has no cycle. `_DeadlockError` detection is the backstop if that ever
      stops being true.
    - The thread is a daemon because warming must never delay exit. CPython freezes daemon threads
      at finalization rather than letting them run against a torn-down import system, so quitting
      mid-import is silent.

    Verified by stress test: a barrier-synchronized four-way race and repeated immediate-exit runs
    produce no deadlock, no exception, and no stderr noise.
    """

    def load() -> None:
        # Warming is only an optimization, and an uncaught failure here would print a thread
        # traceback over the live prompt. Any real problem resurfaces on the request path, which
        # imports the same modules and reports the failure to the user.
        with contextlib.suppress(Exception):
            import anthropic  # noqa: F401 - imported for its side effect of populating sys.modules
            import openai  # noqa: F401

    threading.Thread(target=load, name="sdk-warmup", daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    if sys.platform == "win32":
        print("Error: minacode does not support native Windows; use WSL instead.", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(prog="minacode")
    parser.add_argument("--config", default=None, help="Path to config TOML")
    parser.add_argument("--init-config", action="store_true", help="Create a default config file")
    parser.add_argument("--yolo", action="store_true", help="Skip confirmations for mutating tools")
    parser.add_argument(
        "--theme", choices=["auto", "light", "dark"], default="", help="Color theme (defaults to runtime.theme, then auto-detect via COLORFGBG)"
    )
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume",
        default="",
        nargs="?",
        const="latest",
        help='Resume a session by UID, uid prefix, or name, or "latest"/"last" for this project\'s most recent',
    )
    resume.add_argument("-c", "--last", "--latest", dest="continue_project", action="store_true", help="Resume the latest session in the current project")
    parser.add_argument("-v", "--version", action="store_true", help="Show version")
    parser.add_argument(
        "command", nargs="?", choices=["update", "upgrade"], default=None, help="Maintenance command: update/upgrade minacode to the latest version"
    )
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command in {"update", "upgrade"}:
        return run_update()
    try:
        if args.init_config:
            path, created = ConfigFile.init(args.config)
            print(("Created" if created else "Exists") + " config: " + path)
            return 0
        # Switching sessions ends one run and starts the next rather than re-pointing a live
        # object graph at another Session: everything below is built around one, and this is the
        # only moment nothing is running. Teardown stays in the `finally` that already does it.
        resume = args.resume or ("latest" if args.continue_project else "")
        while True:
            if resume:
                data = ConfigFile.load(args.config)
                config = Config.from_dict(data)
                session = Session.load_snapshot(
                    resume,
                    config=config,
                    settings=RuntimeSettings.from_dict(data, yolo=args.yolo, theme=args.theme),
                    cwd=os.getcwd(),
                )
            else:
                session = Session.from_config_file(path=args.config, yolo=args.yolo, theme=args.theme)
            Theme.set_mode(Theme.resolve(session.settings.theme))
            warm_provider_sdks()
            command_loop = CommandLoop(Agent(session))
            try:
                code = command_loop.run()
            finally:
                command_loop.close_background_output()
                if session.mcp is not None:
                    session.mcp.close()
            resume = command_loop.resume_request
            if not resume:
                return code
    except ConfigError as error:
        print("ConfigError: " + str(error), file=sys.stderr)
        return 2
    except MinacodeError as error:
        print("Error: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
