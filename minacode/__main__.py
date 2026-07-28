"""minacode entry point: command-line argument parsing and dispatch.

Invoked through the ``minacode`` console script or ``python -m minacode``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from minacode.base import Config, ConfigError, ConfigFile, MinacodeError, RuntimeSettings, UpdateStatus, __version__
from minacode.engine import Agent
from minacode.loop import CommandLoop
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
    resume.add_argument("--resume", default="", nargs="?", const="latest", help='Resume a session by UID, or "latest"/"last" for this project\'s most recent')
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
        if args.resume or args.continue_project:
            data = ConfigFile.load(args.config)
            config = Config.from_dict(data)
            session = Session.load_snapshot(
                args.resume or "latest",
                config=config,
                settings=RuntimeSettings.from_dict(data, yolo=args.yolo, theme=args.theme),
                cwd=os.getcwd(),
            )
        else:
            session = Session.from_config_file(path=args.config, yolo=args.yolo, theme=args.theme)
        Theme.set_mode(Theme.resolve(session.settings.theme))
        command_loop = CommandLoop(Agent(session))
        try:
            return command_loop.run()
        finally:
            command_loop.close_background_output()
            if session.mcp is not None:
                session.mcp.close()
    except ConfigError as error:
        print("ConfigError: " + str(error), file=sys.stderr)
        return 2
    except MinacodeError as error:
        print("Error: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
