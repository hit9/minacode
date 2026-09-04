"""wizolt entry point: command-line argument parsing and dispatch.

Invoked through the ``wizolt`` console script or ``python -m wizolt``.

Deliberately imports nothing from the wizolt package at module level: argparse, help, version, and
config initialization should answer before the interactive CLI — prompt_toolkit, the tools, the
TUI, the session machinery — is imported. The interactive-CLI names `main` needs are reached
through the lazy `_cli` namespace below: reading `_cli.Session` imports it on first use and caches
it on this module, so tests can keep substituting fakes here without importing wizolt at module
load time.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import subprocess
import sys
import threading

# Every interactive-CLI name `main` needs, as (module, attribute). Loaded on first use and cached
# on this module, so tests can keep substituting fakes here without `import wizolt.__main__`
# paying for the interactive CLI.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "__version__": ("wizolt.base", "__version__"),
    "Agent": ("wizolt.engine", "Agent"),
    "CatalogError": ("wizolt.providers.schema", "CatalogError"),
    "CatalogRuntime": ("wizolt.providers.sync", "CatalogRuntime"),
    "CommandLoop": ("wizolt.cli", "CommandLoop"),
    "Config": ("wizolt.config", "Config"),
    "ConfigError": ("wizolt.base", "ConfigError"),
    "ConfigFile": ("wizolt.config", "ConfigFile"),
    "RuntimeSettings": ("wizolt.config", "RuntimeSettings"),
    "Session": ("wizolt.session", "Session"),
    "Theme": ("wizolt.render", "Theme"),
    "UpdateChecker": ("wizolt.cli.update", "UpdateChecker"),
    "UpdateStatus": ("wizolt.base", "UpdateStatus"),
    "WizoltError": ("wizolt.base", "WizoltError"),
    "configure_logging": ("wizolt.base", "configure_logging"),
}


def _import_lazy(name: str):
    """Import one interactive-CLI name and cache it on this module (PEP 562)."""
    spec = _LAZY_IMPORTS[name]
    value = getattr(importlib.import_module(spec[0]), spec[1])
    globals()[name] = value
    return value


def __getattr__(name: str):
    """Import a lazy interactive-CLI name on first attribute read (PEP 562)."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _import_lazy(name)


class _EntryCli:
    """The entry point's handle on the interactive CLI. Reading an attribute (`_cli.Session`)
    imports that name on first use and caches it on this module, so later reads are plain
    attribute lookups; a name a test has already bound here (a fake) is returned as-is. This is
    the seam that keeps `wizolt.__main__` importable without the interactive CLI while `main`
    still resolves the real classes through module attributes the way it always has."""

    def __getattr__(self, name: str):
        bound = globals().get(name)
        if bound is not None:
            return bound
        if name not in _LAZY_IMPORTS:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        return _import_lazy(name)


_cli = _EntryCli()


def run_update() -> int:
    """Check PyPI for a newer wizolt and upgrade it via the detected package manager."""
    print(f"wizolt {_cli.__version__}")
    try:
        latest = _cli.UpdateChecker.fetch_latest_sync()
    except Exception as error:  # noqa: BLE001 - update failures from any network/backend layer are reported uniformly.
        print(f"Error: could not check the latest version: {error}", file=sys.stderr)
        return 1
    if not _cli.UpdateStatus(latest=latest).newer_than(_cli.__version__):
        print(f"already up to date ({_cli.__version__})")
        return 0
    command = _cli.UpdateChecker.upgrade_command()
    print(f"updating {_cli.__version__} -> {latest}: {' '.join(command)}")
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
    parser = argparse.ArgumentParser(prog="wizolt", epilog="Documentation: https://wizolt.readthedocs.io")
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
        "command", nargs="?", choices=["update", "upgrade"], default=None, help="Maintenance command: update/upgrade wizolt to the latest version"
    )
    args = parser.parse_args(argv)
    if sys.platform == "win32":
        print("Error: wizolt does not support native Windows; use WSL instead.", file=sys.stderr)
        return 1
    # Cheap exits answer before the interactive CLI is loaded. The explicit update
    # command loads its HTTP/update implementation here, but still never starts a session.
    if args.version:
        print(_cli.__version__)
        return 0
    if args.command in {"update", "upgrade"}:
        return run_update()
    if args.init_config:
        path, created = _cli.ConfigFile.init(args.config)
        print(("Created" if created else "Exists") + " config: " + path)
        return 0

    _cli.configure_logging()
    try:
        # Switching sessions ends one run and starts the next rather than re-pointing a live
        # object graph at another Session: everything below is built around one, and this is the
        # only moment nothing is running. Teardown stays in the `finally` that already does it.
        resume = args.resume or ("latest" if args.continue_project else "")
        while True:
            if resume:
                data = _cli.ConfigFile.load(args.config)
                catalog = _cli.CatalogRuntime(_cli.Config.data_dir_from(data))
                config = _cli.Config.from_dict(data, policy=catalog.policy)
                session = _cli.Session.load_snapshot(
                    resume,
                    config=config,
                    settings=_cli.RuntimeSettings.from_dict(data, yolo=args.yolo, theme=args.theme),
                    cwd=os.getcwd(),
                    catalog=catalog,
                )
            else:
                session = _cli.Session.from_config_file(path=args.config, yolo=args.yolo, theme=args.theme)
            _cli.Theme.set_mode(_cli.Theme.resolve(session.settings.theme))
            warm_provider_sdks()
            command_loop = _cli.CommandLoop(_cli.Agent(session))
            try:
                code = command_loop.run()
            finally:
                # The runtime closes what the session opened, on the loop that opened it; all that
                # is left here is the terminal-output gate, in case the runtime never got that far.
                command_loop.close_background_output()
            resume = command_loop.resume_request
            if not resume:
                return code
    except _cli.ConfigError as error:
        print("ConfigError: " + str(error), file=sys.stderr)
        return 2
    except (_cli.WizoltError, _cli.CatalogError) as error:
        print("Error: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
