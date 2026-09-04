import os
import subprocess
import sys
import time
from io import StringIO
from types import SimpleNamespace

import pytest

import wizolt
import wizolt.__main__ as cli
from wizolt.cli import CommandLoop


def test_package_root_exposes_only_version():
    assert wizolt.__all__ == ["__version__"]
    assert all(not hasattr(wizolt, name) for name in ("Agent", "Session", "TuiApp", "main"))


def test_cli_rejects_native_windows(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "platform", "win32")

    assert cli.main([]) == 1
    assert "use WSL instead" in capsys.readouterr().err


def test_cli_prints_version(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == cli.__version__


def test_cli_help_links_docs(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "https://wizolt.readthedocs.io" in capsys.readouterr().out


def test_loop_help_links_docs():
    assert "https://wizolt.readthedocs.io" in CommandLoop.HELP


@pytest.mark.parametrize(("created", "prefix"), [(True, "Created"), (False, "Exists")])
def test_cli_initializes_config(monkeypatch, capsys, created, prefix):
    monkeypatch.setattr(cli.ConfigFile, "init", lambda path: (path, created))

    assert cli.main(["--init-config", "--config", "/tmp/wizolt.toml"]) == 0
    assert capsys.readouterr().out.strip() == f"{prefix} config: /tmp/wizolt.toml"


def test_cli_runs_session_and_closes_resources(monkeypatch):
    """The entry point owns only the terminal-output gate now.

    Everything the session opened -- the model client, MCP -- is closed by the runtime, on the loop
    that opened it. Closing MCP here would mean closing it after that loop was already gone."""
    closed = []
    mcp = SimpleNamespace(close=lambda: closed.append("mcp"))
    session = SimpleNamespace(settings=SimpleNamespace(theme="dark"), mcp=mcp)
    monkeypatch.setattr(cli.Session, "from_config_file", lambda **kwargs: session)
    monkeypatch.setattr(cli.Theme, "resolve", lambda theme: f"resolved-{theme}")
    monkeypatch.setattr(cli.Theme, "set_mode", lambda theme: closed.append(theme))
    monkeypatch.setattr(cli, "Agent", lambda value: ("agent", value))

    class FakeLoop:
        resume_request = ""

        def __init__(self, agent):
            assert agent == ("agent", session)

        def run(self):
            return 7

        def close_background_output(self):
            closed.append("background")

    monkeypatch.setattr(cli, "CommandLoop", FakeLoop)

    assert cli.main(["--config", "custom.toml", "--yolo", "--theme", "light"]) == 7
    assert closed == ["resolved-dark", "background"]


def test_interactive_banner_precedes_session_and_ui_imports(monkeypatch):
    """A large worktree or a cold render import must not leave the terminal blank."""

    class Tty(StringIO):
        def isatty(self):
            return True

    stdin = Tty()
    stdout = Tty()
    monkeypatch.setattr(cli.sys, "stdin", stdin)
    monkeypatch.setattr(cli.sys, "stdout", stdout)
    calls = []

    def configure_logging():
        calls.append(("configure", stdout.getvalue()))

    session = SimpleNamespace(settings=SimpleNamespace(theme="dark"), mcp=None)
    monkeypatch.setattr(cli, "configure_logging", configure_logging)
    monkeypatch.setattr(cli.Session, "from_config_file", lambda **_kwargs: session)
    monkeypatch.setattr(cli.Theme, "resolve", lambda theme: theme)
    monkeypatch.setattr(cli.Theme, "set_mode", lambda _theme: None)
    monkeypatch.setattr(cli, "Agent", lambda value: value)
    monkeypatch.setattr(cli, "warm_provider_sdks", lambda: None)

    class FakeLoop:
        resume_request = ""

        def __init__(self, _agent):
            pass

        def run(self, *, show_banner=True):
            calls.append(("run", show_banner))
            return 0

        def close_background_output(self):
            pass

    monkeypatch.setattr(cli, "CommandLoop", FakeLoop)

    assert cli.main([]) == 0
    banner = f"wizolt {cli.__version__}. /help for commands.\n"
    assert calls == [("configure", banner), ("run", False)]
    assert stdout.getvalue() == banner


def test_cli_loads_resumed_session_with_runtime_overrides(monkeypatch):
    loaded = {}
    session = SimpleNamespace(settings=SimpleNamespace(theme="auto"), mcp=None)
    catalog = SimpleNamespace(policy="selected-policy")
    monkeypatch.setattr(cli.ConfigFile, "load", lambda path: {"runtime": {"theme": "dark"}})
    monkeypatch.setattr(cli.Config, "data_dir_from", lambda data: "data-dir")
    monkeypatch.setattr(cli, "CatalogRuntime", lambda data_dir: catalog)
    monkeypatch.setattr(cli.Config, "from_dict", lambda data, **kwargs: ("config", data, kwargs))
    monkeypatch.setattr(cli.RuntimeSettings, "from_dict", lambda data, **kwargs: ("settings", data, kwargs))

    def load_snapshot(uid, **kwargs):
        loaded.update(uid=uid, **kwargs)
        return session

    monkeypatch.setattr(cli.Session, "load_snapshot", load_snapshot)
    monkeypatch.setattr(cli.Theme, "resolve", lambda theme: theme)
    monkeypatch.setattr(cli.Theme, "set_mode", lambda _theme: None)
    monkeypatch.setattr(cli, "Agent", lambda value: value)
    monkeypatch.setattr(cli, "CommandLoop", lambda _agent: SimpleNamespace(run=lambda: 0, close_background_output=lambda: None, resume_request=""))
    monkeypatch.setattr(cli.os, "getcwd", lambda: "/workspace")

    assert cli.main(["--resume", "saved", "--config", "custom.toml", "--yolo", "--theme", "light"]) == 0
    assert loaded == {
        "uid": "saved",
        "config": ("config", {"runtime": {"theme": "dark"}}, {"policy": "selected-policy"}),
        "settings": ("settings", {"runtime": {"theme": "dark"}}, {"yolo": True, "theme": "light"}),
        "cwd": "/workspace",
        "catalog": catalog,
    }


@pytest.mark.parametrize(
    ("error", "return_code", "message"),
    [
        (cli.ConfigError("bad config"), 2, "ConfigError: bad config"),
        (cli.WizoltError("bad session"), 1, "Error: bad session"),
    ],
)
def test_cli_reports_domain_errors(monkeypatch, capsys, error, return_code, message):
    def fail(**_kwargs):
        raise error

    monkeypatch.setattr(cli.Session, "from_config_file", fail)

    assert cli.main([]) == return_code
    assert capsys.readouterr().err.strip() == message


def test_cli_update_already_current(monkeypatch, capsys):
    monkeypatch.setattr(cli.UpdateChecker, "fetch_latest_sync", lambda: cli.__version__)
    called = []
    monkeypatch.setattr(cli.subprocess, "call", lambda command: called.append(command) or 0)

    assert cli.main(["update"]) == 0
    assert "already up to date" in capsys.readouterr().out
    assert called == []


def test_cli_upgrade_runs_package_manager(monkeypatch, capsys):
    monkeypatch.setattr(cli.UpdateChecker, "fetch_latest_sync", lambda: "999.0.0")
    monkeypatch.setattr(cli.UpdateChecker, "upgrade_command", lambda: ["uv", "tool", "upgrade", "wizolt"])
    called = []
    monkeypatch.setattr(cli.subprocess, "call", lambda command: called.append(command) or 3)

    assert cli.main(["upgrade"]) == 3
    out = capsys.readouterr().out
    assert f"{cli.__version__} -> 999.0.0" in out
    assert called == [["uv", "tool", "upgrade", "wizolt"]]


def test_cli_update_reports_fetch_error(monkeypatch, capsys):
    def boom():
        raise cli.WizoltError("network down")

    monkeypatch.setattr(cli.UpdateChecker, "fetch_latest_sync", boom)

    assert cli.main(["update"]) == 1
    assert "network down" in capsys.readouterr().err


def test_cli_update_reports_missing_package_manager(monkeypatch, capsys):
    monkeypatch.setattr(cli.UpdateChecker, "fetch_latest_sync", lambda: "999.0.0")
    monkeypatch.setattr(cli.UpdateChecker, "upgrade_command", lambda: ["missing-installer"])

    def missing(_command):
        raise FileNotFoundError("installer not found")

    monkeypatch.setattr(cli.subprocess, "call", missing)

    assert cli.main(["update"]) == 1
    assert "could not run the upgrade command: installer not found" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("/home/u/.local/share/uv/tools/wizolt", ["uv", "tool", "upgrade", "wizolt"]),
        ("/home/u/.local/pipx/venvs/wizolt", ["pipx", "upgrade", "wizolt"]),
    ],
)
def test_upgrade_command_detects_installer(monkeypatch, prefix, expected):
    monkeypatch.setattr(cli.sys, "prefix", prefix)

    assert cli.UpdateChecker.upgrade_command() == expected


def test_upgrade_command_falls_back_to_pip(monkeypatch):
    executable = "/home/u/venvs/foo/bin/python"
    monkeypatch.setattr(cli.sys, "prefix", "/home/u/venvs/foo")
    monkeypatch.setattr(cli.sys, "executable", executable)

    assert cli.UpdateChecker.upgrade_command() == [executable, "-m", "pip", "install", "--upgrade", "wizolt"]


def test_upgrade_command_detects_uv_tool_venv_symlink(monkeypatch, tmp_path):
    """uv tool venvs symlink bin/python to the base interpreter; realpath escapes the venv.

    The venv marker must therefore come from sys.prefix, which stays inside the venv.
    """
    base = tmp_path / "base"
    base.mkdir()
    venv_bin = tmp_path / "uv" / "tools" / "wizolt" / "bin"
    venv_bin.mkdir(parents=True)
    python_link = venv_bin / "python"
    python_link.symlink_to(base / "python")

    monkeypatch.setattr(cli.sys, "executable", str(python_link))
    monkeypatch.setattr(cli.sys, "prefix", str(tmp_path / "uv" / "tools" / "wizolt"))

    assert "/uv/tools/" not in os.path.realpath(str(python_link))
    assert cli.UpdateChecker.upgrade_command() == ["uv", "tool", "upgrade", "wizolt"]


def test_startup_does_not_import_the_provider_sdks():
    """The prompt must accept input immediately, so the ~0.8s SDK imports stay off the startup path.

    A fresh interpreter is used because the test session has already imported both SDKs.
    """
    probe = "import wizolt.__main__, sys; print(int(any(m in sys.modules for m in ('anthropic', 'openai'))))"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "0"


def test_warm_provider_sdks_loads_them_in_the_background():
    cli.warm_provider_sdks()
    for _ in range(200):
        if all(name in sys.modules for name in ("anthropic", "openai")):
            break
        time.sleep(0.05)

    assert {"anthropic", "openai"} <= sys.modules.keys()


def test_py_compile():
    """Package compiles without errors."""
    import py_compile
    from pathlib import Path

    for source in sorted(Path("wizolt").glob("*.py")):
        py_compile.compile(str(source), doraise=True)
