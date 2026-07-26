from types import SimpleNamespace

import pytest

from minacode import __main__ as cli


def test_cli_rejects_native_windows(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "platform", "win32")

    assert cli.main([]) == 1
    assert "use WSL instead" in capsys.readouterr().err


def test_cli_prints_version(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == cli.__version__


@pytest.mark.parametrize(("created", "prefix"), [(True, "Created"), (False, "Exists")])
def test_cli_initializes_config(monkeypatch, capsys, created, prefix):
    monkeypatch.setattr(cli.ConfigFile, "init", lambda path: (path, created))

    assert cli.main(["--init-config", "--config", "/tmp/minacode.toml"]) == 0
    assert capsys.readouterr().out.strip() == f"{prefix} config: /tmp/minacode.toml"


def test_cli_runs_session_and_closes_resources(monkeypatch):
    closed = []
    mcp = SimpleNamespace(close=lambda: closed.append("mcp"))
    session = SimpleNamespace(settings=SimpleNamespace(theme="dark"), mcp=mcp)
    monkeypatch.setattr(cli.Session, "from_config_file", lambda **kwargs: session)
    monkeypatch.setattr(cli.Theme, "resolve", lambda theme: f"resolved-{theme}")
    monkeypatch.setattr(cli.Theme, "set_mode", lambda theme: closed.append(theme))
    monkeypatch.setattr(cli, "Agent", lambda value: ("agent", value))

    class FakeLoop:
        def __init__(self, agent):
            assert agent == ("agent", session)

        def run(self):
            return 7

        def close_background_output(self):
            closed.append("background")

    monkeypatch.setattr(cli, "CommandLoop", FakeLoop)

    assert cli.main(["--config", "custom.toml", "--yolo", "--theme", "light"]) == 7
    assert closed == ["resolved-dark", "background", "mcp"]


def test_cli_loads_resumed_session_with_runtime_overrides(monkeypatch):
    loaded = {}
    session = SimpleNamespace(settings=SimpleNamespace(theme="auto"), mcp=None)
    monkeypatch.setattr(cli.ConfigFile, "load", lambda path: {"runtime": {"theme": "dark"}})
    monkeypatch.setattr(cli.Config, "from_dict", lambda data: ("config", data))
    monkeypatch.setattr(cli.RuntimeSettings, "from_dict", lambda data, **kwargs: ("settings", data, kwargs))

    def load_snapshot(uid, **kwargs):
        loaded.update(uid=uid, **kwargs)
        return session

    monkeypatch.setattr(cli.Session, "load_snapshot", load_snapshot)
    monkeypatch.setattr(cli.Theme, "resolve", lambda theme: theme)
    monkeypatch.setattr(cli.Theme, "set_mode", lambda _theme: None)
    monkeypatch.setattr(cli, "Agent", lambda value: value)
    monkeypatch.setattr(cli, "CommandLoop", lambda _agent: SimpleNamespace(run=lambda: 0, close_background_output=lambda: None))
    monkeypatch.setattr(cli.os, "getcwd", lambda: "/workspace")

    assert cli.main(["--resume", "saved", "--config", "custom.toml", "--yolo", "--theme", "light"]) == 0
    assert loaded == {
        "uid": "saved",
        "config": ("config", {"runtime": {"theme": "dark"}}),
        "settings": ("settings", {"runtime": {"theme": "dark"}}, {"yolo": True, "theme": "light"}),
        "cwd": "/workspace",
    }


@pytest.mark.parametrize(
    ("error", "return_code", "message"),
    [
        (cli.ConfigError("bad config"), 2, "ConfigError: bad config"),
        (cli.MinacodeError("bad session"), 1, "Error: bad session"),
    ],
)
def test_cli_reports_domain_errors(monkeypatch, capsys, error, return_code, message):
    def fail(**_kwargs):
        raise error

    monkeypatch.setattr(cli.Session, "from_config_file", fail)

    assert cli.main([]) == return_code
    assert capsys.readouterr().err.strip() == message
