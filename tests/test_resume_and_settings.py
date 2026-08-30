"""resume and settings (split from tests/test_core_logic.py)."""

from types import SimpleNamespace

import pytest

import wizolt.__main__ as cli
from wizolt.__main__ import main
from wizolt.base import (
    ConfigError,
)
from wizolt.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_OUTPUT_RESERVE_TOKENS,
    Config,
    ConfigFile,
    ProviderConfig,
    RuntimeSettings,
)
from wizolt.providers.compat import bundled_policy
from wizolt.session import Session


def test_default_user_paths_prefer_wizolt_then_minacode_then_nanocode(isolate_home):
    def config_at(directory: str):
        path = isolate_home / directory / "config.toml"
        path.parent.mkdir()
        path.write_text("[provider]\n", encoding="utf-8")
        return path

    nanocode = config_at(".nanocode")
    assert ConfigFile.resolve_path(None) == str(nanocode)
    assert Config.data_dir_from({}) == "~/.nanocode"

    minacode = config_at(".minacode")
    assert ConfigFile.resolve_path(None) == str(minacode)
    assert Config.data_dir_from({}) == "~/.minacode"

    wizolt = config_at(".wizolt")
    assert ConfigFile.resolve_path(None) == str(wizolt)
    assert Config.data_dir_from({}) == "~/.wizolt"


@pytest.mark.parametrize("flag", ["-c", "--last", "--latest"])
def test_continue_flags_resume_latest_session_in_current_project(tmp_path, monkeypatch, flag):
    config = Config(data_dir=str(tmp_path / "data"))
    settings = RuntimeSettings()
    resumed = SimpleNamespace(settings=settings, mcp=None)
    selected = []

    monkeypatch.setattr(ConfigFile, "load", lambda _path: {})
    monkeypatch.setattr(Config, "from_dict", classmethod(lambda _cls, _data, **_kwargs: config))
    monkeypatch.setattr(RuntimeSettings, "from_dict", classmethod(lambda _cls, _data, **_kwargs: settings))
    monkeypatch.setattr(
        Session,
        "load_snapshot",
        classmethod(lambda _cls, uid, config=None, settings=None, cwd="", catalog=None: selected.append((uid, config, settings, cwd)) or resumed),
    )

    class Loop:
        resume_request = ""

        def run(self):
            return 0

        def close_background_output(self):
            pass

    monkeypatch.setattr(cli, "CommandLoop", lambda _agent: Loop())
    monkeypatch.chdir(tmp_path)

    assert main([flag]) == 0
    # The alias is resolved against the current project, not a global pointer.
    assert selected == [("latest", config, settings, str(tmp_path))]


def test_resume_request_starts_the_next_run_on_the_chosen_session(tmp_path, monkeypatch):
    """/sessions ends one run and main starts the next on the session it named, instead of
    re-pointing a live object graph at a different Session."""
    config = Config(data_dir=str(tmp_path / "data"))
    settings = RuntimeSettings()
    loaded = []

    monkeypatch.setattr(ConfigFile, "load", lambda _path: {})
    monkeypatch.setattr(Config, "from_dict", classmethod(lambda _cls, _data, **_kwargs: config))
    monkeypatch.setattr(RuntimeSettings, "from_dict", classmethod(lambda _cls, _data, **_kwargs: settings))
    monkeypatch.setattr(
        Session,
        "load_snapshot",
        classmethod(lambda _cls, uid, config=None, settings=None, cwd="", catalog=None: loaded.append(uid) or SimpleNamespace(settings=settings, mcp=None)),
    )
    closed = []
    handovers = iter(["second-uid", ""])

    class Loop:
        def __init__(self, _agent):
            self.resume_request = ""

        def run(self):
            self.resume_request = next(handovers)
            return 3

        def close_background_output(self):
            closed.append(self.resume_request)

    monkeypatch.setattr(cli, "CommandLoop", Loop)
    monkeypatch.chdir(tmp_path)

    assert main(["--resume", "first-uid"]) == 3
    assert loaded == ["first-uid", "second-uid"]
    # Each run is torn down before the next is built; nothing is carried across.
    assert closed == ["second-uid", ""]


def test_runtime_settings_reads_limits_and_yolo_override():
    settings = RuntimeSettings.from_dict(
        {"runtime": {"shell_timeout": 7, "max_agent_steps": 0, "max_context_tokens": 0, "yolo": False}},
        yolo=True,
    )

    assert settings.shell_timeout == 7
    assert settings.max_steps == 1
    assert settings.max_context_tokens == 1
    assert settings.yolo is True


def test_runtime_language_defaults_normalizes_and_validates():
    assert RuntimeSettings().language == "auto"
    assert RuntimeSettings.from_dict({}).language == "auto"
    assert RuntimeSettings.from_dict({"runtime": {"language": "Chinese"}}).language == "Chinese"
    assert RuntimeSettings.from_dict({"runtime": {"language": "简体中文"}}).language == "简体中文"

    # empty and case-insensitive "auto" (with surrounding whitespace) normalize to "auto"
    assert RuntimeSettings.clean_language("") == "auto"
    assert RuntimeSettings.clean_language("  AUTO  ") == "auto"
    assert RuntimeSettings.clean_language("auto") == "auto"
    # valid values pass through unchanged, spaces included
    assert RuntimeSettings.clean_language("Chinese (Simplified)") == "Chinese (Simplified)"
    assert RuntimeSettings.clean_language("简体中文") == "简体中文"

    # multiline and over-long values are rejected on the config-parsing path
    with pytest.raises(ConfigError, match="runtime.language"):
        RuntimeSettings.from_dict({"runtime": {"language": "Chinese\nJapanese"}})
    with pytest.raises(ConfigError, match="runtime.language"):
        RuntimeSettings.clean_language("x" * 65)


def test_runtime_settings_default_context_budget_is_256k():
    assert RuntimeSettings().max_context_tokens == 256 * 1024
    assert RuntimeSettings.from_dict({}).max_context_tokens == 256 * 1024


def test_provider_timeout_defaults_distinguish_inactivity_from_total_generation():
    assert ProviderConfig().timeout == 120
    assert Config.from_dict({}).provider.timeout == 120
    assert ProviderConfig().response_timeout == 600
    assert Config.from_dict({}).provider.response_timeout == 600
    assert ProviderConfig.from_dict({"response_timeout": 0}).response_timeout == 0
    assert "# response_timeout = 600" in ConfigFile.DEFAULT_TEXT


def test_provider_reasoning_history_defaults_to_catalog_policy():
    assert ProviderConfig().reasoning_history == "auto"
    assert Config.from_dict({}).provider.reasoning_history == "auto"
    assert '# reasoning_history = "auto"' in ConfigFile.DEFAULT_TEXT


def test_provider_max_tokens_defaults_to_zero_and_reserve_stays_bounded():
    assert ProviderConfig().max_tokens == DEFAULT_MAX_TOKENS == 0
    assert Config.from_dict({}).provider.max_tokens == DEFAULT_MAX_TOKENS
    assert ProviderConfig.from_dict({"max_tokens": 0}).max_tokens == 0
    assert "# max_tokens = 0" in ConfigFile.DEFAULT_TEXT
    # Unset leaves the wire cap to the provider, but the output reserve kept out of the input budget
    # stays fixed, so compaction planning does not depend on how the provider treats an absent cap.
    assert DEFAULT_MAX_TOKENS != DEFAULT_OUTPUT_RESERVE_TOKENS
    assert ProviderConfig().output_token_budget() == ProviderConfig.from_dict({"max_tokens": 0}).output_token_budget() == DEFAULT_OUTPUT_RESERVE_TOKENS
    # A wire that requires a concrete cap gets its default from the catalog; explicit config wins.
    policy = bundled_policy()
    expected = policy.snapshot.defaults.wire_defaults["anthropic"]["max_tokens"]
    assert policy.resolve(ProviderConfig(api="anthropic")).output_max_tokens == expected
    assert policy.resolve(ProviderConfig(api="anthropic", max_tokens=2_048)).output_max_tokens == 2_048


def test_provider_stream_defaults_on_and_can_be_disabled():
    assert ProviderConfig().stream is True
    assert ProviderConfig.from_dict({"stream": False}).stream is False
    assert "# stream = true" in ConfigFile.DEFAULT_TEXT


def test_provider_ignores_obsolete_image_input_values():
    assert not hasattr(ProviderConfig(), "image_input")
    assert not hasattr(ProviderConfig.from_dict({"image_input": "off"}), "image_input")
