"""provider command ui (split from tests/test_command_ui.py)."""
import threading
from types import SimpleNamespace

import openai as openai_module
import pytest
from catalog_harness import resolve
from test_command_ui import ModalHarness, diff_loop
from tui_harness import ResizableOutput, loop, rendered_screen_text, run_interactive_tui, wait_until

import minacode.cli.commands as commands_mod
import minacode.cli.modals as modals_mod
from minacode.base import (
    SELECTION_BACK,
)
from minacode.cli import COMMANDS, QUEUE_SAFE_COMMANDS, CommandCompleter, CommandLoop
from minacode.cli.commands import (
    SET_KEYS,
    api,
    catalog_command,
    config,
    language_command,
    model,
    provider,
    reason,
    remote_models,
    set_model,
    set_value,
    strict,
)
from minacode.cli.modals import choice_application, diff_viewer, select_choice
from minacode.config import (
    PROVIDER_API_CHOICES,
    REASONING_CHOICES,
    ProviderConfig,
)
from minacode.providers.sync import CatalogRuntime
from minacode.tui import TUI_MODAL_PENDING, DiffViewState, TabbedViewState, TuiApp


def test_catalog_command_reports_the_selected_snapshot_and_is_not_queue_safe(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.session.catalog = CatalogRuntime(command_loop.session.config.data_dir)

    status = catalog_command(command_loop, "status")

    assert "| version |" in status
    assert "| updated |" in status
    assert "| schema |" in status
    assert "| scope |" in status
    assert catalog_command(command_loop, "unknown") == "Usage: /catalog [status|sync]"
    assert "/catalog" not in QUEUE_SAFE_COMMANDS


def test_choice_navigation_uses_shared_modal_protocol(tmp_path):
    command_loop = loop(tmp_path)
    modal = ModalHarness(["j", "enter"])
    command_loop.tui = modal
    result = choice_application(command_loop, "Pick", ("a", "b", "c"), {"a": "Alpha", "b": "Beta", "c": "Gamma"}, "", set())

    assert result == "b"
    assert "Beta" in "".join(text for frame in modal.frames for _, text in frame)

def test_provider_selection_chains_provider_model_api_and_reasoning(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["other"] = ProviderConfig(model="model-b", available_models=("model-b",), reasoning="low")
    selected = iter(["other", "model-b", "responses", "high"])
    titles = []

    def select(_loop, title, *_args, **_kwargs):
        titles.append(title)
        return next(selected)

    monkeypatch.setattr(commands_mod, "select_choice", select)
    discovered = []
    monkeypatch.setattr(commands_mod, "remote_models", lambda _loop, provider: discovered.append(provider.model) or ())

    result = provider(command_loop, "")

    assert titles == ["Provider", "Model", "Request API", "Reasoning effort"]
    assert command_loop.session.config.active_provider == "other"
    assert command_loop.session.config.provider.model == "model-b"
    assert command_loop.session.config.provider.api == "responses"
    assert command_loop.session.config.provider.reasoning == "high"
    assert discovered == ["model-b"]
    assert "Set provider.model = model-b" in result
    assert "Set provider.api = responses (wire: responses)" in result

def test_runtime_provider_switches_are_recorded_for_resume(tmp_path):
    """Every runtime switch is recorded per entry so a later --resume can restore it."""
    command_loop = loop(tmp_path)
    command_loop.interactive_input = False
    session = command_loop.session
    session.config.providers["other"] = ProviderConfig(model="m", api="chat", reasoning="low")

    assert provider(command_loop, "other") == "Set provider = other"
    assert session.provider_overrides == {"active_provider": "other"}

    assert set_model(command_loop, "model-b") == "Set provider.model = model-b"
    assert session.provider_overrides["providers"]["other"]["model"] == "model-b"

    assert reason(command_loop, "high") == "Set provider.reasoning = high"
    assert session.provider_overrides["providers"]["other"]["reasoning"] == "high"

    assert api(command_loop, "responses") == "Set provider.api = responses (wire: responses)"
    assert session.provider_overrides["providers"]["other"]["api"] == "responses"
    assert session.provider_overrides["active_provider"] == "other"

def test_model_override_binds_to_the_entry_it_was_set_on(tmp_path):
    """model/reasoning/api overrides key on the active entry at switch time, so a /provider after a
    /model restores each switch to the entry it was made on."""
    command_loop = loop(tmp_path)
    command_loop.interactive_input = False
    session = command_loop.session
    session.config.providers["a"] = ProviderConfig(model="ma", api="chat", reasoning="low")
    session.config.providers["b"] = ProviderConfig(model="mb", api="chat", reasoning="low")

    set_model(command_loop, "model-on-default")
    assert session.provider_overrides["providers"]["default"]["model"] == "model-on-default"

    provider(command_loop, "a")
    set_model(command_loop, "model-on-a")
    assert session.provider_overrides["providers"]["a"]["model"] == "model-on-a"
    assert session.provider_overrides["active_provider"] == "a"

    session.provider_overrides["active_provider"] = "default"
    session.apply_provider_overrides()
    assert session.config.active_provider == "default"
    assert session.config.providers["default"].model == "model-on-default"
    assert session.config.providers["a"].model == "model-on-a"

def test_provider_and_model_commands_validate_direct_arguments(tmp_path):
    command_loop = loop(tmp_path)

    assert provider(command_loop, "one two") == "Usage: /provider [NAME]"
    assert provider(command_loop, "missing") == "Unknown provider: missing"
    assert model(command_loop, "one two") == "Usage: /model [MODEL]"

def test_reason_strict_and_set_commands_validate_values(tmp_path):
    from prompt_toolkit.document import Document

    command_loop = loop(tmp_path)

    assert reason(command_loop, "invalid").startswith("Usage: /reason ")
    assert reason(command_loop, "max") == "Set provider.reasoning = max"
    assert command_loop.session.config.provider.reasoning == "max"
    assert strict(command_loop, "on") == "Usage: /strict"
    assert set_value(command_loop, "") == "Usage: /set KEY VALUE"
    assert set_value(command_loop, "unknown value") == "Unknown config key: unknown"
    assert set_value(command_loop, "provider.timeout never") == "Invalid value for provider.timeout"
    assert set_value(command_loop, "provider.response_timeout 900") == "Set provider.response_timeout"
    assert command_loop.session.config.provider.response_timeout == 900
    assert set_value(command_loop, "provider.temperature off") == "Set provider.temperature"
    assert command_loop.session.config.provider.temperature is None
    assert set_value(command_loop, "provider.stream maybe") == "Invalid value for provider.stream"
    assert set_value(command_loop, "provider.stream off") == "Set provider.stream"
    assert command_loop.session.config.provider.stream is False
    stream_values = [item.text for item in CommandCompleter().get_completions(Document("/set provider.stream "), None)]
    assert stream_values == ["on", "off"]
    assert set_value(command_loop, "provider.image_input off") == "Unknown config key: provider.image_input"

def test_reason_only_accepts_efforts_the_active_model_takes(tmp_path):
    """What is offered is what the model accepts, so the chosen effort is the sent effort."""
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider
    provider.url = "https://api.openai.com/v1"
    provider.model = "gpt-5.5"

    assert reason(command_loop, "xhigh") == "Set provider.reasoning = xhigh"
    assert resolve(command_loop.session.config.provider).reasoning_effort == "xhigh"
    # gpt-5.5 has no `max`, so it is not offered and cannot be set: the effort on screen is the
    # effort sent, with nothing rewritten in between.
    assert reason(command_loop, "max").startswith("Usage: /reason ")

def test_reason_accepts_a_level_the_active_model_declares(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.session.config.providers["default"] = ProviderConfig.from_dict(
        {"url": "https://gw.example/v1", "model": "custom-1", "models": {"custom-*": {"reasoning": ["low", "high", "ultra"]}}}
    )

    assert reason(command_loop, "ultra") == "Set provider.reasoning = ultra"
    assert resolve(command_loop.session.config.provider).reasoning_effort == "ultra"
    assert reason(command_loop, "elsewhere").startswith("Usage: /reason ")
    # A level minacode knows but this model does not is refused like any other unavailable one.
    assert reason(command_loop, "medium").startswith("Usage: /reason ")

def test_config_shows_the_reasoning_effort_resolved_for_the_active_model(tmp_path):
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider
    provider.url = "https://api.openai.com/v1"
    provider.model = "gpt-5.5"
    provider.reasoning = "max"

    assert "provider.resolved_reasoning_effort: xhigh" in config(command_loop, "")

def test_language_command_shows_sets_and_resets(tmp_path):
    command_loop = loop(tmp_path)

    assert language_command(command_loop, "") == "Reply language: auto (follows your messages)"

    assert language_command(command_loop, "Chinese") == "Reply language set: Chinese"
    assert language_command(command_loop, "") == "Reply language: Chinese"
    assert command_loop.session.settings.language == "Chinese"

    # the value is normalized (stripped), and free text like CJK names is allowed
    assert language_command(command_loop, "  简体中文  ") == "Reply language set: 简体中文"

    assert language_command(command_loop, "  AUTO  ") == "Reply language reset to auto"
    assert language_command(command_loop, "") == "Reply language: auto (follows your messages)"

    # invalid values return the validation message instead of raising
    assert language_command(command_loop, "Chinese\nJapanese").startswith("runtime.language")
    assert language_command(command_loop, "x" * 65).startswith("runtime.language")
    assert command_loop.session.settings.language == "auto"  # unchanged after the rejected set

def test_config_shows_runtime_language(tmp_path):
    command_loop = loop(tmp_path)
    assert "runtime.language: auto" in config(command_loop, "")

    command_loop.session.settings.language = "Chinese"
    assert "runtime.language: Chinese" in config(command_loop, "")

def test_api_command_switches_the_request_wire_and_names_what_took_effect(tmp_path):
    # A model chosen with /model may not be served over the provider's configured protocol, so the
    # wire has to be switchable in-session rather than only in the config file.
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider
    provider.url = "https://example.com/compatible-mode/v1"
    provider.api = "responses"

    assert api(command_loop, "grpc").startswith("Usage: /api ")
    assert resolve(provider).api == "responses"
    assert api(command_loop, "chat") == "Set provider.api = chat (wire: chat)"
    assert resolve(provider).api == "chat"
    # "auto" reports the wire it inferred rather than echoing "auto" back.
    assert api(command_loop, "auto") == "Set provider.api = auto (wire: chat)"

    provider.url = "https://example.com/v1/responses"
    assert api(command_loop, "auto") == "Set provider.api = auto (wire: responses)"

def test_api_command_selection_offers_every_protocol_with_the_inferred_wire(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    provider = command_loop.session.config.provider
    provider.url = "https://example.com/v1/responses"
    provider.api = "chat"
    shown = {}

    def choose(_loop, title, choices, labels, current, _disabled, preview_fn=None):
        shown.update(title=title, choices=choices, labels=labels, current=current)
        return "auto"

    monkeypatch.setattr(modals_mod, "choice_application", choose)

    assert api(command_loop, "") == "Set provider.api = auto (wire: responses)"
    assert shown["title"] == "Request API"
    assert shown["choices"] == PROVIDER_API_CHOICES
    assert shown["current"] == "chat"
    assert shown["labels"]["auto"] == "auto - infer from the endpoint URL and model (responses)"
    assert shown["labels"]["chat"] == "chat (current)"

def test_reasoning_picker_offers_only_what_the_model_takes(tmp_path, monkeypatch):
    """The picker is the model's scale, not minacode's."""
    import minacode.cli.modals as modals_mod

    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    provider = command_loop.session.config.provider
    provider.url = "https://api.openai.com/v1"
    provider.model = "gpt-5.5"
    shown = {}

    def choose(_loop, title, choices, labels, current, _disabled, preview_fn=None):
        shown.update(choices=choices, labels=labels, footer=preview_fn and "".join(text for _style, text in preview_fn(choices[0])))
        shown["footer_styles"] = {style for style, _text in preview_fn(choices[0])} if preview_fn else set()
        return "high"

    monkeypatch.setattr(modals_mod, "choice_application", choose)

    assert reason(command_loop, "") == "Set provider.reasoning = high"
    # gpt-5.5 documents low/medium/high/xhigh. `minimal` and `max` are not choices here, so no row
    # can be picked that the request would then have to rewrite.
    assert shown["choices"] == ("off", "low", "medium", "high", "xhigh")
    assert shown["labels"]["off"] == "off - disable reasoning"
    # A shortened list has to account for itself where it is shown, or it just looks broken.
    assert shown["footer"] == (
        "  │ Why these levels\n"
        "  │ this generation dropped minimal, and max is not a spelling here\n"
        "  │ https://developers.openai.com/api/docs/models/gpt-5.5\n"
    )
    # The dim style every other secondary line in a modal uses, not the preview default, which is
    # green italic and reads as content rather than as a note about the screen.
    assert shown["footer_styles"] == {"class:choice.disabled"}

def test_reasoning_picker_offers_the_levels_the_model_declares(tmp_path, monkeypatch):
    import minacode.cli.modals as modals_mod

    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["default"] = ProviderConfig.from_dict(
        {"url": "https://gw.example/v1", "model": "custom-1", "models": {"custom-*": {"reasoning": ["low", "high", "ultra"]}}}
    )
    shown = {}

    def choose(_loop, title, choices, labels, current, _disabled, preview_fn=None):
        shown.update(choices=choices, labels=labels, footer=preview_fn and "".join(text for _style, text in preview_fn(choices[0])))
        shown["footer_styles"] = {style for style, _text in preview_fn(choices[0])} if preview_fn else set()
        return "ultra"

    monkeypatch.setattr(modals_mod, "choice_application", choose)

    assert reason(command_loop, "") == "Set provider.reasoning = ultra"
    assert shown["choices"] == ("off", "low", "high", "ultra")
    # A config declaration is its own account of itself; there is no page to cite for it.
    assert shown["footer"] == "  │ Why these levels\n  │ declared for this model in your config\n"

def test_api_is_registered_like_reason_and_completes_its_choices(tmp_path):
    from prompt_toolkit.document import Document

    command_loop = loop(tmp_path)

    assert "/api" in CommandLoop.COMMANDS
    command_loop.command("/api anthropic")
    assert command_loop.session.config.provider.api == "anthropic"

    texts = [c.text for c in CommandCompleter().get_completions(Document("/api "), None)]
    assert set(texts) == set(PROVIDER_API_CHOICES)
    # The wire is a command, not a /set key, so it must not be reachable both ways.
    assert "provider.api" not in SET_KEYS
    assert set_value(command_loop, "provider.api chat") == "Unknown config key: provider.api"

def test_model_chain_steps_back_from_the_wire_to_the_model_and_from_reasoning_to_the_wire(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    provider = command_loop.session.config.provider
    provider.available_models = ("model-a", "model-b")
    scripted = iter(
        [
            ("Model", "model-a"),
            ("Request API", SELECTION_BACK),  # back lands on the model picker again
            ("Model", "model-a"),
            ("Request API", "chat"),
            ("Reasoning effort", SELECTION_BACK),  # back lands on the wire, not the model
            ("Request API", "responses"),
            ("Reasoning effort", "high"),
        ]
    )
    titles = []

    def select(_loop, title, *_args, **_kwargs):
        expected_title, value = next(scripted)
        assert title == expected_title
        titles.append(title)
        return value

    monkeypatch.setattr(commands_mod, "select_choice", select)
    monkeypatch.setattr(commands_mod, "remote_models", lambda _loop, _provider: ())

    result = model(command_loop, "")

    assert titles == ["Model", "Request API", "Model", "Request API", "Reasoning effort", "Request API", "Reasoning effort"]
    assert provider.model == "model-a"
    assert provider.api == "responses"
    assert provider.reasoning == "high"
    assert "Set provider.api = responses (wire: responses)" in result

def test_model_chain_leaves_the_wire_alone_when_selection_is_unavailable(tmp_path):
    # Non-interactive input returns None from every picker; the model still applies, the wire is untouched.
    command_loop = loop(tmp_path)
    command_loop.interactive_input = False
    provider = command_loop.session.config.provider
    provider.api = "responses"
    provider.reasoning = "low"

    result = set_model(command_loop, "model-a")

    assert result == "Set provider.model = model-a"
    assert provider.model == "model-a"
    assert provider.api == "responses"
    assert provider.reasoning == "low"

def test_remote_models_normalizes_sdk_results(monkeypatch, tmp_path):
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider
    provider.url = "https://example.com/v1"
    provider.key = "secret"
    calls = []

    class Models:
        def list(self):
            return SimpleNamespace(data=[{"id": "zeta"}, SimpleNamespace(id="alpha"), {"id": "zeta"}, {"missing": True}, None])

    def openai(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(models=Models())

    monkeypatch.setattr(openai_module, "OpenAI", openai)

    assert remote_models(command_loop, provider) == ("alpha", "zeta")
    assert calls[0]["api_key"] == "secret"
    assert calls[0]["max_retries"] == 0

def test_remote_models_is_optional_and_failure_safe(monkeypatch, tmp_path):
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider

    assert remote_models(command_loop, provider) == ()

    provider.url = "https://example.com/v1"
    provider.key = "secret"
    monkeypatch.setattr(openai_module, "OpenAI", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")))

    assert remote_models(command_loop, provider) == ()

def test_effort_is_an_alias_for_reason(tmp_path):
    command_loop = loop(tmp_path)

    # Registered as a command that dispatches to the same handler as /reason.
    assert "/effort" in CommandLoop.COMMANDS
    reason_command = next(command for command in COMMANDS if command.name == "/reason")
    assert "/effort" in reason_command.aliases

    # Dispatch sets reasoning effort exactly like /reason.
    command_loop.command("/effort high")
    assert command_loop.session.config.provider.reasoning == "high"

    # Tab completion offers the same reasoning choices.
    from prompt_toolkit.document import Document

    texts = [c.text for c in CommandCompleter().get_completions(Document("/effort "), None)]
    assert set(texts) == set(REASONING_CHOICES)

def test_model_selection_groups_configured_and_remote_choices_like_master(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    provider = command_loop.session.config.provider
    provider.model = "configured-model"
    provider.available_models = ("configured-model",)
    provider.url = "https://example.com/v1"
    provider.key = "key"
    shown = []

    def select(_loop, title, choices, **_kwargs):
        shown.append((title, choices))
        if title == "Reasoning effort":
            return "off"
        if title == "Request API":
            return "auto"
        return "remote-model"

    monkeypatch.setattr(commands_mod, "select_choice", select)
    monkeypatch.setattr(commands_mod, "remote_models", lambda _loop, _provider: ("remote-model",))

    assert "Set provider.model = remote-model" in model(command_loop, "")
    assert shown[0] == (
        "Model",
        (
            commands_mod.MODEL_CONFIGURED_LABEL,
            "configured-model",
            commands_mod.MODEL_DISCOVERED_LABEL,
            "remote-model",
        ),
    )

def test_model_discovery_shows_loading_state_for_selected_provider(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    provider = command_loop.session.config.provider
    provider.model = "configured-model"
    provider.available_models = ("configured-model",)
    provider.url = "https://example.com/v1"
    provider.key = "key"
    transitions = []
    command_loop.tui = TuiApp()
    command_loop.tui.set_dispatching = lambda prompt="": transitions.append(prompt)
    monkeypatch.setattr(commands_mod, "remote_models", lambda _loop, selected: ("remote-model",))
    selected = iter(["remote-model", "auto", "off"])
    monkeypatch.setattr(commands_mod, "select_choice", lambda *_args, **_kwargs: next(selected))

    assert "Set provider.model = remote-model" in model(command_loop, "")
    assert transitions == ["Loading models...", ""]

def test_interactive_provider_chain_uses_one_inline_tui_and_real_navigation(monkeypatch, tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["zz-other"] = ProviderConfig(
        model="model-a",
        available_models=("model-a", "model-b"),
        reasoning="low",
    )
    app = TuiApp()
    command_loop.tui = app
    output = ResizableOutput(rows=20, columns=80)
    result = []
    application_ids = []

    def modal_title():
        modal = app.modal
        if modal is None:
            return ""
        # The modal opens with a blank spacer line and indents its title, so the title is the
        # first non-empty line, stripped.
        return next((line.strip() for line in "".join(text for _, text in modal.fragments_fn()).splitlines() if line.strip()), "")

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        application_ids.append(id(app.app))
        worker = threading.Thread(target=lambda: result.append(provider(command_loop, "")), daemon=True)
        worker.start()
        for title in ("Provider", "Model", "Request API", "Reasoning effort"):
            wait_until(lambda title=title: modal_title().startswith(title))
            wait_until(lambda title=title: title in rendered_screen_text(app.app, output))
            application_ids.append(id(app.app))
            pipe_input.send_text("j\r")
        worker.join(timeout=1)
        assert not worker.is_alive()
        app.set_idle()
        wait_until(lambda: app.modal is None)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive, output=output)

    assert len(set(application_ids)) == 1
    assert command_loop.session.config.active_provider == "zz-other"
    assert command_loop.session.config.provider.model == "model-b"
    assert command_loop.session.config.provider.reasoning == "medium"
    assert "Set provider.model = model-b" in result[0]

def test_single_enabled_choice_is_selected_without_opening_modal(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    monkeypatch.setattr(modals_mod, "choice_application", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("modal should not open")))

    assert select_choice(command_loop, "Provider", ("only",), current="only") == "only"
    assert select_choice(command_loop, "Model", ("heading", "only"), disabled={"heading"}) == "only"

def test_provider_auto_selects_sole_provider_and_model(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    provider_config = command_loop.session.config.provider
    provider_config.available_models = ("only-model",)
    provider_config.model = "only-model"
    provider_config.url = ""
    provider_config.key = ""
    titles = []

    def choose(_loop, title, _choices, _labels, current, _disabled, preview_fn=None):
        titles.append(title)
        return current

    monkeypatch.setattr(modals_mod, "choice_application", choose)

    result = provider(command_loop, "")

    assert titles == ["Request API", "Reasoning effort"]
    assert "Set provider.model = only-model" in result

def test_diff_viewer_switches_tabs_and_opens_selected_file(tmp_path):
    command_loop = diff_loop(tmp_path)
    switched = ModalHarness(["l", "q"])
    command_loop.tui = switched
    diff_viewer(command_loop)
    opened = ModalHarness(["j", "enter", "q"])
    command_loop.tui = opened
    diff_viewer(command_loop)

    assert any(("class:tab.active", " Session ") in frame for frame in switched.frames)
    assert switched.exclusive == [True]
    assert opened.exclusive == [True]
    text = "".join(text for frame in opened.frames for _, text in frame)
    assert "Edit · b.py" in text
    assert "[diff]" in text

def test_diff_viewer_ctrl_d_scrolls_file_preview(tmp_path):
    command_loop = diff_loop(tmp_path)
    initial = ModalHarness(["enter", "q"])
    command_loop.tui = initial
    diff_viewer(command_loop)
    scrolled = ModalHarness(["enter", "c-d", "c-d", "q"])
    command_loop.tui = scrolled
    diff_viewer(command_loop)

    initial_text = "".join(text for frame in initial.frames for _, text in frame)
    scrolled_text = "".join(text for frame in scrolled.frames for _, text in frame)
    assert initial_text != scrolled_text
    assert "[diff]" in scrolled_text

def test_empty_diff_viewer_reports_zero_position(tmp_path):
    command_loop = loop(tmp_path)
    modal = ModalHarness(["q"])
    command_loop.tui = modal
    diff_viewer(command_loop)
    text = "".join(text for frame in modal.frames for _, text in frame)

    assert "No diffs" in text
    assert "[0/0]" in text

def test_diff_view_state_owns_navigation_transitions():
    state = DiffViewState(TabbedViewState(("Latest", "Session")))

    state.handle_key("down", 3, 10)
    assert state.file == 1
    state.handle_key("enter", 3, 10)
    assert state.mode is DiffViewState.Mode.FILE
    state.handle_key("c-d", 3, 10)
    assert state.view.scroll == 5
    assert state.handle_key("escape", 3, 10) is TUI_MODAL_PENDING
    assert state.mode is DiffViewState.Mode.LIST

    state.handle_key("right", 3, 10)
    assert state.view.tab == 1
    assert state.file == 0
    assert state.handle_key("r", 3, 10) is DiffViewState.REFRESH
    assert state.handle_key("q", 3, 10) is None

def test_diff_view_g_and_shift_g_jump_top_and_bottom():
    state = DiffViewState(TabbedViewState(("Latest", "Session")))

    # LIST mode: jump file selection to last / first.
    state.handle_key("G", 5, 10)
    assert state.file == 4
    state.handle_key("g", 5, 10)
    assert state.file == 0

    # FILE mode: jump scroll to bottom (clamped on render) / top.
    state.handle_key("enter", 5, 10)
    assert state.mode is DiffViewState.Mode.FILE
    state.handle_key("G", 5, 10)
    assert state.view.scroll > 0
    state.handle_key("g", 5, 10)
    assert state.view.scroll == 0

@pytest.mark.parametrize(("key", "expected_tab"), [("l", 1), ("tab", 1), ("h", 0)])
def test_diff_view_h_l_and_tab_switch_tabs_from_file_preview(key, expected_tab):
    state = DiffViewState(TabbedViewState(("Latest", "Session"), tab=0 if key != "h" else 1))
    state.open_file(3)

    state.handle_key(key, 3, 10)

    assert state.view.tab == expected_tab
    assert state.mode is DiffViewState.Mode.LIST
    assert state.file == 0

def test_switching_provider_says_when_it_had_to_move_the_effort(tmp_path):
    """The one moment an effort changes without being chosen. Saying it is the price of never
    rewriting one silently on the way to the provider."""
    command_loop = loop(tmp_path)
    session = command_loop.session
    # `medium` is the default effort and DeepSeek has no such level, so an entry can arrive at its
    # own model already holding one it cannot use.
    session.config.providers["deep"] = ProviderConfig(url="https://api.deepseek.com", key="k", model="deepseek-v4-flash", reasoning="medium")

    result = provider(command_loop, "deep")

    assert result == "Set provider = deep\nReasoning medium is not offered by deepseek-v4-flash, using high"
    assert session.config.provider.reasoning == "high"
    assert session.provider_overrides["providers"]["deep"]["reasoning"] == "high"

def test_switching_provider_stays_quiet_when_the_effort_still_fits(tmp_path):
    command_loop = loop(tmp_path)
    session = command_loop.session
    session.config.providers["deep"] = ProviderConfig(url="https://api.deepseek.com", key="k", model="deepseek-v4-flash", reasoning="high")

    assert provider(command_loop, "deep") == "Set provider = deep"
    assert session.config.provider.reasoning == "high"
