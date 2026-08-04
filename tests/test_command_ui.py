"""Interactive command surfaces: the provider/model/api/reason selection chains, the diff
viewer, and the stored Bash output viewer."""

import os
import shutil
import threading
from types import SimpleNamespace

import openai as openai_module
import pytest
from prompt_toolkit.utils import get_cwidth
from tui_harness import ResizableOutput, loop, rendered_screen_text, run_interactive_tui, session, wait_until

from minacode.base import (
    PROVIDER_API_CHOICES,
    REASONING_CHOICES,
    SELECTION_BACK,
    ModelError,
    ProviderConfig,
)
from minacode.engine import Agent
from minacode.loop import SET_KEYS, CommandCompleter, CommandLoop
from minacode.model import ModelClient
from minacode.session import Session
from minacode.tools import Tool
from minacode.tui import TUI_MODAL_PENDING, DiffViewState, TabbedViewState, TuiApp


def diff_loop(tmp_path):
    command_loop = loop(tmp_path)
    before = "".join(f"old {index}\n" for index in range(20))
    after = "".join(f"new {index}\n" for index in range(20))
    command_loop.session.store_turn_diff("tr.1", 1, "a.py", "unused", before=before, after=after, round=1)
    command_loop.session.store_turn_diff("tr.2", 2, "b.py", "unused", before="old\n", after="new\n", round=1)
    return command_loop


class ModalHarness:
    def __init__(self, keys):
        self.keys = keys
        self.frames = []
        self.exclusive = []

    def show_modal(self, fragments_fn, key_fn, *, exclusive=False):
        self.exclusive.append(exclusive)
        self.frames.append(fragments_fn())
        result = TUI_MODAL_PENDING
        for key in self.keys:
            result = key_fn(key, key if len(key) == 1 else "")
            self.frames.append(fragments_fn())
            if result is not TUI_MODAL_PENDING:
                return result
        return None


def test_bash_output_viewer_browses_latest_ten_bounded_previews(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    for index in range(12):
        stdout = "\n".join(f"line {line}" for line in range(40)) if index == 10 else f"output {index}"
        stderr = "detail stderr" if index == 10 else ""
        command_loop.session.store_tool_result("Bash", [f"printf command-{index}"], Tool.process_result("BashToolResult", 0, stdout, stderr))
    command_loop.session.store_tool_result("Bash", ["true"], Tool.process_result("BashToolResult", 0, "", ""))
    modal = ModalHarness(["j", "enter", "escape", "G", "enter", "c-o"])
    command_loop.tui = modal

    # ``shutil`` is a shared module object also used by pytest's terminal reporter. Restore the
    # patch before pytest reports this test result, rather than waiting for fixture teardown.
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((50, 20)))
        command_loop.bash_output_viewer()

    listing = "".join(value for _style, value in modal.frames[0])
    assert listing.startswith("\n──── Bash outputs · latest 10 ")
    assert get_cwidth(listing.splitlines()[1]) == 48
    assert "command-11" in listing and "command-2" in listing
    assert "Bash printf command-1\n" not in listing and "Bash printf command-0\n" not in listing and "Bash true" not in listing
    second_detail = "".join(value for _style, value in modal.frames[2])
    assert second_detail.startswith("\n──── Bash output · tr.11 ")
    assert get_cwidth(second_detail.splitlines()[1]) == 48
    assert "command-10" in second_detail
    assert "line 0" in second_detail and "line 39" in second_detail
    assert "... 16 lines omitted ..." in second_detail
    assert "detail stderr" in second_detail
    assert "──── Bash outputs · latest 10 " in "".join(value for _style, value in modal.frames[3])
    oldest_detail = "".join(value for _style, value in modal.frames[5])
    assert "command-2" in oldest_detail and "output 2" in oldest_detail
    assert modal.exclusive == [False]


def test_bash_output_viewer_is_noop_without_stored_bash_output(tmp_path):
    command_loop = loop(tmp_path)
    modal = ModalHarness([])
    command_loop.tui = modal

    command_loop.bash_output_viewer()

    assert modal.frames == []


def test_bash_output_viewer_reads_resumed_history(tmp_path):
    saved = session(tmp_path)
    saved.store_tool_result("Bash", ["printf persisted"], Tool.process_result("BashToolResult", 0, "persisted output", ""))
    saved.save_snapshot()
    restored = Session.load_snapshot(saved.uid, config=saved.config)
    command_loop = CommandLoop(Agent(restored, output_fn=lambda _text: None), input_fn=lambda prompt="": "", output_fn=lambda _text: None)
    modal = ModalHarness(["enter", "q"])
    command_loop.tui = modal

    command_loop.bash_output_viewer()

    detail = "".join(value for _style, value in modal.frames[1])
    assert "Bash printf persisted" in detail
    assert "persisted output" in detail


def test_choice_navigation_uses_shared_modal_protocol(tmp_path):
    command_loop = loop(tmp_path)
    modal = ModalHarness(["j", "enter"])
    command_loop.tui = modal
    result = command_loop.choice_application("Pick", ("a", "b", "c"), {"a": "Alpha", "b": "Beta", "c": "Gamma"}, "", set())

    assert result == "b"
    assert "Beta" in "".join(text for frame in modal.frames for _style, text in frame)


def test_provider_selection_chains_provider_model_api_and_reasoning(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["other"] = ProviderConfig(model="model-b", available_models=("model-b",), reasoning="low")
    selected = iter(["other", "model-b", "responses", "high"])
    titles = []

    def select(title, *_args, **_kwargs):
        titles.append(title)
        return next(selected)

    command_loop.select_choice = select
    discovered = []
    command_loop.remote_models = lambda provider: discovered.append(provider.model) or ()

    result = command_loop.provider("")

    assert titles == ["Provider", "Model", "Request API", "Reasoning effort"]
    assert command_loop.session.config.active_provider == "other"
    assert command_loop.session.config.provider.model == "model-b"
    assert command_loop.session.config.provider.api == "responses"
    assert command_loop.session.config.provider.reasoning == "high"
    assert discovered == ["model-b"]
    assert "Set provider.model = model-b" in result
    assert "Set provider.api = responses (wire: responses)" in result


def test_provider_and_model_commands_validate_direct_arguments(tmp_path):
    command_loop = loop(tmp_path)

    assert command_loop.provider("one two") == "Usage: /provider [NAME]"
    assert command_loop.provider("missing") == "Unknown provider: missing"
    assert command_loop.model("one two") == "Usage: /model [MODEL]"


def test_reason_strict_and_set_commands_validate_values(tmp_path):
    from prompt_toolkit.document import Document

    command_loop = loop(tmp_path)

    assert command_loop.reason("invalid").startswith("Usage: /reason ")
    assert command_loop.reason("max") == "Set provider.reasoning = max"
    assert command_loop.session.config.provider.reasoning == "max"
    assert command_loop.strict("on") == "Usage: /strict"
    assert command_loop.set_value("") == "Usage: /set KEY VALUE"
    assert command_loop.set_value("unknown value") == "Unknown config key: unknown"
    assert command_loop.set_value("provider.timeout never") == "Invalid value for provider.timeout"
    assert command_loop.set_value("provider.response_timeout 900") == "Set provider.response_timeout"
    assert command_loop.session.config.provider.response_timeout == 900
    assert command_loop.set_value("provider.temperature off") == "Set provider.temperature"
    assert command_loop.session.config.provider.temperature is None
    assert command_loop.set_value("provider.stream maybe") == "Invalid value for provider.stream"
    assert command_loop.set_value("provider.stream off") == "Set provider.stream"
    assert command_loop.session.config.provider.stream is False
    stream_values = [item.text for item in CommandCompleter().get_completions(Document("/set provider.stream "), None)]
    assert stream_values == ["on", "off"]
    assert command_loop.set_value("provider.image_input maybe") == "Invalid value for provider.image_input"
    assert command_loop.set_value("provider.image_input off") == "Set provider.image_input"
    assert command_loop.session.config.provider.image_input == "off"


def test_config_shows_the_reasoning_effort_resolved_for_the_active_model(tmp_path):
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider
    provider.url = "https://api.openai.com/v1"
    provider.model = "gpt-5.5"
    provider.reasoning = "max"

    assert "provider.resolved_reasoning_effort: xhigh" in command_loop.config("")


def test_api_command_switches_the_request_wire_and_names_what_took_effect(tmp_path):
    # A model chosen with /model may not be served over the provider's configured protocol, so the
    # wire has to be switchable in-session rather than only in the config file.
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider
    provider.url = "https://example.com/compatible-mode/v1"
    provider.api = "responses"

    assert command_loop.api("grpc").startswith("Usage: /api ")
    assert provider.resolve().api == "responses"
    assert command_loop.api("chat") == "Set provider.api = chat (wire: chat)"
    assert provider.resolve().api == "chat"
    # "auto" reports the wire it inferred rather than echoing "auto" back.
    assert command_loop.api("auto") == "Set provider.api = auto (wire: chat)"

    provider.url = "https://example.com/v1/responses"
    assert command_loop.api("auto") == "Set provider.api = auto (wire: responses)"


def test_api_command_selection_offers_every_protocol_with_the_inferred_wire(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    provider = command_loop.session.config.provider
    provider.url = "https://example.com/v1/responses"
    provider.api = "chat"
    shown = {}

    def choose(title, choices, labels, current, _disabled):
        shown.update(title=title, choices=choices, labels=labels, current=current)
        return "auto"

    command_loop.choice_application = choose

    assert command_loop.api("") == "Set provider.api = auto (wire: responses)"
    assert shown["title"] == "Request API"
    assert shown["choices"] == PROVIDER_API_CHOICES
    assert shown["current"] == "chat"
    assert shown["labels"]["auto"] == "auto - infer from the endpoint URL and model (responses)"
    assert shown["labels"]["chat"] == "chat (current)"


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
    assert command_loop.set_value("provider.api chat") == "Unknown config key: provider.api"


def test_model_chain_steps_back_from_the_wire_to_the_model_and_from_reasoning_to_the_wire(tmp_path):
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

    def select(title, *_args, **_kwargs):
        expected_title, value = next(scripted)
        assert title == expected_title
        titles.append(title)
        return value

    command_loop.select_choice = select
    command_loop.remote_models = lambda _provider: ()

    result = command_loop.model("")

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

    result = command_loop.set_model("model-a")

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

    assert command_loop.remote_models(provider) == ("alpha", "zeta")
    assert calls[0]["api_key"] == "secret"
    assert calls[0]["max_retries"] == 0


def test_remote_models_is_optional_and_failure_safe(monkeypatch, tmp_path):
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider

    assert command_loop.remote_models(provider) == ()

    provider.url = "https://example.com/v1"
    provider.key = "secret"
    monkeypatch.setattr(openai_module, "OpenAI", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")))

    assert command_loop.remote_models(provider) == ()


def test_effort_is_an_alias_for_reason(tmp_path):
    command_loop = loop(tmp_path)

    # Registered as a command that dispatches to the same handler as /reason.
    assert "/effort" in CommandLoop.COMMANDS
    assert CommandLoop.COMMAND_HANDLERS["/effort"] == CommandLoop.COMMAND_HANDLERS["/reason"]

    # Dispatch sets reasoning effort exactly like /reason.
    command_loop.command("/effort high")
    assert command_loop.session.config.provider.reasoning == "high"

    # Tab completion offers the same reasoning choices.
    from prompt_toolkit.document import Document

    texts = [c.text for c in CommandCompleter().get_completions(Document("/effort "), None)]
    assert set(texts) == set(REASONING_CHOICES)


def test_model_selection_groups_configured_and_remote_choices_like_master(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    provider = command_loop.session.config.provider
    provider.model = "configured-model"
    provider.available_models = ("configured-model",)
    provider.url = "https://example.com/v1"
    provider.key = "key"
    shown = []

    def select(title, choices, **_kwargs):
        shown.append((title, choices))
        if title == "Reasoning effort":
            return "off"
        if title == "Request API":
            return "auto"
        return "remote-model"

    command_loop.select_choice = select
    command_loop.remote_models = lambda _provider: ("remote-model",)

    assert "Set provider.model = remote-model" in command_loop.model("")
    assert shown[0] == (
        "Model",
        (
            command_loop.MODEL_CONFIGURED_LABEL,
            "configured-model",
            command_loop.MODEL_DISCOVERED_LABEL,
            "remote-model",
        ),
    )


def test_model_discovery_shows_loading_state_for_selected_provider(tmp_path):
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
    command_loop.remote_models = lambda selected: ("remote-model",)
    selected = iter(["remote-model", "auto", "off"])
    command_loop.select_choice = lambda *_args, **_kwargs: next(selected)

    assert "Set provider.model = remote-model" in command_loop.model("")
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
        return "".join(text for _style, text in modal.fragments_fn()).splitlines()[0]

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        application_ids.append(id(app.app))
        worker = threading.Thread(target=lambda: result.append(command_loop.provider("")), daemon=True)
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


def test_single_enabled_choice_is_selected_without_opening_modal(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.choice_application = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("modal should not open"))

    assert command_loop.select_choice("Provider", ("only",), current="only") == "only"
    assert command_loop.select_choice("Model", ("heading", "only"), disabled={"heading"}) == "only"


def test_provider_auto_selects_sole_provider_and_model(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    provider = command_loop.session.config.provider
    provider.available_models = ("only-model",)
    provider.model = "only-model"
    provider.url = ""
    provider.key = ""
    titles = []

    def choose(title, _choices, _labels, current, _disabled):
        titles.append(title)
        return current

    command_loop.choice_application = choose

    result = command_loop.provider("")

    assert titles == ["Request API", "Reasoning effort"]
    assert "Set provider.model = only-model" in result


def test_diff_viewer_switches_tabs_and_opens_selected_file(tmp_path):
    command_loop = diff_loop(tmp_path)
    switched = ModalHarness(["l", "q"])
    command_loop.tui = switched
    command_loop.diff_viewer()
    opened = ModalHarness(["j", "enter", "q"])
    command_loop.tui = opened
    command_loop.diff_viewer()

    assert any(("class:tab.active", " Session ") in frame for frame in switched.frames)
    assert switched.exclusive == [True]
    assert opened.exclusive == [True]
    text = "".join(text for frame in opened.frames for _style, text in frame)
    assert "Edit · b.py" in text
    assert "[diff]" in text


def test_diff_viewer_ctrl_d_scrolls_file_preview(tmp_path):
    command_loop = diff_loop(tmp_path)
    initial = ModalHarness(["enter", "q"])
    command_loop.tui = initial
    command_loop.diff_viewer()
    scrolled = ModalHarness(["enter", "c-d", "c-d", "q"])
    command_loop.tui = scrolled
    command_loop.diff_viewer()

    initial_text = "".join(text for frame in initial.frames for _style, text in frame)
    scrolled_text = "".join(text for frame in scrolled.frames for _style, text in frame)
    assert initial_text != scrolled_text
    assert "[diff]" in scrolled_text


def test_empty_diff_viewer_reports_zero_position(tmp_path):
    command_loop = loop(tmp_path)
    modal = ModalHarness(["q"])
    command_loop.tui = modal
    command_loop.diff_viewer()
    text = "".join(text for frame in modal.frames for _style, text in frame)

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


def test_api_command_reports_an_incompatible_builtin_tools_configuration_without_clearing_it(tmp_path):
    """Switching /api reports inactive builtin tools and never rewrites provider config."""
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider
    provider.url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    provider.model = "qwen3.8-max-preview"
    provider.key = "sk-test"
    provider.api = "responses"
    provider.builtin_tools = ({"type": "web_search"}, {"type": "web_extractor"})

    assert command_loop.api("chat") == "Set provider.api = chat (wire: chat); builtin_tools inactive on chat"
    # The requested API value is applied and the provider configuration is left intact.
    assert provider.api == "chat"
    assert provider.builtin_tools == ({"type": "web_search"}, {"type": "web_extractor"})

    # The next request projects no provider-native tools on the mismatched wire.
    assert ModelClient(command_loop.session).builtin_tools() == []

    # Switching back restores the working Responses configuration without erasing it.
    assert command_loop.api("responses") == "Set provider.api = responses (wire: responses)"
    assert provider.builtin_tools == ({"type": "web_search"}, {"type": "web_extractor"})


def test_api_command_reports_when_no_wire_accepts_the_configured_builtin_tools(tmp_path):
    """DeepSeek has no provider-side tools channel, so the shared config stays inactive."""
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider
    provider.url = "https://api.deepseek.com/v1"
    provider.model = "deepseek-chat"
    provider.key = "sk-test"
    provider.builtin_tools = ({"type": "web_search"},)

    assert command_loop.api("chat") == "Set provider.api = chat (wire: chat); builtin_tools inactive on chat"
    assert provider.builtin_tools == ({"type": "web_search"},)


def test_config_distinguishes_configured_and_active_builtin_tools(tmp_path):
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider
    provider.url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    provider.model = "qwen3.8-max-preview"
    provider.api = "chat"
    provider.builtin_tools = ({"type": "web_search"}, {"type": "web_extractor"})

    inactive = command_loop.config("")
    assert "provider.builtin_tools: web_search, web_extractor" in inactive
    assert "provider.resolved_builtin_tools: inactive on chat: web_search, web_extractor" in inactive

    provider.api = "responses"
    active = command_loop.config("")
    assert "provider.resolved_builtin_tools: active: web_search, web_extractor" in active


def test_api_command_uses_the_same_entry_policy_as_the_request_boundary(tmp_path):
    """A valid wire with an unsupported entry is reported immediately, not only on send."""
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider
    provider.url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    provider.model = "qwen3.8-max-preview"
    provider.key = "sk-test"
    provider.builtin_tools = ({"type": "code_interpreter"},)

    assert command_loop.api("responses") == "Set provider.api = responses (wire: responses); unsupported builtin_tools: code_interpreter"
    with pytest.raises(ModelError):
        ModelClient(command_loop.session).builtin_tools()


# --- /worker pickers and tab completion, mirroring the /provider /model /reason surfaces. ---


def test_worker_command_completion(tmp_path):
    from prompt_toolkit.document import Document

    command_loop = loop(tmp_path)
    command_loop.session.config.providers["alt"] = ProviderConfig(model="m", available_models=("w-a", "w-b"))
    command_loop.session.config.worker_provider = "alt"
    completer = CommandCompleter(
        providers=lambda: tuple(sorted(command_loop.session.config.providers)),
        worker_models=lambda: tuple(
            dict.fromkeys(
                (*command_loop.session.config.providers[command_loop.session.config.worker_provider or command_loop.session.config.active_provider].available_models, "default")
            )
        ),
    )

    sub_texts = [c.text for c in completer.get_completions(Document("/worker "), None)]
    assert set(sub_texts) == {"status", "reset", "on", "off", "provider", "model", "reason"}

    provider_texts = [c.text for c in completer.get_completions(Document("/worker provider "), None)]
    assert set(provider_texts) == {"default", "alt", "off"}

    model_texts = [c.text for c in completer.get_completions(Document("/worker model "), None)]
    assert set(model_texts) == {"w-a", "w-b", "default"}

    reason_texts = [c.text for c in completer.get_completions(Document("/worker reason "), None)]
    assert set(reason_texts) == set(REASONING_CHOICES) | {"default"}


# The no-arg pickers follow the /provider picker pattern: select_choice is stubbed, and the
# selection runs the exact same set path as the typed form (live-apply, frozen-gate note).
def test_worker_provider_picker_sets_and_clears_like_the_typed_form(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["alt"] = ProviderConfig(model="m")
    calls = []
    picks = iter(["alt", "off"])

    def select(title, choices, **kwargs):
        calls.append((title, choices, kwargs))
        return next(picks)

    command_loop.select_choice = select

    first = command_loop.worker_command("provider")
    assert calls[0][0] == "Worker provider"
    assert "off" in calls[0][1]
    assert calls[0][1][-1] == "off"  # the clear entry trails the provider names
    assert first.startswith("Set worker provider = alt")
    assert command_loop.session.config.worker_provider == "alt"

    cleared = command_loop.worker_command("provider")
    assert calls[1][2]["labels"] == {"alt": "alt (current)"}  # the live entry is marked
    assert cleared == "worker provider: off"
    assert command_loop.session.config.worker_provider == ""


def test_worker_model_picker_sets_the_override_without_the_model_chain(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["alt"] = ProviderConfig(model="m-a", available_models=("m-a", "m-b"))
    command_loop.session.config.worker_provider = "alt"
    command_loop.session.config.worker_model = "m-c"
    titles = []
    discovered = []
    picks = iter(["m-b"])

    def select(title, choices, **kwargs):
        titles.append(title)
        assert "default" in choices and "m-c" in choices and "m-a" in choices
        return next(picks)

    command_loop.select_choice = select
    command_loop.remote_models = lambda entry: discovered.append(entry.model) or ("m-remote",)

    result = command_loop.worker_command("model")
    assert titles == ["Worker model"]
    assert discovered == ["m-a"]  # discovery ran against the worker's entry, not the parent's
    assert command_loop.session.config.worker_model == "m-b"
    assert result == "Set worker.model = m-b"


def test_worker_reason_picker_covers_efforts_and_default(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.worker_reasoning = "high"
    picks = iter(["low"])

    def select(title, choices, **kwargs):
        assert set(choices) == set(REASONING_CHOICES) | {"default"}
        assert kwargs["labels"] == {"default": "default - inherit the provider entry's reasoning", "high": "high (current)"}
        return next(picks)

    command_loop.select_choice = select
    result = command_loop.worker_command("reason")
    assert command_loop.session.config.worker_reasoning == "low"
    assert result == "Set worker.reasoning = low"


def test_worker_pickers_return_no_change_on_back(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["alt"] = ProviderConfig(model="m", available_models=("m",))
    command_loop.session.config.worker_provider = "alt"
    command_loop.session.config.worker_model = "m-x"
    command_loop.session.config.worker_reasoning = "high"
    command_loop.select_choice = lambda *_args, **_kwargs: SELECTION_BACK
    assert command_loop.worker_command("provider") == "No change"
    assert command_loop.worker_command("model") == "No change"
    assert command_loop.worker_command("reason") == "No change"
    assert command_loop.session.config.worker_provider == "alt"
    assert command_loop.session.config.worker_model == "m-x"
    assert command_loop.session.config.worker_reasoning == "high"


def test_worker_model_and_reason_pickers_clear_via_default(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.worker_model = "m-x"
    command_loop.session.config.worker_reasoning = "high"
    picks = iter(["default", "default"])
    command_loop.select_choice = lambda *_args, **_kwargs: next(picks)
    assert command_loop.worker_command("model") == "worker model: (inherit)"
    assert command_loop.worker_command("reason") == "worker reasoning: (inherit)"
    assert command_loop.session.config.worker_model == ""
    assert command_loop.session.config.worker_reasoning == ""
