"""worker command ui (split from tests/test_command_ui.py)."""

from types import SimpleNamespace

from test_command_ui import ModalHarness
from tui_harness import loop

import wizolt.cli.commands as commands_mod
from wizolt.base import (
    SELECTION_BACK,
)
from wizolt.cli import CommandCompleter
from wizolt.cli import worker as worker_mod
from wizolt.cli.modals import tool_output_viewer
from wizolt.cli.worker import WorkerFlow, worker_command
from wizolt.config import (
    PROVIDER_API_CHOICES,
    Config,
    ProviderConfig,
)
from wizolt.tools import Tool


def async_callable(fn):
    async def call(*args, **kwargs):
        return fn(*args, **kwargs)

    return call


async def test_worker_command_completion(tmp_path):
    from prompt_toolkit.document import Document

    command_loop = loop(tmp_path)
    command_loop.session.config.providers["alt"] = ProviderConfig(model="m", available_models=("w-a", "w-b"))
    command_loop.session.config.worker_provider = "alt"
    completer = CommandCompleter(
        providers=lambda: tuple(sorted(command_loop.session.config.providers)),
        worker_reasoning_choices=lambda: ("off", "cheap", "deep"),
        worker_models=lambda: tuple(
            dict.fromkeys(
                (
                    *command_loop.session.config.providers[
                        command_loop.session.config.worker_provider or command_loop.session.config.active_provider
                    ].available_models,
                    "default",
                )
            )
        ),
    )

    sub_texts = [c.text for c in completer.get_completions(Document("/worker "), None)]
    assert set(sub_texts) == {"status", "reset", "on", "off", "provider", "model", "reason", "api"}

    provider_texts = [c.text for c in completer.get_completions(Document("/worker provider "), None)]
    assert set(provider_texts) == {"default", "alt", "off"}

    model_texts = [c.text for c in completer.get_completions(Document("/worker model "), None)]
    assert set(model_texts) == {"w-a", "w-b", "default"}

    reason_texts = [c.text for c in completer.get_completions(Document("/worker reason "), None)]
    assert set(reason_texts) == {"off", "cheap", "deep", "default"}

    api_texts = [c.text for c in completer.get_completions(Document("/worker api "), None)]
    assert set(api_texts) == set(PROVIDER_API_CHOICES) | {"default"}


async def test_worker_api_subcommand_sets_clears_and_rejects(tmp_path):
    command_loop = loop(tmp_path)

    assert await worker_command(command_loop, "api responses") == "Set worker.api = responses"
    assert command_loop.session.config.worker_api == "responses"

    assert await worker_command(command_loop, "api default") == "worker api: (inherit)"
    assert command_loop.session.config.worker_api == ""

    assert await worker_command(command_loop, "api oai") == "Usage: /worker api " + "|".join(PROVIDER_API_CHOICES)
    assert command_loop.session.config.worker_api == ""

    assert await worker_command(command_loop, "api chat responses") == "Usage: /worker api [API]"


async def test_worker_api_picker_sets_and_clears_like_the_typed_form(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    picks = iter(["chat", "default"])
    calls = []

    def select(_loop, title, choices, **kwargs):
        calls.append((title, choices, kwargs))
        return next(picks)

    monkeypatch.setattr(worker_mod, "select_choice", async_callable(select))

    assert await worker_command(command_loop, "api") == "Set worker.api = chat"
    assert command_loop.session.config.worker_api == "chat"
    assert calls[0][0] == "Worker api"
    assert set(calls[0][1]) == set(PROVIDER_API_CHOICES) | {"default"}
    assert calls[0][2]["labels"]["default"].startswith("default")

    assert await worker_command(command_loop, "api") == "worker api: (inherit)"
    assert command_loop.session.config.worker_api == ""


async def test_worker_status_line_reports_worker_config(tmp_path):
    command_loop = loop(tmp_path)

    assert "worker: no active session" in await worker_command(command_loop, "")


async def test_run_worker_config_drives_pickers_until_done(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    picks = iter(["provider", "api", "done"])
    calls = []
    driven = []

    def select(_loop, title, choices, **kwargs):
        calls.append((title, choices, kwargs))
        return next(picks)

    monkeypatch.setattr(worker_mod, "select_choice", async_callable(select))
    monkeypatch.setattr(WorkerFlow, "_worker_provider_picker", async_callable(lambda self: driven.append("provider")))
    monkeypatch.setattr(WorkerFlow, "_worker_model_picker", async_callable(lambda self: driven.append("model")))
    monkeypatch.setattr(WorkerFlow, "_worker_reason_picker", async_callable(lambda self: driven.append("effort")))
    monkeypatch.setattr(WorkerFlow, "_worker_api_picker", async_callable(lambda self: driven.append("api")))

    await WorkerFlow(command_loop).run_worker_config()

    assert driven == ["provider", "api"]
    assert calls[0][0] == "Worker config"
    assert calls[0][1] == ("provider", "model", "effort", "api", "done")
    assert calls[0][2]["current"] == "done"  # Enter with nothing selected exits
    assert calls[0][2]["labels"]["provider"].startswith("provider:")
    assert calls[0][2]["labels"]["model"].startswith("model:")

    # Esc (SELECTION_BACK) and a non-interactive select (None) both exit without driving pickers.
    for value in (SELECTION_BACK, None):
        monkeypatch.setattr(worker_mod, "select_choice", async_callable(lambda *a, value=value, **k: value))
        await WorkerFlow(command_loop).run_worker_config()
    assert driven == ["provider", "api"]


async def test_worker_provider_picker_sets_and_clears_like_the_typed_form(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["alt"] = ProviderConfig(model="m")
    calls = []
    # First picker: provider "alt", then the cascade's model/reason pickers (default keeps the
    # entry's values), then a second /worker provider that clears with "off".
    picks = iter(["alt", "default", "default", "off"])

    def select(_loop, title, choices, **kwargs):
        calls.append((title, choices, kwargs))
        return next(picks)

    monkeypatch.setattr(worker_mod, "select_choice", async_callable(select))

    first = await worker_command(command_loop, "provider")
    assert calls[0][0] == "Worker provider"
    assert "off" in calls[0][1]
    assert calls[0][1][-1] == "off"  # the clear entry trails the provider names
    assert [call[0] for call in calls] == ["Worker provider", "Worker model", "Worker reasoning"]
    assert first.startswith("Set worker provider = alt")
    assert "worker model: (inherit)" in first
    assert "worker reasoning: (inherit)" in first
    assert command_loop.session.config.worker_provider == "alt"
    assert command_loop.session.config.worker_model == ""
    assert command_loop.session.config.worker_reasoning == ""

    cleared = await worker_command(command_loop, "provider")
    assert calls[3][0] == "Worker provider"
    assert calls[3][2]["labels"] == {"alt": "alt (current)"}  # the live entry is marked
    assert cleared == "worker provider: off"  # picking "off" clears without cascading
    assert command_loop.session.config.worker_provider == ""


async def test_worker_model_picker_sets_the_override_without_the_model_chain(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["alt"] = ProviderConfig(model="m-a", available_models=("m-a", "m-b"))
    command_loop.session.config.worker_provider = "alt"
    command_loop.session.config.worker_model = "m-c"
    titles = []
    discovered = []
    picks = iter(["m-b"])

    def select(_loop, title, choices, **kwargs):
        titles.append(title)
        assert "default" in choices and "m-c" in choices and "m-a" in choices
        return next(picks)

    monkeypatch.setattr(worker_mod, "select_choice", async_callable(select))
    monkeypatch.setattr(commands_mod, "remote_models", async_callable(lambda _loop, entry: discovered.append(entry.model) or ("m-remote",)))

    result = await worker_command(command_loop, "model")
    assert titles == ["Worker model"]
    assert discovered == ["m-a"]  # discovery ran against the worker's entry, not the parent's
    assert command_loop.session.config.worker_model == "m-b"
    assert result == "Set worker.model = m-b"


async def test_worker_reason_picker_covers_efforts_and_default(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.worker_reasoning = "high"
    picks = iter(["low"])

    def select(_loop, title, choices, **kwargs):
        assert set(choices) == {"off", *command_loop.session.policy.effort_order, "default"}
        assert kwargs["labels"] == {"default": "default - inherit the provider entry's reasoning", "high": "high (current)"}
        return next(picks)

    monkeypatch.setattr(worker_mod, "select_choice", async_callable(select))
    result = await worker_command(command_loop, "reason")
    assert command_loop.session.config.worker_reasoning == "low"
    assert result == "Set worker.reasoning = low"


async def test_worker_reason_picker_uses_the_effective_models_declared_scale(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["custom"] = ProviderConfig.from_dict(
        {
            "model": "main",
            "models": {"worker-*": {"reasoning": ["cheap", "deep"]}},
        }
    )
    command_loop.session.config.worker_provider = "custom"
    command_loop.session.config.worker_model = "worker-small"

    def select(_loop, _title, choices, **_kwargs):
        assert choices == ("off", "cheap", "deep", "default")
        return "deep"

    monkeypatch.setattr(worker_mod, "select_choice", async_callable(select))

    assert await worker_command(command_loop, "reason") == "Set worker.reasoning = deep"


async def test_worker_pickers_return_no_change_on_back(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["alt"] = ProviderConfig(model="m", available_models=("m",))
    command_loop.session.config.worker_provider = "alt"
    command_loop.session.config.worker_model = "m-x"
    command_loop.session.config.worker_reasoning = "high"
    monkeypatch.setattr(worker_mod, "select_choice", async_callable(lambda *_args, **_kwargs: SELECTION_BACK))
    assert await worker_command(command_loop, "provider") == "No change"
    assert await worker_command(command_loop, "model") == "No change"
    assert await worker_command(command_loop, "reason") == "No change"
    assert command_loop.session.config.worker_provider == "alt"
    assert command_loop.session.config.worker_model == "m-x"
    assert command_loop.session.config.worker_reasoning == "high"


async def test_worker_model_and_reason_pickers_clear_via_default(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.worker_model = "m-x"
    command_loop.session.config.worker_reasoning = "high"
    picks = iter(["default", "default"])
    monkeypatch.setattr(worker_mod, "select_choice", async_callable(lambda *_args, **_kwargs: next(picks)))
    assert await worker_command(command_loop, "model") == "worker model: (inherit)"
    assert await worker_command(command_loop, "reason") == "worker reasoning: (inherit)"
    assert command_loop.session.config.worker_model == ""
    assert command_loop.session.config.worker_reasoning == ""


async def test_worker_provider_picker_cascades_into_model_and_reasoning(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["fast"] = ProviderConfig(model="fast-model", available_models=("fast-model", "fast-mini"))
    worker = SimpleNamespace(config=Config())  # a live worker session, like session.worker after a spawn
    command_loop.session.worker = worker
    titles = []
    discovered = []
    picks = iter(["fast", "fast-mini", "high"])

    def select(_loop, title, choices, **kwargs):
        titles.append(title)
        return next(picks)

    monkeypatch.setattr(worker_mod, "select_choice", async_callable(select))
    monkeypatch.setattr(commands_mod, "remote_models", async_callable(lambda _loop, entry: discovered.append(entry.model) or ("remote-mini",)))

    result = await worker_command(command_loop, "provider")

    assert titles == ["Worker provider", "Worker model", "Worker reasoning"]
    assert discovered == ["fast-model"]  # discovery ran against the newly selected entry
    assert command_loop.session.config.worker_provider == "fast"
    assert command_loop.session.config.worker_model == "fast-mini"
    assert command_loop.session.config.worker_reasoning == "high"
    assert "Set worker provider = fast" in result
    assert "Set worker.model = fast-mini" in result
    assert "Set worker.reasoning = high" in result
    # the live worker's detached entry reflects all three stages
    assert worker.config.active_provider == "fast"
    assert worker.config.providers["fast"].model == "fast-mini"
    assert worker.config.providers["fast"].reasoning == "high"


async def test_worker_provider_cascade_aborts_at_model_stage_keeping_earlier_stages(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["fast"] = ProviderConfig(model="fast-model", available_models=("fast-model",))
    command_loop.session.config.worker_model = "m-x"
    command_loop.session.config.worker_reasoning = "high"
    picks = iter(["fast", SELECTION_BACK])
    monkeypatch.setattr(worker_mod, "select_choice", async_callable(lambda *_args, **_kwargs: next(picks)))
    monkeypatch.setattr(commands_mod, "remote_models", async_callable(lambda _loop, _entry: ()))

    result = await worker_command(command_loop, "provider")

    assert command_loop.session.config.worker_provider == "fast"  # the provider stage landed
    assert command_loop.session.config.worker_model == "m-x"  # model/reasoning untouched
    assert command_loop.session.config.worker_reasoning == "high"
    assert "Set worker provider = fast" in result
    assert "worker model: unchanged" in result
    assert "worker reasoning" not in result


async def test_worker_provider_cascade_aborts_at_reason_stage_keeping_model(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    command_loop.session.config.providers["fast"] = ProviderConfig(model="fast-model", available_models=("fast-model",))
    command_loop.session.config.worker_reasoning = "high"
    picks = iter(["fast", "fast-model", None])  # None = the picker was dismissed
    monkeypatch.setattr(worker_mod, "select_choice", async_callable(lambda *_args, **_kwargs: next(picks)))
    monkeypatch.setattr(commands_mod, "remote_models", async_callable(lambda _loop, _entry: ()))

    result = await worker_command(command_loop, "provider")

    assert command_loop.session.config.worker_provider == "fast"
    assert command_loop.session.config.worker_model == "fast-model"
    assert command_loop.session.config.worker_reasoning == "high"  # untouched by the dismissal
    assert "Set worker.model = fast-model" in result
    assert "worker reasoning: unchanged" in result


async def test_worker_provider_typed_form_does_not_cascade(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.session.config.providers["alt"] = ProviderConfig(model="m")
    monkeypatch.setattr(worker_mod, "select_choice", async_callable(lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("the typed form opens no picker"))))

    result = await worker_command(command_loop, "provider alt")

    assert result.startswith("Set worker provider = alt")
    assert command_loop.session.config.worker_provider == "alt"
    assert command_loop.session.config.worker_model == ""
    assert command_loop.session.config.worker_reasoning == ""


async def test_tool_output_viewer_opens_a_stored_script_in_the_scrolling_viewer(tmp_path):
    """Ctrl-O is the only door to a script under yolo, where no prompt ever offered `v`: the entry
    hands the stored source to the same read-only viewer the confirm-time key opens."""
    command_loop = loop(tmp_path)
    code = "\n".join(f"x{index} = {index}" for index in range(30))
    envelope = "ToolScript ok\ncalls: 2 [tr.1-2]\nstdout:\ncounted 30 rows"
    command_loop.session.store_tool_result("ToolScript", [{"action": "call", "code": code}], envelope)
    modal = ModalHarness(["enter", "G"])  # open the entry, then scroll the viewer to the bottom
    command_loop.tui = modal

    tool_output_viewer(command_loop)

    listing = "".join(value for _, value in modal.frames[0])
    assert "ToolScript call 30 lines" in listing
    frames = ["".join(value for _, value in frame) for frame in modal.frames]
    viewer = [frame for frame in frames if "Script · tr.1 · read-only" in frame]
    assert viewer, "the entry hands off to the read-only script viewer"
    assert " 1  x0 = 0" in viewer[0]  # numbered, so a traceback's line N is findable
    assert "x29 = 29" in viewer[-1]  # the whole script is reachable, not just the excerpt
    assert "calls  2" in viewer[0]
    # A script is a question and its printed output is the answer, so the entry carries both.
    assert "── result " in viewer[-1]
    assert "counted 30 rows" in viewer[-1]


async def test_tool_output_viewer_skips_a_describe_with_no_script(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.session.store_tool_result("ToolScript", [{"action": "describe", "tools": ["Read"]}], "Read\njson:    no")
    modal = ModalHarness([])
    command_loop.tui = modal

    tool_output_viewer(command_loop)

    assert modal.frames == []


async def test_tool_output_viewer_shows_a_failed_script_with_its_traceback(tmp_path):
    """The log line clips a failure to one row. Here the whole traceback sits under the numbered
    source, so `File "<toolscript>", line N` resolves against the line it names."""
    command_loop = loop(tmp_path)
    code = "rows = []\nprint(rows[2])\n"
    envelope = 'ToolScript failed\ncalls: 0\nerror:\nTraceback (most recent call last):\n  File "<toolscript>", line 2, in <module>\nIndexError: list index out of range'
    command_loop.session.store_tool_result("ToolScript", [{"action": "call", "code": code}], envelope)
    modal = ModalHarness(["enter"])
    command_loop.tui = modal

    tool_output_viewer(command_loop)

    viewer = "".join(value for _, value in modal.frames[-1])
    assert " 2  print(rows[2])" in viewer
    assert "IndexError: list index out of range" in viewer
    assert 'File "<toolscript>", line 2' in viewer


async def test_tool_output_viewer_shows_the_whole_command_not_the_clipped_log_line(tmp_path):
    """The transcript row collapses and clips a command at 200 characters. A viewer opened to see
    what was run has to show what was run."""
    command_loop = loop(tmp_path)
    command = "rg --json " + " ".join(f"--glob '!vendor/{index}/**'" for index in range(30)) + " pattern"
    command_loop.session.store_tool_result("Bash", [command], Tool.process_result("BashToolResult", 0, "hit", ""))
    modal = ModalHarness(["enter"])
    command_loop.tui = modal

    tool_output_viewer(command_loop)

    viewer = next(frame for frame in ("".join(v for _, v in f) for f in modal.frames) if "read-only" in frame)
    assert "vendor/29" in viewer  # the tail of the command survived
    assert "..." not in viewer.split("── result")[0]
