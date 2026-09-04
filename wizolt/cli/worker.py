"""The /worker command: inspect or reconfigure the worker session.

Its multi-stage configuration flow (provider -> model -> reason -> api) is a small state machine
rather than one handler, which is why it lives beside the command implementations instead of among
them. Depends on commands.py for remote model discovery; nothing there imports this module back,
so the registry in wizolt/cli/__init__.py wires `/worker` straight to `worker_command`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from wizolt.base import SELECTION_BACK, SESSION_EVENT_KEY
from wizolt.cli import commands
from wizolt.cli.modals import select_choice
from wizolt.config import PROVIDER_API_CHOICES
from wizolt.tools.delegate import DelegateTool, refresh_worker_entry, worker_provider_config

if TYPE_CHECKING:
    from wizolt.cli import CommandLoop

WORKER_SUBCOMMANDS = ("status", "reset", "on", "off", "provider", "model", "reason", "api")


class WorkerFlow:
    """The multi-stage /worker configuration flow (provider -> model -> reason -> api).

    All stages go through the shared choice selector; backing out of any stage keeps the stages
    already set and reports what landed. Externally only `worker_command` is exposed.
    """

    def __init__(self, loop: CommandLoop) -> None:
        self.loop = loop

    async def worker_command(self, args: str) -> str:
        parts = args.split()
        subcommand = parts[0].lower() if parts else ""
        rest = parts[1:]
        if subcommand == "reset" and not rest:
            result = await DelegateTool(self.loop.session, [{"action": "reset"}])._reset()
            if 'action="reset"' not in result:
                return result
            # The parent model does not know the user reset the worker; without this event the next
            # delegation would write "continue where you left off" against a clean context. Tail
            # append, ages with compaction, render-hidden, never filtered from the model history.
            self.loop.session.messages.append(
                {
                    "role": "user",
                    "content": "[Worker context was reset by the user. The next delegation starts from scratch.]",
                    SESSION_EVENT_KEY: "worker_reset",
                }
            )
            await self.loop.session.save_snapshot()
            if 'alive="false"' in result:
                return "[worker] reset · no worker session to reset."
            return "[worker] reset · worker context cleared; file changes and merged diffs kept. The next delegation starts from scratch."
        if subcommand == "on" and not rest:
            self.loop.session.settings.worker = True
            return "worker: on (the tool block changes, so the prompt-cache scope is recompiled once)"
        if subcommand == "off" and not rest:
            self.loop.session.settings.worker = False
            return "worker: off (the worker's context stays on disk; /worker on resumes it)"
        if subcommand in {"", "status"} and not rest:
            return self._worker_status()
        if subcommand == "provider":
            if len(rest) > 1:
                return "Usage: /worker provider [NAME]"
            if not rest:
                return await self._worker_provider_picker()
            return self._worker_set_provider(rest[0])
        if subcommand == "model":
            if len(rest) > 1:
                return "Usage: /worker model [MODEL]"
            if not rest:
                return await self._worker_model_picker()
            return self._worker_set_model(rest[0])
        if subcommand == "reason":
            if len(rest) > 1:
                return "Usage: /worker reason [EFFORT]"
            if not rest:
                return await self._worker_reason_picker()
            return self._worker_set_reasoning(rest[0])
        if subcommand == "api":
            if len(rest) > 1:
                return "Usage: /worker api [API]"
            if not rest:
                return await self._worker_api_picker()
            return self._worker_set_api(rest[0])
        return "Usage: /worker [" + "|".join(WORKER_SUBCOMMANDS) + "]"

    def _worker_status(self) -> str:
        """Readable /worker status for the human; the model-facing envelope stays in DelegateTool."""
        worker = self.loop.session.worker
        if worker is None:
            return "worker: no active session\nworker provider: " + (self.loop.session.config.worker_provider or "(off)")
        usage = worker.usage
        percent = min(100, usage.last_prompt_tokens * 100 // usage.last_prompt_budget) if usage.last_prompt_budget else worker.state.context_percent
        provider = worker.config.provider
        state = "delegating" if worker._active_turn_messages else "idle"
        return "\n".join(
            [
                f"worker: {worker.config.active_provider}/{provider.model or '(no model)'}",
                "worker reasoning: " + provider.reasoning,
                "worker state: " + state,
                "worker rounds: " + str(worker.state.round_count),
                "worker context: " + str(percent) + "%",
            ]
        )

    async def _worker_provider_picker(self) -> str:
        summary = (
            "worker provider: "
            + (self.loop.session.config.worker_provider or "(off)")
            + "\nproviders: "
            + ", ".join(sorted(self.loop.session.config.providers))
        )
        choices = tuple(sorted(self.loop.session.config.providers))
        if "off" not in choices:
            choices = (*choices, "off")
        current = self.loop.session.config.worker_provider
        choice = await select_choice(self.loop, "Worker provider", choices, labels={current: current + " (current)"} if current else {}, current=current)
        if not isinstance(choice, str):
            return "No change" if choice is SELECTION_BACK else summary
        provider_result = self._worker_set_provider(choice)
        if self.loop.session.config.worker_provider != choice:
            # Picking "off" cleared the entry (or the set failed): there is no newly selected
            # provider entry to pick a model for, so the cascade stops after the provider set.
            return provider_result
        # One setup flow, like /provider: worker provider -> worker model -> worker reasoning.
        # Backing out of any stage keeps the stages already set and reports what landed.
        lines = [provider_result]
        set_ok, model_result = await self._worker_model_stage()
        if not set_ok:
            lines.append("worker model: unchanged")
            return "\n".join(lines)
        lines.append(model_result)
        set_ok, reason_result = await self._worker_reason_stage()
        if not set_ok:
            lines.append("worker reasoning: unchanged")
            return "\n".join(lines)
        lines.append(reason_result)
        return "\n".join(lines)

    def _worker_set_provider(self, name: str) -> str:
        if name == "off" and "off" not in self.loop.session.config.providers:
            # "off" names the clearing action unless a provider entry is literally named "off"
            # (existence in config.providers wins). The Delegate gate is frozen per session, so
            # this only clears the next spawn's provider; the live worker keeps running on its
            # current provider and the tool block never flips mid-session.
            self.loop.session.config.worker_provider = ""
            return "worker provider: off"
        if name not in self.loop.session.config.providers:
            return "Unknown provider: " + name
        self.loop.session.config.worker_provider = name
        refresh_worker_entry(self.loop.session.config, self.loop.session.worker, name)
        result = "Set worker provider = " + name
        if not self.loop.session.worker_tool_enabled:
            # Delegation was off at session start: the frozen gate keeps the tool block off no
            # matter what the live config says, so the change only counts after a restart.
            result += " (delegation is off this session; takes effect after a restart)"
        return result

    async def _worker_simple_field(
        self,
        *,
        title: str,
        label: str,
        choices: tuple[str, ...],
        current: str,
        labels: dict[str, str],
        apply: Callable[[str], str],
    ) -> tuple[bool, str]:
        """Pick one value through the shared selector and apply it; returns (set, message) so both
        the standalone /worker pickers and the /worker provider cascade can tell a set from an
        abort. Shared by worker model/reasoning/api: same shape, no cascade, and each `apply`
        writes the config and refreshes a live worker itself."""
        choice = await select_choice(self.loop, title, choices, labels=labels, current=current)
        if not isinstance(choice, str):
            return False, ("No change" if choice is SELECTION_BACK else (f"{label}: " + (current or "(inherit)")))
        return True, apply(choice)

    async def _worker_model_picker(self) -> str:
        """Standalone /worker model picker: one selection, no cascade."""
        return (await self._worker_model_stage())[1]

    async def _worker_model_stage(self) -> tuple[bool, str]:
        """Pick a worker model override; returns (set, message). Shared by /worker model and the
        /worker provider cascade so the cascade can tell a set from an abort."""
        entry = self.loop.session.config.providers[self.loop.session.config.worker_provider or self.loop.session.config.active_provider]
        configured = tuple(dict.fromkeys(entry.available_models))
        tui = self.loop.tui
        # Remote discovery on a freshly set provider entry is the slow step of the cascade; show
        # the same dispatch note /model does so the pause does not read as a hang.
        show_loading = tui is not None and bool(entry.url and entry.key)
        if show_loading and tui is not None:
            tui.set_dispatching("Loading models...")
        try:
            remote = tuple(model for model in await commands.remote_models(self.loop, entry) if model not in configured)
        finally:
            if show_loading and tui is not None:
                tui.set_dispatching()
        override = self.loop.session.config.worker_model
        choices = [*configured, *remote]
        if override and override not in choices:
            choices.append(override)
        choices.append("default")
        choice_values = tuple(dict.fromkeys(choices))
        labels = {override: override + " (current)"} if override in choice_values else {}
        labels["default"] = "default - inherit the provider entry's model"
        return await self._worker_simple_field(
            title="Worker model", label="worker model", choices=choice_values, current=override, labels=labels, apply=self._worker_set_model
        )

    def _worker_set_model(self, value: str) -> str:
        if value != "default":
            self.loop.session.config.worker_model = value
        else:
            self.loop.session.config.worker_model = ""
        refresh_worker_entry(self.loop.session.config, self.loop.session.worker)
        if value == "default":
            return "worker model: (inherit)"
        return "Set worker.model = " + value

    async def _worker_reason_picker(self) -> str:
        """Standalone /worker reason picker: one selection, no cascade."""
        return (await self._worker_reason_stage())[1]

    async def _worker_reason_stage(self) -> tuple[bool, str]:
        """Pick a worker reasoning effort; returns (set, message). Shared by /worker reason and
        the /worker provider cascade."""
        current = self.loop.session.config.worker_reasoning
        choices = (*self._reasoning_choices(), "default")
        labels = {"default": "default - inherit the provider entry's reasoning"}
        if current:
            labels[current] = current + " (current)"
        return await self._worker_simple_field(
            title="Worker reasoning", label="worker reasoning", choices=choices, current=current, labels=labels, apply=self._worker_set_reasoning
        )

    def _worker_set_reasoning(self, value: str) -> str:
        if value != "default":
            # "off" is a valid effort, never the clearing word; only "default" clears.
            choices = self._reasoning_choices()
            if value not in choices:
                return "Usage: /worker reason " + "|".join(choices)
            self.loop.session.config.worker_reasoning = value
        else:
            self.loop.session.config.worker_reasoning = ""
        refresh_worker_entry(self.loop.session.config, self.loop.session.worker)
        if value == "default":
            return "worker reasoning: (inherit)"
        return "Set worker.reasoning = " + value

    def _reasoning_choices(self) -> tuple[str, ...]:
        config = self.loop.session.config
        provider_name = config.worker_provider or config.active_provider
        return self.loop.session.policy.reasoning_choices(worker_provider_config(config, provider_name))

    async def _worker_api_picker(self) -> str:
        """Standalone /worker api picker: one selection, no cascade."""
        current = self.loop.session.config.worker_api
        choices = (*PROVIDER_API_CHOICES, "default")
        labels = {"default": "default - inherit the provider entry's api"}
        if current:
            labels[current] = current + " (current)"
        return (
            await self._worker_simple_field(title="Worker api", label="worker api", choices=choices, current=current, labels=labels, apply=self._worker_set_api)
        )[1]

    def _worker_set_api(self, value: str) -> str:
        if value != "default":
            if value not in PROVIDER_API_CHOICES:
                return "Usage: /worker api " + "|".join(PROVIDER_API_CHOICES)
            self.loop.session.config.worker_api = value
        else:
            self.loop.session.config.worker_api = ""
        refresh_worker_entry(self.loop.session.config, self.loop.session.worker)
        if value == "default":
            return "worker api: (inherit)"
        return "Set worker.api = " + value

    async def run_worker_config(self) -> None:
        """The Delegate confirm-time `c` loop: pick which worker knob to adjust with the shared
        choice selector (each field labeled with its current value, done preselected), then drive
        the corresponding /worker picker -- which writes the config and refreshes a live worker
        itself. done or Esc returns to the confirmation prompt; non-interactive runs return at
        once (select_choice yields nothing)."""
        while True:
            config = self.loop.session.config
            provider_name = config.worker_provider or config.active_provider
            entry = config.providers[provider_name]
            provider_value = config.worker_provider or f"(inherit) {provider_name}"
            model_value = config.worker_model or f"(inherit) {entry.model or '(no model)'}"
            effort_value = config.worker_reasoning or f"(inherit) {entry.reasoning}"
            api_value = config.worker_api or f"(inherit) {entry.api}"
            labels = {
                "provider": f"provider: {provider_value}",
                "model": f"model: {model_value}",
                "effort": f"effort: {effort_value}",
                "api": f"api: {api_value}",
                "done": "done - return to the confirmation prompt",
            }
            choice = await select_choice(self.loop, "Worker config", ("provider", "model", "effort", "api", "done"), labels=labels, current="done")
            if choice == "provider":
                await self._worker_provider_picker()
            elif choice == "model":
                await self._worker_model_picker()
            elif choice == "effort":
                await self._worker_reason_picker()
            elif choice == "api":
                await self._worker_api_picker()
            else:
                return


async def worker_command(loop: CommandLoop, args: str) -> str:
    """Dispatch a /worker subcommand through a fresh WorkerFlow."""
    return await WorkerFlow(loop).worker_command(args)
