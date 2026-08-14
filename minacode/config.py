"""minacode configuration: provider entries, runtime settings, and the config file."""

from __future__ import annotations

import os
import platform
import shutil
import sys
import tomllib
from dataclasses import dataclass, field, replace
from typing import ClassVar
from urllib.parse import urlparse

from minacode.base import ConfigError, Json, builtin_function_names
from minacode.model_catalog import REASONING_LEVELS
from minacode.provider_compat import (
    COMPATIBILITY_PROFILES,
    CompatibilityProfile,
    ResolvedProvider,
    compatibility_for_host,
)

DEFAULT_MAX_CONTEXT_TOKENS = 256 * 1024
PROVIDER_API_CHOICES = ("auto", "chat", "responses", "anthropic")
IMAGE_INPUT_CHOICES = ("auto", "on", "off")
REASONING_CHOICES = ("off", *REASONING_LEVELS)
CHAT_REASONING_CHOICES = (
    "auto",
    "off",
    "reasoning",
    "reasoning_effort",
    "thinking",
    "thinking_toggle",
    "thinking_effort",
    "enable_thinking",
    "mandatory_thinking",
)


# Output room kept out of the input budget for one request's answer. It is a planning reserve, not a
# wire parameter, so it stays fixed whether or not the user configured a cap (see output_token_budget).
DEFAULT_OUTPUT_RESERVE_TOKENS = 16_384
# Conservative `max_tokens` sent on the Anthropic wire when the user left it unset: Anthropic requires
# the parameter, and 8K covers every current model (Claude 3.5 Haiku's ceiling is 8K). Legacy Claude 3
# models that cap lower are retired from the API.
ANTHROPIC_DEFAULT_MAX_TOKENS = 8_192
# Unset: Chat and Responses omit the cap and let the provider apply its own default; the Anthropic
# wire substitutes ANTHROPIC_DEFAULT_MAX_TOKENS because the parameter is mandatory there.
DEFAULT_MAX_TOKENS = 0
MIN_CONTEXT_SAFETY_TOKENS = 4_096


def request_budget_for(max_context_tokens: int, output_budget: int) -> int:
    """The input budget one request is measured against: the context limit less the output reserve and
    a safety margin. Pure, so ContextManager and the usage recorder share the same denominator."""
    safety = max(MIN_CONTEXT_SAFETY_TOKENS, (max_context_tokens + 49) // 50)
    return max(1, max_context_tokens - output_budget - safety)


@dataclass
class SystemInfo:
    # fmt: off
    COMMANDS: ClassVar[tuple[str, ...]] = (
        "bash", "git", "rg", "sed", "grep", "find", "awk", "python3", "jq", "xargs", "cat", "head", "tail", "wc",
        "sort", "uniq", "make", "cmake", "gcc", "g++", "clang", "clang++", "node", "npm", "uv", "pytest",
    )
    # fmt: on

    AGENTS_MD_FILES: ClassVar[tuple[str, ...]] = ("AGENTS.md", "CLAUDE.md")

    cwd: str
    os: str
    arch: str
    commands: tuple[str, ...]
    agents_md: str = ""  # loaded project-instructions text; "" when no candidate file was found
    agents_md_source: str = ""  # the file it came from, e.g. "AGENTS.md" or "CLAUDE.md"; "" when none

    @classmethod
    def load_agents_md(cls, cwd: str) -> tuple[str, str]:
        """Read the first existing candidate file under cwd; return (content, source), or ("", "").

        No upward traversal, no merging. UTF-8 decoded; OSError/UnicodeDecodeError return ("", "")."""
        for name in cls.AGENTS_MD_FILES:
            try:
                with open(os.path.join(cwd, name), encoding="utf-8") as file:
                    return file.read(), name
            except (OSError, UnicodeDecodeError):
                continue
        return "", ""

    @classmethod
    def detect(cls, cwd: str) -> SystemInfo:
        agents_md, agents_md_source = cls.load_agents_md(cwd)
        return cls(
            cwd=cwd,
            os=platform.system() or sys.platform,
            arch=platform.machine() or "unknown",
            commands=tuple(name for name in cls.COMMANDS if shutil.which(name)),
            agents_md=agents_md,
            agents_md_source=agents_md_source,
        )


@dataclass
class ProviderConfig:
    COMPATIBILITY: ClassVar[dict[str, CompatibilityProfile]] = COMPATIBILITY_PROFILES

    url: str = ""
    key: str = ""
    model: str = ""
    api: str = "auto"
    stream: bool = True
    image_input: str = "auto"
    prompt_cache_key: str = "auto"
    available_models: tuple[str, ...] = ()
    temperature: float | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    # How much of *this* entry's model window to use; 0 inherits runtime.max_context_tokens. Entries
    # are effectively per-model, so a 1M-window model and a 128K one no longer have to share one
    # number -- see context_token_limit.
    max_context_tokens: int = 0
    strict_tools: bool = False
    reasoning: str = "medium"
    chat_reasoning: str = "auto"
    timeout: int = 120
    response_timeout: int = 600
    extra_body: Json = field(default_factory=dict)
    builtin_tools: tuple[Json, ...] = ()

    @classmethod
    def from_dict(cls, data: Json) -> ProviderConfig:
        api = Config.str(data, "api", "auto")
        image_input = Config.str(data, "image_input", "auto")
        prompt_cache_key = cls.clean_prompt_cache_key(Config.str(data, "prompt_cache_key", "auto"))
        reasoning = Config.str(data, "reasoning", "medium")
        chat_reasoning = Config.str(data, "chat_reasoning", "auto")
        for key, value, choices in (
            ("api", api, PROVIDER_API_CHOICES),
            ("image_input", image_input, IMAGE_INPUT_CHOICES),
            ("reasoning", reasoning, REASONING_CHOICES),
            ("chat_reasoning", chat_reasoning, CHAT_REASONING_CHOICES),
        ):
            if value not in choices:
                raise ConfigError("provider." + key + " must be one of " + ", ".join(choices))
        return cls(
            url=Config.str(data, "url"),
            key=Config.str(data, "key"),
            model=Config.str(data, "model"),
            api=api,
            stream=Config.bool(data, "stream", True),
            image_input=image_input,
            prompt_cache_key=prompt_cache_key,
            available_models=Config.str_tuple(data, "available_models"),
            temperature=Config.float(data, "temperature", None),
            max_tokens=max(0, Config.int(data, "max_tokens", DEFAULT_MAX_TOKENS)),
            max_context_tokens=max(0, Config.int(data, "max_context_tokens", 0)),
            strict_tools=Config.bool(data, "strict_tools", False),
            reasoning=reasoning,
            chat_reasoning=chat_reasoning,
            timeout=Config.int(data, "timeout", 120),
            response_timeout=max(0, Config.int(data, "response_timeout", 600)),
            extra_body=Config.table(data, "extra_body"),
            builtin_tools=Config.table_tuple(data, "builtin_tools"),
        )

    def builtin_function_names(self) -> tuple[str, ...]:
        """Declared builtin functions, which the runner answers instead of rejecting as unknown.
        Evidence: https://platform.kimi.ai/docs/guide/use-web-search"""
        return builtin_function_names(self.builtin_tools)

    def resolve(self) -> ResolvedProvider:
        """Fold explicit configuration and documented compatibility into one request policy."""

        url = self.url.rstrip("/").removesuffix("/chat/completions").removesuffix("/responses").removesuffix("/messages")
        host = (urlparse(url).hostname or "").lower()
        profile = compatibility_for_host(host, self.COMPATIBILITY)
        model = self.model.lower()

        api = self.api
        if api == "auto":
            path = urlparse(self.url.rstrip("/")).path
            suffix_api = next(
                (value for suffix, value in (("/responses", "responses"), ("/messages", "anthropic"), ("/chat/completions", "chat")) if path.endswith(suffix)),
                None,
            )
            api = suffix_api or profile.rule_value(profile.api_rules, model) or "chat"

        chat_reasoning = self.chat_reasoning
        if chat_reasoning == "auto":
            chat_reasoning = profile.rule_value(profile.chat_reasoning_rules, model) or profile.chat_reasoning or "off"

        if self.reasoning == "off":
            reasoning_effort = profile.rule_value(profile.reasoning_effort_off_rules, model)
            if api == "responses":
                reasoning_effort = profile.rule_value(profile.responses_reasoning_effort_off_rules, model) or reasoning_effort
        else:
            effort = self.reasoning_effort()
            reasoning_effort = profile.reasoning_effort_value(model, effort)

        suppress_temperature = profile.suppress_temperature or any(model.startswith(prefix) for prefix in profile.suppress_temperature_models)
        if not suppress_temperature:
            reasoning_enabled = self.reasoning != "off"
            suppress_temperature = reasoning_enabled and chat_reasoning in ("thinking", "enable_thinking")

        strict_tools_active = self.strict_tools and profile.strict_tools and api in ("chat", "responses")
        if strict_tools_active and profile.strict_beta and not url.endswith("/beta"):
            url += "/beta"

        return ResolvedProvider(
            api=api,
            base_url=url,
            host=host,
            chat_reasoning=chat_reasoning,
            chat_reasoning_history=profile.rule_value(profile.chat_reasoning_history_rules, model) or profile.chat_reasoning_history,
            reasoning_effort=reasoning_effort,
            responses_reasoning=profile.responses_reasoning_models is None or any(model.startswith(prefix) for prefix in profile.responses_reasoning_models),
            suppress_temperature=suppress_temperature,
            prompt_cache_key=profile.prompt_cache_key,
            strict_tools_active=strict_tools_active,
            builtin_tools_by_wire=profile.builtin_tools_by_wire,
        )

    def reasoning_effort(self) -> str:
        return self.reasoning if self.reasoning in REASONING_LEVELS else "medium"

    def output_token_budget(self) -> int:
        return self.max_tokens or DEFAULT_OUTPUT_RESERVE_TOKENS

    def context_token_limit(self, fallback: int) -> int:
        """How much context this entry may use: its own `max_context_tokens`, else the runtime
        default. Resolved per call, never cached, so `/provider` moves the budget with the entry —
        and so a worker on a small model stops borrowing the parent model's window."""
        return self.max_context_tokens or fallback

    def anthropic_output_cap(self) -> int:
        """The `max_tokens` sent on the Anthropic wire: the configured cap, or a conservative default
        (Anthropic requires the parameter, unlike Chat and Responses)."""
        return self.max_tokens or ANTHROPIC_DEFAULT_MAX_TOKENS

    @staticmethod
    def clean_prompt_cache_key(value: str) -> str:
        value = value.strip()
        if not value:
            return "auto"
        lower = value.lower()
        if lower in {"auto", "off"}:
            return lower
        if len(value) > 64 or any(char.isspace() for char in value):
            raise ConfigError("provider.prompt_cache_key must be auto, off, or a stable key up to 64 chars without whitespace")
        return value


@dataclass
class RuntimeSettings:
    shell_timeout: int = 60
    # Bash foreground wait budget: if the command hasn't exited within this many seconds the running
    # process is promoted to a background job (see BashTool.stream_process) and control returns to
    # the model with a partial-output payload. Set to 0 to disable promotion (fall back to killing
    # on shell_timeout).
    bash_wait_timeout: int = 10
    max_steps: int = 400
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    session_retention_days: int = 7
    # Max read-only tool calls from one model batch to execute concurrently; 1 disables parallelism.
    max_parallel_tools: int = 4
    yolo: bool = False
    quick_hints: bool = True
    worker: bool = False  # register the Delegate tool (see [worker] in ConfigFile.DEFAULT_TEXT)
    theme: str = "auto"
    language: str = "auto"  # forced reply language; "auto" injects nothing (see /language)
    agents_md: bool = True  # inject the project's AGENTS.md (or CLAUDE.md fallback) into every request

    @classmethod
    def from_dict(cls, data: Json, *, yolo: bool = False, theme: str = "") -> RuntimeSettings:
        runtime = Config.table(data, "runtime")
        return cls(
            shell_timeout=Config.int(runtime, "shell_timeout", 60),
            bash_wait_timeout=max(0, Config.int(runtime, "bash_wait_timeout", 10)),
            max_steps=max(1, Config.int(runtime, "max_agent_steps", 400)),
            max_context_tokens=max(1, Config.int(runtime, "max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS)),
            max_parallel_tools=max(1, Config.int(runtime, "max_parallel_tools", 4)),
            session_retention_days=max(0, Config.int(runtime, "session_retention_days", 7)),
            yolo=yolo or Config.bool(runtime, "yolo", False),
            quick_hints=Config.bool(runtime, "quick_hints", True),
            worker=Config.bool(runtime, "worker", False),
            theme=theme or Config.str(runtime, "theme", "auto"),
            language=RuntimeSettings.clean_language(Config.str(runtime, "language", "auto")),
            agents_md=Config.bool(runtime, "agents_md", True),
        )

    @staticmethod
    def clean_language(value: str) -> str:
        value = value.strip()
        if not value or value.lower() == "auto":
            return "auto"
        if len(value) > 64 or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ConfigError("runtime.language must be a single-line language name up to 64 chars, or auto")
        return value


@dataclass
class Config:
    active_provider: str = "default"
    providers: dict[str, ProviderConfig] = field(default_factory=lambda: {"default": ProviderConfig()})
    data_dir: str = "~/.minacode"
    mcp: Json = field(default_factory=dict)
    # The provider entry a Delegate sends its worker to; empty disables the tool entirely. The
    # registration gate reads Session.worker_tool_enabled, the value frozen from this field at
    # session start, never the live field: a runtime /worker provider switch tunes an already-
    # enabled delegation and prepares the next session, but never flips the tool block (and thus
    # the prompt-cache scope) mid-session. worker_model/worker_reasoning/worker_api are
    # runtime-switchable via /worker model|reason|api (temporary, like /provider: snapshots
    # rebuild Config from the config file), and also come from [worker] model/reasoning/api:
    # an empty string means "inherit the chosen provider entry's value".
    worker_provider: str = ""
    worker_model: str = ""
    worker_reasoning: str = ""
    worker_api: str = ""

    # The provider entry compaction summaries run on, mirroring [worker]: compaction_provider names
    # a base provider entry (empty = the active provider), and compaction_model/reasoning/api
    # override that entry per field (empty = inherit the entry's value). Resolved per call by
    # compaction_provider_config, never cached, so a runtime /provider switch is picked up by the
    # next compaction.
    compaction_provider: str = ""
    compaction_model: str = ""
    compaction_reasoning: str = ""
    compaction_api: str = ""

    # Backward compatibility: the data dir moved from ~/.nanocode to ~/.minacode.
    LEGACY_DATA_DIR: ClassVar[str] = "~/.nanocode"

    def __post_init__(self) -> None:
        # When the data dir is still the new default but does not exist yet and the legacy
        # ~/.nanocode dir does, keep using the legacy dir so existing sessions, skills, and
        # cache are found without a migration step.
        if (
            self.data_dir == "~/.minacode"
            and not os.path.exists(os.path.expanduser(self.data_dir))
            and os.path.exists(os.path.expanduser(self.LEGACY_DATA_DIR))
        ):
            self.data_dir = self.LEGACY_DATA_DIR

    @property
    def provider(self) -> ProviderConfig:
        return self.providers[self.active_provider]

    @classmethod
    def from_dict(cls, data: Json) -> Config:
        provider_root = cls.table(data, "provider")
        active = cls.str(provider_root, "active", "default")
        providers = {name: ProviderConfig.from_dict(value) for name, value in provider_root.items() if name != "active" and isinstance(value, dict)}
        if not providers:
            providers = {active: ProviderConfig.from_dict(provider_root)}
        if active not in providers:
            raise ConfigError(f"provider.active `{active}` does not exist")
        paths = cls.table(data, "paths")
        worker_root = cls.table(data, "worker")
        worker_provider = cls.str(worker_root, "provider", "")
        if worker_provider and worker_provider not in providers:
            raise ConfigError(f"worker.provider `{worker_provider}` does not exist")
        worker_reasoning = cls.str(worker_root, "reasoning", "")
        if worker_reasoning and worker_reasoning not in REASONING_CHOICES:
            raise ConfigError("worker.reasoning must be one of " + ", ".join(REASONING_CHOICES))
        worker_api = cls.str(worker_root, "api", "")
        if worker_api and worker_api not in PROVIDER_API_CHOICES:
            raise ConfigError("worker.api must be one of " + ", ".join(PROVIDER_API_CHOICES))
        compaction_root = cls.table(data, "compaction")
        compaction_provider = cls.str(compaction_root, "provider", "")
        if compaction_provider and compaction_provider not in providers:
            raise ConfigError(f"compaction.provider `{compaction_provider}` does not exist")
        compaction_reasoning = cls.str(compaction_root, "reasoning", "")
        if compaction_reasoning and compaction_reasoning not in REASONING_CHOICES:
            raise ConfigError("compaction.reasoning must be one of " + ", ".join(REASONING_CHOICES))
        compaction_api = cls.str(compaction_root, "api", "")
        if compaction_api and compaction_api not in PROVIDER_API_CHOICES:
            raise ConfigError("compaction.api must be one of " + ", ".join(PROVIDER_API_CHOICES))
        return cls(
            active_provider=active,
            providers=providers,
            data_dir=cls.str(paths, "data_dir", "~/.minacode"),
            mcp=cls.table(data, "mcp"),
            worker_provider=worker_provider,
            worker_model=cls.str(worker_root, "model", ""),
            worker_reasoning=worker_reasoning,
            worker_api=worker_api,
            compaction_provider=compaction_provider,
            compaction_model=cls.str(compaction_root, "model", ""),
            compaction_reasoning=compaction_reasoning,
            compaction_api=compaction_api,
        )

    @staticmethod
    def table(data: Json, key: str) -> Json:
        return value if isinstance((value := data.get(key)), dict) else {}

    @staticmethod
    def table_tuple(data: Json, key: str) -> tuple[Json, ...]:
        """A list of tables passed through verbatim, checked only for the shape every host shares.

        Entries reach the wire unmodified, so validating their contents would mean tracking each
        host's tool catalog. `type` is the one field every documented builtin tool carries, and
        requiring it turns a typo into a config error instead of a provider 400."""
        value = data.get(key)
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"config value `{key}` must be a list of tables")
        entries: list[Json] = []
        for item in value:
            if not isinstance(item, dict):
                raise ConfigError(f"config value `{key}` must be a list of tables")
            if not (isinstance(item.get("type"), str) and item["type"]):
                raise ConfigError(f"config value `{key}` entries must each set a non-empty `type`")
            entries.append(dict(item))
        return tuple(entries)

    @staticmethod
    def str(data: Json, key: str, default: str = "") -> str:
        return default if (value := data.get(key)) is None else str(value)

    @staticmethod
    def str_tuple(data: Json, key: str) -> tuple[str, ...]:
        value = data.get(key)
        if value is None:
            return ()
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            return tuple(value)
        raise ConfigError(f"config value `{key}` must be a string list")

    @staticmethod
    def bool(data: Json, key: str, default: bool = False) -> bool:
        value = data.get(key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        lower = value.lower() if isinstance(value, str) else ""
        if lower in {"on", "true", "yes", "1", "off", "false", "no", "0"}:
            return lower in {"on", "true", "yes", "1"}
        raise ConfigError(f"config value `{key}` must be boolean")

    @staticmethod
    def int(data: Json, key: str, default: int) -> int:
        value = data.get(key)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"config value `{key}` must be integer")
        return value

    @staticmethod
    def float(data: Json, key: str, default: float | None) -> float | None:
        value = data.get(key)
        if value is None:
            return default
        if value is False or (isinstance(value, str) and value.lower() == "off"):
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"config value `{key}` must be number or off")
        return float(value)


class ConfigFile:
    DEFAULT_PATH: ClassVar[str] = os.path.join(os.path.expanduser("~"), ".minacode", "config.toml")
    LEGACY_PATH: ClassVar[str] = os.path.join(os.path.expanduser("~"), ".nanocode", "config.toml")
    # Only the provider block is required; every other key falls back to its built-in default, so the
    # commented lines below just document the common knobs and their defaults.
    DEFAULT_TEXT: ClassVar[str] = """# minacode configuration — unset keys use built-in defaults.

[provider]
active = "default"

[provider.default]
url = ""
key = ""
model = ""
# api = "auto"                 # auto | chat | responses | anthropic
# stream = true
# image_input = "auto"         # auto | on | off
# reasoning = "medium"
# max_context_tokens = 0       # how much of THIS model's window to use; 0 inherits runtime.max_context_tokens.
                               # Set it per entry when models differ: 1048576 for a 1M-window model,
                               # 131072 for a 128K one. Fewer compactions on the big one, no overflow
                               # on the small one.
# max_tokens = 0               # output cap per request, reasoning included; 0 leaves it to the provider
                               # (Anthropic sends a conservative 8K). 16K is still reserved from the
                               # input budget, trading against runtime.max_context_tokens one for one
# timeout = 120                # transport inactivity
# response_timeout = 600       # total generation time; 0 disables
# available_models = ["gpt-5", "gpt-5-mini"]

# builtin_tools = [{ type = "web_search" }]   # provider-side tools, passed through verbatim
                                              # OpenAI/Qwen: { type = "web_search" }
                                              # Anthropic:   { type = "web_search_20250305", name = "web_search" }
                                              # Z.AI:        { type = "web_search", web_search = { enable = "True" } }

# [runtime]                    # optional overrides (defaults shown)
# yolo = false
# quick_hints = true           # model-suggested next-step chips; toggle with /hints
# max_context_tokens = 262144      # 256K; how much of the model's window to use, not its size.
                               # Raise it for a 1M-window model; lower it for a smaller one.
# max_agent_steps = 400
# shell_timeout = 60
# worker = false               # register the Delegate tool; toggle with /worker on|off
                               # (flipping it changes the tool block and thus the prompt-cache scope)
# language = "auto"           # auto follows your messages and injects nothing; set a language
                               # name (e.g. "Chinese") to force the reply language
# agents_md = true               # inject the project's AGENTS.md (or CLAUDE.md fallback) into every
                                 # request as a bounded Project-instructions section of Environment

# [worker]                     # optional: hand tasks to a second minacode session (Delegate tool)
# provider = "fast"           # a provider entry; pick one from a DIFFERENT vendor than
                               # provider.active, so the worker's reviews cross-validate the
                               # parent's -- same-family models share blind spots
# model = ""                  # optional: override the entry's model (inherit by default)
# reasoning = ""              # optional: override the entry's reasoning; /worker reason at runtime
# api = ""                    # optional: override the entry's api protocol; empty = inherit the entry's own
# [mcp.example]                # url (+ auth = "oauth") for remote, or command/args for stdio
# url = "https://example.com/mcp"
# auto_connect = false
"""

    @classmethod
    def resolve_path(cls, path: str | None) -> str:
        if path:
            return os.path.expanduser(path)
        # Backward compatibility: read the legacy ~/.nanocode/config.toml when the new
        # ~/.minacode/config.toml does not exist yet.
        if not os.path.exists(cls.DEFAULT_PATH) and os.path.exists(cls.LEGACY_PATH):
            return cls.LEGACY_PATH
        return cls.DEFAULT_PATH

    @classmethod
    def init(cls, path: str | None = None) -> tuple[str, bool]:
        config_path = cls.resolve_path(path)
        if os.path.exists(config_path):
            return config_path, False
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as file:
            file.write(cls.DEFAULT_TEXT)
        return config_path, True

    @classmethod
    def load(cls, path: str | None = None) -> Json:
        config_path = cls.resolve_path(path)
        try:
            with open(config_path, "rb") as file:
                data = tomllib.load(file)
        except FileNotFoundError as error:
            raise ConfigError(f"config not found: {config_path}; run --init-config") from error
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"invalid config {config_path}: {error}") from error
        return data if isinstance(data, dict) else {}


def compaction_provider_config(config: Config) -> ProviderConfig:
    """The provider entry compaction summaries run on, with [compaction] overrides applied.

    Base entry is `config.compaction_provider` or the active provider when unset; non-empty
    compaction_model/reasoning/api fold over it. Never shares or mutates the base entry object:
    dataclasses.replace is shallow, so folding onto a shared object would leak compaction-only
    overrides into the main provider (same reasoning as worker_provider_config in tools/delegate.py).
    Resolved per call, never cached, so a runtime /provider switch is picked up by the next
    compaction.
    """
    provider = replace(config.providers[config.compaction_provider or config.active_provider])
    if config.compaction_model:
        provider.model = config.compaction_model
    if config.compaction_reasoning:
        provider.reasoning = config.compaction_reasoning
    if config.compaction_api:
        provider.api = config.compaction_api
    return provider
