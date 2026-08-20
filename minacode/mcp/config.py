"""MCP server configuration: the MCPServerConfig model and pure config-file parsing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from minacode.base import Json
from minacode.config import Config


@dataclass
class MCPServerConfig:
    name: str
    url: str = ""
    command: str = ""
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    auth: str = ""
    bearer_token_env_var: str = ""
    env_http_headers: dict[str, str] = field(default_factory=dict)
    auto_connect: bool = False
    error: str = ""


def parse_config(name: str, raw: Json) -> MCPServerConfig:
    """Parse one `[mcp.<name>]` block into an MCPServerConfig, recording validation errors on it."""
    config = MCPServerConfig(
        name=name,
        url=Config.str(raw, "url"),
        command=Config.str(raw, "command"),
        auth=Config.str(raw, "auth").lower(),
        bearer_token_env_var=Config.str(raw, "bearer_token_env_var"),
        auto_connect=Config.bool(raw, "auto_connect", False),
    )

    def config_error(message: str) -> None:
        if not config.error:
            config.error = message

    def string_list(value: object) -> tuple[str, ...] | None:
        return tuple(value) if isinstance(value, list) and all(isinstance(item, str) for item in value) else None

    def string_map(value: object) -> dict[str, str] | None:
        return dict(value) if isinstance(value, dict) and all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()) else None

    def read_field(key: str, parse: Callable[[object], object | None], error: str) -> None:
        if (value := raw.get(key)) is None:
            return
        parsed = parse(value)
        if parsed is None:
            config_error(error)
        else:
            setattr(config, key, parsed)

    read_field("args", string_list, "args must be a string list")
    read_field("env", string_map, "env must be a string map")
    read_field("env_http_headers", string_map, "env_http_headers must be a string map")
    if bool(config.url) == bool(config.command):
        config_error("exactly one of url or command is required")
    elif config.command and (config.auth or config.bearer_token_env_var or raw.get("env_http_headers")):
        config_error("command (stdio) servers cannot use auth/bearer_token_env_var/env_http_headers")
    if config.auth not in {"", "oauth"}:
        config_error("auth must be oauth")
    if config.auth == "oauth" and config.bearer_token_env_var:
        config_error("auth=oauth conflicts with bearer_token_env_var")
    if config.auth == "oauth" and has_header(config.env_http_headers, "authorization"):
        config_error("auth=oauth conflicts with env_http_headers.Authorization")
    return config


def has_header(headers: dict[str, str], name: str) -> bool:
    """Whether `headers` contains `name`, case-insensitively."""
    return any(header.lower() == name.lower() for header in headers)
