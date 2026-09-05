"""wizolt MCP manager: configured-server lifecycle and the bounded model- and user-facing views."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
from collections.abc import Awaitable, Callable, Coroutine
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from wizolt.base import Json, ToolError
from wizolt.mcp.config import MCPServerConfig, has_header, parse_config
from wizolt.mcp.rendering import (
    MCPResourceInfo,
    MCPToolInfo,
    dump_object,
    extract_uris,
    format_resource_line,
    format_tool_line,
    index_body,
    markdown_cell,
    normalize_resource,
    normalize_result,
    render_describe,
    resources_info,
    server_lines,
    tool_args_summary,
)
from wizolt.mcp.tokens import MCPFileTokenStore
from wizolt.mentions import scan_mentions
from wizolt.session import Session

if TYPE_CHECKING:
    from fastmcp.client import Client
    from fastmcp.client.auth import OAuth
    from fastmcp.client.client import CallToolResult
    from fastmcp.client.tasks import ResourceTask, ToolTask
    from fastmcp.client.transports import ClientTransport
    from mcp.types import BlobResourceContents, Resource, TextResourceContents, Tool

_MCPResultT = TypeVar("_MCPResultT")


class MCPManager:
    """Manage configured MCP servers and expose bounded model- and user-facing views.

    Servers are external, so nothing here may be load-bearing. Discovery runs concurrently in the
    background, and a slow, broken, or unauthorized server records its reason and drops out of the
    index rather than failing the session. Connection state is therefore something to display, not an
    error to raise.

    Only a bounded summary of a catalog reaches the model: schemas and descriptions are capped per
    tool and overall, because a verbose server would otherwise spend the context budget every turn
    merely by existing. Full schemas stay available on demand through describe. The same normalized
    catalog produces command listings and connection status without making the command loop
    understand MCP schemas or failure states.

    Each operation opens its own short-lived client, so no connection is durable state. That costs a
    process start per stdio call and is why the lifecycle rework is on the roadmap in DESIGN.md.

    Every operation is a coroutine on the caller's loop -- the one the runtime owns -- so a client
    is entered, used, and exited in one place, and a cancelled turn brings its MCP call down with
    it. Discovery, the catalog, and the status the TUI renders from it therefore all live on that
    one loop, and the catalog needs no lock of its own.
    """

    RAW_OUTPUT_LIMIT: ClassVar[int] = 200_000
    DISCOVERY_TIMEOUT: ClassVar[int] = 10
    MAX_DISCOVERY_WORKERS: ClassVar[int] = 8
    DESCRIBE_DESCRIPTION_LIMIT: ClassVar[int] = 1_000
    DESCRIBE_ARGUMENT_LIMIT: ClassVar[int] = 50
    DESCRIBE_ARGUMENT_DESCRIPTION_LIMIT: ClassVar[int] = 160
    INDEX_SCHEMA_LIMIT: ClassVar[int] = 700  # per-tool schema cap in the early (cached) tools index
    INDEX_TOTAL_LIMIT: ClassVar[int] = 16_000  # overall cap for the tools index block
    STATUS_MARKER: ClassVar[str] = "●"
    AUTH_STATUS_RE: ClassVar[re.Pattern] = re.compile(r"\b(?:401|403)\b")

    def __init__(self, session: Session):
        self.session = session
        self.tools: dict[str, list[MCPToolInfo]] = {}
        self.resources: dict[str, list[MCPResourceInfo]] = {}
        self._auto_read_done: set[tuple[str, str]] = set()
        self.server_errors: dict[str, str] = {}
        self.server_skips: dict[str, str] = {}
        self.discovery_status: str = "stale"  # stale | discovering | ready | error
        self.index_truncated: bool = False  # set by render_tools_index when even name-only overflows the cap
        self._configs_cache: list[MCPServerConfig] | None = None
        self._oauth_token_store = MCPFileTokenStore(self.session.data_path("mcp-oauth", "tokens.json"))
        self._oauth_lock: asyncio.Lock | None = None
        self._oauth_lock_loop: asyncio.AbstractEventLoop | None = None
        self._discovering_servers: dict[str, int] = {}
        self._discovery_failed = False
        # Operations in flight on the runtime loop, so shutdown has something to cancel and await.
        self._tasks: set[asyncio.Task] = set()
        self._closed = False

    def parse_configs(self) -> list[MCPServerConfig]:
        # Config and selector are immutable for the session, so parse once and reuse.
        if self._configs_cache is None:
            self._configs_cache = self._parse_configs()
        return self._configs_cache

    def _parse_configs(self) -> list[MCPServerConfig]:
        mcp_config = self.session.config.mcp
        if not isinstance(mcp_config, dict):
            return []
        configs = [parse_config(str(name), raw) for name, raw in mcp_config.items() if isinstance(raw, dict)]
        return configs

    @staticmethod
    def _has_header(headers: dict[str, str], name: str) -> bool:
        return has_header(headers, name)

    def find_config(self, name: str) -> MCPServerConfig | None:
        return next((config for config in self.parse_configs() if config.name == name), None)

    @contextlib.contextmanager
    def _discovery(self, names: tuple[str, ...]):
        if not self._discovering_servers:
            self._discovery_failed = False
        for name in names:
            self._discovering_servers[name] = self._discovering_servers.get(name, 0) + 1
        self.discovery_status = "discovering"
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            self._discovery_failed |= failed
            for name in names:
                remaining = self._discovering_servers.get(name, 0) - 1
                if remaining > 0:
                    self._discovering_servers[name] = remaining
                else:
                    self._discovering_servers.pop(name, None)
            if not self._discovering_servers:
                self.discovery_status = "error" if self._discovery_failed else "ready"

    def discovering(self, name: str) -> bool:
        return name in self._discovering_servers

    def _forget(self, name: str) -> None:
        self.tools.pop(name, None)
        self.resources.pop(name, None)
        self._auto_read_done = {entry for entry in self._auto_read_done if entry[0] != name}
        self.server_errors.pop(name, None)
        self.server_skips.pop(name, None)

    async def discover_auto(self) -> None:
        configs = self.parse_configs()
        discoverable = [config for config in configs if config.auto_connect]
        names = tuple(config.name for config in discoverable)
        try:
            with self._discovery(names):
                configured = {config.name for config in configs}
                for name in list(self.tools.keys() | self.resources.keys()):
                    if name not in configured:
                        self._forget(name)
                await self._gather_bounded([self._discover_one(config) for config in discoverable])
        except Exception as error:  # noqa: BLE001 - discovery aggregates failures from arbitrary MCP transports.
            self.server_errors["-"] = str(error)

    async def _gather_bounded(self, coroutines: list[Coroutine[Any, Any, Any]]) -> list[Any]:
        """Run discovery-shaped work concurrently under the worker cap, preserving order.

        Each child records its own failure, so `gather` here sees successes: one unreachable server
        must not cancel the healthy siblings mid-flight, which is exactly what a raising child in a
        TaskGroup would do."""

        if not coroutines:
            return []
        limit = asyncio.Semaphore(min(self.MAX_DISCOVERY_WORKERS, len(coroutines)))

        async def bounded(coroutine: Coroutine[Any, Any, Any]) -> Any:
            async with limit:
                return await coroutine

        return list(await asyncio.gather(*(bounded(coroutine) for coroutine in coroutines)))

    async def discover_server(self, name: str) -> None:
        config = self.find_config(name)
        if config is None:
            self._forget(name)
            self.server_errors[name] = "server not found"
            return
        with self._discovery((name,)):
            await self._discover_one(config)

    async def disconnect_server(self, name: str) -> str:
        config = self.find_config(name)
        if config is None:
            return "MCP server not found: " + name
        if config.auth == "oauth" and config.url:
            await self._oauth_token_store.clear_server(config.url)
        self._forget(name)
        return "MCP server disconnected: " + name

    def connected(self, name: str) -> bool:
        return name in self.tools or name in self.resources

    async def connect_server(
        self,
        name: str,
        *,
        interactive: bool = False,
        notify: Callable[[str], None] | None = None,
        _compact: bool = False,
    ) -> str:
        config = self.find_config(name)
        if config is None:
            return self._compact_line("error", name, "server not found") if _compact else "MCP server not found: " + name
        if not config.error and config.auth == "oauth":
            has_tokens = await self._oauth_token_store.has_server_tokens(config.url)
            if not interactive and not has_tokens:
                message = f"authentication required; run `/mcp connect {name}` interactively"
                if _compact:
                    return self._compact_line("error", name, message)
                return f"MCP server authentication required: {name}; run /mcp connect {name} interactively"
            if interactive:
                if has_tokens:
                    await self.discover_server(name)
                    if not self._oauth_reauthorization_required(name):
                        return self._connect_result(name, compact=_compact)
                async with self._oauth_gate():
                    # The token and registered OAuth client form one credential set. If
                    # either is rejected, discard both so the new random callback port is
                    # registered together with the replacement token.
                    await self._oauth_token_store.clear_server(config.url)
                    if error := await self._authenticate_oauth(config, notify=notify):
                        if _compact:
                            prefix = f"MCP OAuth authentication failed for {name}: "
                            return self._compact_line("error", name, error.removeprefix(prefix))
                        return error
        await self.discover_server(name)
        return self._connect_result(name, compact=_compact)

    def _compact_line(self, kind: str, name: str, detail: str) -> str:
        """One-line server status used by the batch connect/manager UIs: '● kind  `name` — detail'."""
        return f"{self.STATUS_MARKER} {kind}  `{name}` — {detail}"

    def _oauth_reauthorization_required(self, name: str) -> bool:
        issue = self.server_issue(name)
        if issue is None or issue[0] != "error":
            return False
        message = issue[1].lower()
        markers = ("authentication required", "unauthorized", "invalid token", "invalid_token", "invalid_request", "invalid client")
        return any(marker in message for marker in markers) or MCPManager.AUTH_STATUS_RE.search(message) is not None

    def _connect_result(self, name: str, *, compact: bool = False) -> str:
        if issue := self.server_issue(name):
            kind, message = issue
            if compact:
                return self._compact_line(kind, name, message)
            return f"MCP server {kind}: {name}: {message}"
        tool_count = len(self.tools.get(name, []))
        resource_count = len(self.resources.get(name, []))
        if compact:
            assets = f"{tool_count} tool" + ("" if tool_count == 1 else "s")
            if resource_count:
                assets += f", {resource_count} resource" + ("" if resource_count == 1 else "s")
            return self._compact_line("connected", name, assets)
        return f"MCP server connected: {name}; tools={tool_count}; resources={resource_count}"

    async def connect_servers(
        self,
        names: list[str],
        *,
        interactive: bool = False,
        notify: Callable[[str], None] | None = None,
    ) -> str:
        """Connect a de-duplicated batch concurrently while preserving result order."""
        selected = list(dict.fromkeys(names))
        if len(selected) == 1:
            return await self.connect_server(selected[0], interactive=interactive, notify=notify)

        connected = await self._gather_bounded([self.connect_server(name, interactive=interactive, notify=notify, _compact=True) for name in selected])
        results = dict(zip(selected, connected, strict=True))
        items = ("- " + results[name].replace("\n", "\n    ") for name in selected)
        return "MCP connection results:\n\n" + "\n".join(items)

    async def _discover_one(self, config: MCPServerConfig) -> None:
        if config.error:
            self.set_server_error(config.name, config.error)
            return
        headers = self._build_mcp_headers(config)
        if isinstance(headers, str):
            if self.can_skip_auth_error(headers):
                self.set_server_skip(config.name, headers)
            else:
                self.set_server_error(config.name, headers)
            return

        if config.auth == "oauth" and not await self._oauth_token_store.has_server_tokens(config.url):
            self.set_server_error(config.name, "authentication required; run /mcp connect " + config.name)
            return
        try:
            tools, resources = await self._bounded(self._gather_assets(config, headers), timeout=self.discovery_timeout())
            self.tools[config.name] = self._tools_info(config.name, tools)
            self.resources[config.name] = self._resources_info(config.name, resources)
            self.server_errors.pop(config.name, None)
            self.server_skips.pop(config.name, None)
        except BaseException as error:
            if self.is_cancelled_error(error):
                self.server_errors.pop(config.name, None)
                return
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            self.set_server_error(config.name, self.error_text(error, timeout=self.discovery_timeout()))

    async def _gather_assets(self, config: MCPServerConfig, headers: dict[str, str]) -> tuple[list[Tool], list[Resource]]:
        """Fetch tools and resources concurrently. Tool failure aborts discovery; resources are best-effort."""
        tools_co = self._list_tools(config, headers)
        resources_co = self._list_resources(config, headers)
        tools, resources = await asyncio.gather(tools_co, resources_co, return_exceptions=True)
        if isinstance(tools, BaseException):
            raise tools
        if isinstance(resources, BaseException):
            resources = []
        return tools, resources

    def set_server_error(self, name: str, error: str) -> None:
        self._forget(name)
        self.server_errors[name] = error

    def set_server_skip(self, name: str, reason: str) -> None:
        self._forget(name)
        self.server_skips[name] = reason

    @classmethod
    def is_cancelled_error(cls, error: BaseException) -> bool:
        seen: set[int] = set()

        def visit(item: BaseException) -> bool:
            identity = id(item)
            if identity in seen:
                return False
            seen.add(identity)
            if type(item).__name__ == "CancelledError":
                return True
            nested = getattr(item, "exceptions", ())
            if nested:
                return all(isinstance(child, BaseException) and visit(child) for child in nested)
            cause = item.__cause__ or item.__context__
            return isinstance(cause, BaseException) and visit(cause)

        return visit(error)

    @staticmethod
    def can_skip_auth_error(error: str) -> bool:
        return error.startswith("missing environment variable ")

    def call_timeout(self) -> int:
        return max(1, self.session.settings.shell_timeout)

    def discovery_timeout(self) -> int:
        return min(self.call_timeout(), self.DISCOVERY_TIMEOUT)

    def error_text(self, error: BaseException, *, timeout: int | None = None) -> str:
        if isinstance(error, TimeoutError):
            return f"timeout after {timeout or self.call_timeout()}s"
        text = str(error).strip()
        return text or error.__class__.__name__

    def _tools_info(self, server: str, tools: list[Tool]) -> list[MCPToolInfo]:
        return [
            MCPToolInfo(
                server=server,
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema,
                output_schema=self.tool_output_schema(t),
                annotations=self.tool_annotations(t),
            )
            for t in tools
        ]

    def _resources_info(self, server: str, resources: list[Resource]) -> list[MCPResourceInfo]:
        return resources_info(server, resources)

    @staticmethod
    def tool_output_schema(tool: Tool) -> Json:
        """The tool's declared `outputSchema`, or {} when it declares none.

        Read under both spellings: the wire field is camelCase and SDK models expose it that way,
        but a server object built by hand may carry the snake_case name instead."""
        for attribute in ("outputSchema", "output_schema"):
            schema = getattr(tool, attribute, None)
            if isinstance(schema, dict) and schema:
                return schema
        return {}

    @staticmethod
    def tool_annotations(tool: Tool) -> Json:
        annotations = getattr(tool, "annotations", None)
        if annotations is None:
            return {}
        if isinstance(annotations, dict):
            return annotations
        if hasattr(annotations, "model_dump"):
            data = annotations.model_dump(mode="json", exclude_none=True)
            return data if isinstance(data, dict) else {}
        return {}

    def tool_needs_confirmation(self, server: str, tool_name: str) -> bool:
        info = self.tool_info(server, tool_name)
        if info is None:
            return True
        annotations = info.annotations
        if annotations.get("readOnlyHint") is True:
            return False
        return annotations.get("destructiveHint") is not False

    def tool_info(self, server: str, tool_name: str) -> MCPToolInfo | None:
        return next((tool for tool in self.tools.get(server, []) if tool.name == tool_name), None)

    def _oauth_gate(self) -> asyncio.Lock:
        """The gate that keeps OAuth logins one at a time, on whichever loop is running.

        A login opens a browser and a local callback server, so two at once collide on both. An
        asyncio lock, not a threading one: this is held across the login's awaits, and a blocking
        lock there would stop the loop that has to run the other half of the handshake.

        Rebound rather than kept for the manager's lifetime -- a lock belongs to the loop that
        created it, and separate `asyncio.run()` boundaries do not share one."""

        loop = asyncio.get_running_loop()
        if self._oauth_lock is None or self._oauth_lock_loop is not loop:
            self._oauth_lock, self._oauth_lock_loop = asyncio.Lock(), loop
        return self._oauth_lock

    async def _bounded(self, coroutine: Coroutine[Any, Any, _MCPResultT], *, timeout: int | None = None) -> _MCPResultT:
        """Await one MCP operation under a deadline, as a task this manager owns.

        A task rather than a bare await so `close()` has something to cancel: an operation
        started by a turn that is gone must still be brought down before the loop closes.

        `wait_for` cancels the operation and *awaits* it, so the FastMCP client's `async with` has
        unwound -- process reaped, HTTP session closed -- by the time the timeout is raised here.
        Cancellation from outside travels the same way, which is why neither is special-cased."""

        if self._closed:
            # Closed before this one was ever started: the operation is closed rather than dropped,
            # or the caller's un-awaited coroutine would surface as a warning at collection.
            coroutine.close()
            raise ToolError("MCP manager is closed")
        timeout = self.call_timeout() if timeout is None else timeout
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout)
            if done:
                return task.result()
            await self._settle(task)
            raise ToolError(f"MCP call timed out after {timeout}s")
        except asyncio.CancelledError:
            await self._settle(task)
            raise
        finally:
            self._tasks.discard(task)

    @staticmethod
    async def _settle(task: asyncio.Task) -> None:
        """Cancel one operation and wait for its client to finish unwinding.

        `wait` rather than awaiting the task: the caller already has the answer that matters -- a
        timeout, or its own cancellation -- and a client whose teardown then fails on its own (an
        HTTP read timeout on the request it was abandoning) must neither replace that answer nor be
        left unretrieved for the loop's exception handler to print during collection."""

        task.cancel()
        await asyncio.wait({task})
        if not task.cancelled():
            task.exception()

    async def close(self) -> None:
        """Cancel and await everything this manager still has in flight.

        Called by the runtime's shutdown, on the loop that owns those tasks -- a FastMCP client
        must not be left to the interpreter's teardown, where an in-flight cleanup (an HTTP session
        termination, a DNS lookup in the default executor) races the executor's own atexit."""

        if self._closed:
            return
        self._closed = True
        pending = list(self._tasks)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()

    def oauth_client(self, config: MCPServerConfig, *, interactive: bool = False, notify: Callable[[str], None] | None = None) -> OAuth:
        from fastmcp.client.auth import OAuth

        class WizoltOAuth(OAuth):
            async def redirect_handler(self, authorization_url: str) -> None:
                if not interactive:
                    raise RuntimeError("authentication required; run /mcp connect " + config.name)
                if notify:
                    notify("Open this URL to authorize MCP server `" + config.name + "`:\n" + authorization_url)
                await super().redirect_handler(authorization_url)

        return WizoltOAuth(
            # FastMCP types this as its full AsyncKeyValue protocol, although TokenStorageAdapter
            # only calls get/put/delete. MCPFileTokenStore deliberately implements that used subset.
            token_storage=self._oauth_token_store,  # pyright: ignore[reportArgumentType]
            client_name="wizolt",
            callback_timeout=self.session.settings.shell_timeout,
        )

    def _transport(self, config: MCPServerConfig, headers: dict[str, str]) -> ClientTransport:
        from fastmcp.client.transports import StdioTransport, StreamableHttpTransport

        if config.command:
            # The MCP SDK replaces (not merges) the subprocess environment when env is set,
            # so layer the configured vars over the inherited environment to keep PATH etc.
            env = {**os.environ, **config.env} if config.env else None
            return StdioTransport(command=config.command, args=list(config.args), env=env)
        return StreamableHttpTransport(config.url, headers=headers)

    async def _run_op(
        self,
        config: MCPServerConfig,
        headers: dict[str, str],
        operation: Callable[[Client], Awaitable[_MCPResultT]],
        *,
        long_timeout: bool = False,
        interactive: bool = False,
        notify: Callable[[str], None] | None = None,
    ) -> _MCPResultT:
        """Enter a fastmcp Client (with OAuth if config.auth=='oauth') and await one operation."""
        from fastmcp.client import Client

        timeout = self.call_timeout() if long_timeout or interactive else self.discovery_timeout()
        auth = self.oauth_client(config, interactive=interactive, notify=notify) if config.auth == "oauth" else None
        async with Client(self._transport(config, headers), auth=auth, timeout=timeout, init_timeout=timeout) as client:
            return await asyncio.wait_for(operation(client), timeout=timeout)

    async def _list_tools(self, config: MCPServerConfig, headers: dict[str, str]) -> list[Tool]:
        return await self._run_op(config, headers, lambda client: client.list_tools())

    async def _list_resources(self, config: MCPServerConfig, headers: dict[str, str]) -> list[Resource]:
        return await self._run_op(config, headers, lambda client: client.list_resources())

    async def _call_tool(self, config: MCPServerConfig, headers: dict[str, str], name: str, arguments: Json) -> CallToolResult | ToolTask:
        return await self._run_op(config, headers, lambda client: client.call_tool(name, arguments), long_timeout=True)

    async def _read_resource(
        self, config: MCPServerConfig, headers: dict[str, str], uri: str
    ) -> list[TextResourceContents | BlobResourceContents] | ResourceTask:
        return await self._run_op(config, headers, lambda client: client.read_resource(uri), long_timeout=True)

    def _build_mcp_headers(self, config: MCPServerConfig) -> dict[str, str] | str:
        headers: dict[str, str] = {}
        if config.bearer_token_env_var:
            token = os.environ.get(config.bearer_token_env_var)
            if not token:
                return f"missing environment variable {config.bearer_token_env_var}"
            headers["Authorization"] = f"Bearer {token}"
        if config.env_http_headers:
            for header_name, env_var in config.env_http_headers.items():
                value = os.environ.get(env_var)
                if not value:
                    return f"missing environment variable {env_var}"
                if header_name.lower() == "authorization":
                    if config.auth == "oauth":
                        return "conflicting Authorization header; use auth=oauth instead"
                    if self._has_header(headers, "authorization"):
                        return "conflicting Authorization header; use only one authorization source"
                headers[header_name] = value
        return headers

    async def _resolve_server(self, server: str) -> tuple[MCPServerConfig, dict[str, str]]:
        """Look up a configured server and build its request headers, raising ToolError with a
        user-facing message on a missing, errored, or unauthenticated server. Shared by tool and
        resource calls."""
        config = self.find_config(server)
        if config is None:
            raise ToolError(f"MCP server '{server}' not found")
        if config.error:
            raise ToolError(config.error)
        headers = self._build_mcp_headers(config)
        if isinstance(headers, str):
            raise ToolError(headers)
        if config.auth == "oauth" and not await self._oauth_token_store.has_server_tokens(config.url):
            raise ToolError(f"MCP server '{server}' requires authentication; run /mcp connect {server}")
        self._require_available(server)
        return config, headers

    def _require_available(self, server: str) -> None:
        """Raise ToolError if a configured server has a failure state or is not connected."""
        if issue := self.server_issue(server):
            raise ToolError(f"MCP server '{server}' {issue[0]}: {issue[1]}")
        if not self.connected(server):
            raise ToolError(f"MCP server '{server}' is not connected; run /mcp connect {server}")

    async def call_tool(self, server: str, tool_name: str, arguments: Json) -> str:
        result = await self._call_result(server, tool_name, arguments)

        text = self.normalize_result(result)
        return f"<MCPCall server={json.dumps(server)} tool={json.dumps(tool_name)}>\n{text}\n</MCPCall>"

    async def _call_result(self, server: str, tool_name: str, arguments: Json) -> Any:
        """Shared transport path for call_tool and call_tool_structured: resolve, run, normalize errors."""
        config, headers = await self._resolve_server(server)
        try:
            return await self._bounded(self._call_tool(config, headers, tool_name, arguments))
        except Exception as e:
            raise ToolError("MCP call failed: " + self.error_text(e)) from e

    async def call_tool_structured(self, server: str, tool_name: str, arguments: Json) -> Any:
        """Call an MCP tool and return its payload as a parsed JSON value.

        `Any`, not `Json`: a declared outputSchema may describe an array as legitimately as an
        object, and a tool returning a list of hits must reach the script as that list.

        When the tool declared an outputSchema the structuredContent payload is authoritative: it is
        returned as-is, and a declared-but-missing payload is an error, never a silent downgrade to
        text. Without a declared schema the call's text body is parsed as JSON; a non-JSON body is
        an error so the caller can decide whether to fall back to text.
        """
        result = await self._call_result(server, tool_name, arguments)

        info = self.tool_info(server, tool_name)
        if info is not None and info.output_schema:
            # Asked for by attribute, not through _structured_content: that helper renders "" both
            # for "no payload" and for an empty one, and a search that legitimately matched nothing
            # returns exactly `{}` or `[]`. Treating that as a missing payload would fail every
            # such call.
            for attribute in ("structuredContent", "structured_content"):
                structured = getattr(result, attribute, None)
                if isinstance(structured, (dict, list)):
                    return structured
                if structured is not None:
                    try:
                        return json.loads(self._dump_object(structured))
                    except (json.JSONDecodeError, ValueError):
                        raise ToolError(f'server returned a structuredContent payload that is not JSON for tool "{tool_name}"') from None
            raise ToolError(f'server declared outputSchema but no structuredContent for tool "{tool_name}"')

        # No declared schema: parse the call's text body (the same text call_tool wraps in
        # <MCPCall>, i.e. what runner.MCP_CALL_RE unwraps before its own json.loads).
        text = self.normalize_result(result)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            raise ToolError(f'MCP returned text that is not JSON for tool "{tool_name}"')

    async def list_resources(self, server: str) -> str:
        await self._resolve_server(server)
        resources = self.resources.get(server, [])
        lines = [f"<MCPResources server={json.dumps(server)}>"]
        if resources:
            lines.extend(self._format_resource_line(res) for res in resources)
        else:
            lines.append("(no resources advertised by this server)")
        lines.append("</MCPResources>")
        return "\n".join(lines)

    async def read_resource(self, server: str, uri: str) -> str:
        if not uri:
            raise ToolError("MCP read_resource requires a uri")
        config, headers = await self._resolve_server(server)
        try:
            result = await self._bounded(self._read_resource(config, headers, uri))
        except Exception as e:  # noqa: BLE001 - normalize arbitrary MCP transport errors as ToolError.
            raise ToolError("MCP resource read failed: " + self.error_text(e))
        text = self.normalize_resource(result)
        return f"<MCPResource server={json.dumps(server)} uri={json.dumps(uri)}>\n{text}\n</MCPResource>"

    AUTO_READ_LIMIT: ClassVar[int] = 6_000  # per-doc cap for resources auto-injected on first tool call

    async def auto_read_prefix(self, server: str, tool_name: str) -> str:
        """On the first call to a tool whose description references a resource doc, fetch it once.

        Returns a block to attach to that call's result (so the grammar reaches the model on the
        first attempt and lands in cached history), or "" when there is nothing new to inject.
        Best-effort: failures are swallowed and never retried for the same uri.
        """
        info = self.tool_info(server, tool_name)
        if info is None:
            return ""
        advertised = {res.uri for res in self.resources.get(server, [])}
        blocks: list[str] = []
        for uri in self._extract_uris(info.description):
            if (server, uri) in self._auto_read_done:
                continue
            scheme = uri.split("://", 1)[0].lower()
            # Only fetch things we can actually read over MCP: advertised resources or custom
            # (non-web) schemes. Plain http(s) links are left for the model to read explicitly.
            if uri not in advertised and scheme in ("http", "https"):
                continue
            self._auto_read_done.add((server, uri))  # mark before fetching so failures don't retry
            try:
                blocks.append((await self.read_resource(server, uri))[: self.AUTO_READ_LIMIT])
            except Exception:  # noqa: BLE001, S112 - referenced resources are injected best-effort.
                continue
        if not blocks:
            return ""
        body = "\n".join(blocks)
        return f'<MCPAutoResources note="docs referenced by {server}.{tool_name}; injected once">\n{body}\n</MCPAutoResources>\n'

    @staticmethod
    def _dump_object(item: Any) -> str:
        return dump_object(item)

    def normalize_resource(self, result: Any) -> str:
        return normalize_resource(result, raw_output_limit=self.RAW_OUTPUT_LIMIT)

    def _format_resource_line(self, info: MCPResourceInfo) -> str:
        return format_resource_line(info)

    def normalize_result(self, result: Any) -> str:
        """Render a tool result as the text the model reads.

        A tool that declares an `outputSchema` returns its payload as `structuredContent`, and only
        *should* also repeat it as text for older clients. A server that skips the repeat would
        otherwise arrive here as an empty result -- indistinguishable, to the model, from a query
        that matched nothing -- so the structured payload stands in when the content blocks are
        empty. It is not appended when they are not: servers that honor the repeat send the same
        payload twice, and printing both would double every result.
        """
        return normalize_result(result, raw_output_limit=self.RAW_OUTPUT_LIMIT)

    async def _authenticate_oauth(self, config: MCPServerConfig, notify: Callable[[str], None] | None = None) -> str | None:
        """Validate cached OAuth credentials or complete interactive authorization."""
        headers = self._build_mcp_headers(config)
        if isinstance(headers, str):
            return headers
        try:
            await self._bounded(self._run_op(config, headers, lambda c: c.list_tools(), interactive=True, notify=notify))
        except Exception as error:  # noqa: BLE001 - OAuth probes cross third-party MCP transports.
            text = self.error_text(error, timeout=self.call_timeout())
            self.set_server_error(config.name, text)
            return self.oauth_auth_failure(config, text)
        self.server_errors.pop(config.name, None)
        return None

    @staticmethod
    def oauth_auth_failure(config: MCPServerConfig, error: str) -> str:
        return "\n".join(
            [
                "MCP OAuth authentication failed for " + config.name + ": " + error,
                "No authorization URL was provided by the server.",
                "Open MCP URL: " + config.url,
            ]
        )

    def describe_tool(self, server: str, tool_name: str) -> str:
        if self.find_config(server) is None:
            raise ToolError(f"MCP server '{server}' not found")
        self._require_available(server)

        info = self.tool_info(server, tool_name)
        if info is None:
            raise ToolError(f"MCP tool '{tool_name}' not found on server '{server}'")

        return self._render_describe(server, info)

    def describe_tool_block(self, server: str, tool_name: str) -> tuple[str, MCPToolInfo]:
        """Public variant of describe_tool that also hands back the tool info, so callers
        (ToolScript) can append their own gate line after the shared _render_describe
        rendering. Same checks and rendering as describe_tool; only the return differs."""
        if self.find_config(server) is None:
            raise ToolError(f"MCP server '{server}' not found")
        self._require_available(server)

        info = self.tool_info(server, tool_name)
        if info is None:
            raise ToolError(f"MCP tool '{tool_name}' not found on server '{server}'")

        return self._render_describe(server, info), info

    def _render_describe(self, server: str, info: MCPToolInfo) -> str:
        return render_describe(
            server,
            info,
            description_limit=self.DESCRIBE_DESCRIPTION_LIMIT,
            argument_limit=self.DESCRIBE_ARGUMENT_LIMIT,
            argument_description_limit=self.DESCRIBE_ARGUMENT_DESCRIPTION_LIMIT,
        )

    def render_tools_index(self) -> str:
        """Render the MCP tools block injected into every model turn (in the cached prefix).

        The block is capped at INDEX_TOTAL_LIMIT so it cannot bloat each request. When it
        would overflow we degrade by shedding *detail*, never *entities*: the model can
        always re-fetch a dropped schema via `describe`, but it can never call a server or
        tool it was never told exists. So we try progressively cheaper renderings and emit
        the richest one that fits:

            tier 1 "schema" — full per-tool JSON schemas inline (normal case)
            tier 2 "args"   — schemas dropped, name + arg summary per tool
            tier 3 "names"  — name-only, grouped per server
            tier 4          — hard truncate (only at thousands of tools, where 16KB
                              physically cannot hold them); server headers come first so
                              the model still sees most servers exist.

        Tiers 1–3 keep every connected server and tool name visible. See _index_body for how
        each detail level is rendered, and test_mcp.TestToolIndexBudget for the guarantees.
        """
        activated = self.tools.keys() | self.resources.keys()
        configs = [config for config in self.parse_configs() if config.name in activated]
        if not configs:
            return ""

        intro = [
            "--- MCP TOOLS ---",
            'Use MCP(action="call", server, tool, arguments) for external MCP server tools.',
            'Use MCP(action="describe", server, tool) for the full schema when one is truncated below; the result stays in the conversation, so do not describe the same tool again once its schema is shown — just call it.',
            'Use MCP(action="read_resource", server, uri) to read a listed resource (e.g. docs describing how to build a tool\'s arguments). Read relevant resources before calling.',
            "Format: server.tool(req: type; opt: type) - description",
            "        schema: <JSON Schema for the arguments object>",
            "",
        ]

        # A note tells the model what was shed (and that describe recovers it) so it does not
        # assume a tool is argument-less. Tier 1 ("schema") needs no note; tier 4 reuses the
        # last (tier 3) text below.
        notes = {
            "args": ['Schemas omitted to fit; use MCP(action="describe", server, tool) for a tool\'s arguments.', ""],
            "names": ['Only tool names shown to fit; use MCP(action="describe", server, tool) before calling.', ""],
        }
        for detail in ("schema", "args", "names"):
            body = self._index_body(configs, detail=detail)
            text = "\n".join(intro + notes.get(detail, []) + body)
            if len(text) <= self.INDEX_TOTAL_LIMIT:
                self.index_truncated = False
                return text

        # Tier 4: even name-only overflows, so some tools are dropped entirely (not just
        # detail). Flag it so the CLI can warn the user — unlike tiers 1-3 these tools are
        # not callable until the index fits (fewer servers, or consult /mcp tools).
        self.index_truncated = True
        return text[: self.INDEX_TOTAL_LIMIT - 10] + "\n... MCP tools truncated; use /mcp tools for full list."

    def _server_lines(self, server: str, tools: list[MCPToolInfo], resources: list[MCPResourceInfo], *, include_schema: bool = True) -> list[str]:
        return server_lines(server, tools, resources, include_schema=include_schema, schema_limit=self.INDEX_SCHEMA_LIMIT)

    def _index_body(self, configs: list[MCPServerConfig], *, detail: str = "schema") -> list[str]:
        return index_body(
            configs,
            detail=detail,
            tools=self.tools,
            resources=self.resources,
            pending_status=self._pending_status,
            schema_limit=self.INDEX_SCHEMA_LIMIT,
        )

    def server_issue(self, name: str) -> tuple[str, str] | None:
        """Classify a server's failure state as (kind, message); error takes precedence over skip."""
        if (error := self.server_errors.get(name)) is not None:
            return "error", error
        if (skip := self.server_skips.get(name)) is not None:
            return "skipped", skip
        return None

    def _pending_status(self, name: str) -> str:
        if issue := self.server_issue(name):
            kind, message = issue
            return message if kind == "error" else "skipped: " + message
        if self.discovering(name):
            return "discovering — tools not loaded yet; retry shortly"
        if self.connected(name):
            return "connected; no tools or resources advertised"
        return "not connected"

    MAX_MENTION_BLOCKS = 50

    async def resolve_mentions(self, text: str) -> str:
        configs = {config.name: config for config in self.parse_configs()}
        if not configs:
            return ""
        lower = {name.lower(): name for name in configs}
        seen: set[tuple[str, str]] = set()
        blocks: list[str] = []
        for span in scan_mentions(text):
            if span.kind not in {"bare", "mcp"} or not span.complete or not span.payload:
                continue
            raw_server, _, raw_tool = span.payload.partition(".")
            name = raw_server if raw_server in configs else lower.get(raw_server.lower())
            if name is None:  # not a configured server — leave the literal @token alone
                continue
            key = (name, raw_tool)
            if key in seen:
                continue
            seen.add(key)
            blocks.append(await self._mention_block(name, raw_tool))
            if len(blocks) >= self.MAX_MENTION_BLOCKS:
                break
        if not blocks:
            return ""
        header = [
            "--- MCP MENTIONS ---",
            'The user explicitly referenced these MCP servers/tools. Prefer them via MCP(action="call", ...) unless clearly irrelevant.',
            "",
        ]
        return "\n".join(header + blocks).strip()

    async def _mention_block(self, server: str, tool: str) -> str:
        if not self.connected(server) and not self.discovering(server):
            await self.discover_server(server)
        if issue := self.server_issue(server):
            kind, message = issue
            return f"[{server}] {'unavailable' if kind == 'error' else 'skipped'}: {message}"
        tools = self.tools.get(server, [])
        resources = self.resources.get(server, [])
        if not tools and not resources:
            return f"[{server}] {self._pending_status(server)}"
        if tool:
            info = self.tool_info(server, tool)
            if info is not None:
                return self._render_describe(server, info)
            available = ", ".join(t.name for t in tools) or "(none)"
            return f"[{server}] tool '{tool}' not found; available: {available}"
        return "\n".join(self._server_lines(server, tools, resources))

    def _format_tool_line(self, server: str, info: MCPToolInfo, *, include_schema: bool = True) -> str:
        return format_tool_line(server, info, include_schema=include_schema, schema_limit=self.INDEX_SCHEMA_LIMIT)

    @classmethod
    def _extract_uris(cls, text: str, limit: int = 5) -> list[str]:
        return extract_uris(text, limit)

    def _tool_args_summary(self, info: MCPToolInfo) -> str:
        return tool_args_summary(info)

    def render_tool_listing(self, server: str | None = None) -> str:
        from wizolt.tools import Tool  # local import: tools is built on top of mcp

        sections: list[str] = []
        configs = self.parse_configs()
        if server:
            config = self.find_config(server)
            if config is None:
                return f"MCP server not found: {server}"
            if not self.connected(server):
                return f"MCP server '{server}' is not connected; run /mcp connect {server}"
            configs = [config]
        elif not configs:
            return "(no MCP servers configured)"
        else:
            configs = [config for config in configs if self.connected(config.name)]
        for config in configs:
            lines = [f"### `{config.name}`", "", "| tool | args | description |", "| --- | --- | --- |"]
            tools = self.tools.get(config.name, [])
            if not tools:
                lines.append("| (none) |  | no tools discovered |")
            else:
                for tool in tools:
                    args_str = self._tool_args_summary(tool)
                    desc = Tool.compact((tool.description or "").split("\n")[0].strip(), 80)
                    lines.append(
                        "| `" + self.markdown_cell(tool.name) + "` | `" + self.markdown_cell(args_str) + "` | " + self.markdown_cell(desc or "-") + " |"
                    )
            resources = self.resources.get(config.name, [])
            if resources:
                lines.extend(["", "| resource | description |", "| --- | --- |"])
                for resource in resources:
                    lines.append("| `" + self.markdown_cell(resource.uri) + "` | " + self.markdown_cell(resource.description or "-") + " |")
            sections.append("\n".join(lines))
        return "\n\n".join(sections) if sections else "(no connected MCP servers)"

    def render_server_status(self) -> str:
        headers = ("server", "mode", "status", "tools", "auth")
        rows: list[tuple[str, ...]] = []
        configs = self.parse_configs()
        for config in configs:
            tools = ""
            if issue := self.server_issue(config.name):
                kind, message = issue
                status = self.STATUS_MARKER + " " + kind + ": " + message
            else:
                if self.connected(config.name):
                    status = self.STATUS_MARKER + " connected"
                    tools = str(len(self.tools.get(config.name, [])))
                else:
                    status = self.STATUS_MARKER + " disconnected"
            auth = []
            if config.auth:
                auth.append(config.auth)
            if config.bearer_token_env_var:
                auth.append("bearer_token_env_var(" + config.bearer_token_env_var + ")")
            auth.extend("env_header(" + name + ")" for name in config.env_http_headers)
            rows.append(
                (
                    "`" + self.markdown_cell(config.name) + "`",
                    "auto" if config.auto_connect else "manual",
                    self.markdown_cell(status),
                    self.markdown_cell(tools or "-"),
                    self.markdown_cell(", ".join(auth) or "-"),
                )
            )
        if not rows:
            return "(no MCP servers configured)"
        widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]

        def table_row(cells: tuple[str, ...]) -> str:
            return "| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells)) + " |"

        separators = tuple("-" * (width - 1) + (":" if index == 3 else "-") for index, width in enumerate(widths))
        lines = [table_row(headers), table_row(separators), *(table_row(row) for row in rows)]
        lines.extend(["", "Manage in the TUI with `/mcp`; fallback: `/mcp connect|disconnect NAME`. Mention `@NAME` to connect on demand."])
        return "\n".join(lines)

    @staticmethod
    def markdown_cell(text: str) -> str:
        return markdown_cell(text)
