"""mcp user scenarios (split from tests/test_mcp_commands.py)."""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from mcp_harness import _fake_resource, as_async
from test_mcp_commands import oauth_store, oauth_value, put_oauth_state

from wizolt.cli import CommandLoop
from wizolt.cli.commands import mcp_command
from wizolt.config import (
    Config,
)
from wizolt.engine import Agent
from wizolt.render import StatusBar
from wizolt.session import Session, bootstrap_features


class TestMCPUserScenarios:
    @staticmethod
    def tool(name, description):
        return SimpleNamespace(
            name=name,
            description=description,
            inputSchema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            annotations=None,
        )

    @staticmethod
    def model():
        class RecordingModel:
            def __init__(self):
                self.requests = []
                self.tools = []

            async def request(self, messages, tools=None):
                self.requests.append(messages)
                self.tools.append(tools or [])
                return {"role": "assistant", "content": "done"}, [], "done"

        return RecordingModel()

    @staticmethod
    def mcp_context(model):
        return next(
            (message["content"] for message in model.requests[-1] if str(message.get("content", "")).startswith("--- MCP TOOLS ---")),
            "",
        )

    @staticmethod
    def wait_until(predicate, timeout=1):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.001)
        return predicate()

    @staticmethod
    async def wait_for(predicate, timeout=2.0):
        """Yield to the loop until `predicate` holds. The timeout bounds a deadlock, not a pace."""
        deadline = time.monotonic() + timeout
        while not predicate():
            assert time.monotonic() < deadline, "condition was not reached"
            await asyncio.sleep(0.001)

    async def test_startup_manual_connect_and_disconnect_update_next_model_request(self, monkeypatch):
        raw = {
            "mcp": {
                "search": {"url": "https://search.example/mcp", "auto_connect": True},
                "docs": {"url": "https://docs.example/mcp"},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)
        tools = {
            "search": [self.tool("find", "Search source code")],
            "docs": [self.tool("lookup", "Look up documentation")],
        }

        async def list_tools(config, _headers):
            return tools[config.name]

        async def list_resources(config, _headers):
            return [_fake_resource(uri="docs://guide.md", description="Project guide")] if config.name == "docs" else []

        monkeypatch.setattr(s.mcp, "_list_tools", list_tools)
        monkeypatch.setattr(s.mcp, "_list_resources", list_resources)
        await s.mcp.discover_auto()
        agent = Agent(s, output_fn=lambda _text: None)
        agent.model = self.model()
        loop = CommandLoop(agent, input_fn=lambda _: "", output_fn=lambda _text: None)

        assert await agent.run("Search the project") == "done"
        assert "[search]" in self.mcp_context(agent.model)
        assert "[docs]" not in self.mcp_context(agent.model)

        assert await mcp_command(loop, "connect docs") == "MCP server connected: docs; tools=1; resources=1"
        assert await agent.run("Read the project guide") == "done"
        context = self.mcp_context(agent.model)
        assert "[search]" in context
        assert "[docs]" in context
        assert "docs://guide.md" in context

        assert await mcp_command(loop, "disconnect search") == "MCP server disconnected: search"
        assert await agent.run("Continue with the documentation") == "done"
        context = self.mcp_context(agent.model)
        assert "[search]" not in context
        assert "[docs]" in context

    async def test_resource_only_mention_connects_server_and_reaches_model(self, monkeypatch):
        raw = {"mcp": {"handbook": {"url": "https://handbook.example/mcp"}}}
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)

        async def no_tools(_config, _headers):
            return []

        async def handbook(_config, _headers):
            return [_fake_resource(uri="handbook://operations.md", description="Operations handbook")]

        monkeypatch.setattr(s.mcp, "_list_tools", no_tools)
        monkeypatch.setattr(s.mcp, "_list_resources", handbook)
        agent = Agent(s, output_fn=lambda _text: None)
        agent.model = self.model()

        assert await agent.run("Use @handbook to check the deployment process") == "done"

        request_text = "\n".join(str(message.get("content", "")) for message in agent.model.requests[-1])
        assert "--- MCP MENTIONS ---" in request_text
        assert "handbook://operations.md" in request_text
        assert "MCP" in {schema["function"]["name"] for schema in agent.model.tools[-1]}

    async def test_reauthorization_replaces_cached_token_and_client_as_one_unit(self, tmp_path, monkeypatch):
        url = "https://metabase.example/mcp"
        raw = {"mcp": {"metabase": {"url": url, "auth": "oauth"}}}
        s = Session(cwd=str(tmp_path), config=Config.from_dict(raw))
        bootstrap_features(s)
        store = oauth_store(tmp_path, {url: "stale"})
        s.mcp._oauth_token_store = store
        authorized = False

        async def list_tools(_config, _headers):
            if not authorized:
                raise RuntimeError("authentication required; run /mcp connect metabase")
            return [self.tool("query", "Query analytics")]

        async def no_resources(_config, _headers):
            return []

        async def authorize(_config, notify=None):
            nonlocal authorized
            assert oauth_value(store, url, "mcp-oauth-token", "/tokens") is None
            assert oauth_value(store, url, "mcp-oauth-client-info", "/client_info") is None
            put_oauth_state(store, url, "fresh")
            authorized = True

        monkeypatch.setattr(s.mcp, "_list_tools", list_tools)
        monkeypatch.setattr(s.mcp, "_list_resources", no_resources)
        monkeypatch.setattr(s.mcp, "_authenticate_oauth", authorize)
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _text: None)
        loop.interactive_input = True

        result = await mcp_command(loop, "connect metabase")

        assert result == "MCP server connected: metabase; tools=1; resources=0"
        assert oauth_value(store, url, "mcp-oauth-token", "/tokens")["access_token"] == "fresh-token"
        assert oauth_value(store, url, "mcp-oauth-client-info", "/client_info")["client_id"] == "fresh-client"

    async def test_noninteractive_connect_preserves_rejected_cached_oauth_state(self, tmp_path, monkeypatch):
        url = "https://metabase.example/mcp"
        raw = {"mcp": {"metabase": {"url": url, "auth": "oauth"}}}
        s = Session(cwd=str(tmp_path), config=Config.from_dict(raw))
        bootstrap_features(s)
        store = oauth_store(tmp_path, {url: "cached"})
        s.mcp._oauth_token_store = store

        async def rejected(_config, _headers):
            raise RuntimeError("authentication required; run /mcp connect metabase")

        monkeypatch.setattr(s.mcp, "_list_tools", rejected)
        monkeypatch.setattr(s.mcp, "_list_resources", rejected)
        monkeypatch.setattr(s.mcp, "_authenticate_oauth", as_async(lambda *_args, **_kwargs: pytest.fail("non-interactive connect opened OAuth")))
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _text: None)

        result = await mcp_command(loop, "connect metabase")

        assert result == "MCP server error: metabase: authentication required; run /mcp connect metabase"
        assert oauth_value(store, url, "mcp-oauth-token", "/tokens")["access_token"] == "cached-token"
        assert oauth_value(store, url, "mcp-oauth-client-info", "/client_info")["client_id"] == "cached-client"

    async def test_oauth_mention_reports_rejection_without_starting_login(self, tmp_path, monkeypatch):
        url = "https://metabase.example/mcp"
        raw = {"mcp": {"metabase": {"url": url, "auth": "oauth"}}}
        s = Session(cwd=str(tmp_path), config=Config.from_dict(raw))
        bootstrap_features(s)
        store = oauth_store(tmp_path, {url: "cached"})
        s.mcp._oauth_token_store = store

        async def rejected(_config, _headers):
            raise RuntimeError("authentication required; run /mcp connect metabase")

        monkeypatch.setattr(s.mcp, "_list_tools", rejected)
        monkeypatch.setattr(s.mcp, "_list_resources", rejected)
        monkeypatch.setattr(s.mcp, "_authenticate_oauth", as_async(lambda *_args, **_kwargs: pytest.fail("mention opened OAuth")))
        agent = Agent(s, output_fn=lambda _text: None)
        agent.model = self.model()

        assert await agent.run("Use @metabase to inspect the dashboard") == "done"

        request_text = "\n".join(str(message.get("content", "")) for message in agent.model.requests[-1])
        assert "[metabase] unavailable: authentication required" in request_text
        assert oauth_value(store, url, "mcp-oauth-client-info", "/client_info")["client_id"] == "cached-client"

    async def test_mixed_batch_reauthorizes_only_rejected_oauth_server(self, tmp_path, monkeypatch):
        valid_url = "https://valid.example/mcp"
        stale_url = "https://stale.example/mcp"
        raw = {
            "mcp": {
                "valid": {"url": valid_url, "auth": "oauth"},
                "stale": {"url": stale_url, "auth": "oauth"},
                "plain": {"url": "https://plain.example/mcp"},
            }
        }
        s = Session(cwd=str(tmp_path), config=Config.from_dict(raw))
        bootstrap_features(s)
        store = oauth_store(tmp_path, {valid_url: "valid", stale_url: "stale"})
        s.mcp._oauth_token_store = store
        refreshed: set[str] = set()
        authorized = []

        async def list_tools(config, _headers):
            if config.name == "stale" and config.name not in refreshed:
                raise RuntimeError("HTTP 401 unauthorized")
            return [self.tool(config.name + "_tool", "Tool for " + config.name)]

        async def no_resources(_config, _headers):
            return []

        async def authorize(config, notify=None):
            authorized.append(config.name)
            assert config.name == "stale"
            assert oauth_value(store, stale_url, "mcp-oauth-client-info", "/client_info") is None
            assert oauth_value(store, valid_url, "mcp-oauth-client-info", "/client_info")["client_id"] == "valid-client"
            put_oauth_state(store, stale_url, "fresh")
            refreshed.add(config.name)

        monkeypatch.setattr(s.mcp, "_list_tools", list_tools)
        monkeypatch.setattr(s.mcp, "_list_resources", no_resources)
        monkeypatch.setattr(s.mcp, "_authenticate_oauth", authorize)
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _text: None)
        loop.interactive_input = True

        result = await mcp_command(loop, "connect valid stale plain")

        assert authorized == ["stale"]
        assert all(s.mcp.connected(name) for name in ("valid", "stale", "plain"))
        assert result.index("`valid`") < result.index("`stale`") < result.index("`plain`")
        assert oauth_value(store, valid_url, "mcp-oauth-client-info", "/client_info")["client_id"] == "valid-client"
        assert oauth_value(store, stale_url, "mcp-oauth-client-info", "/client_info")["client_id"] == "fresh-client"

    async def test_batch_connection_isolates_failed_server_from_model_context(self, monkeypatch):
        raw = {
            "mcp": {
                "catalog": {"url": "https://catalog.example/mcp"},
                "offline": {"url": "https://offline.example/mcp"},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)

        async def list_tools(config, _headers):
            if config.name == "offline":
                raise ConnectionError("service unavailable")
            return [self.tool("search", "Search the catalog")]

        async def no_resources(_config, _headers):
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", list_tools)
        monkeypatch.setattr(s.mcp, "_list_resources", no_resources)
        agent = Agent(s, output_fn=lambda _text: None)
        agent.model = self.model()
        loop = CommandLoop(agent, input_fn=lambda _: "", output_fn=lambda _text: None)

        result = await mcp_command(loop, "connect catalog offline")
        assert "● connected  `catalog`" in result
        assert "● error  `offline` — service unavailable" in result

        assert await agent.run("Search available products") == "done"
        context = self.mcp_context(agent.model)
        assert "[catalog]" in context
        assert "[offline]" not in context

    async def test_batch_command_reports_live_progress_until_every_server_finishes(self, monkeypatch):
        raw = {
            "mcp": {
                "alpha": {"url": "https://alpha.example/mcp"},
                "beta": {"url": "https://beta.example/mcp"},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)
        started = {name: threading.Event() for name in ("alpha", "beta")}
        release = {name: threading.Event() for name in ("alpha", "beta")}

        async def list_tools(config, _headers):
            started[config.name].set()
            while not release[config.name].is_set():
                await asyncio.sleep(0.001)
            return [self.tool("run", "Run workflow")]

        async def no_resources(_config, _headers):
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", list_tools)
        monkeypatch.setattr(s.mcp, "_list_resources", no_resources)
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _text: None)
        # The status row has no animation of its own, but each render reads the current connected
        # count while both servers connect concurrently on this loop.
        connecting = asyncio.ensure_future(mcp_command(loop, "connect alpha beta"))
        await self.wait_for(lambda: all(event.is_set() for event in started.values()))
        assert "mcp 0" in "".join(text for _, text in StatusBar(s).fragments())

        release["alpha"].set()
        await self.wait_for(lambda: s.mcp.connected("alpha"))
        assert s.mcp.discovery_status == "discovering"
        assert "mcp 1" in "".join(text for _, text in StatusBar(s).fragments())

        release["beta"].set()
        result = await connecting

        assert s.mcp.discovery_status == "ready"
        assert "mcp 2" in "".join(text for _, text in StatusBar(s).fragments())
        assert result and "`alpha`" in result and "`beta`" in result
