"""MCP on the runtime loop: partial discovery, and what shutdown does to work still in flight."""
import asyncio

import pytest
from mcp_harness import mcp_cfg, session

from wizolt.base import ToolError
from wizolt.config import Config
from wizolt.session import Session, bootstrap_features


def two_server_session(tmp_path):
    raw = {
        "mcp": {
            "healthy": {"url": "https://healthy.example/mcp", "auto_connect": True},
            "broken": {"url": "https://broken.example/mcp", "auto_connect": True},
        }
    }
    s = Session(cwd=str(tmp_path), config=Config.from_dict(raw))
    bootstrap_features(s)
    return s


async def test_one_unreachable_server_does_not_cancel_its_healthy_siblings(tmp_path, monkeypatch):
    """Discovery gathers children that each record their own failure.

    A raising child in a TaskGroup would cancel the siblings mid-flight, which is how a single
    unreachable server would take the whole catalog down with it."""
    s = two_server_session(tmp_path)

    async def list_tools(config, _headers):
        if config.name == "broken":
            raise RuntimeError("service unavailable")
        await asyncio.sleep(0.01)  # still in flight when the sibling fails
        return []

    async def list_resources(_config, _headers):
        return []

    monkeypatch.setattr(s.mcp, "_list_tools", list_tools)
    monkeypatch.setattr(s.mcp, "_list_resources", list_resources)

    await s.mcp.discover_auto_async()

    assert s.mcp.connected("healthy")
    assert "service unavailable" in s.mcp.server_errors["broken"]
    assert s.mcp.discovery_status == "ready"


async def test_close_cancels_an_operation_and_waits_for_its_client(tmp_path, monkeypatch):
    """Shutdown owns the operations still in flight: each is cancelled and awaited on this loop.

    A client left to the interpreter's teardown is exactly what the private MCP loop used to hide —
    an HTTP session closing while the default executor is already shutting down."""
    s = session(tmp_path)
    s.config.mcp = mcp_cfg()["mcp"]
    s.mcp.tools["test"] = []
    entered, unwound = asyncio.Event(), asyncio.Event()

    async def slow_call(_config, _headers, _name, _arguments):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            unwound.set()  # the client's `async with` closing
            raise

    monkeypatch.setattr(s.mcp, "_call_tool", slow_call)
    call = asyncio.ensure_future(s.mcp.call_tool_async("test", "echo", {}))
    await asyncio.wait_for(entered.wait(), 2)

    await s.mcp.close_async()

    assert unwound.is_set()
    assert s.mcp._tasks == set()
    with pytest.raises((asyncio.CancelledError, ToolError)):
        await call


async def test_a_closed_manager_refuses_new_operations(tmp_path):
    """After shutdown there is no loop left to own a client, so a late call is refused, not started."""
    s = session(tmp_path)
    s.config.mcp = mcp_cfg()["mcp"]
    s.mcp.tools["test"] = []
    await s.mcp.close_async()

    with pytest.raises(ToolError, match="MCP manager is closed"):
        await s.mcp.call_tool_async("test", "echo", {})


async def test_runtime_shutdown_drains_the_discovery_task(tmp_path, monkeypatch):
    """Discovery is a runtime-owned task, so exiting during it cancels and awaits it.

    The whole point of moving it off its daemon thread: a discovery still opening clients when the
    loop closes is work nobody is watching."""
    from test_tui_runtime_shutdown import run_until, runtime_for

    runtime, command_loop, _tui = runtime_for(tmp_path, monkeypatch)
    started, unwound = asyncio.Event(), asyncio.Event()

    async def discovery():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            unwound.set()
            raise

    monkeypatch.setattr(command_loop, "discover_mcp_async", discovery)

    async def stop_once_discovering():
        await asyncio.wait_for(started.wait(), 2)
        runtime.request_shutdown()

    assert await run_until(runtime, stop_once_discovering) == 0
    assert unwound.is_set()
    assert runtime.tasks == set()
