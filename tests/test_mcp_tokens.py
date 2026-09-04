"""MCP OAuth token storage runs its JSON transactions off the loop."""

import asyncio
import os
import threading

import pytest

from wizolt.mcp import MCPFileTokenStore


async def test_async_store_roundtrip_delete_and_server_clear(tmp_path):
    store = MCPFileTokenStore(str(tmp_path / "tokens.json"))
    url = "https://example.com/mcp"
    await store.put(store.token_key(url, "/tokens"), {"access_token": "secret", "token_type": "Bearer"}, collection="mcp-oauth-token")
    await store.put(store.token_key(url, "/client_info"), {"client_id": "c"}, collection="mcp-oauth-client-info")

    assert await store.has_server_tokens(url) is True
    assert await store.get(store.token_key(url, "/tokens"), collection="mcp-oauth-token") == {"access_token": "secret", "token_type": "Bearer"}

    await store.delete(store.token_key(url, "/tokens"), collection="mcp-oauth-token")
    assert await store.get(store.token_key(url, "/tokens"), collection="mcp-oauth-token") is None
    assert await store.delete(store.token_key(url, "/tokens"), collection="mcp-oauth-token") is False  # idempotent

    await store.clear_server(url)
    assert await store.has_server_tokens(url) is False
    assert await store.get(store.token_key(url, "/client_info"), collection="mcp-oauth-client-info") is None
    # The file survives with the collections' structure intact.
    assert os.path.isfile(store.path)


async def test_expired_token_is_dropped_on_read(tmp_path):
    store = MCPFileTokenStore(str(tmp_path / "tokens.json"))
    key = store.token_key("https://example.com/mcp", "/tokens")
    await store.put(key, {"access_token": "stale"}, collection="mcp-oauth-token", ttl=-1)

    assert await store.get(key, collection="mcp-oauth-token") is None
    assert await store.has_server_tokens("https://example.com/mcp") is False


async def test_blocked_token_write_does_not_block_an_unrelated_task(tmp_path, monkeypatch):
    """The store's read/modify/write transaction runs on the executor; a token file on a slow disk
    cannot stall the rest of the loop."""
    store = MCPFileTokenStore(str(tmp_path / "tokens.json"))
    entered, release = threading.Event(), threading.Event()
    real_replace = os.replace

    def slow_replace(source, destination):
        entered.set()
        release.wait(5)
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", slow_replace)

    beats = 0

    async def heartbeat():
        nonlocal beats
        while True:
            beats += 1
            await asyncio.sleep(0.001)

    pulse = asyncio.create_task(heartbeat())
    writing = asyncio.create_task(store.put("k", {"v": 1}))
    await asyncio.to_thread(entered.wait, 5)
    await asyncio.sleep(0.02)
    assert beats > 0, "the loop stalled behind an MCP token write"
    release.set()
    await writing
    pulse.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pulse

    assert await store.get("k") == {"v": 1}


async def test_cancelled_token_write_quiesces_and_finishes_the_file(tmp_path, monkeypatch):
    """Cancelling a token write waits for the worker that owns the atomic replace, so the token
    file is never left half-written and no `.tmp` staging file survives."""
    store = MCPFileTokenStore(str(tmp_path / "tokens.json"))
    entered, release = threading.Event(), threading.Event()
    real_replace = os.replace

    def slow_replace(source, destination):
        entered.set()
        release.wait(5)
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", slow_replace)
    writing = asyncio.create_task(store.put("k", {"v": 1}))
    await asyncio.to_thread(entered.wait, 5)
    writing.cancel()
    await asyncio.sleep(0.05)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await writing

    assert not os.path.exists(store.path + ".tmp")
    assert await store.get("k") == {"v": 1}
