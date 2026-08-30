"""The /mcp command surface: subcommands, tab completion, and end-to-end user scenarios."""



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from wizolt.mcp import MCPFileTokenStore


def oauth_value(store: MCPFileTokenStore, url: str, collection: str, suffix: str) -> dict | None:
    entry = store.load().get(collection, {}).get(store.token_key(url, suffix))
    return entry.get("value") if entry else None


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def oauth_store(tmp_path, states: dict[str, str]) -> MCPFileTokenStore:
    """Create a real token store containing one token/client pair per server URL."""
    store = MCPFileTokenStore(str(tmp_path / "mcp-oauth.json"))
    for url, label in states.items():
        put_oauth_state(store, url, label)
    return store


def put_oauth_state(store: MCPFileTokenStore, url: str, label: str) -> None:
    data = store.load()
    data.setdefault("mcp-oauth-token", {})[store.token_key(url, "/tokens")] = {"value": {"access_token": label + "-token", "token_type": "Bearer"}}
    data.setdefault("mcp-oauth-client-info", {})[store.token_key(url, "/client_info")] = {
        "value": {"client_id": label + "-client", "redirect_uris": ["http://localhost:12345/callback"]}
    }
    store.save(data)




# ---------------------------------------------------------------------------
# Tab completion
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# MCPManager — discover_server with nonexistent server
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# render_tools_index truncation
# ---------------------------------------------------------------------------


