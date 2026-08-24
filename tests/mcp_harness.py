"""Shared harness for the MCP test modules: session builders, config fixtures, tool-info
factories, and the OAuth token-store helpers."""

from types import SimpleNamespace

from minacode.mcp import MCPToolInfo
from minacode.session import Session, bootstrap_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def session(tmp_path):
    session = Session(cwd=str(tmp_path))
    bootstrap_features(session)
    return session


def mcp_cfg(**overrides) -> dict:
    """Return a full [mcp.x] config dict for one server."""
    cfg = {
        "mcp": {
            "test": {
                "url": "http://localhost:9999/mcp",
                "auto_connect": True,
            }
        }
    }
    server = cfg["mcp"]["test"]
    server.update(overrides)
    return cfg


def mcp_tool_info(server: str, name: str, **kw) -> MCPToolInfo:
    """Create an MCPToolInfo suitable for tests."""
    return MCPToolInfo(
        server=server,
        name=name,
        description=kw.pop("description", "A test tool."),
        input_schema=kw.pop(
            "input_schema",
            {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Input text."}},
                "required": ["text"],
            },
        ),
        annotations=kw.pop("annotations", {}),
        **kw,
    )


def _fake_resource(uri="docs://x.md", name="x", description="A doc", mime="text/markdown"):
    return SimpleNamespace(uri=uri, name=name, description=description, mimeType=mime)
