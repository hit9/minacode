# MCP

minacode can connect to [Model Context Protocol](https://modelcontextprotocol.io) servers and
call their tools through its `MCP` tool. Servers can be **remote** (HTTP) or **local**
(stdio), and <span class="marker">nothing about a server reaches the model until you connect
it</span>.

## Configuring servers

Each server is an `[mcp.<name>]` block.

### Remote (HTTP)

```toml
[mcp.example]
url = "https://example.com/mcp"
bearer_token_env_var = "EXAMPLE_MCP_TOKEN"  # optional: send a bearer token from this env var
# auth = "oauth"                            # optional: use interactive OAuth instead
# auto_connect = true                       # optional: connect at startup (default false)
```

### Local (stdio)

```toml
[mcp.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
# env = { FOO = "bar" }   # optional extra environment variables
# auto_connect = true
```

### Options

| Key | Applies to | Meaning |
|---|---|---|
| `url` | remote | Server endpoint (mutually exclusive with `command`) |
| `command`, `args` | local | Program and arguments to launch |
| `env` | local | Extra environment variables for the subprocess |
| `auth = "oauth"` | remote | Authenticate via interactive OAuth |
| `bearer_token_env_var` | remote | Send `Authorization: Bearer <env var>` |
| `env_http_headers` | remote | Map of HTTP header → env var holding its value |
| `auto_connect` | both | Connect at startup instead of on demand (default `false`) |

## Connecting

Servers are <span class="marker"><strong>manual by default</strong></span> — they stay inactive,
and cost nothing, until you connect them. Set `auto_connect = true` for servers you always
want. Ways to connect:

- **`/mcp`** — open the interactive manager and toggle a server on or off.

```{figure} ../snapshots/minacode-mcp-list.png
:alt: MCP server manager listing all configured servers and their connection status
:width: 600px
:align: center

The /mcp interactive server manager.
```
- **`@server`** in a message — connect on demand. The connection remains active until you
  disconnect it.

```{figure} ../snapshots/minacode-mcp-mention.png
:alt: Using @server mention to connect an MCP server on demand
:width: 600px
:align: center

Connecting a server on demand with an @-mention.
```
- **`/mcp connect <server> [server ...]`** / **`/mcp disconnect <server>`** — terminal
  fallbacks.
- **`/mcp tools [server]`** — list the tools of connected servers.

Connecting several servers in one command runs them concurrently; interactive OAuth browser
flows are serialized so they do not interfere with each other.

Once a server is connected, minacode can use its tools like any other. Tools the server marks
read-only run without a prompt; anything that may change state asks for
{ref}`confirmation <built-in-guardrails>` first.

## Scripting tool calls

When the agent expects several <span class="marker">same-shape MCP calls</span> — the same tool
with a handful of different arguments — it can write one Python script instead of emitting each
call separately, and only what the script prints returns to the conversation. See
[ToolScript](tools.md#toolscript), which covers scripting built-in tools as well.

### Authentication

- **Bearer token** — set `bearer_token_env_var` (or a custom header via `env_http_headers`).
- **OAuth** — set `auth = "oauth"`. Connecting runs the authorization flow; disconnecting
  clears the saved login.

```{admonition} Trust
:class: warning
Local (stdio) servers run programs on your machine, and remote servers receive whatever the
agent sends them. Only connect servers you trust.
```
