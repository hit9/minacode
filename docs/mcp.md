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
with a handful of different arguments — the `ToolScript` tool lets it write one Python script
instead of emitting each call separately. Only what the script prints returns to the
conversation, so a batch of calls collapses into a single result. Like `MCP`, it appears only
after a server is connected.

### Describing shapes in bulk

`ToolScript(action="describe", tools=["server.tool", ...])` batch-reports each tool's call
shape, return shape, and a json gate:

- `json: yes` — the server declared an `outputSchema`, so `format="json"` below can rely on it.
- `json: unknown` — no declared schema; `format="json"` will try to parse the returned text and
  may fail.

The rendering matches `MCP(action="describe")` with the gate line added, so a shape learned here
is what a scripted call will actually receive.

### Running a script

`ToolScript(action="call", code="...")` runs the code as Python. Inside it, `call()` invokes an
MCP tool:

```python
rows = []
for lang in ("zh", "en", "ja"):
    d = call("MCP", {"server": "orionTemplate", "tool": "get_template_detail",
                     "arguments": {"template_key": KEY, "language": lang}},
             format="json")
    rows.append((lang, d["title"]))
print(rows)  # only this line returns to the conversation
```

- `call(name, args, format="text")` — `name` must be `"MCP"`; built-in tools answer
  `not scriptable yet`. `format="text"` returns the rendered result; `format="json"` returns a
  parsed dict.
- Nested calls go through the <span class="marker">same confirmation, live logging, and `tr.N`
  retention as top-level calls</span>, and they never add extra tool messages — the agent still
  sees exactly one result per call it emitted.
- The result envelope lists the nested `tr.N` keys, so a shortened output can still be recalled.
- Refusing a nested call aborts the script; the rest of the outer batch continues.
- A script that fails returns with a traceback (source lines included) so the agent can correct
  itself.
- Pure script execution is budgeted at about 60 seconds, excluding time spent inside nested
  calls.

### The json gate in practice

`format="json"` prefers the server's declared `structuredContent`. Two errors are possible:

- The tool declared an `outputSchema` but the call returned no `structuredContent` — reported as
  `server declared outputSchema but no structuredContent`.
- No schema was declared and the returned text is not JSON — reported as
  `MCP returned text that is not JSON`.

### Safety

The script is <span class="marker">not sandboxed</span>: it runs with the same privileges as
`Bash`. `ToolScript` asks for confirmation on every `action="call"` and the prompt shows the
script; `--yolo` skips that confirmation exactly as it does for `Bash`.

### Authentication

- **Bearer token** — set `bearer_token_env_var` (or a custom header via `env_http_headers`).
- **OAuth** — set `auth = "oauth"`. Connecting runs the authorization flow; disconnecting
  clears the saved login.

```{admonition} Trust
:class: warning
Local (stdio) servers run programs on your machine, and remote servers receive whatever the
agent sends them. Only connect servers you trust.
```
