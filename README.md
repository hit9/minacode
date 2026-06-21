# nanocode

A small terminal coding agent written in Python.

[简体中文](README.zh-CN.md)

[Chinese blog](https://hit9.dev/post/nanocode)

nanocode is pre-1.0 software. Commands, configuration, and tool behavior may change before a stable release.

![nanocode screenshot](snapshots/nanocode-snapshot.png)

## Overview

nanocode is a terminal-first coding agent for local development work. It keeps the interaction in one CLI: model selection, history search, confirmations, live command output, queued input, session recovery, and status display.

Core capabilities:

- Live turn control with the `+>` prompt while the agent is still working.
- File-aware context from `Read`, `Search`, `InspectCode`, and `Edit`.
- Stale-edit protection with current `line:hash` anchors.
- Project navigation through the optional code symbol index.
- Recoverable tool results through compact `tr.N` references and `Recall`.
- Focused working memory through `Note`.
- MCP integration for remote HTTP and local stdio servers.
- Append-only session recovery with `nanocode --resume`.

## Install

Install with uv:

```sh
uv tool install nanocode-cli
```

Upgrade:

```sh
uv tool upgrade nanocode-cli
```

For local development:

```sh
uv sync --extra dev
uv run nanocode
```

## Quick Start

Create a config file:

```sh
nanocode --init-config
```

Edit `~/.nanocode/config.toml`, then start:

```sh
nanocode
```

Useful arguments:

- `--config <path>`: use a TOML config file.
- `--init-config`: create a default config file.
- `--resume [UID]`: resume a saved session; without `UID`, resumes `latest`.
- `--yolo`: skip confirmations for mutating tools.
- `--mcp <selector>`: choose which configured MCP servers to enable.
- `--debug`: write model I/O debug traces.
- `-v`, `--version`: show the version.

During a running turn, type into the `+>` prompt to add follow-up input for the next model request.

## Sessions

nanocode saves recoverable sessions under `[paths] data_dir` as append-only JSONL snapshots. Empty sessions are not saved.

On exit, nanocode prints the command needed to restore the session:

```sh
Resume with: nanocode --resume <session-id>
```

Resume a session with:

```sh
nanocode --resume <session-id>
nanocode --resume latest
nanocode --resume last
```

Restored sessions render the conversation history once on startup. Tool execution summaries are shown again, but raw tool result bodies are not printed. `/status` shows the active session id.

Snapshots store only the recovery data nanocode needs: conversation messages, usage, working notes, tool records, and tool errors. Runtime settings, config, git branch, and other rebuildable state are loaded from the current environment/config instead of the snapshot.

Session files older than `runtime.session_retention_days` are removed on startup. The default is `7`; set it to `0` to disable retention cleanup.

## CLI

Commands:

- `/help`: show commands and tools.
- `/status`: show runtime status, including the active session id.
- `/config`: show active config.
- `/api [auto|chat|anthropic]`: show or set provider API format.
- `/debug [on|off]`: toggle model I/O debug traces.
- `/compact`: compact context now.
- `/index [force]`: sync or rebuild the code symbol index.
- `/mcp [tools|login|logout|refresh] ...`: manage MCP servers and tools.
- `/provider [NAME]`: show or set provider.
- `/model [MODEL]`: show or set model.
- `/reason`: choose reasoning effort.
- `/set KEY VALUE`: set supported provider/runtime values for the current session.
- `/yolo`: toggle tool confirmations.
- `/exit`, `/quit`: exit.

Interactive selectors support `j`/`k`, arrows, `/` search, Enter, and Esc. Input supports history, completion, and `Ctrl-R` history search.

Tools:

- File: `Read`, `LineCount`, `List`, `Find`, `Search`.
- Code index: `InspectCode`.
- Edit: `Edit` creates or patches file content.
- Shell: `Bash`, `Git`.
- Tool results: `Recall`.
- Working notes: `Note`.
- Ask the user: `Question`.
- MCP: `MCP`.

`Read`, `Search`, and `InspectCode` return line anchors where useful. `Edit` uses current `line:hash` anchors to reject stale edits.

## Configuration

Default config location:

```text
~/.nanocode/config.toml
```

Main fields:

- `[provider] active = "name"`
- `[provider.<name>]`: `url`, `key`, `model`, `api`, `prompt_cache_key`, `available_models`, `reasoning`, `chat_reasoning`, `temperature`, `timeout`
- `[paths] data_dir`
- `[runtime] shell_timeout`, `max_agent_steps`, `max_context_tokens`, `check_updates`, `update_check_interval_hours`, `session_retention_days`, `yolo`, `debug`

`api = "auto"` chooses between Chat Completions and Anthropic Messages using provider/model profiles. `prompt_cache_key = "auto"` derives a stable key from provider, model, workspace, and tool schema names.

Runtime flags such as `--yolo`, `--debug`, and `--mcp` apply to resumed sessions too. Saved sessions do not carry their old runtime config forward.

## MCP

nanocode connects to [Model Context Protocol](https://modelcontextprotocol.io) servers and exposes their tools through the `MCP` tool. Configure each server under `[mcp.<name>]`. A server is either `url` (remote) or `command` (local), never both.

Remote server over streamable HTTP:

```toml
[mcp.example]
url = "https://example.com/mcp"
bearer_token_env_var = "EXAMPLE_MCP_TOKEN"  # optional; sends Authorization: Bearer
enabled = true

[mcp.oauth_example]
url = "https://example.com/mcp"
auth = "oauth"                              # browser login via /mcp login <server>
enabled = true
```

Local server over stdio:

```toml
[mcp.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
env = { SOME_TOKEN = "value" }              # optional; merged over the inherited environment
enabled = true
```

The HTTP auth options (`auth`, `bearer_token_env_var`, `env_http_headers`) apply to `url` servers only. `env_http_headers` maps a header name to the environment variable holding its value.

Manage servers at runtime:

- `/mcp`: list configured servers and connection status.
- `/mcp tools [server]`: list discovered tools.
- `/mcp refresh [server]`: rediscover servers.
- `/mcp login <server>` / `/mcp logout <server>`: OAuth login and logout.

## Providers

The following providers have been tested with nanocode:

- **deepseek**: DeepSeek API
- **opencode**: OpenCode API
- **aliyun**: Alibaba Cloud Tongyi Qianwen API via Chat Completions
- **llama.cpp**: Local inference via llama.cpp server

## Context Model

Each model request is built manually from explicit messages. Stable context comes first, conversation stays as messages, working memory follows, and the latest file state is appended at the end.

```text
model request
+--------------------------------------------------+
| system                                           |
|   concise agent contract and tool rules          |
+--------------------------------------------------+
| user                                             |
|   Environment                                    |
+--------------------------------------------------+
| user/assistant                                   |
|   conversation, compacted summaries, tools       |
+--------------------------------------------------+
| user                                             |
|   Memory: goal, plan, known, date                |
+--------------------------------------------------+
| user                                             |
|   FILE STATE: latest Read/Edit file view         |
+--------------------------------------------------+
```

Core rules:

- Mid-turn assistant text and appended user input are kept as conversation.
- Earlier conversation is compacted into an explicit summary when the context grows too large.
- FILE STATE is updated by successful `Read` and `Edit` tools and shows current listed file ranges, with recent files first.
- Newer file lines overwrite older lines; edit invalidations clear stale ranges.
- File lines are checked against current file stat or line hash before being shown.
- Successful `Read` and `Edit` tool messages point to FILE STATE instead of repeating file bodies.
- Other tool outputs are bounded in conversation messages and can be recalled by `tr.N`.

## Safety

nanocode can edit files and run shell commands in the environment where it is started. It does not provide sandbox protection. Run it inside your own sandbox, container, VM, or other isolated environment when needed.
