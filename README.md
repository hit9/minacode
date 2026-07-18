<h1 align="center">nanocode-cli</h1>

<p align="center">
  A small terminal coding agent I use, maintain, and customize in one Python file —
  reads and edits code, runs commands, resumes sessions, connects MCP servers,
  and loads reusable skills, all from an interactive conversation.
</p>

<p align="center">
  <img src="snapshots/nanocode1.gif" alt="nanocode editing code and running tools" width="600">
</p>

<p align="center"><a href="README.zh-CN.md">中文</a></p>
## Install

Requires macOS or Linux, Python 3.11+, and [uv](https://docs.astral.sh/uv/).

```sh
uv tool install nanocode-cli
nanocode --init-config
```

Add your provider to `~/.nanocode/config.toml`:

```toml
[provider]
active = "default"

[provider.default]
url = "https://api.deepseek.com"
key = "sk-..."
model = "deepseek-v4-flash"
```

Then run:

```sh
nanocode
```

Upgrade with `uv tool upgrade nanocode-cli`.

The full documentation is at [nanocode.readthedocs.io](https://nanocode.readthedocs.io).

## What it is

nanocode does not try to invent a new kind of coding agent. It combines familiar ideas — reading and editing files, running commands, live follow-ups, sessions, diffs, MCP, and skills — into a single Python module I actually use and maintain.

Everything ships in one file, so changing the behavior is one edit away.

<p align="center">
  <img src="snapshots/nanocode2.gif" alt="nanocode resuming a saved session" width="600">
</p>
<p align="center"><sub>Resuming a saved session with its conversation and tool history.</sub></p>

## Highlights

- **Live follow-ups:** type while the agent works; queued input joins the next turn or interrupts the current one.
- **Anchored edits:** structured edits use `line:hash` anchors and reject stale file content.
- **Resumable sessions:** conversation, tool calls, diffs, and working memory survive `-c` or `--resume`.
- **Built-in diff viewer:** `/diff` shows the latest round and the net session result.
- **MCP and skills:** connect Model Context Protocol servers and load Markdown instruction packs on demand.
- **Provider compatibility:** OpenAI-compatible APIs and Anthropic.

## Common commands

| Command | Description |
|---|---|
| `/help` | Command and tool reference |
| `/status` | Runtime status, context, cache, and MCP |
| `/diff` | Review latest edits and session diff |
| `/mcp` | Manage MCP server connections |
| `/model [MODEL]` | Show or switch model |
| `/yolo` | Toggle confirmations |

The full documentation is at [nanocode.readthedocs.io](https://nanocode.readthedocs.io).

## Safety

**Use at your own risk.** nanocode can edit files and run shell commands in the environment where it starts. It does not provide sandbox isolation; use a container or VM when needed.
