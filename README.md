<h1 align="center">nanocode-cli</h1>

<p align="center">
  A single-file terminal coding agent with explicit control flow and cache-aware sessions.
</p>

<p align="center">
  <img src="snapshots/nanocode1.gif" alt="nanocode demo" width="600">
</p>
<p align="center"><sub>Editing code and running tools in one interactive session.</sub></p>

<p align="center"><a href="README.zh-CN.md">中文</a></p>

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv tool install nanocode-cli
nanocode --init-config
```

Edit `~/.nanocode/config.toml` to use [DeepSeek Flash](https://api-docs.deepseek.com/):

```toml
[provider]
active = "default"

[provider.default]
url = "https://api.deepseek.com"
key = "YOUR_DEEPSEEK_API_KEY"
model = "deepseek-v4-flash"
```

Then start:

```sh
nanocode
```

Upgrade: `uv tool upgrade nanocode-cli`

Common flags:

- `--resume [UID]`: resume a saved session; without UID, resumes latest
- `--yolo`: skip tool confirmations
- `--mcp <selector>`: choose which MCP servers to enable
- `--config <path>`: use a specific TOML config file

## What it is

nanocode is the coding agent I use day to day. It runs in the terminal, handles the usual read-edit-run loop, and lives in a single Python file that I can change whenever I want the workflow to behave differently.

It is not a minimal code sample, but the implementation stays direct: one agent loop, explicit state, plain tool calls, and no plugin framework at the center.

<p align="center">
  <img src="snapshots/nanocode2.gif" alt="nanocode session" width="600">
</p>
<p align="center"><sub>Resuming a saved session with its conversation and tool history.</sub></p>

## Highlights

- **Live follow-ups:** type while the agent works; queued input joins the next model request or can interrupt the current one.
- **Anchored edits:** structured edits use `line:hash` anchors and reject stale file content instead of guessing.
- **Resumable sessions:** conversation, completed tool calls, diffs, and working memory survive interruption and `--resume`.
- **Built-in diff viewer:** `/diff` shows changes from the latest user round and the net session result.
- **Prompt-cache aware:** stable instructions, environment, and tool schemas preserve reusable request prefixes.
- **Provider compatibility:** tested with DeepSeek, OpenCode, Alibaba Cloud, and ZenMux; other OpenAI-compatible endpoints should work in principle. Anthropic APIs, remote/local MCP servers, and Markdown skills are also supported.

## Common commands

| Command | Description |
|---|---|
| `/help` | Show the complete command and tool reference |
| `/status` | Runtime status: token usage, context %, cache hit rate |
| `/context` | Model context frame — environment and memory (goal/plan/known/check) |
| `/diff` | Latest edit diffs and net session diff (interactive, tabbed) |
| `/compact` | Compact context now |
| `/mcp` | Manage MCP servers and tools |
| `/model [MODEL]` | Show or switch model |
| `/yolo` | Toggle tool confirmations |

Run `/help` for every command, tool, and shortcut. Interactive selectors support `j`/`k`, arrows, `/` search, Enter, and Esc; input supports history completion and `Ctrl-R` search.

## Configuration

Config file: `~/.nanocode/config.toml`

The generated file documents common provider and runtime options. Multiple `[provider.<name>]` sections are supported; select one with `[provider] active = "name"`. Use `/config` to inspect the active configuration and `/help` for runtime commands.

## MCP

Connect to [Model Context Protocol](https://modelcontextprotocol.io) servers and expose their tools through `MCP`.

Remote server (HTTP):

```toml
[mcp.example]
url = "https://example.com/mcp"
bearer_token_env_var = "EXAMPLE_MCP_TOKEN"  # optional
enabled = true
```

Local server (stdio):

```toml
[mcp.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
enabled = true
```

Runtime management: `/mcp` to view status, `/mcp tools [server]` to list tools, `/mcp login/logout <server>` for OAuth.

## Skills

Skills are instruction packs the agent loads on demand. Each skill is a folder with a `SKILL.md`.

- **Discovery**: `.nanocode/skills/` (project) and `~/.nanocode/skills/` (user); project wins on name clash
- **Model view**: only a name + description index sits in context; the full body loads on `Skill(name)` call
- **Inline reference**: type `$name` in a message (Tab-completes) to inject that skill's instructions
- **Bundled scripts**: `{skill_dir}` expands to the skill's absolute path for running via `Bash`
- **Built-in**: ships with a `nanocode-help` skill containing a self-contained manual and auto-generated command/tool/config lists

## Safety

**Use at your own risk.** nanocode can edit files and run shell commands in the environment where it is started. It does not provide sandbox protection. Run it inside your own sandbox, container, VM, or other isolated environment when needed.
