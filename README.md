<h1 align="center">nanocode-cli</h1>

<p align="center">
  A compact, cache-friendly coding agent for developers who want an inspectable CLI.
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

Edit `~/.nanocode/config.toml` with an OpenAI-compatible endpoint:

```toml
[provider]
active = "default"

[provider.default]
url = "https://api.openai.com/v1"
key = "YOUR_API_KEY"
model = "gpt-5"
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

## Why nanocode

nanocode keeps the agent loop small enough to understand while retaining the workflows needed for real repository work.

**Inspectable by design.** The complete agent lives in `nanocode.py`. You can read how prompts, tools, approvals, persistence, and model requests work without tracing a framework.

**Changes fail safely.** Edits use `line:hash` anchors. If a file changes after it was read, the edit is rejected instead of being applied to stale content.

**Keep steering while it works.** Type follow-ups during a running turn. They queue for the next model request, so you can refine direction without waiting for the entire task to finish.

**Resume without losing the thread.** Sessions persist conversation, completed tool calls, diffs, and working memory. Stop at any time and continue with `nanocode --resume`.

**Built for prompt caching.** Stable instructions, environment data, and tool schemas remain byte-identical across requests, improving cache reuse on providers that support it.

<p align="center">
  <img src="snapshots/nanocode2.gif" alt="nanocode session" width="600">
</p>
<p align="center"><sub>Resuming a saved session with its conversation and tool history.</sub></p>

## At a glance

| | |
|---|---|
| Providers | OpenAI, Anthropic, DeepSeek, OpenRouter, llama.cpp, and any Chat-Completions endpoint |
| Editing | Structured patch ops (`replace`, `insert_before`, `insert_after`, …) with `line:hash` anchors |
| Sessions | Auto-saved JSONL snapshots, `--resume latest` / `--resume <id>` |
| MCP | Remote (HTTP streamable) and local (stdio) servers, with OAuth support |
| Skills | Reusable Markdown instruction packs from project and user directories |
| Architecture | The complete agent lives in the inspectable `nanocode.py` module |

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
