<h1 align="center">nanocode-cli</h1>

<p align="center">
  A coding agent I use, maintain, and customize in one Python file.
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

Configure any OpenAI-compatible provider in `~/.nanocode/config.toml`. For example, [DeepSeek Flash](https://api-docs.deepseek.com/):

```toml
[provider]
active = "default"

[provider.default]
url = "https://api.deepseek.com"
key = "API_KEY"
model = "deepseek-v4-flash"
```

Replace `url`, `key`, and `model` with the values from your provider.

Then start:

```sh
nanocode
```

Upgrade: `nanocode upgrade`

Common flags:

- `-c`, `--last`, `--latest`: resume the latest session in the current project
- `--resume [UID]`: resume a saved session; without UID, resumes latest
- `--yolo`: skip tool confirmations
- `--mcp <selector>`: choose which MCP servers to enable
- `--config <path>`: use a specific TOML config file

## A working personal agent

nanocode does not introduce a new kind of coding agent. It combines familiar features—reading and editing files, running commands, follow-ups, sessions, diffs, MCP, and skills—into a tool I use personally.

It works on real repositories, including its own: I use nanocode to build and maintain nanocode. Everything ships in one Python module, so I can change the behavior directly whenever I want the workflow to work differently.

<p align="center">
  <img src="snapshots/nanocode2.gif" alt="nanocode session" width="600">
</p>
<p align="center"><sub>Resuming a saved session with its conversation and tool history.</sub></p>

## Highlights

- **Live follow-ups:** type while the agent works; queued input joins the next model request or can interrupt the current one.
- **Anchored edits:** structured edits use `line:hash` anchors and reject stale file content instead of guessing.
- **Resumable sessions:** conversation, completed tool calls, diffs, and working memory survive interruption and `-c` or `--resume`.
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

Run `/help` for every command and tool. Interactive selectors support `j`/`k`, arrows, `/` search, Enter, and Esc; input supports history completion, `Ctrl-R` search, and `Ctrl-X` to edit the current input in `$EDITOR` (falling back to vim).

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
