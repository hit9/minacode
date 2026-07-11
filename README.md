# nanocode-cli

<img src="snapshots/nanocode1.gif" alt="nanocode demo" width="680">

A coding agent that fits in one file. Describe a task — it reads, edits, runs commands, and reports back.

[中文](README.zh-CN.md)

## Install

```sh
uv tool install nanocode-cli
```

Create your config and start:

```sh
nanocode --init-config
# edit ~/.nanocode/config.toml → set provider.url, provider.key, provider.model
nanocode
```

Upgrade: `uv tool upgrade nanocode-cli`

Common flags:

- `--resume [UID]`: resume a saved session; without UID, resumes latest
- `--yolo`: skip tool confirmations
- `--mcp <selector>`: choose which MCP servers to enable
- `--config <path>`: use a specific TOML config file

## Why nanocode

**Live typing.** Keep giving input while the agent is still working — your next message queues up instead of blocking the turn.

**Edits that don't drift.** Every edit carries a `line:hash` anchor. If the code changes underneath, the edit is rejected — no silent corruption.

**Sessions survive.** Exit anytime and resume later with `nanocode --resume`. Conversation, tool results, and working memory come back.

**Prompt-cache friendly.** Stable context (system prompt, environment, tool schemas) stays byte-identical across turns so providers that cache prompts reuse them, saving cost and latency.

**One file.** `nanocode.py` is the whole agent — readable, hackable, easy to vendor.

<img src="snapshots/nanocode2.gif" alt="nanocode session" width="680">

## At a glance

| | |
|---|---|
| Tools | Read, Search, Edit, Bash, InspectCode, Job, Recall, Note, Ask, MCP, Skill |
| Providers | OpenAI, Anthropic, DeepSeek, OpenRouter, llama.cpp, and any Chat-Completions endpoint |
| MCP | Remote (HTTP streamable) and local (stdio) servers, OAuth support |
| Skills | Reusable instruction packs (Markdown); project `.nanocode/skills/` and user `~/.nanocode/skills/` |
| Editing | Structured patch ops (`replace`, `insert_before`, `insert_after`, …) with `line:hash` anchors |
| Sessions | Auto-saved JSONL snapshots, `--resume latest` / `--resume <id>` |
| Index | Code symbol index — outline, references, implementors, call chains (`InspectCode`) |
| Context | cache-stable prefix + conversation + Memory (goal/plan/known/check) three-part layout |
| Config | TOML — `~/.nanocode/config.toml` |

## Commands

| Command | Description |
|---|---|
| `/help` | Show commands and tools |
| `/status` | Runtime status: token usage, context %, cache hit rate |
| `/context` | Model context frame — environment and memory (goal/plan/known/check) |
| `/diff` | Latest edit diffs and net session diff (interactive, tabbed) |
| `/skills` | List installed skills |
| `/config` | Show active config |
| `/debug` | Last three in-memory diagnostics (cache-prefix mismatches, etc.) |
| `/compact` | Compact context now |
| `/mcp` | Manage MCP servers and tools |
| `/provider [NAME]` | Show or switch provider |
| `/model [MODEL]` | Show or switch model |
| `/reason` | Adjust reasoning effort |
| `/strict` | Toggle strict tool-call schemas (OpenAI / DeepSeek) |
| `/set KEY VALUE` | Set a config value for this session |
| `/yolo` | Toggle tool confirmations |
| `/exit`, `/quit` | Exit |

Interactive selectors support `j`/`k`, arrows, `/` search, Enter, Esc. Input supports history completion and `Ctrl-R` search.

## Configuration

Config file: `~/.nanocode/config.toml`

Key sections:

- `[provider] active = "name"`
- `[provider.<name>]`: `url`, `key`, `model`, `api`, `prompt_cache_key`, `reasoning`, `temperature`, `max_tokens`, `strict_tools`, `timeout`
- `[paths] data_dir`
- `[runtime]`: `shell_timeout`, `max_agent_steps`, `max_context_tokens`, `max_parallel_tools`, `session_retention_days`, `yolo`, `tips`

`api = "auto"` picks between Chat Completions and Anthropic Messages based on provider/model profiles. `prompt_cache_key = "auto"` derives a stable key from provider, model, workspace, and tool schema names.

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
