# nanocode

A small terminal coding agent written in Python.

[简体中文](README.zh-CN.md)

[Chinese blog](https://hit9.dev/post/nanocode)

nanocode is pre-1.0 software. Commands, configuration, and tool behavior may change before a stable release.

![nanocode screenshot](snapshots/nanocode-snapshot.png)

## Features

- **Live turn control**: Add follow-up input while the agent is still working, without losing the current tool flow.
- **File-state brain**: Reads and edits build a current, line-numbered view of the files that matter now.
- **Stale-edit protection**: `line:hash` anchors reject edits when the target code has drifted.
- **Project-aware navigation**: Use the symbol index to jump through outlines, references, implementors, call chains, and changed files quickly.
- **Recoverable context**: Tool output stays bounded in the prompt, while raw `tr.N` results remain recallable.
- **Session recovery**: Resume saved work with `nanocode --resume`, including restored conversation history.
- **Cache-aware context**: Stable sections stay early and noisy working state stays late to improve prompt-cache reuse.
- **Focused working memory**: `Note` separates goal, plan, and known facts from noisy execution logs.
- **MCP integration**: Connect to remote (HTTP) or local (stdio) Model Context Protocol servers and call their tools.
- **Terminal-first workflow**: Model selection, history search, confirmations, live command output, appended input, and status all stay in one CLI.

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
- `update` / `upgrade`: update nanocode to the latest release, using the right installer (uv tool, pipx, or pip) for how it was installed.

During a running turn, type into the `+>` prompt to add follow-up input for the next model request.

## Sessions

nanocode saves recoverable non-empty sessions under `[paths] data_dir` and prints a restore command on exit:

```sh
Resume with: nanocode --resume <session-id>
```

Resume by id, or use the latest saved session:

```sh
nanocode --resume <session-id>
nanocode --resume latest
nanocode --resume last
```

Restored sessions replay the visible conversation history once; tool summaries are shown, raw tool result bodies are not. `/status` shows the active session id. Old session files are cleaned up after `runtime.session_retention_days` days, defaulting to `7`; set it to `0` to disable cleanup.

## CLI

Commands:

- `/help`: show commands and tools.
- `/status`: show runtime status, including the active session id.
- `/context [PATH]`: show the model's context frame — environment, memory (goal, plan, known facts, check), and file state; `PATH` shows that file's current in-context lines.
- `/skills`: list installed skills (load with `Skill(name)` or reference inline with `$name`).
- `/config`: show active config.
- `/api [auto|chat|anthropic]`: show or set provider API format.
- `/debug [on|off]`: toggle model I/O debug traces.
- `/compact`: compact context now.
- `/index [force]`: sync or rebuild the code symbol index.
- `/mcp [tools|login|logout|refresh] ...`: manage MCP servers and tools.
- `/provider [NAME]`: show or set provider.
- `/model [MODEL]`: show or set model.
- `/reason`: choose reasoning effort.
- `/strict`: toggle strict tool-call schemas (OpenAI / DeepSeek only).
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
- Skills: `Skill` loads a skill's full instructions on demand (offered whenever any skill exists — the built-in `nanocode-help` means it is normally always available).

`Read`, `Search`, and `InspectCode` return line anchors where useful. `Edit` uses current `line:hash` anchors to reject stale edits.

## Configuration

Default config location:

```text
~/.nanocode/config.toml
```

Main fields:

- `[provider] active = "name"`
- `[provider.<name>]`: `url`, `key`, `model`, `api`, `prompt_cache_key`, `available_models`, `reasoning`, `chat_reasoning`, `temperature`, `max_tokens`, `strict_tools`, `timeout`
- `[paths] data_dir`
- `[runtime] shell_timeout`, `max_agent_steps`, `max_context_tokens`, `max_parallel_tools`, `check_updates`, `update_check_interval_hours`, `session_retention_days`, `yolo`, `debug`, `tips`

`api = "auto"` chooses between Chat Completions and Anthropic Messages using provider/model profiles. `prompt_cache_key = "auto"` derives a stable key from provider, model, workspace, and tool schema names.

`strict_tools = true` (toggle with `/strict`) constrains tool-call arguments to each tool's JSON schema. It only takes effect on hosts that support it (OpenAI and DeepSeek) and on the Chat Completions path; it is a no-op elsewhere. For DeepSeek, enabling it routes requests to the `/beta` endpoint. Tools whose schemas can't be represented under strict function calling fall back to non-strict automatically.

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

## Skills

Skills are reusable instruction packs the agent can pull in on demand. Each skill is a folder with a `SKILL.md`:

```text
.nanocode/skills/                 # project skills (travel with the repo)
  release-notes/
    SKILL.md
    scripts/
      collect_commits.py
~/.nanocode/skills/               # personal skills (all projects)
```

`SKILL.md` has `name`/`description` frontmatter and a Markdown body of instructions:

```markdown
---
name: release-notes
description: Draft a CHANGELOG entry from commits since the last release.
---
Run `python "{skill_dir}/scripts/collect_commits.py" <last-tag>` to gather commits,
then group them by type and write entries in the house style.
```

- **Discovery**: `.nanocode/skills/` (project) and `~/.nanocode/skills/` (user). On a name clash the project skill wins.
- **How the model sees them**: only a compact `SKILLS` index (name + description) sits in context; the full body is loaded on demand when the model calls `Skill(name)`. A repeated load of the same skill collapses to a pointer so the instructions are not re-billed. When no skills are installed, nothing is added to the prompt.
- **Reference one inline**: type `$name` in your message (Tab-completes) to nudge the model to use that skill; its instructions are injected for that turn.
- **Bundled scripts**: `{skill_dir}` (or `${SKILL_DIR}`) in the body expands to the skill's absolute folder path, so the model can run bundled scripts via `Bash` (subject to normal confirmation unless `/yolo`).
- **Inspect**: `/skills` lists installed skills; the status bar and `/status` show the count.
- **Built-in**: a `nanocode-help` skill ships by default and carries a self-contained manual — authored prose on how to use nanocode, its features, and common problems, plus command/tool/config lists assembled from nanocode's own `/help` text, tool descriptions, and config keys. So "how do I / what does X / why is Y" questions are answered from the manual without searching the source, and the lists can't drift from the running version. Drop a `nanocode-help` skill of your own to override it.

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
|   Memory: goal, plan, known, code index, date    |
+--------------------------------------------------+
| user                                             |
|   FILE STATE: latest Read/Edit file view         |
+--------------------------------------------------+
```

Core rules:

- Environment holds only stable host facts (cwd, os, arch, detected commands) so it stays byte-identical across turns and keeps the conversation prefix cacheable; volatile state like the code index status lives in the late Memory section instead.
- Mid-turn assistant text and appended user input are kept as conversation.
- Earlier conversation is compacted into an explicit summary when the context grows too large.
- FILE STATE is updated by successful `Read` and `Edit` tools and shows current listed file ranges, with recent files first.
- Newer file lines overwrite older lines; edit invalidations clear stale ranges.
- File lines are checked against current file stat or line hash before being shown.
- Successful `Read` and `Edit` tool messages point to FILE STATE instead of repeating file bodies.
- Other tool outputs are bounded in conversation messages and can be recalled by `tr.N`.

## Safety

nanocode can edit files and run shell commands in the environment where it is started. It does not provide sandbox protection. Run it inside your own sandbox, container, VM, or other isolated environment when needed.
