# nanocode

A coding agent that fits in one file. Describe a task — it reads, edits, runs commands, and reports back.

![nanocode demo](snapshots/nanocode1.gif)

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

## Why nanocode

**Live typing.** Keep giving input while the agent is still working — your next message queues up instead of blocking the turn.

**Edits that don't drift.** Every edit carries a `line:hash` anchor. If the code changes underneath, the edit is rejected — no silent corruption.

**Sessions survive.** Exit anytime and resume later with `nanocode --resume`. Conversation, tool results, and working memory come back.

**Prompt-cache friendly.** Stable context (system prompt, environment, tools) stays byte-identical across turns so providers that cache prompts reuse them.

**One file.** `nanocode.py` is the whole agent — readable, hackable, easy to vendor.

![nanocode session](snapshots/nanocode2.gif)

## At a glance

| | |
|---|---|
| Tools | Read, Search, Edit, Bash, InspectCode, Job, Recall, Note, Ask, MCP, Skill |
| Providers | OpenAI, Anthropic, DeepSeek, OpenRouter, llama.cpp, any Chat-Completions endpoint |
| MCP | Remote (HTTP) and local (stdio) servers, OAuth login |
| Skills | Reusable instruction packs in Markdown; project `.nanocode/skills/` and user `~/.nanocode/skills/` |
| Editing | Structured patch ops (`replace`, `insert_before`, `insert_after`, …) with `line:hash` anchors |
| Sessions | Auto-saved JSONL snapshots, resume by ID or `--resume latest` |
| Index | Code symbol index — jump through outlines, references, call chains via `InspectCode` |
| Config | TOML — `~/.nanocode/config.toml` |

## Commands

`/help` `/status` `/context` `/diff` `/skills` `/config` `/debug` `/compact` `/mcp` `/provider` `/model` `/reason` `/strict` `/set` `/yolo` `/exit`

## Safety

nanocode edits files and runs shell commands. Run it in a sandbox, container, or VM.