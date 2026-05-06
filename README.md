# nanocode

A lightweight terminal-based AI coding assistant.

nanocode is used to help building itself, including features such as `@file` path completion.

## Screenshots

| | |
|---|---|
| ![Screenshot 1](https://raw.githubusercontent.com/hit9/nanocode/master/snapshots/nanocode-snapshot1.png) | ![Screenshot 2](https://raw.githubusercontent.com/hit9/nanocode/master/snapshots/nanocode-snapshot2.png) |

## Features

- **Constrained Output**: Force model replies into auditable action frames.
- **Verified Edits**: Reject stale range edits before they touch files.
- **Autonomous Loop**: Chain reading, editing, running, and verification.
- **Live Telemetry**: Stream tool intent, token use, cost, and status.

## Install

```sh
uv tool install nanocode-cli
```

Upgrade an existing install:

```sh
uv tool upgrade nanocode-cli
```

For local development:

```sh
uv sync --extra dev
uv run nanocode
```

## Usage

Start nanocode:

```sh
nanocode
```

Show available commands:

```text
/help
```

Ask a source-aware question about nanocode itself:

```text
/help how does compact work?
```

## Safety

nanocode does NOT provide sandbox protection. It can run shell commands and edit files in the environment where you start it.

If you do not fully trust the model, tools, prompts, or workspace, run nanocode inside your own sandbox, container, VM, or other isolated environment.

USE AT YOUR OWN RISK.

## Tools

- File: `Read`, `LineCount`, `ListDir`, `Search`.
- Edit: `Edit`, `ReplaceRange`, `BatchReplaceRanges`, `ApplyPatch`.
- Shell: `Bash`, `Git`.
- Memory: `Details` reads hidden detail values by key.

## Commands

- Info: `/help [question]`, `/status`, `/details [clear]`.
- Session: `/compact`.
- Config: `/model`, `/compact-at`, `/reason`, `/reason_effort`, `/stream`, `/yolo`.
- Exit: `/exit`, `/quit`.

## Configuration

- Required: `NANOCODE_API_URL`, `NANOCODE_API_KEY`, `NANOCODE_MODEL`.
- Runtime: `NANOCODE_DIR`, `NANOCODE_TEMPERATURE`, `NANOCODE_STREAM`.
- Reasoning: `NANOCODE_REASONING`, `NANOCODE_REASONING_EFFORT`.
- Limits: `NANOCODE_MODEL_TIMEOUT`, `NANOCODE_SHELL_TIMEOUT`, `NANOCODE_COMPACT_AT`, `NANOCODE_MAX_AGENT_STEPS`.
- Cost: `NANOCODE_PROMPT_PRICE_PER_1M_TOKENS`, `NANOCODE_COMPLETION_PRICE_PER_1M_TOKENS`.

## Status

- Status bar: model, reasoning, context, tokens/cost, elapsed time, and active model-call time.
- `/status`: model, reasoning, stream, yolo, conversation, details, tokens/cost, goal, and verification.
