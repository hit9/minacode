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

CLI arguments:

- `--yolo`: Skip tool execution confirmations.
- `--debug`: Write request prompts to `.nanocode/debug`.
- `--config <path>`: Path to config file (default: `~/.nanocode/config.toml`).
- `--init-config`: Create a default config file.
- `-v`, `--version`: Show program version.

## Safety

nanocode does NOT provide sandbox protection. It can run shell commands and edit files in the environment where you start it.

If you do not fully trust the model, tools, prompts, or workspace, run nanocode inside your own sandbox, container, VM, or other isolated environment.

USE AT YOUR OWN RISK.

## Tools

- File: `Read`, `LineCount`, `ListDir`, `Search`.
- Edit: `Edit`, `ReplaceRange`, `ApplyPatch`.
- Shell: `Bash`, `Git`.
- Memory: `Recall` reads stored tool results by key.

## Commands

- Info: `/help [question]`, `/status`, `/learn [prompt]`, `/compact`.
- Config: `/config`, `/set <key> <value>`.
- Maintenance: `/clean-logs`.
- Exit: `/exit`, `/quit`.

## Configuration

Run `nanocode --init-config` to create `~/.nanocode/config.toml`.

- API config: `[api]` url and key.
- Model layers: `[main_model]` and `[worker_model]`.
- Explore config: `[explore_agent]` max_turns.
- Runtime config: `[paths]` and `[runtime]`.

## Status

- Status bar: active model, reasoning, context, tool calls, tokens/cost, elapsed time, and active model-call time.
- `/status`: main/worker model state, per-model calls/tokens, total cost, runtime state, goal, and verification.
