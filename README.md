# nanocode

A lightweight terminal-based AI coding assistant.

nanocode is used to help building itself, including features such as `@file` path completion.

Pre-1.0 note: nanocode is still evolving quickly. Functionality, commands, configuration, and behavior may change incompatibly before a 1.0 release.

## Screenshots

| | |
|---|---|
| ![Screenshot 1](https://raw.githubusercontent.com/hit9/nanocode/master/snapshots/nanocode-snapshot1.png) | ![Screenshot 2](https://raw.githubusercontent.com/hit9/nanocode/master/snapshots/nanocode-snapshot2.png) |

## Features

- **Function Tools**: Route model decisions through auditable tools.
- **Verified Edits**: Reject stale range edits before they touch files.
- **Autonomous Loop**: Chain reading, editing, running, and verification.
- **Live Telemetry**: Stream tool intent, token use, and status.

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
- `--plan`: Plan changes without editing files or running commands.
- `--debug`: Write request prompts to the current session directory under `~/.nanocode/sessions/`.
- `--config <path>`: Path to config file (default: `~/.nanocode/config.toml`).
- `--init-config`: Create a default config file.
- `-v`, `--version`: Show program version.

## Safety

nanocode does NOT provide sandbox protection. It can run shell commands and edit files in the environment where you start it.

If you do not fully trust the model, tools, prompts, or workspace, run nanocode inside your own sandbox, container, VM, or other isolated environment.

USE AT YOUR OWN RISK.

nanocode currently targets macOS and Linux. Windows is not supported.

## Tools

- File: `Read`, `LineCount`, `List`, `Search`.
- Codebase: `CodeGraph` when the local `codegraph` CLI is installed and initialized.
- Edit: `Edit`, `ReplaceRange`.
- Shell: `Bash`, `Git`.
- Memory: `Recall` reads stored tool results by key.

## Commands

- Info: `/help [question]`, `/status`, `/rules`, `/compact`.
- Config: `/config`, `/set <key> <value>`, `/api [auto|chat|responses]`, `/model [model_name]`, `/reason`, `/reason-payload [value]`, `/provider [name]`, `/plan [on|off|question]`, `/yolo`.
- CodeGraph: `/codegraph [status|sync|init|index]`.
- Maintenance: `/clean`.
- Exit: `/exit`, `/quit`.

Selectors support `j`/`k`, arrows, `/keyword`, Enter, and Esc. `/api responses` switches the current provider to Responses format. `/reason` sets `provider.reasoning` to `off` or an effort value; `/reason-payload` controls the Chat-only reasoning payload shape. `/model` lists configured models before discovered ones, then prompts for reasoning.
During a slow model request, press `Ctrl-G` to cancel that request and resend the same prompt.

## Configuration

Run `nanocode --init-config` to create `~/.nanocode/config.toml`.

- Provider config: `[provider] active = "<name>"` plus `[provider.<name>]` url, key, model, `available_models`, and model options. `api` selects `chat`, `responses`, or `auto`; auto uses exact-host profiles. Responses uses standard `reasoning.effort`; Chat reasoning is mapped by provider/model profile when known.
- Provider auto-detection covers common providers: OpenAI/OpenRouter prefer Responses API; DeepSeek, selected OpenCode models, and DashScope models use their matching Chat reasoning payload shapes.
- Path config: `[paths] data_dir = "~/.nanocode"`.
- Runtime config: `[runtime]`.
- `/context [low|medium|high]` shows or switches tool-result context budgets; lower budgets reduce token usage and observe overhead.
- Session data: debug prompts and tool-result logs are stored under `~/.nanocode/sessions/<session_id>/`.
- Old inactive session directories are auto-cleaned after `runtime.auto_clean_recent` (default `1d`; use `off` to disable). `/clean` removes inactive sessions immediately.
- Project data: user rules are stored under `~/.nanocode/projects/<project_key>/`.

## Status

- Status bar: active model, reasoning, active yolo/plan modes, conversation context, current-turn tool calls, tokens, elapsed time, and active model-call time.
- `/status`: active provider, model state, session id, runtime state, conversation/tool counters, per-model calls/tokens, task, goal, and verification.
