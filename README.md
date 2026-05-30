# nanocode

A small terminal coding agent written in Python.

nanocode is pre-1.0 software. Commands, configuration, and tool behavior may change before a stable release.

![nanocode screenshot](snapshots/nanocode-snapshot.png)

## Features

- **Snapshot context**: Every turn starts from a clean workspace snapshot instead of a bloated chat replay.
- **File memory**: Recent reads and edits stay available as focused, line-numbered file ranges.
- **Anchored edits**: `line:hash` anchors catch stale edits before they touch the wrong code.
- **Symbol index**: Jump from names to outlines, references, and changed files without searching blindly.
- **Tool recall**: Big outputs are bounded in the prompt but still recallable later by `tr.N`.
- **Terminal-native UI**: Model picking, history search, confirmations, live command output, and status all stay in the terminal.

## Install

```sh
uv tool install nanocode-cli
```

For local development:

```sh
uv sync --extra dev
uv run nanocode
```

## Usage

Start the CLI:

```sh
nanocode
```

Useful arguments:

- `--config <path>`: use a TOML config file.
- `--init-config`: create a default config file.
- `--yolo`: skip confirmations for mutating tools.
- `-v`, `--version`: show the version.

## Commands

- `/help`: show commands and tools.
- `/status`: show runtime status.
- `/config`: show active config.
- `/api [auto|chat|anthropic]`: show or set provider API format.
- `/debug [on|off]`: toggle model I/O debug traces.
- `/compact`: compact context now.
- `/index [force]`: sync or rebuild the code symbol index.
- `/provider [NAME]`: show or set provider.
- `/model [MODEL]`: show or set model.
- `/reason`: choose reasoning effort.
- `/set KEY VALUE`: set provider/runtime values.
- `/yolo`: toggle tool confirmations.
- `/exit`, `/quit`: exit.

Interactive selectors support `j`/`k`, arrows, `/` search, Enter, and Esc. Input supports history, completion, and `Ctrl-R` history search.

## Tools

- File: `Read`, `LineCount`, `List`, `Find`, `Search`.
- Code index: `InspectCode`.
- Edit: `CreateFile`, `Edit`.
- Shell: `Bash`, `Git`.
- Tool results: `Recall`.
- Working notes: `Note`.

`Read`, `Search`, and `InspectCode` return line anchors where useful. `Edit` uses current `line:hash` anchors to reject stale edits.

## Configuration

Run:

```sh
nanocode --init-config
```

Default config location is `~/.nanocode/config.toml`.

Main fields:

- `[provider] active = "name"`
- `[provider.<name>]`: `url`, `key`, `model`, `api`, `prompt_cache_key`, `available_models`, `reasoning`, `chat_reasoning`, `temperature`, `timeout`
- `[paths] data_dir`
- `[runtime] shell_timeout`, `max_agent_steps`, `max_context_tokens`, `yolo`

`api = "auto"` chooses between Chat Completions and Anthropic Messages using provider/model profiles. `prompt_cache_key = "auto"` derives a stable key from provider, model, workspace, and tool schema names.

## Context Design

Each model request is built manually from explicit messages. Stable context comes first, the live conversation stays as messages, and current file content is appended at the end.

```text
model request
+--------------------------------------------------+
| system                                           |
|   concise agent contract and tool rules          |
+--------------------------------------------------+
| user                                             |
|   Environment                                   |
+--------------------------------------------------+
| user/assistant                                  |
|   conversation and bounded tool results         |
+--------------------------------------------------+
| user                                             |
|   Memory: goal, summary, plan, known, date       |
+--------------------------------------------------+
| user                                             |
|   ACTIVE FILE VIEW                              |
+--------------------------------------------------+
```

Core rules:

- ACTIVE FILE VIEW includes the latest active `Read`, `Edit`, and `CreateFile` file ranges.
- Newer file lines overwrite older lines; edit invalidations clear stale ranges.
- File lines are checked against current file stat or line hash before being shown.
- Successful `Read`, `Edit`, and `CreateFile` tool messages point to ACTIVE FILE VIEW instead of repeating file bodies.
- Other tool outputs are bounded in conversation messages.

## Safety

nanocode can edit files and run shell commands in the environment where it is started. It does not provide sandbox protection. Run it inside your own sandbox, container, VM, or other isolated environment when needed.
