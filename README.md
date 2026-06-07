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
- **Project-aware navigation**: Use the symbol index to jump through outlines, references, and changed files quickly.
- **Recoverable context**: Tool output stays bounded in the prompt, while raw `tr.N` results remain recallable.
- **Cache-aware context**: Stable sections stay early and noisy working state stays late to improve prompt-cache reuse.
- **Focused working memory**: `Note` separates goal, plan, and known facts from noisy execution logs.
- **Terminal-first workflow**: Model selection, history search, confirmations, live command output, appended input, and status all stay in one CLI.

## Install

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

During a running turn, the `+>` prompt accepts follow-up input for the next model request.

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
- Edit: `Edit` creates or patches file content.
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


## Tested Providers

The following providers have been tested with nanocode:

- **deepseek**: DeepSeek API
- **opencode**: OpenCode API
- **aliyun**: Alibaba Cloud (Tongyi Qianwen) API via Chat Completions
- **llama.cpp**: Local inference via llama.cpp server

## Context Design

Each model request is built manually from explicit messages. Stable context comes first, conversation stays as messages, working memory follows, and the latest file state is appended at the end.

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
|   conversation, compacted summaries, tools      |
+--------------------------------------------------+
| user                                             |
|   Memory: goal, plan, known, date               |
+--------------------------------------------------+
| user                                             |
|   FILE STATE: latest Read/Edit file view        |
+--------------------------------------------------+
```

Core rules:

- Mid-turn assistant text and appended user input are kept as conversation.
- Earlier conversation is compacted into an explicit summary when the context grows too large.
- FILE STATE is updated by successful `Read` and `Edit` tools and shows current listed file ranges, with recent files first.
- Newer file lines overwrite older lines; edit invalidations clear stale ranges.
- File lines are checked against current file stat or line hash before being shown.
- Successful `Read` and `Edit` tool messages point to FILE STATE instead of repeating file bodies.
- Other tool outputs are bounded in conversation messages and can be recalled by `tr.N`.

## Safety

nanocode can edit files and run shell commands in the environment where it is started. It does not provide sandbox protection. Run it inside your own sandbox, container, VM, or other isolated environment when needed.
