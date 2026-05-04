# nanocode


## Screenshots

| | |
|---|---|
| ![Screenshot 1](https://raw.githubusercontent.com/hit9/nanocode/main/snapshots/nanocode-snapshot1.png) | ![Screenshot 2](https://raw.githubusercontent.com/hit9/nanocode/main/snapshots/nanocode-snapshot2.png) |

## Install

```sh
uv tool install nanocode-cli
```

For local development:

```sh
uv sync --extra dev
uv run nanocode
```

## Environment Variables

Required:

```sh
export NANOCODE_API_URL="https://api.example.com/v1/chat/completions"
export NANOCODE_API_KEY="your-api-key"
export NANOCODE_MODEL="your-model"
```

Optional:

```sh
export NANOCODE_DIR=".nanocode"
export NANOCODE_TEMPERATURE="0.7"
export NANOCODE_REASONING="on"
export NANOCODE_REASONING_EFFORT="medium"
export NANOCODE_MODEL_TIMEOUT="60"
export NANOCODE_SHELL_TIMEOUT="60"
export NANOCODE_COMPACT_AT="100"
```

## Safety

nanocode does not provide sandbox protection. It can run shell commands and edit files in the environment where you start it.

If you do not fully trust the model, tools, prompts, or workspace, run nanocode inside your own sandbox, container, VM, or other isolated environment.

Use at your own risk.
