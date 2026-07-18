<h1 align="center">nanocode-cli</h1>

<p align="center">
  A small terminal coding agent, implemented in one Python module you can read and change.
</p>

<p align="center">
  <img src="snapshots/nanocode1.gif" alt="nanocode editing code and running tools" width="600">
</p>

<p align="center"><a href="README.zh-CN.md">中文</a></p>

nanocode reads and edits code, runs commands, resumes sessions, connects MCP servers, and
loads reusable skills—all from an interactive terminal conversation.

## Requirements

- macOS or Linux
- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

Native Windows is not supported. Use nanocode inside
[Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/windows/wsl/) instead.

## Install

```sh
uv tool install nanocode-cli
nanocode --init-config
```

Add your provider URL, API key, and model to `~/.nanocode/config.toml`, then start:

```sh
nanocode
```

Upgrade with `uv tool upgrade nanocode-cli`.

## Highlights

- Live follow-ups while the agent works
- Anchored edits that reject stale file content
- Resumable sessions and a built-in diff viewer
- OpenAI-compatible and Anthropic providers
- On-demand MCP servers and Markdown skills
- One Python module for straightforward customization

<p align="center">
  <img src="snapshots/nanocode2.gif" alt="nanocode resuming a saved session" width="600">
</p>
<p align="center"><sub>Resuming a saved session with its conversation and tool history.</sub></p>

## Documentation

The full documentation is at [nanocode.readthedocs.io](https://nanocode.readthedocs.io).
## Safety

**Use at your own risk.** nanocode can edit files and run shell commands in the environment
where it starts. It does not provide sandbox isolation; use a container or VM when needed.
