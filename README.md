<h1 align="center">nanocode-cli</h1>

<p align="center">
  <img src="snapshots/nanocode1.gif" alt="nanocode editing code and running tools" width="600">
</p>

<p align="center">
  A coding agent I use, maintain, and customize in one Python file.
</p>

<p align="center"><a href="README.zh-CN.md">中文</a></p>

## Safety

**Use at your own risk.** nanocode can edit files and run shell commands in the environment where it starts. It does not provide sandbox isolation; use a container or VM when needed.

## Install

Requires macOS or Linux, Python 3.11+, and [uv](https://docs.astral.sh/uv/).

```sh
uv tool install nanocode-cli
nanocode --init-config
```

Add your provider to `~/.nanocode/config.toml`:

```toml
[provider]
active = "default"

[provider.default]
url = "https://api.deepseek.com"
key = "sk-..."
model = "deepseek-v4-flash"
```

Then run:

```sh
nanocode
```

Upgrade with `uv tool upgrade nanocode-cli`.

## What it is

nanocode does not introduce a new kind of coding agent. It combines familiar features — reading and editing files, running commands, follow-ups, sessions, diffs, MCP, and skills — into a tool I use personally.

It works on real repositories, including its own: I use nanocode to build and maintain nanocode. Everything ships in one Python module, so I can change the behavior directly whenever I want the workflow to work differently.

<p align="center">
  <img src="snapshots/nanocode2.gif" alt="nanocode resuming a saved session" width="600">
</p>
<p align="center"><sub>Resuming a saved session with its conversation and tool history.</sub></p>

## Highlights

- **Prompt-cache aware:** stable instructions, environment, and tool schemas preserve reusable request prefixes, routinely hitting 98–99% cache rates.
- **Code navigation:** jump to definitions, callers, and implementations with a searchable code index.
- **Live follow-ups:** type while the agent works; queued input joins the next turn or interrupts the current one.
- **Anchored edits:** structured edits use `line:hash` anchors and reject stale file content.
- **Resumable sessions:** conversation, tool calls, diffs, and working memory survive `-c` or `--resume`.
- **Built-in diff viewer:** `/diff` shows the latest round and the net session result.
- **MCP and skills:** connect Model Context Protocol servers and load Markdown instruction packs on demand.
- **Provider compatibility:** OpenAI-compatible APIs and Anthropic.

## Links

- [Documentation](https://nanocode.readthedocs.io) — full usage guide and reference.
- [Blog post](https://hit9.dev/post/nanocode) — why and how it was built.
- [code-symbol-index](https://github.com/hit9/code-symbol-index) — the code index library nanocode uses.
