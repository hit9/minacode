# nanocode

A small terminal coding agent in one Python file

nanocode works in your terminal: you describe a task, and it reads code, edits files, runs
commands, and reports back. It keeps a searchable code index, runs background jobs, tracks
its own working notes, and resumes where you left off. Everything ships as a single Python
module, so any behaviour you want to change is one file away.

```{figure} ../snapshots/nanocode1.gif
:alt: nanocode editing code and running tools in one interactive session
:width: 600px
:align: center

Editing code and running tools in one interactive session.
```

```{admonition} Use at your own risk
:class: warning
nanocode edits files and runs shell commands in the directory where you start it. It has
**no sandbox of its own**. Run it inside a container, VM, or another isolated environment
when you need isolation. See [Safety](safety.md).
```

## Install and run

```sh
uv tool install nanocode-cli
nanocode --init-config          # write ~/.nanocode/config.toml
# add your provider's url, key, and model to that file
nanocode
```

Full walkthrough: [Getting started](getting-started.md).

## What it does

```{figure} ../snapshots/nanocode2.gif
:alt: nanocode working through a repository task
:width: 600px
:align: center

Working through a repository task in an interactive session.
```

| Area | In short |
|---|---|
| **[Interactive session](usage.md)** | Describe a task; keep typing while it works to steer or add context. |
| **[What it can do](tools.md)** | Read, search, and navigate code; edit files; run commands and background jobs. |
| **[Sessions](usage.md#sessions)** | Your work is saved and resumable with `-c` or `--resume`. |
| **[MCP](mcp.md)** | Connect external Model Context Protocol servers and use their tools. |
| **[Skills](skills.md)** | Load reusable instruction packs on demand. |
| **[Providers](configuration.md#providers)** | Any OpenAI-compatible API, plus Anthropic. |

```{toctree}
:hidden:

getting-started
usage
tools
mcp
skills
configuration
safety
```

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:

changelog
```
