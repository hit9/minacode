# nanocode

A small terminal coding agent

nanocode works in your terminal: you describe a task, and it reads code, edits files, runs
commands, and reports back. It reuses prompt prefixes (up to 90–99% cache hit rate) to reduce API cost, keeps a
searchable code index, runs background jobs, tracks its own working notes, and resumes
where you left off.

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
| **[Interaction](usage.md)** | Follow-ups, commands, keys — how you drive the agent. |
| **[Tools](tools.md)** | Read, search, navigate code; edit files; run commands; background jobs. |
| **[Sessions](usage.md#sessions)** | Your work is saved and resumable with `-c` or `--resume`. |
| **[MCP](mcp.md)** | Connect external Model Context Protocol servers and use their tools. |
| **[Skills](skills.md)** | Load reusable instruction packs on demand. |
| **[Providers](configuration.md#providers)** | Any OpenAI-compatible API, plus Anthropic. |

```{toctree}
:hidden:

getting-started
configuration
cache
usage
tools
mcp
skills
safety
```

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:

changelog
```
