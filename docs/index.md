# minacode

A terminal coding agent

minacode works in your terminal: you describe a task, and it reads code, edits files, runs
commands, and reports back. It keeps <span class="marker">stable prompt prefixes</span> so
supported providers can reuse work, maintains a searchable code index, runs background jobs,
tracks its own working notes, and <span class="marker">resumes where you left off</span>.

minacode is the former nanocode, renamed once it outgrew both the single file and the small size
that made it *nano*.

```{figure} ../snapshots/minacode1.gif
:alt: minacode editing code and running tools in one interactive session
:width: 600px
:align: center

Editing code and running tools in one interactive session.
```

```{admonition} Use at your own risk
:class: warning
minacode edits files and runs shell commands in the directory where you start it. It has
**no sandbox of its own**. Run it inside a container, VM, or another isolated environment
when you need isolation. See [Safety](safety.md).
```

## Install and run

```sh
uv tool install minacode
minacode --init-config          # write ~/.minacode/config.toml
# add your provider's url, key, and model to that file
minacode
```

Full walkthrough: [Getting started](getting-started.md).

## What it does

```{figure} ../snapshots/minacode2.gif
:alt: minacode working through a repository task
:width: 600px
:align: center

Working through a repository task in an interactive session.
```

| Area | In short |
|---|---|
| **[Interaction](usage.md)** | Follow-ups, streaming, keys — how you drive the agent. |
| **[Commands](commands.md)** | The `/` command reference: status, models, sessions, MCP. |
| **[Tools](tools.md)** | Read, search, navigate code; edit files; run commands; background jobs; optional provider-side web search. |
| **[Sessions](usage.md#sessions)** | Your work is saved, named, and resumable with `/sessions`, `-c`, or `--resume`. |
| **[MCP](mcp.md)** | Connect external Model Context Protocol servers and use their tools. |
| **[Worker](worker.md)** | Delegate bounded tasks to a second in-process session on its own provider, with context kept until reset. |
| **[Skills](skills.md)** | Load reusable instruction packs on demand. |
| **[Configuration](configuration.md)** | Providers, runtime settings, and data location. |
| **[Compatibility catalog](catalog.md)** | How documented provider/model exceptions are selected, updated, and overridden. |
| **[Context](context.md)** | How the window is filled, summarized when it fills up, and reused by the provider's cache. |
| **[Safety](safety.md)** | What minacode can reach, and how to keep that bounded. |
| **[Troubleshooting](troubleshooting.md)** | What a symptom means and what to do about it. |

```{toctree}
:hidden:
:caption: Guide

getting-started
usage
context
safety
troubleshooting
```

```{toctree}
:hidden:
:caption: Reference

commands
tools
configuration
catalog
worker
mcp
skills
```

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:

changelog
```
