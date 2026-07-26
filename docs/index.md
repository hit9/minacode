# minacode

A small terminal coding agent

minacode works in your terminal: you describe a task, and it reads code, edits files, runs
commands, and reports back. It keeps <span class="marker">stable prompt prefixes</span> so
supported providers can reuse work, maintains a searchable code index, runs background jobs,
tracks its own working notes, and <span class="marker">resumes where you left off</span>.

minacode is the former nanocode, renamed once it outgrew the single file that made it *nano*.

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
| **[Interaction](usage.md)** | Follow-ups, commands, keys — how you drive the agent. |
| **[Tools](tools.md)** | Read, search, navigate code; edit files; run commands; background jobs. |
| **[Sessions](usage.md#sessions)** | Your work is saved and resumable with `-c` or `--resume`. |
| **[MCP](mcp.md)** | Connect external Model Context Protocol servers and use their tools. |
| **[Skills](skills.md)** | Load reusable instruction packs on demand. |
| **[Providers](providers.md)** | OpenAI-compatible Chat and Responses APIs, plus Anthropic. |

```{toctree}
:hidden:

getting-started
providers
configuration
context
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
