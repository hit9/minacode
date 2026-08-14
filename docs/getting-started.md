# Getting started

## Install

- minacode supports <span class="marker">macOS and Linux only</span>
- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) to install and run

```sh
uv tool install minacode
```

### Upgrade

```sh
uv tool upgrade minacode
```

minacode checks PyPI at most once a day and reports an available update at startup and in
`/status`.

## Configure

minacode needs one thing to start: <span class="marker">a provider to talk to</span>. Generate a
starter config:

```sh
minacode --init-config
```

This writes `~/.minacode/config.toml`. Only the `[provider]` block is required; every other
setting has a built-in default, and the file lists the common ones as comments.

### Point it at a provider

minacode speaks to any OpenAI-compatible API (and to Anthropic). Open the config and fill in
a provider — for example [DeepSeek](https://api-docs.deepseek.com/):

```toml
[provider]
active = "default"

[provider.default]
url = "https://api.deepseek.com"
key = "sk-..."
model = "deepseek-v4-flash"
```

| Key | Meaning |
|---|---|
| `url` | Base URL of the API |
| `key` | Your API key |
| `model` | Model name to use |

You can define several `[provider.<name>]` blocks and switch between them with `active` (or
`/provider` inside a session). See [Configuration](configuration.md#providers) for optional
provider, runtime, and data settings.

## Start a session

```sh
minacode
```

Type a request in plain language and the agent starts working — reading files, proposing
edits, running commands. Before anything that changes files or runs a command, it asks for
confirmation (unless you pass `--yolo`). You can keep typing while it works; see
[Follow-ups](usage.md#follow-ups).

Exit with `/exit`, `/quit`, or `Ctrl-D`.

## Your first turn

A turn starts with your request and ends with an answer. In between the agent reads what it
needs, asks before it changes anything, and shows you what it did.

<div class="term-shot" role="img" aria-label="One complete turn: the user's request, two read-only tool lines, a proposed edit shown as a unified diff with old and new line numbers and colored backgrounds for the removed and added rows, an Approve or Refuse action row, the applied edit and a test run storing its output, then the agent's answer and the idle prompt."><span class="fs-user">• fix the tokenizer crash on empty input</span><span> </span><span><span class="fs-i fs-rule">--</span><span class="fs-i fs-glow">-</span><span class="fs-i fs-rule"> </span><span class="fs-i fs-add">●</span><span class="fs-i fs-rule"> </span><span class="fs-i fs-working">working (4s)</span><span class="fs-i fs-rule"> ------------------------------</span></span><span> </span><span class="fs-tool">  Read minacode/parser.py</span><span class="fs-tool">  Search def tokenize minacode/</span><span> </span><span class="fs-tool">  Edit minacode/parser.py</span><span><span class="fs-i fs-dim">      10   10 | </span>def tokenize(text):</span><span class="fs-diff-del"><span class="fs-i fs-dim">      11      | </span><span class="fs-i fs-del">-</span>    first = text[0]</span><span class="fs-diff-add"><span class="fs-i fs-dim">           11 | </span><span class="fs-i fs-add">+</span>    if not text:</span><span class="fs-diff-add"><span class="fs-i fs-dim">           12 | </span><span class="fs-i fs-add">+</span>        return []</span><span><span class="fs-i fs-dim">      12   13 | </span>    return text.split()</span><span><span class="fs-i fs-sel"> Approve </span><span class="fs-dim">   Refuse     Tab to move</span></span><span> </span><span class="fs-tool">  Bash uv run pytest -q</span><span class="fs-dim">    └ stored tr.3</span><span class="fs-output">      41 passed in 2.10s</span><span> </span><span>Empty input returned the first character before the length check. Guarded it and the suite passes.</span><span class="fs-prompt">&gt; <span class="fs-caret">▏</span></span></div>

1. **You ask.** Plain language; no special syntax. `Enter` sends.
2. **It works.** Read-only steps — reading, searching, listing — happen without asking. The
   divider counts the elapsed time, and the status bar below shows the model and context fill.
3. **It asks before changing anything.** An edit arrives as a diff with an action row: `Enter`
   approves, `Tab` moves to `Refuse`, and typing anything writes a reason the agent will read.
   Commands work the same way.
4. **It reports.** The answer lands in your scrollback, and the prompt returns.

From here: `/diff` reviews everything changed so far, `/status` shows where the context stands,
and `/help` lists every command. Your work is saved as you go — close the terminal and
`minacode -c` picks the session back up.

## Command-line flags

| Flag | Effect |
|---|---|
| `-c`, `--last`, `--latest` | Resume the most recent session in this project |
| `--resume [UID]` | Resume a saved session; with no `UID`, resumes this project's latest |
| `--yolo` | Skip confirmation prompts for mutating tools |
| `--theme {auto,light,dark}` | Override the configured terminal color theme |
| `--config <path>` | Use a specific config file instead of `~/.minacode/config.toml` |
| `--init-config` | Write a starter config file and exit |
| `-h`, `--help` | Show command-line help and exit |
| `-v`, `--version` | Print the version and exit |
