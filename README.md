<h1 align="center">
  <img src="https://raw.githubusercontent.com/hit9/wizolt/master/docs/_static/wizolt-logo.png" alt="Wizolt logo: a blue hooded terminal mage holding a lightning staff" width="64">
  wizolt
</h1>

<p align="center">
  <img src="https://raw.githubusercontent.com/hit9/wizolt/master/snapshots/wizolt1.gif" alt="wizolt editing code and running tools" width="600">
</p>

<p align="center">
  A terminal coding agent I use, maintain, and customize, shipped as a self-contained Python package.
</p>

## Safety

**Use at your own risk.** wizolt can edit files and run shell commands in the environment where it starts. It does not provide sandbox isolation; use a container or VM when needed.

## What it is

wizolt does not introduce a new kind of coding agent. It combines familiar features — reading and editing files, running commands, follow-ups, sessions, diffs, MCP, and skills — into a tool I use personally.

It works on real repositories, including its own: I use wizolt to build and maintain wizolt. Everything ships in one self-contained Python package, so I can change the behavior directly whenever I want the workflow to work differently.

Wizolt is the former minacode, which began as the single-file nanocode. The implementation outgrew both earlier names; the project history remains continuous.

<p align="center">
  <img src="https://raw.githubusercontent.com/hit9/wizolt/master/snapshots/wizolt2.gif" alt="wizolt resuming a saved session" width="600">
</p>
<p align="center"><sub>Resuming a saved session with its conversation and tool history.</sub></p>

## Highlights

- **Worker delegation:** hand a bounded task to a second in-process session on its own provider with `/worker`; the `Delegate` tool keeps worker context across delegations until reset.
- **Forced reply language:** `/language` or `[runtime] language` pins the reply language for the session.
- **Smarter retries:** exponential backoff with jitter and provider `Retry-After`, shown as a live `retrying` phase with a countdown.
- **Prompt-cache aware:** stable request prefixes let supported providers reuse work and can reach 90–99% cache hit rates; `/status` shows the reported result.
- **Code navigation:** jump to definitions, callers, and implementations with a searchable code index.
- **Live follow-ups:** type while the agent works; `Enter` queues a message for the next model step, while `Ctrl-C` discards a draft or interrupts the task once the input is empty.
- **Anchored edits:** structured edits use `line:hash` anchors and reject stale file content.
- **Resumable sessions:** conversation, tool calls, diffs, and working memory survive `-c` or `--resume`.
- **Built-in diff viewer:** `/diff` shows the latest round and the net session result.
- **MCP and skills:** connect Model Context Protocol servers and load Markdown instruction packs on demand.
- **Provider-side web search:** opt in to a provider's own search tool (OpenAI, Qwen, Anthropic, Z.AI) and see each search and its sources in the transcript.
- **Provider compatibility:** OpenAI-compatible APIs and Anthropic.

## Install

Requires macOS or Linux, Python 3.11+, and [uv](https://docs.astral.sh/uv/).

```sh
uv tool install wizolt
wizolt --init-config
```

Add your provider to `~/.wizolt/config.toml`:

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
wizolt
```

Upgrade with `uv tool upgrade wizolt`.

## Links

- [Documentation](https://wizolt.readthedocs.io/en/latest/) — full usage guide and reference.
- [Blog post](https://hit9.dev/post/nanocode) — why and how it was built.
- [code-symbol-index](https://github.com/hit9/code-symbol-index) — the code index library wizolt uses.
