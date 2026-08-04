# Configuration

minacode reads a single TOML file, `~/.minacode/config.toml` by default. Generate a
commented starter with `minacode --init-config`, or point at another file with
`--config <path>`.

<span class="marker">Only the `[provider]` block is required.</span> Every other key falls back to
a built-in default, so a minimal config is just a provider. Inspect the resolved configuration
at any time with `/config`.

## Providers

minacode supports OpenAI-compatible Chat Completions and Responses APIs, plus the Anthropic
Messages API. Define one or more `[provider.<name>]` blocks and select one with
`[provider] active`:

```toml
[provider]
active = "default"

[provider.default]
url = "https://api.deepseek.com"
key = "sk-..."
model = "deepseek-v4-flash"
```

These three fields are enough for most endpoints. minacode selects the usual protocol and applies
only necessary, documented compatibility adjustments. Explicit settings always take precedence.
Use `/config` to inspect the result.

Define additional blocks to use more providers. Switch between them with `/provider [NAME]`, and
switch the active model with `/model [MODEL]`.

### API protocol

Leave `api = "auto"` unless your endpoint needs an explicit protocol:

| Value | Meaning |
|---|---|
| `auto` | Infer the protocol when possible; otherwise use Chat Completions |
| `chat` | OpenAI-compatible Chat Completions |
| `responses` | OpenAI-compatible Responses |
| `anthropic` | Anthropic-compatible Messages |

A URL ending in `/chat/completions`, `/responses`, or `/messages` also selects that protocol.

### Optional provider settings

Most users can leave these unset.

| Key | Default | Meaning |
|---|---|---|
| `api` | `auto` | API protocol shown above |
| `stream` | `true` | Stream model output; disable for endpoints that reject streaming or Chat `stream_options` |
| `image_input` | `auto` | Image capability: learn automatically, force `on`, or disable with `off` |
| `reasoning` | `medium` | Reasoning effort: `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`; change it during a session with `/reason` |
| `available_models` | — | Additional models shown by `/model` |
| `temperature` | — | Sampling temperature; omitted by default |
| `max_tokens` | `0` | Output-token cap per model request, reasoning included; `0` leaves it to the provider (Anthropic sends a conservative 8K). 16K is still reserved from the input budget for the answer, trading against `max_context_tokens` one for one |
| `timeout` | `120` | Transport inactivity timeout in seconds |
| `response_timeout` | `600` | Total generation limit in seconds; `0` disables it |
| `prompt_cache_key` | `auto` | Stable prompt-cache key; set `off` to omit it |
| `strict_tools` | `false` | Request strict function schemas where supported; toggle with `/strict` |
| `extra_body` | `{}` | Extra fields for an OpenAI-compatible request body |
| `builtin_tools` | `[]` | Tools the provider runs itself, passed through verbatim; see below |
| `chat_reasoning` | `auto` | Provider-specific Chat reasoning format; normally leave on `auto` |

Streaming is enabled by default for all three protocols. If a compatible endpoint does not
support it, set `stream = false` in that provider block, or use `/set provider.stream off` for
the current session.

`timeout` detects a connection that stops delivering data. Streaming reasoning can keep that
timer active indefinitely, so `response_timeout` separately limits the complete model response to
ten minutes by default. Reaching the total limit cancels the request without automatic retries;
set it to `0` only when deliberately allowing unbounded generations.

For provider/model combinations with documented reasoning constraints, minacode maps the selected
effort to the nearest accepted value. Unknown OpenAI-compatible endpoints and model names stay on
the generic path rather than an allowlist; set `api` and `chat_reasoning` explicitly if automatic
selection is wrong. `/config` shows the resolved reasoning effort, while `/status` shows the active
model and cache usage reported by the provider.

## Provider-side tools

Some providers can run web search themselves; see
[Provider-side tools](tools.md#provider-side-tools) for what that looks like in a session. List
the ones you want in `builtin_tools`, written the way your provider documents them:

```toml
[provider]
url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = "qwen3-max"
api = "responses"
builtin_tools = [{ type = "web_search" }, { type = "web_extractor" }]
```

| Provider | Entry |
|---|---|
| OpenAI (Responses) | `{ type = "web_search" }`, optionally with `search_context_size` or `filters` |
| Qwen (Responses) | `{ type = "web_search" }`; also `web_extractor` |
| Anthropic | `{ type = "web_search_20250305", name = "web_search", max_uses = 5 }` |
| Z.AI / BigModel | `{ type = "web_search", web_search = { enable = "True" } }` |
| Kimi / Moonshot | `{ type = "builtin_function", function = { name = "$web_search" } }` |
| OpenRouter | `{ type = "openrouter:web_search" }`; also `openrouter:web_fetch`, `openrouter:datetime` |

One provider configures search elsewhere, through [`extra_body`](#optional-provider-settings):
Qwen's Chat Completions endpoint takes `enable_search`. DeepSeek has no web search.

Builtin tools only work with the APIs shown in the table. If you switch to another API, minacode
keeps the setting but does not send those tools; switching back enables them again. Use `/config`
to check whether they are active. If minacode reports an unsupported entry, compare it with the
example for your provider.

With `image_input = "auto"`, minacode sends attached images using the selected standard API. A
successful image request is remembered for that provider and model during the session; only an
explicit image-not-supported response disables later image submissions. Set `on` or `off` when the
endpoint's capability is already known. Historical images remain readable as text labels after
switching to a provider or model with image input disabled.

## Runtime

Optional; the defaults shown are used when omitted.

| Key | Default | Meaning |
|---|---|---|
| `yolo` | `false` | Start without confirmation prompts |
| `quick_hints` | `true` | Let the model offer selectable next-step chips; toggle with `/hints` |
| `max_context_tokens` | `262144` (256K) | How much of the model's context window to use, which sets the automatic-compaction budget. It is a budget, not the window's size: raise it for a 1M-window model, lower it for a smaller one |
| `max_agent_steps` | `200` | Maximum tool steps in one turn |
| `shell_timeout` | `60` | Maximum shell-command lifetime, in seconds |
| `bash_wait_timeout` | `10` | Foreground wait before a running command becomes a background job; `0` disables promotion |
| `max_parallel_tools` | `4` | Maximum read-only tool calls executed concurrently; `1` disables parallelism |
| `session_retention_days` | `7` | Delete saved sessions untouched for this many days, swept in the background at startup; `0` keeps them indefinitely |
| `theme` | `auto` | Terminal theme: `auto`, `light`, or `dark`; overridden by `--theme` |
| `worker` | `false` | Let the model delegate to a second in-process session; see below |
| `language` | `auto` | Force the reply language (`auto` follows your messages and injects nothing); set a name like `Chinese` to append a fixed `LANGUAGE OVERRIDE` block to the system prompt. Change for the current session with `/language` |

Selected tuning values can be changed for the current session with `/set` (Tab completion
lists the supported keys). `/yolo` toggles `yolo`.

(worker-delegation)=
## Worker delegation

With `[worker] provider` set and `runtime.worker` on, the model gains a `Delegate` tool that hands a
bounded task to a second session **in the same process** (the worker). The worker runs on its own
configured provider — usually a different vendor than `provider.active`, so its reviews
cross-validate the parent's — with its own system prompt and a reduced tool set, and it keeps its
context across delegations until you reset it.

```toml
[worker]
provider = "anthropic"   # worker provider key; unset disables delegation
model = ""               # optional: override the entry's model (inherit by default)
reasoning = ""           # optional: override the entry's reasoning effort (inherit by default)

[runtime]
worker = true            # or /worker on
```

A `Delegate` call runs to completion before the parent's turn continues: you see the worker's
progress in the status bar, and the worker's final text is returned to the parent model, which
decides what to do next. A worker is a full minacode session — compaction, Recall, tools, skills,
confirmation — projected from the parent's state: it shares the parent's cwd, environment, Note
tool, skill library, and MCP servers, but only files and the repo really belong to both.

`/worker` shows status, `/worker on`/`/worker off` toggle `runtime.worker` for the session, and
`/worker reset` clears the worker's context. Turning delegation off or resetting the worker never
removes file changes or merged diffs — the worker owns only its conversation, and reset drops that
conversation (and the wrong beliefs it may have accumulated) on purpose.

The worker's provider, model, and reasoning effort can be changed at runtime, mirroring the
parent's `/provider` `/model` `/reason` temporary switches: `/worker provider NAME` re-targets the
worker (`/worker provider off` clears the entry), `/worker model MODEL` and `/worker reason EFFORT`
override the entry's model and reasoning effort, and `/worker model default` /
`/worker reason default` clear an override. A change applies to a live worker immediately and to
future spawns, and is session-scoped like `/provider` — it is not written back to the config file.
The `Delegate` tool block is fixed at session start from `[worker] provider`: changing the provider
mid-session tunes an already-enabled delegation, but enabling delegation from scratch (provider
unset at start) takes effect after a restart. Set the same defaults persistently with the
`[worker] model` and `[worker] reasoning` keys above.

## Data location

```toml
[paths]
data_dir = "~/.minacode"   # sessions, input history, OAuth tokens, user skills, update cache
```

Sessions live under `<data_dir>/projects/<project>/`, one directory per working directory. Each
holds that project's session logs and a `latest` pointer, so a resume stays scoped to the project
it belongs to. A project directory is removed once its last session expires.

Beside each log sits a small `<uid>.meta.json` holding what the session picker shows — name,
opening line, round count. The log stays the source of truth; deleting a sidecar only costs that
session its label in the list.

`<data_dir>/history.txt` holds the input history that Up and Ctrl-P recall, across every project.
It is capped at 512 KB, keeping the most recent entries.
