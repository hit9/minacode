# Changelog

## Unreleased

### Added

- `Edit` can now patch a file straight from exact text seen anywhere -- `Bash` output included --
  by giving `old` (the exact original text) and `content` instead of `source=view.N` and line
  numbers, which saves the `Read` that used to sit between finding a line and changing it. The
  text must appear exactly once in the file; a target that is missing, repeated, or overlaps
  another target in the same call refuses the whole call and writes nothing, and a repeated one
  comes back with bounded context around its occurrences. Source views still work unchanged, are
  still preferred when one is in hand, and one path cannot use both kinds of evidence in the same
  batch.

### Fixed

- Agent guidance now tells models to emit every ready tool call in one response and wait only for
  true data dependencies. Shorter role, tool, and compaction prompts remove conflicting serial
  instructions and stale claims that shell-located text requires another `Read` before `Edit`.
- Edit matching and batch planning no longer block the event loop on large files.
- The status bar now keeps one static, semantically colored layout while work is running:
  `[yolo] model · level | mcp N · skills N | ctx N% · cache N% | index*`. Its facts still
  refresh, but it no longer sweeps, swaps to worker or compaction details, or starts a dedicated
  repaint thread in the simple frontend.
- The working divider now leaves a blank row above queued follow-up inputs, so the queue reads
  as its own region below the boundary instead of a list glued to the divider's label.
- The `Ask` modal opens with a blank row above the question, matching every other selector, so
  the question no longer butts straight against the activity region above it.
- A very short pane no longer makes the `Ask` modal render nearly every option: when the title,
  footer, and gaps leave no room for the rows, the rows are dropped instead of the list being
  sliced from the end. A search with no matches still shows its query and follows the same height
  limit.
- Phase rules and `/status`-style compact command output keep one blank row at each boundary, so
  the transcript's seams and a command's table no longer butt straight against what follows them.
- The agent prompt now asks for a blank line between Markdown blocks (paragraphs, lists,
  headings, code fences), avoiding dense stacking and over-sectioning.
- `Edit` tolerates a provider omitting `op` only when `source`, `start`, `end`, and explicit
  replacement content make `replace` unambiguous; incomplete create/delete shapes remain errors.
- A rejected multi-range `Edit` now returns one fresh editable view covering every requested range,
  with a direct-retry hint, so recovery does not require reading the same lines again.
- `Note` ignores a redundant `fields` projection on an otherwise explicit update instead of
  rejecting the working-state change.

### Added

- `ToolScript` scripts can hand a batch of independent calls over at once with
  `call_many([(name, args), ...])`, which returns one value per entry in the order given and runs
  read-only, auto-approved entries concurrently (up to `max_parallel_tools`). It follows the same
  rule the tool runner uses for a batch from the model: only consecutive parallel-safe calls
  overlap, so a write or a confirmation never runs beside a concurrent read, and every result and
  its log line lands in the order the script listed them. A failed entry comes back as the error
  object instead of ending the script — a loop of `call()` needs a try/except per item for that —
  while a refusal still ends it. Scripts should not start threads of their own: the execution time
  budget is measured on the script's own thread and the runner's bookkeeping is not thread-safe.

### Changed

- Colors now come from one semantic palette per appearance instead of ANSI names scattered through
  the renderer, and both built-in schemes were retuned around it. The greys are warm, so the
  interface no longer reads as washed out; the accent is a single teal note worn by few things
  instead of blue on blue; code highlighting is warm too; hints and other supporting text are grey;
  rules are visible enough to part the transcript into blocks; and selected rows paint a band of
  their own instead of borrowing the terminal's reverse video. `theme` still takes `auto`, `light`,
  and `dark`, and diff colors are unchanged.
- Markdown in answers is laid out for reading in a terminal rather than for a web page. It has one
  vertical rhythm: a blank row between blocks, two above a heading because a heading opens a
  section, list items tight until one of them wraps, and no outer margin of its own. Headings are
  left aligned and step down by weight, fenced code keeps the terminal's own background instead of
  a filled band, tables lose their empty outer edge, and list markers are set apart by weight (an
  ordered item keeps its period). One hue is spent, on the two things a reader looks for rather
  than reads: the top of a section and code inside a sentence. Nothing else in a document is
  coloured -- links are underlined, emphasis is weight and slant.
- The system prompt now asks for short paragraphs, few headings, and lists only where they aid
  reading, so answers arrive shaped for a terminal rather than for a web page.
- A restored transcript is spaced like the live turn it replays. Replayed tool calls went
  straight to the printer while live ones were parted from what came before, so a resumed
  session ran its narration and every call together in one dense block, and the next user
  message arrived under a doubled blank row.
- Blank rows between transcript blocks are decided in one place, so a block can no longer arrive
  glued to the one above it or open a second gap where a rule already left one.
- Starting wizolt loads less code before dispatch, and its banner appears before session/UI imports
  and terminal cursor probing. Cheap exits (`wizolt --version`, `--help`, `--init-config`) answer
  before the interactive CLI is loaded.
- **Breaking.** Async Python APIs now use the natural operation name (`Agent.run`,
  `ToolRunner.run`, `ModelClient.request`, and `TuiApp.run`); retained synchronous entry points use
  an explicit `_sync` suffix. Internal async APIs follow the same convention.
- The prompt stays live and responsive for the whole turn. The TUI, the running turn, model
  requests, MCP, compaction, and vision now share one event loop instead of the TUI having a thread
  of its own, so input, redraws, the status line, and the queue keep working while a request, an
  automatic compaction, a slow MCP server, or a long `Bash` call is in flight. `/compact` and
  queued slash commands run on that loop too, and no longer block the terminal while they reach the
  provider.
- Ctrl-C reports `cancelled` only once the turn has actually let go. The status stays on
  `cancelling` while a tool that cannot be interrupted finishes unwinding — a `Bash` process being
  reaped, an MCP client closing, a `ToolScript` worker returning — so a cancelled turn is never
  still writing files or talking to a server behind the next prompt. `/resend` and automatic model
  retries stay separate from cancelling the turn.
- Answers promoted into scrollback and the tool output that follows them are ordered by an awaited
  queue, so a promoted response can no longer land under the batch it introduced.
- `/model` and `/worker model` now discover remote models through the async provider client, so a
  slow model-list endpoint no longer freezes input or redraws.
- `Search` now runs ripgrep as an asyncio subprocess and kills and reaps it directly on
  cancellation. Its Python filesystem fallback remains off-loop and is awaited to quiescence.
- `@file` completion, the fzf picker, and the external editor (Ctrl-X Ctrl-E / Ctrl-G) no longer
  hold the prompt. Escape closes the picker while a large worktree is still being scanned, the
  completion menu always shows the query you are typing now rather than an earlier one, and
  quitting the session ends the editor or picker instead of leaving it attached to the terminal.
- `/index` and `/catalog sync` keep the prompt and status line live while they run, and Ctrl-O can
  no longer stack a second output browser behind the one already open.
- The startup version check, the provider-catalog refresh, and the saved-session retention sweep
  are owned by the session that starts them: they no longer print or change state after you exit,
  and retention finishes the deletion pass it started rather than stopping half-way.
- Exiting closes what the session opened, in order, before the process ends: the active turn is
  cancelled and awaited, background work is drained, and the model client and MCP are closed. A
  Ctrl-D during a model request or during MCP discovery now exits cleanly instead of leaving
  interpreter warnings behind.
- MCP no longer runs on an event loop thread of its own. Discovery, connect/disconnect, tool and
  resource calls, and the `/mcp` manager all run on the session's loop, so a cancelled or timed-out
  MCP call closes its client before it returns, and one unreachable server no longer interferes
  with discovering the healthy ones.
- **Breaking.** `Read`, `Search`, and `InspectCode` now return numbered source views (`view.N`)
  instead of `line:hash` anchors, and `Edit` targets an existing file by naming a source view plus
  ordinary one-based line numbers. The whole selected range is validated against the file before
  anything is written, not just its first and last line; an unchanged target that merely moved is
  relocated within 50 lines and the move is reported. A failed edit returns a fresh bounded view of
  the current file, and a successful one returns the changed region as a new view. A source view
  expires once it leaves the conversation, and an edit naming an expired one comes back with the
  lines it asked for as they are now, under a new id, so the retry needs no separate read. Output
  from `Bash` is never a source view: code found with `rg` must be read through `Read`, `Search`,
  or `InspectCode` before it can be edited.
- **Breaking.** `Edit` drops the `replace_all` and `replace_unique` operations; a repeated change is
  made with `Search` plus range edits.
- **Breaking.** `Edit` drops `insert_before` and `insert_after`. An insertion is a `replace` over
  a single line whose content is that line's final text, so the content of every operation is the
  complete final text of a stated range and there is no boundary line the call preserves
  implicitly. This removes the accidental-duplicate class that the previous adjacent-duplicate
  warning existed to report, and that warning is removed with it. Writing into an existing empty
  file is now `create`.
- **Breaking.** The session snapshot format is now version 3. Sessions written by earlier releases
  are refused with the existing unsupported-format error rather than resumed, because their stored
  messages teach the removed anchor syntax.
- `Edit` warns when a replacement's first or last line is identical to the preserved line just
  outside the range, and the file had no such adjacent pair before the call: that is what a
  neighbouring line copied into content as context looks like. The edit still applies; the
  advisory names the duplicated pair and its new line numbers. Repeats inside a range and pairs
  that already existed are never reported.

### Fixed

- Ctrl-C in the `/mcp` manager now cancels an in-progress connection immediately instead of
  waiting for the server's full timeout.
- Cancelling a non-TTY run no longer leaves an injected blocking input callback holding
  `asyncio.run()` open, and inputs submitted around a turn boundary retain their original order.
- A saved-session preview that becomes unreadable is reported inside the picker instead of
  surfacing as an unhandled task exception.
- `Read` and `Recall` accept line-range endpoints that a provider serialized as decimal strings,
  while malformed, signed, boolean, and fractional endpoints remain rejected.
- Interactive selectors and read-only viewers no longer block the runtime event loop while waiting
  for a key. This fixes hangs in `/provider`, `/model`, `/worker`, `/sessions`, `/diff`, compaction
  history, and the approval `v`/`c` side trips.
- `wizolt update` (including `/upgrade`) under uv tool and pipx installs previously misread the
  install source because the venv's `bin/python` is a symlink, ran
  `python -m pip install --upgrade wizolt`, and failed with `No module named pip`; the upgrade
  command and the startup update prompt now detect uv tool and pipx installs correctly.
- A drifted single-line `Edit` target that repeats within the relocation window is now
  disambiguated by the lines the source view showed around it, instead of being refused as
  ambiguous. The comparison stays exact, and the surrounding lines only ever narrow a set of
  candidates that already matched, so a target whose own text changed is still never guessed at.
- An `Edit` target whose text still matches at the line number its view named is no longer applied
  there on that basis alone. When the file has drifted and the target's text repeats, the line
  sitting at that number can be a different occurrence than the one the model was shown, which
  would edit the wrong place silently. The view's surrounding lines now have to agree as well, and
  when they do not the position is re-derived by relocation, which resolves it or refuses. A target
  that is unique nearby is unaffected, as is one whose position a batch is already tracking through
  its own earlier edits.
- Repeated tmux zoom/unzoom no longer walks the prompt up the pane. prompt-toolkit handled a
  resize by erasing from the row it remembered and trusting the next cursor position report;
  after a multiplexer reflow that report carries the drifted row, so every cycle inflated the
  application's height until the prompt reached the top and stale copies piled into scrollback.
  A resize now erases from the terminal's actual cursor and re-anchors the prompt at the pane
  bottom first, so the reported position describes the app: the prompt stays put and only one
  copy of it is ever visible. Transcript already scrolled off by the reflow itself stays
  reachable in scrollback.
- Picking or setting a worker provider now shows "Loading models..." while remote model
  discovery runs, like /model does, so the cascade's pause before the model selector does not
  read as a hang.

## 0.37.1 - 2026-08-29

### Fixed

- Avoid a rare crash when the terminal UI exits while background work requests a redraw.

## 0.37.0 - 2026-08-29

### Changed

- Rename the project, PyPI distribution, Python package, console command, data directory, and
  documentation from `minacode` to `wizolt`. Existing `~/.minacode` and `~/.nanocode`
  configuration and session data remain readable in place when `~/.wizolt` does not exist.

## 0.36.0 - 2026-08-29

### Added

- `Ctrl-O` lists `Job` calls and opens the associated background process log while that job is
  available in the current session. The stored tool result remains available after a resume or
  after the log is removed.
- `/catalog` completes its `status` and `sync` subcommands, and its status table points to the
  manual sync command.

### Changed

- Read-only viewers leave a blank row around section rules so headers, content, and returned
  results remain visually distinct.

## 0.35.0 - 2026-08-29

### Added

- A provider entry can set `reasoning_history = "all"`, `"current_turn"`, or `"tool_calls"` when
  an unknown gateway needs a replay policy different from the catalog. The default `auto` follows
  the selected catalog; request-body extensions no longer change history behavior implicitly. The
  selected policy governs both the request and its context estimate on Chat, Responses, and
  Anthropic.

- A provider entry can state what a model accepts, for when the built-in guess is wrong:
  `[provider.X.models]` maps a model name or glob to an ordered `reasoning` list, weakest first.
  Effort support belongs to the model rather than the endpoint, so this is where it is declared;
  the levels become what `/reason` offers for matching models and follow the entry into worker and
  compaction requests. A declared level may be one minacode has never heard of — `["low", "high",
  "ultra"]` puts `ultra` in the picker and sends it as written. Worker and compaction overrides
  validate against their effective model's declaration, saved sessions restore custom levels, and
  a declaration cannot add `off` to a model the catalog documents as always reasoning.

- `omit_body = ["reasoning_effort"]` leaves named fields out of the request. `extra_body` could
  only add and merge, so an endpoint that answers 400 for a field minacode sends had no
  configuration answer at all. A name is dropped wherever it sits in the built request, on all
  three wires, as the last step before sending; the fields that carry the request itself (`model`,
  `messages`, `input`) and the local response-parser selector (`stream`) are refused. `/config`
  lists what is omitted.

- A provider entry can send extra HTTP headers with `headers = { x-cmd-zdr = "1" }`. `extra_body`
  reaches the request body only, so a provider feature documented as a header had no expression at
  all — Command Code's zero-retention routing, a gateway's tenant or routing key. The entry's
  headers are merged over minacode's own on both the OpenAI-compatible and Anthropic wires, follow
  the entry into worker and compaction requests and model discovery, and are listed by `/config`.
  A header change also separates opaque reasoning issued under a different tenant or routing key.
  Values are ASCII strings or plain integers; `key` still supplies authentication.

- Provider and model compatibility knowledge now ships as one complete, versioned JSON catalog.
  At startup minacode validates the bundled and previously synced copies and uses the whole copy
  with the higher numeric version; it never merges partial records. GitHub is checked at most once
  every 72 hours, while `/catalog sync` checks immediately. `/catalog` reports the active version,
  publication date, maintenance scope, source, bundled/cached versions, and the latest sync result.

### Changed

- The explanation under `/reason` uses a readable informational color instead of the disabled-row
  gray, while keeping the selected effort visually dominant.

- Kimi K3 and K2.7 Code are no longer routed as text-only on either product: the open-platform
  IDs (`kimi-k3`, `kimi-k2.7-code`) and Kimi Code IDs (`k3`, `kimi-for-coding`, including their
  suffixed variants) all document image input. Attachments now stay on the selected model instead
  of being diverted to the image fallback. Kimi K2 remains text-only.

- A well-known model now behaves the same wherever it is served. How a model takes reasoning — the
  thinking format, the replay rule, the effort scale — is a fact about the model, so it is matched
  on the model name and applies on every endpoint, including one minacode has never seen. Until
  now that knowledge was reachable only through a recognized host, so `deepseek-v4-flash` on a
  gateway or a self-hosted proxy was treated as an unknown model and sent generic requests.
  Catalog-declared `vendor/model` IDs now use the model suffix for these rules too, including an
  endpoint's model-based `api = "auto"` routing.
  Endpoint facts (wire, caching, strict schemas, provider-side tools) still come from the host, and
  a gateway that re-encodes reasoning into its own format — OpenRouter — keeps its own spelling.

- `/reason` now offers the levels the active model documents, and sends the one you pick. It used
  to offer all seven and quietly rewrite the request when the model had no such level — picking
  `max` on a DeepSeek model sent `high`, with nothing saying so. DeepSeek models now offer
  `off, low, high, max`; a model minacode has no evidence about keeps the full scale. Switching
  model or provider can leave an effort the new model has no level for: it moves to the nearest
  one and says so once (`Reasoning medium is not offered by deepseek-v4-flash, using high`).
  `/config` lists the levels as `provider.supported_reasoning`.

- `/reason` explains a shortened list where it shows it: under the levels, one line saying why the
  model offers only those, and the vendor page it was read from. Nothing appears when the full
  scale is on offer, since there is nothing to account for. A `[provider.X.models]` declaration
  says so instead of citing a page.

- Grok and Gemini now resolve like every other known family, on any endpoint that serves them:
  `grok-4.6` offers low/medium/high/xhigh and `grok-4.5` low/medium/high (an `xhigh` request is
  served as `high` there, so it is not a level); Gemini's OpenAI-compatible layer takes
  minimal/low/medium/high and never `xhigh`, with `minimal` left out where it maps to the same
  thinking level as `low` (3.1 Pro, and 2.5, where both are a 1,024-token budget).

- `off` is no longer offered for a model that documents it always reasons — Grok, Kimi K3, and
  GLM-5.3. Gemini 2.5 keeps it, since it documents `none`. GLM-5.3 also stops being treated as an
  ordinary GLM-5: it was matching the family rule and being sent a disable it does not honour. It
  offers low/high/max.

- Models served through Volcengine Ark offer `minimal`, `low`, `medium` and `high` — the scale the
  endpoint gives everything it hosts, in place of each model's own. A DeepSeek V4 model there was
  being offered its vendor's `max`, which Ark does not take.

- DeepSeek V4 Pro offers `high` and `max`: it serves a `low` request as `high`, so `low` was a
  third menu entry that behaved like the second. Flash keeps all three.

- DeepSeek and GLM models served through Alibaba's endpoint offer `high` and `max` only, which is
  what that endpoint distinguishes: it serves `low` and `medium` as `high`. Their own scales still
  apply everywhere else, GLM-5.3 keeping low/high/max there too.

- DeepSeek's effort levels are `low`, `high` and `max`. `medium` and `xhigh` are accepted for
  backward compatibility and both resolve to `high` server-side, so they were two menu entries
  that did the same thing as a third.

- minacode now requires `openai >= 3.0.0` and `anthropic >= 1.0.0`. Both SDKs moved their HTTP
  client to httpx2, which no longer installs `certifi` and verifies TLS against the operating
  system's trust store instead. Nothing changes on a normal desktop install, but a minimal
  container without system CA certificates, or a corporate proxy presenting its own certificate,
  can now fail to connect where it previously succeeded — install the system CA bundle, or point
  `SSL_CERT_FILE` or `SSL_CERT_DIR` at the bundle to use. The floors are pinned rather than left
  to the resolver so every install lands on the httpx2 pair that minacode is tested against.

### Fixed

- Restoring a session through the default library path validates provider settings against the
  active bundled-or-cached catalog, so a newer cached dialect or effort is not rejected using the
  older bundled vocabulary before that same cached catalog becomes active.

- Synced catalogs now distinguish image-capable (`auto`) models from text-only models and apply
  provider-local image rules with the same precedence as every other compatibility setting.

- `/catalog sync` reports a same-version content conflict as such and prints network failures with
  one error prefix. If a newly activated catalog no longer contains an explicitly configured
  reasoning dialect, the next request reports a configuration error instead of a raw `KeyError`.

- Responses requests keep configured `extra_body` extensions when a catalog recipe also writes
  under `extra_body`; catalog-managed fields still take precedence on direct conflicts.

- A connection dropped mid-stream is retried again. The provider SDKs re-raise transport failures
  from their streaming reads unwrapped, and their move to httpx2 put those errors in a hierarchy
  that shares no base class with the httpx one the retry policy matched, so a dropped connection
  stopped counting as transient and surfaced on the first attempt. Both httpx generations are now
  matched, since the MCP client transports still raise the original ones.

## 0.34.2 - 2026-08-26

### Fixed

- An MCP call that times out no longer prints an asyncio `Task exception was never retrieved`
  traceback over the session. The abandoned call's client can raise its own error while tearing
  the connection down (an HTTP read timeout on the dropped request); that late failure is now
  swallowed, since the caller already has its timeout error.

## 0.34.1 - 2026-08-26

### Changed

- The transcript now parts where the agent actually speaks: an interim reply opens with a
  full-width rule, the same line the turn's `done in` rule draws but without the label, and the
  rule lands above the text so it announces the new phase rather than closing the old one. The
  user's message opens its turn the same way, and a resumed session replays all of them. Two
  rules closer than about six rendered rows are collapsed to one, and a stretch of about four
  tool batches with nothing said gets a rule of its own.
- A call whose arguments span many lines (e.g. a heredoc script) now shows only its first
  three lines on the call row, followed by `… +N more lines`; the full text is still in the
  viewer.

## 0.34.0 - 2026-08-24

## 0.33.0 - 2026-08-24

### Changed

- `--help` and `/help` now point to the full documentation at https://minacode.readthedocs.io.

## 0.32.0 - 2026-08-24

### Added

- The worker session (Delegate) now has `ViewImage` and `ToolScript`: it can inspect any local
  image path, not just user attachments, and batch repetitive same-shape tool calls.
- A session's tool whitelist is now enforced at execution time, not only in the schema
  projection: a hallucinated or scripted call to an excluded tool is rejected with "not
  available in this session" instead of running.

### Changed

- Enter with a Tab-highlighted completion row now commits that row into the input instead of
  sending the message; a second Enter sends.
- A streamed response that ends without a terminal event no longer fails the turn: it is a
  transient provider-side drop, retried automatically like a dropped connection.

## 0.31.1 - 2026-08-23

### Changed

- The live spark now shows one star per breath and swaps at the crest, so both the fine
  four-point star and the heavy six-point one appear at full brightness: the opening breath
  is the four-point star, the next the six-point one, then back.

## 0.31.0 - 2026-08-23

### Added

- A `[vision]` provider entry gives image input a fallback: when the active model rejects an
  attached image or is known to be text-only, `[vision]` produces one plain-text observation that
  replaces the raw image in the conversation. Each routing prints a compact image log with the
  input name or count and a `described by <provider>` child matching `ViewImage`.
- A blocking `Job` wait now streams the job's output into the same live preview as Bash, and the
  status line shows the seconds left in the wait (e.g. `· 12s left`).

### Changed

- **Breaking:** `Edit` now uses `content` as the single field for text written by every operation,
  including `replace_all` and `replace_unique`; their former `new` field is rejected. Exact-text
  replacements must include `content`, with an explicit empty string meaning deletion, so an
  omitted replacement can no longer silently delete the match.
- Worker replies printed between tool calls now render as markdown, like the worker's final
  report. Tool lines stay as the log tree.
- The live stream preview now marks inline markdown in the streaming text: `**bold**`,
  `` `code` ``, and `*italic*` render with weight and underline once their closing marker
  arrives. The marks stay in the preview's own gray tones, so the region keeps its all-gray
  look; block constructs and unclosed markers stay literal, so a stream mid-token never
  flickers.
- The Ctrl-O browser marks each Bash row with its exit verdict in the first column: a green ✓
  for exit 0, a red ✗ for any other exit, and a blank cell for entries with no exit code (a
  script, an order). The list is mostly bash, so the failures become scannable by color, and it
  now lists every stored result (up to the session's 400) instead of the latest fifty.
- A divider label wider than the default rule (e.g. the worker's `[worker]` label with status,
  elapsed time, rate, and a queued line) widens the rule so both sides keep at least twelve dashes and the label
  stays whole, instead of squeezing the comet's track to two or three dashes; on a narrow
  terminal the trail shrinks and the comet slows down so it never reads as frantic.
- Quick-hint chips are picked with `Enter` instead of `Space`: the pick returns focus to the
  prompt, so one `Tab` advances to the next chip, `Enter` again combines it, and a final
  `Enter` sends. Refocus a picked chip and press `Enter` to unpick it; `Space` is plain again.
- **Breaking:** Quick hints are now always available in the TUI. The `/hints` command and
  `[runtime] quick_hints` setting have been removed; existing config values are ignored and
  `/hints` is now an unknown command.
- **Breaking:** Image routing is main-first with a bounded `[vision]` fallback. A raw image is
  always tried on the active model first unless minacode knows the route is text-only; when that
  request fails with HTTP 400, the image is described once through `[vision]` and the turn retries
  without raw image blocks. Text-only knowledge comes from a static catalog of documented
  text-only model families (DeepSeek chat/reasoner, GLM 5 and 4.x, Kimi K3, Moonshot v1, gpt-oss,
  and a few others) or is learned for the session from that first 400; learned evidence is never
  saved, so a resumed session starts unknown again while the text observation itself stays in
  history. The removed `provider.image_input` setting is ignored in old configs; `off` no longer
  disables submission and `auto` no longer manages vision routing.
- Each `Job(action="wait")` call now waits at most 20 seconds, including the default wait; a
  still-running job remains available for a later status check or another wait.
- Quick-hint chips flow left to right, up to three per line, and wrap to new lines only
  between chips when the terminal is too narrow, so every suggestion stays visible instead of
  being clipped off the side.
- Documentation is English-only now: the Simplified Chinese pages, their gettext catalogs and
  `sphinx-intl` tooling, the language switcher, and the Chinese README have been removed. The
  README screenshots now use absolute GitHub URLs so they render on PyPI instead of breaking.

### Fixed

- Attached images now include their stable session-owned paths in model context, so a model that
  silently ignores image input can call `ViewImage` directly instead of searching the filesystem.
- A rejected image no longer makes every later request replay the same unsupported image. The
  failed occurrence is retained as readable text with stable session asset paths for `ViewImage`,
  while a newly submitted image is still attempted normally.
- A stale `Edit` anchor now returns a small, freshly anchored file neighborhood in the failed
  result, so a verified nearby target can be retried without another `Read`; ambiguous targets
  still require a read and are never guessed. Batched edits now also return the same duplicate-line
  and large-edit warnings as direct edits, and the tool guidance no longer tells models to copy
  unchanged boundary lines into replacement content.
- Pressing `Ctrl-C` while a `ViewImage` vision request is running now cancels the read at once,
  instead of hanging until the vision provider's response timeout; the turn then settles like any
  other interrupted model call.
- A failed `NextHints` batch counts as a tool batch, so the next ordinary tool batch shows the
  `·2` suffix instead of presenting as the first batch.
- The live preview of a short `Job` log no longer repeats already-shown lines when the command
  writes more output: each new line is streamed once instead of the whole tail again.
- Crash residue in the session's assets directory -- a `.image-*` staging file left behind by an
  interrupted save -- is cleaned up once it is old enough, so abandoned files no longer pile up or
  keep the directory around.
- Batched terminal output is queued before the event-loop handoff and drained when the TUI stops,
  so the turn's final lines cannot disappear or race ahead of a restored transcript.
- The Ctrl-O output browser no longer hides the oldest stored result while a ToolScript is running:
  the live running entry is listed on top of all stored records instead of counting against them.

- Anthropic requests with a configured `temperature` no longer crash with
  `unexpected keyword argument 'temperature'` on anthropic SDK 1.0 or newer, which removed the
  top-level parameter; the value still reaches the request body as before.
- An all-`NextHints` tool batch now ends the turn even when the model returns no answer text:
  the suggestions stay at the idle prompt instead of being cleared by a follow-up model request,
  and no blank answer line is printed. If every `NextHints` call in such a batch fails (no
  answer text, no suggestions), the turn continues instead of ending as a blank reply, and
  several `NextHints` calls in one batch merge their suggestions rather than the last call
  overwriting the rest.

## 0.30.0 - 2026-08-21

### Fixed
- In the Ctrl-O output browser, `Esc`, `q`, or `Ctrl-C` inside a detail now returns to the list
  with the cursor on the entry it came from instead of closing the whole browser; `Ctrl-O` still
  closes it.
- Resuming a long session no longer replays the whole transcript onto the terminal: only the
  twenty most recent turns are redrawn, with a line noting that the earlier ones stay in context.
  The replay is printed in a single flush, so it appears at once instead of line by line. In the
  full-TUI shell a quiet gray `resuming session…` status shows while it is being restored.
- Opening `/sessions` / `/resume` no longer crashes with `UnicodeDecodeError` when the session
  log's last chunk happens to cut a multi-byte (e.g. Chinese) character in half; the preview
  reads the tail in binary and skips the torn line.

### Changed

- The `/sessions` / `/resume` picker is redesigned. It opens full-screen like the other browsers,
  with the session list scrolling in a viewport, and its rows line up in columns: session name,
  age, and round count padded to the widest value in each (measured in terminal cells, so CJK
  names align too), the name plain, age and round count dimmed, the current session marked in the
  live colour. The preview below shows the session's facts plus its most recent messages, read
  lazily when the cursor lands on a session rather than by scanning every log on open. Internal
  `<session_event ...>` resume markers are hidden so the preview shows real conversation, tool-only
  turns collapse into one counted line (`-> Bash ×3, Read`), and the messages are laid out like the
  transcript -- a `• ` bullet in the prompt colour with the message in the transcript's warm tone,
  replies indented in the default colour, newest exchange at the bottom, anchored on the opening
  question when the recent turns are all replies. The preview widens its tail window only until
  it holds the recent conversation (capped), so a session whose tool output runs to megabytes no
  longer shows just the last message or two.
- The breathing spark on a streaming response and a running command now swaps between the
  fine star and a heavier one at the darkest point of its breath (so the change is barely
  visible) and is bold throughout: bigger and still continuous. A blank row now separates the
  spark from the output rail below it, so the star reads as capping the region rather than
  sitting on it. A gray phase word rides beside the spark -- `thinking` while the model reasons,
  `responding` once it answers -- and the streamed text starts on its own row below, so the first
  line no longer shares the spark's row.

## 0.29.1 - 2026-08-20

### Fixed

- A TUI redraw no longer crashes with `RuntimeError: no running event loop` when the agent thread
  asks for a repaint while a long task runs. prompt_toolkit made `Application.invalidate()` safe
  to call from any thread in 3.0.53; the dependency floor now guarantees that fix instead of
  depending on whatever version a fresh install resolves.

## 0.29.0 - 2026-08-20

### Fixed

- Editing a `.yaml` or `.yml` file no longer fails with `ToolError: Token.Literal.Scalar.Plain`.
  Rendering the edit's own diff preview highlights the file, and the YAML lexer emits tokens
  neither theme's syntax style names — the lookup raised instead of falling back, so the error
  reached the model as a failed `Edit` and the file was never touched. CI workflows, compose
  files, and k8s manifests were all unreachable. Perl's `.pl` had the same hole.
  A token the style does not name now inherits from its ancestors, so those files are highlighted
  rather than merely spared, and a lexer that fails costs the color instead of the edit.

### Changed

- Resuming a session restores the provider entry, model, reasoning effort, and API wire you
  last switched to with `/provider`, `/model`, `/reason`, and `/api`, instead of falling back
  to the config file's choices. A switch that no longer exists is skipped, never an error.
- Everything a turn produces now starts in the same column. The user's message already sat there
  with its `• ` bullet hanging in the margin, and the model's text and tool lines followed it;
  the turn outcome (`Cancelled`, an error), a slash command's reply, and the live stream
  preview did not. They do now, so the transcript has one left edge
  instead of two. Session chrome — the startup banner, the restored-session notice, the resume
  line — stays flush left and frames the conversation.
- The live stream preview is drawn with the same rail as a tool call's output, and no longer
  opens with a `├` — a `├` claims a line joining it from above, and there was never one there.
- The preview has also lost its `thinking`/`responding` heading. The divider right below it
  already names the phase and times it, so the word was printed twice on one screen.
- A spark caps the preview's rail in its place, breathing slowly from near-black up to near-white
  in the divider's own accent, so a long silent stretch of reasoning still shows a pulse. Only
  the spark moves; the streamed text under it holds still. It opens at its brightest and is timed
  from the moment the region appears, so the frame that announces it is the loudest one rather
  than wherever the clock happened to be.
- The live output of a running `Bash` command gets the same treatment, from the same definition:
  its status row had the same phantom `├` — drawn under a root that frame never had — and it now
  carries the same breathing spark. A command that goes quiet for minutes no longer leaves a
  frozen-looking frame with only a clock ticking in it.
- A resumed session renders the final answer in that column too. It was replayed flush left
  while the live turn had already moved it, so the same turn changed shape across `--resume`.

## 0.28.0 - 2026-08-20

### Changed

- The final answer of a turn is now indented to the same column as the text the model writes
  between tool calls, and the `Sources` footer after a web search follows it there instead of
  starting at column 0.
- The output speed on the working divider is now marked with a `↓`: `responding (12s · ↓ 48
  tok/s)`. The arrow names the model's incoming stream and replaces the `~` that stood for the
  estimate; the speed is still estimated from the text as it arrives.

### Fixed

- `Ctrl-P`/`Up` pressed immediately after submitting a message now recalls that message. Every
  submit resets the input buffer, and the history copy that refills it only runs at the next
  repaint, so a recall key arriving in between found an empty list and recalled nothing.
- The worker's final report now prints into the scrollback rendered like the agent's own
  answers, instead of hiding inside the tool result; the folded answer preview under the
  `worker done` divider is unchanged.
- A piped or non-TTY run (`echo ... | minacode`) no longer prints each final answer twice. The
  engine publishes the answer itself now, and the simple REPL was still printing the returned
  value on top of it.

## 0.27.0 - 2026-08-19

### Added

- The working divider shows the model's output speed while a response streams, next to the elapsed
  time: `responding (12s · ~48 tok/s)`. Estimated from the text as it arrives, hence the `~`, and
  absent between requests and on providers that do not stream.
- `Ctrl-O` lists delegation orders alongside Bash outputs and ToolScript scripts, and opens one
  with the worker's answer below it. Judging an answer means reading the order again, and the
  transcript keeps only the `Delegate send` line.
- An `Edit` call that writes more than 6000 characters now says so in its own result. Everything
  one call writes is generated inside a single assistant message, and a timeout partway through
  loses all of it, so a change that size is safer as several calls.

### Changed

- The status bar marks a running compaction with `[compaction]` even when the summary runs on the
  session's own provider and model, which is the default. The marker was previously shown only
  when a `[compaction]` entry overrode the model, so an ordinary compaction pause looked exactly
  like a slow reply.
- `Ctrl-O` browses the 50 most recent results instead of 10, through a scrolling window about ten
  rows tall with a counter under it. The session keeps 400 results; the old limit was the list
  filling the screen, not the storage.

### Fixed

- Anthropic requests now cache the conversation, not just the prompt header. Anthropic has no
  implicit caching and writes only at a `cache_control` breakpoint, and minacode set one on the
  system block — so tools and the system prompt were cached while the conversation body, the part
  that grows to a hundred thousand tokens, was re-read at full price on every turn. A second,
  rolling breakpoint on the last block of the request writes the history through this turn and
  reads it back on the next, which is what the OpenAI-shaped providers were already doing
  implicitly. `/status` shows the difference in its cache row.

- A failed `Delegate send` now reports what the worker did before dying — steps, elapsed time,
  changed files, rounds, and context fill — and that its context survives, so the parent can
  decide between resending and resetting instead of guessing. The failed turn is settled in the
  worker's history: every unanswered tool call gets a `Failed` result and the turn ends with a
  `[This turn ended early: …]` marker, so the next delegation goes out normally, and
  `Delegate status` keeps showing the last failure until a send succeeds.
- A multi-line command no longer breaks its row in the `Ctrl-O` list. A `git commit -m` with a real
  message spilled over several lines and took the numbering and the selection bar with it; the row
  now folds to one line, and the viewer still shows the command exactly as it was run.

## 0.26.0 - 2026-08-19

### Added

- ToolScript scripts can name an MCP tool the way the tool list spells it: `call("server.tool",
  {...})` is now the `call("MCP", {"server": ..., "tool": ..., "arguments": ...})` form, which is
  the spelling models reach for first and `action="describe"` already accepted. A dotted name that
  resolves to no configured server now says which server is missing instead of "unknown tool".
- The running divider shows `running script` while a ToolScript body executes, so a long batch no
  longer reads as plain `working` from approval until the last nested call returns.
- `Ctrl-O` lists the ToolScript that is running right now, above the stored entries, and opens its
  source in the same read-only viewer. A long batch is exactly when the script is worth reading,
  and until it returns there is no stored record to open.
- `/status` gained a `compaction cache` row. Summaries are billed to their own counter, and this is
  the only place that says whether one reused the conversation's cached prefix.

### Changed

- Compaction keeps far more of the recent conversation. The recent window was measured only over
  what follows the latest user message, which made it a cap rather than a floor: `/compact` run
  just after a turn answered kept two messages out of 118, and everything concrete - the last tool
  results and file contents - survived only as prose in the checkpoint. It now spans the whole
  tail, bounded by size as well as count (at most 8 messages and a quarter of the request budget),
  so ordinary messages get the full window while very large ones still collapse it rather than
  leaving the request over budget with nothing left to compact.
- The summary request is now the agent's own request with one instruction appended - same system,
  same tools, the conversation as real messages - instead of a fresh request carrying a flattened
  re-rendering. It rides the prefix the turn just paid for on providers with prefix caching, and
  the compactor sees tool calls, which the flattening dropped. It falls back to the old payload
  when the summary runs on a separate `[compaction]` provider.
- A compactor that replies in prose, or that copies the conversation into `summary` instead of
  summarizing it, is asked once more with a correction before compaction degrades to the
  deterministic trim. Where the provider supports it (OpenAI, DeepSeek, Alibaba Model Studio), the
  reply is additionally constrained to a JSON object. Compaction failures now name the provider
  entry that served them.
- ToolScript's description now states what `call()` returns for each `format`, that a failed nested
  call raises and ends the script, and shows a fan-out example that contains per-item failures.
- The `Ctrl-O` browser colours each row by part - dim `tr.N` key, green tool name, plain arguments -
  the way the transcript colours the same call, instead of one flat grey line. The selected row is
  still a single reverse bar.


## 0.25.1 - 2026-08-17

### Changed

- Upgraded the `code-symbol-index` dependency from 0.4.0 to 0.5.1, picking up its latest
  index and query improvements for the code search tools.


## 0.25.0 - 2026-08-17

### Added

- New `ToolScript` tool for batches of same-shape MCP calls. `action="describe"` batch-reports
  each tool's call shape and a json gate ("yes" when the server declared an `outputSchema`,
  "unknown" otherwise); `action="call"` runs a Python script whose `call("MCP", {...})` performs
  nested MCP invocations with the usual confirmation, logging, and `tr.N` retention - only printed
  output returns to the context. Nested calls never add tool messages, a refused nested call aborts
  the script while the rest of the batch proceeds, and `format="json"` prefers the declared
  `structuredContent` payload over parsing rendered text. Built-in tools are scriptable with
  `format="text"` too (Delegate/Job/ToolScript cannot be nested); `format="json"` for built-ins
  waits for structured results.
- ToolScript scripts are readable in the UI. The approval block shows the opening lines
  syntax-highlighted and numbered, `v` at the confirmation prompt opens the whole script in a
  read-only scrolling viewer, and `Ctrl-O` reopens it afterwards - which is the only way to read a
  script under `--yolo`, where nothing stops to ask.
- `Ctrl-O` lists ToolScript calls alongside Bash outputs, and every entry opens that same viewer,
  showing what was run above what it returned: a Bash command with both its streams, or a script
  with its result - the printed output, or the whole traceback when it failed, against the numbered
  line the traceback names. Large results are bounded there (head and tail kept, lines over 1000
  characters clipped) because stored output has no cap of its own; the header says whenever
  anything was cut, and the complete result stays under its `tr.N` key.

### Changed

- The system prompt now says when to reach for `ToolScript` rather than leaving it to the tool
  description: 4+ same-shape calls whose individual results are not needed, since only what the
  script prints returns - and plain batching when each result matters or a step needs the model's
  judgment, because a script runs to the end on its own. `ToolScript`'s own description states the
  same trade rather than a bare threshold, and it gains examples.
- Calls a ToolScript makes are logged indented under the script, on a `|` rail that runs unbroken
  from the script through every call it made - including whatever each of those calls logged below
  itself, a diff or a command's output - down to the result line, so the batch reads as work the
  script did rather than as calls the model made itself. The closing line reports how many nested
  calls ran plus the first lines of what the script printed. The call line now names the script's
  size instead of echoing its first line, which was usually setup.

### Fixed

- A `ToolScript` script can no longer end the session: `sys.exit()` raised a `SystemExit` that flew
  past the tool, past the runner, and out of the agent loop. It is now a failed script like any
  other, while Ctrl-C still cancels the turn.
- Nested calls inside a script log to the terminal again instead of into the script's own output.
  The capture meant for what the script prints was swallowing every nested call's log line and
  handing it back to the model as the script's result; on the headless path the confirmation prompt
  went with it, stopping the run at a prompt nobody could see.
- `ToolScript` no longer requires the describe-only `tools` argument on every call. Under a
  strict-tools provider that made running a script at all a schema violation; `action` is now the
  required field.
- A failed script now reads as failed in the transcript: the result line says so and carries the
  error, instead of looking exactly like a script that finished.
- `call()` inside a script returns the result of tools whose output is not retained (`Recall`,
  `RecallContext`, `Note`) rather than the model-facing envelope wrapped around it.
- An MCP tool that declares an `outputSchema` and returns an empty `{}` or `[]` payload - a search
  matching nothing - no longer fails a scripted `format="json"` call as though the payload were
  missing, and a non-JSON structured payload is reported as a tool error rather than escaping.
- The Delegate `v` viewer reflects a worker configuration just changed with `c`, instead of the one
  captured when the prompt was first drawn.
- The `Ctrl-O` viewer shows the whole command that was run; it was rendering the transcript's
  one-line display, clipped at 200 characters.
- The script time budget no longer enables line tracing inside every tool a script calls, which
  cost a callback per line of `Read`, `Search`, and the MCP transport.


## 0.24.6 - 2026-08-17

### Added

- `Edit` gains a `replace_unique` operation: like `replace_all`, but the old text must occur
  exactly once - zero or multiple matches are rejected with the hit lines and leave the file
  untouched.

### Changed

- Stale-anchor errors now end with guidance to retry with the returned current anchor only when
  its content is the intended line; otherwise the model re-reads instead of relying on a guessed
  range. Successful edits return three lines of anchor context on each side of the change, so a
  follow-up edit can anchor next to the hunk.
- NextHints chips are picked with `Space` instead of `Enter`, so `Enter` only ever sends the
  input; `Space` toggles the focused chip into or out of the picked set.
- Delegate `send` results now include the worker's `context_percent` and `rounds` so the model
  can decide when to reset the worker's context; the done line shows both as well.
- The external editor (`Ctrl-X Ctrl-E` / `Ctrl-G`) now includes several recent AI replies as
  reference, not just the last one, so the reply you are answering is usually in view.

### Fixed

- `Edit` now treats `content` and `new` as the JSON-decoded strings they already are, preserving
  literal `\n` and `\t` text instead of guessing that it represents another layer of escaping.
- `replace_unique` remains an exact substring replacement when the match ends at a line boundary,
  and rejecting repeated text now finds and reports its hit lines in linear time.
- Worker-mode `Note` output now shows the same per-line colors (goal/plan/known) as the main
  session; it was rendered as a plain block before.
- A `Job wait` no longer prints its call line twice when the confirmation prompt already showed
  it.


## 0.24.5 - 2026-08-16

### Fixed

- An MCP tool that returns only `structuredContent` no longer arrives as an empty result. A tool
  declaring an `outputSchema` returns its payload there and only *should* also repeat it as text
  for older clients; minacode read the text alone, so a server that skips the repeat produced a
  result the model reads as a query that matched nothing. The structured payload now stands in when
  the content blocks are empty, and is not appended when they are not — servers that honor the
  repeat send the same payload twice.

### Added

- `MCP(action="describe")` reports what a tool returns, on the servers that declare it. The result
  shape is read from the tool's `outputSchema` and rendered as `<returns>` in the same form as its
  arguments, with the full schema below. Without it the only way to learn a tool's result shape was
  to call it, spending an exploratory call on every unfamiliar tool. Tools that declare no schema
  read exactly as before.


## 0.24.4 - 2026-08-16

### Fixed

- GLM-5.2 reasoning effort is folded to the two levels it documents. The model resolves anything
  that is not `high` to `max`, so `/reason low` and `/reason medium` were buying its most expensive
  setting for a request that asked for its cheapest. Levels at or below `high` now send `high` —
  the model's low end — and `xhigh` folds up to `max`.
- Qwen3.8-Max reasoning effort is folded to the levels it documents (`low`, `medium`, `xhigh`).
  `/reason high` and `/reason max` were sending spellings the model does not accept; both now fold
  to `xhigh`. Disabling reasoning still sends `none`.


## 0.24.3 - 2026-08-16

### Fixed

- Assistant turns now record which endpoint, model, and credential produced their provider echo,
  and the echo is replayed only back to that issuer. Switching between two hosts that share a
  protocol — one Responses endpoint to another, or a Claude gateway to the first-party API — or
  between two entries on one host holding different credentials, no longer replays
  encrypted reasoning or a thinking signature that the new host cannot verify; those turns are
  rebuilt from their text and tool calls instead. Sessions recorded before this release carry no
  origin and keep replaying as before.
- DeepSeek's tool-call reasoning-replay rule now follows the model as well as the endpoint, so a
  gateway serving DeepSeek stops sending reasoning on turns the model ignores it on.
- `extra_body` fields that live inside an object minacode also manages are merged into it instead
  of replacing it on the Responses path. Configuring `extra_body.reasoning.context` no longer takes
  the resolved `reasoning.effort` down with it; `/reason` still wins on the fields it manages.
- `/config` reports the resolved Chat reasoning-history policy.


## 0.24.2 - 2026-08-15

### Added

- `@file:path` mentions point the agent at a file in the project. `@` lists the bounded
  namespaces without scanning repository files; selecting one opens its candidates. Typing or
  selecting `@file:` launches fzf directly when supported, with a non-blocking literal fallback
  otherwise. The picker shows navigation keys including `Ctrl-N/P`. Candidates exclude `.git` and
  every Git-ignored path. Quoted mentions round-trip spaces, Unicode, and punctuation.
  The canonical `@mcp:` and `@skill:` forms and legacy `@server` and `$skill` forms still resolve.
  Mentioned files become a bounded FILE MENTIONS block in initial and queued messages: small text
  files are inlined, while large, binary, outside-workspace, missing, or excess files degrade to
  explicit pointers or diagnostics.
- Quick-hint chips can be picked into the input one by one and combined before sending: Enter
  on a focused chip fills the input (again unpicks it), Tab keeps cycling the chips, and a
  manual edit hands the input back to normal editing.

### Fixed

- Selecting a partially typed mention namespace now opens its second-stage candidates, and the
  merged mention menu keeps its 50-row bound.
- Warm file-picker candidates after the prompt appears, reuse an existing snapshot while it
  refreshes, and overlap the two read-only Git queries. Opening `@file:` no longer waits on a
  fresh full-worktree scan in large repositories.
- Quick-hint acceptance now uses one immutable hint snapshot and clears stale focus/picks after
  the hints or input change, avoiding crashes and selecting the wrong suggestion during refreshes.
- File, MCP, and skill mentions now share one scanner, so reserved namespaces do not collide with
  legacy MCP syntax and email-like text is left alone. File aliases deduplicate by canonical path,
  per-file read failures no longer abort the turn, and unterminated final lines are counted.


## 0.24.1 - 2026-08-14

### Changed

- Compaction keeps the newest 50 history segments instead of every span a session ever evicted.
  `seg.N` keys keep counting past the bound, so a key is never reused for different content, and
  `RecallContext` answers a key below the window with what happened to it and where the retained
  ones start rather than a bare `missing`. `/compact log` already showed compactions and stored
  segments separately, so a capped session reads as what it is.

### Fixed

- Compaction no longer evicts the request a turn is executing when the runtime appended a message
  of its own behind it: an `@server` or `$skill` mention expansion, or the protocol correction sent
  after a model prints a tool call as text. Those are session events now, not user turns, so the
  boundary compaction never crosses stays the request itself. A worker delegation is where this
  cost the most — the order is the entire spec, the worker cannot see the parent's history, and
  nothing re-sends it — but a long parent turn could lose its own request the same way.


## 0.24.0 - 2026-08-14

### Added

- The Delegate order viewer renders the order as markdown and aligns its field header.
- Compaction now names the span it evicts, in the same reply that produces the summary, so a
  history segment is titled by what the work was about instead of by the first user message in the
  window — often `ok` or `continue` once a span starts mid-work. Titles show in `/compact log` and
  in the agent's `RecallContext` listing. A compaction that returns no name, including a
  deterministic trim after a summarizer failure, keeps the old first-message title.
- `/compact log` reviews what compaction kept: stored segments newest first with when each ran,
  whether it was automatic or manual, whether it covered prior conversation or the running turn,
  and how much it evicted; opening one shows the summary it produced, which the active context
  keeps only until the next compaction. `/compact log seg.N` prints that summary without the
  viewer.
- A `[compaction]` config section gives context compaction its own provider entry: `provider`
  names a base entry (empty = the active provider), and `model`/`reasoning`/`api` override it per
  field (empty inherits the entry's value), mirroring `[worker]`. The summary request runs on the
  resolved entry — including its `response_timeout` — while the context budget still measures the
  active provider's window; `/compact log` records which model produced each segment and `/config`
  shows the effective `compaction.*` values. While a summary is in flight the status bar names the
  entry serving it, `[compaction] entry/model effort`, the way it already names an in-flight
  worker; a compaction that resolves to the row's own entry leaves the row alone. An incomplete
  compaction entry is refused by name before the request, instead of reaching the provider SDK and
  coming back as a credentials error that names nothing. Summary tokens are counted apart from the
  conversation and reported on their own `/status` row: they can be billed to another account at
  another price, and a summary is a fresh prefix that never hits the conversation's cache, so one
  blended total could be read against neither model.
- Each provider entry can nest its own `[provider.NAME.compaction]` table with the same
  `model`/`reasoning`/`api` keys, so a provider can route summaries to a cheap model per entry
  instead of one global choice. Per field the most specific value wins: the entry's nested table,
  then the global `[compaction]` section, then the entry's own value; `/config` shows the resolved
  effective values.


### Changed

- Typing `@` or `$` opens the completion list as you type, instead of waiting for `Tab`. Both
  symbols exist only to name an MCP server or a skill, so the list belongs on the keystroke that
  declares one; prose, `/` commands, and everything else still complete on `Tab`.

- Provider `429` responses whose error body carries account/billing wording (e.g. OpenAI
  `insufficient_quota`, Kimi `exceeded_current_quota_error`, `z.ai` "Insufficient balance ...") are
  treated as permanent quota/billing failures: the request fails immediately instead of retrying
  through the backoff. Transient rate-limit 429s retry exactly as before, including the ones
  phrased with the same vocabulary — Google/Vertex `Quota exceeded for quota metric …` and
  DashScope `Throttling.RateQuota` are per-minute limits, not billing failures.
- Compaction's summary request honors `provider.response_timeout` instead of an undocumented 60s
  cap, which also made the timeout's "set it to 0 to disable" advice a no-op for compaction. A
  summary that hits the limit now says so in the fallback error.


## 0.23.0 - 2026-08-11

### Added
- `[provider.NAME] max_context_tokens` overrides `runtime.max_context_tokens` for that entry, so a
  1M-window model and a 128K one no longer have to share one global number — the big one compacts
  far less often without risking overflow on the small one. `0` (the default) inherits. The budget
  resolves per request, so `/provider` moves it with the entry and a worker on its own entry stops
  borrowing the parent model's window. `/status` shows the effective limit, and `/set
  provider.max_context_tokens` changes it for the session.
- The project's `AGENTS.md` (or `CLAUDE.md` fallback) is now injected into every request as a
  bounded "Project instructions" section of the Environment block, so repo-specific conventions
  ride the cache-stable prefix. Controlled by `[runtime] agents_md` (default on); `/status` and
  `/config` show whether it is loaded.
- Tool approvals are a small form instead of a line to type at. The actions sit in a row above the
  input line with `Approve` selected — `Tab` and the arrows move along it, `Enter` fires what is
  selected, `Esc` refuses — so nothing has to be memorized and, for a Delegate send, viewing the
  order or editing the worker config no longer takes a letter plus Enter. Typing still goes to the
  refusal reason, and because no letter is a shortcut, a reason can start with any word at all.
  Each action submits exactly the line you could have typed, so the protocol underneath is
  unchanged and headless runs keep `y`/`n`/`v`/`c` verbatim.
- Context at 100% no longer sits there without compacting. The compactable head is everything
  before the latest user message, and after a compaction that is only the previous summary — which
  is filtered out — so a session whose latest user message was followed by fewer than eight
  messages had nothing to compact and went over budget on every request from then on. Since the
  recent window is a message count rather than a size, a few large messages were enough to trigger
  it. A smaller window is now tried when the ordinary one leaves nothing, and when nothing can be
  compacted at all the run says so instead of silently sending an over-budget request.
- Automatic compaction is now explicitly limited to one pass per scope until the messages change.
  Nothing enforced that before — the loop was prevented only as a side effect of the bug above, so
  fixing it without a real guard would have turned a stuck context into a compaction loop.
- A rejected `Note` no longer prints its whole body in grey. An argument rejection is meant to be a
  quiet one-line summary, but it reused the tool's display, and `Note` keeps the entire rendered
  note there so a successful call can print it — so a rejected update dimmed thirty lines and left
  the reason hanging off the end of the last one. The rejection line is one line now, and a failed
  call's red root is collapsed the same way; a successful `Note` still prints in full.
- A tool approval no longer reprints its whole brief every time you come back to it. Viewing a
  Delegate order or editing the worker config re-asked by redrawing the entire brief, stacking a
  copy in the transcript per visit; the brief is printed once and the config cycle now reports the
  values the picker left behind rather than the ones it started from.
- The Delegate send confirmation prompt now accepts `v` (or `view`) to open a read-only,
  full-screen viewer of the complete delegation order before approving (Esc/q closes back to
  the prompt). The approval brief still shows only a one-line excerpt; the viewer is for reading
  the whole spec. Headless/no-TUI falls back to printing the full order.
- A tool result too large for the context budget is now materialized to a file alongside its
  truncated marker. The `<bounded_output>` marker carries the file's absolute path under
  `file="..."`, so the model can `Read`, `Search`, or `grep` the full output instead of paging
  through `Recall` slices - which are themselves re-truncated and can't show the whole thing at
  once. The marker also carries a `hint="..."` naming the cheaper move, because an attribute name
  says where the rest of the output went, not what to do about it. The file lives in the session's
  `.assets` directory and is cleaned up with it; only outputs over the budget are written, and a
  read-only or full disk leaves truncation itself untouched.

### Fixed
- `Read` and `Search` no longer ask for confirmation to open the session's own assets. The file a
  truncated tool result is materialized to lives outside the workspace, so following the path the
  `<bounded_output>` marker just handed the model tripped the out-of-workspace prompt. Every other
  path outside the workspace still asks.
- The file a truncated tool result is materialized to is no longer deleted on the next save. The
  session's asset collector kept only image references, so it collected the file immediately while
  the `<bounded_output>` marker went on advertising its path - the model was sent looking for a file
  that no longer existed. The file is now retained for as long as its tool result is, and collected
  with it.
- A truncated tool result no longer collapses to nothing when the output has very long lines. The
  head and tail excerpts snap to a line boundary, which threw away the whole excerpt when a single
  line was longer than the budget - the common shape of an MCP server returning compact JSON, where
  the model was left with a marker announcing that nearly everything was omitted and no content on
  either side of it. Snapping is now abandoned when it would cost more than half the budget, so such
  a result keeps a real head and tail, and `Recall` over it returns real content too.
- An `Ask` free-text question no longer renders its line breaks as stray `^J` control characters.
  The prompt reaches the input row through a single-line prefix, which draws a literal newline as
  `^J` rather than breaking the line - so the blank line above the question, and every line break in
  a multi-line question, arrived as `^J`. Each line but the last is now its own row above the input.
- `Job(action="wait")` can no longer hold the agent indefinitely. Waiting with no timeout blocked
  until the process exited, which is exactly what a model does right after a slow `Bash` is
  backgrounded — so backgrounding handed control back and the very next call gave it away again,
  for as long as the command ran. A wait now lasts 60s by default, the model can ask for up to
  900s when it knows the job is slow, and a wait that ends with the job still running says so
  instead of looking like a result.
- `Ctrl-C` now interrupts a `Job` wait. Only `Bash` was wired to the runner's cancellation, so a
  wait was unreachable from the cancelling thread and could not be taken back. Interrupting a wait
  abandons the wait only: the command keeps running and stays addressable through `Job`.
- Ctrl-C at a tool approval prompt no longer approves the call. The approval cancel returned an
  empty reply, which the confirmation loop read as the default "yes", so interrupting a Delegate
  (or any tool) confirmation silently started it instead of cancelling. Ctrl-C at an approval now
  refuses the call.
- Cancelling an approval is now its own signal rather than a stand-in string, closing the two
  remaining ways a cancel could still be read as an answer: `Ctrl-D` on an empty approval line
  approved the call, and so did quitting the app while an approval was waiting. Both now refuse.
  The placeholder text a cancel used to submit also no longer leaks — it reached the model as the
  refusal reason, and it became the literal answer to an `Ask` free-text question; Ctrl-C on an
  `Ask` page now dismisses the batch.
- The Delegate order viewer no longer overflows on CJK text. It wrapped by character count, so an
  order written in Chinese produced rows twice the terminal width and the overflow was cut off.
  Wrapping now measures terminal cells, follows a terminal resize while the viewer is open, keeps
  the indentation of wrapped code lines, and shortens its key legend on narrow terminals.

### Changed
- `runtime.max_agent_steps` defaults to `400` instead of `200`. The cap is a runaway stop, not a
  budget, and long tool loops were hitting it as ordinary work rather than as a fault.
- The note on a backgrounded `Bash` leads with `status` rather than `wait`, and says to keep
  working. It is read at the moment control comes back, and waiting is what gives it away.


## 0.22.1 - 2026-08-09

### Fixed
- A whole-file `Read` no longer echoes `1:0` back to the model. The tool message repeats the call's
  ranges into the conversation, and `end` 0 is the "to the end of the file" sentinel, so a read the
  model made by omitting `ranges` came back showing a range it never wrote — one that, under
  1-based inclusive bounds, reads as empty. A whole-file read now echoes just the path, and
  `[[50, 0]]` echoes `50:`. Reading from a line to the end of the file is documented in the `Read`
  schema instead of being an undocumented sentinel.


## 0.22.0 - 2026-08-09

### Changed
- Line numbers are now 1-based everywhere the model sees them, and line ranges include both ends.
  `Read`, `Search`, `InspectCode`, edit anchors, and `Recall` ranges previously counted from 0 with
  an exclusive end, which disagreed with every other source of line numbers in the same context —
  `grep -n`, Bash output, tracebacks, diff hunks, editors. A line reported as `anchor=91:49shj` is
  now the line `grep -n` calls 91, and `Read` with `ranges: [[91, 91]]` returns exactly that line.
  Requires `code-symbol-index>=0.4.0`, which makes the same change to `InspectCode`'s output.
- Anchors captured in sessions created before this change decode one line low. The content hash
  makes that safe: such an anchor is either relocated to the line it actually describes or refused
  with `stale anchor`, and cannot silently apply to the neighbouring line. Re-read the file when
  one is refused.


## 0.21.4 - 2026-08-07

### Changed
- `Ctrl-R` history search: `Enter` now ends the search with the match placed in the input box
  instead of submitting it immediately, so the text can be reviewed or edited first; a second
  `Enter` sends it. `Ctrl-C` and `Ctrl-U` during a search abort it and restore the input that
  was there before the search started.
- A focused quick-hint chip: `Enter` now only fills the chip text into the input box instead of
  submitting it immediately, so the text can be reviewed or edited first; a second `Enter` sends
  it. The input placeholder now reflects the two-step flow.

## 0.21.3 - 2026-08-06

### Fixed
- Keep a separate append-only CLI transcript when model context is compacted, so resuming a session
  still shows its original user/assistant turns, tool calls, failures, and Edit previews. Transcript
  checkpoints no longer re-hash the full history, provider-only continuation data and large tool
  arguments are excluded, and tool results replay by call id instead of name/order. Successful tool
  output containing the text `status: failed` is no longer misclassified, and interrupted calls get
  an explicit failed transcript result. Existing snapshots remain loadable and migrate the history
  they still contain on their next save; if an older minacode later writes the same log, resume warns
  that the transcript may be incomplete.
- Persist automatic current-turn compaction in the live turn instead of rewriting a throwaway
  request copy, so the next step or a post-Ctrl-C “continue” does not compact the same oversized
  prefix again. A queued live follow-up now commits its staged compacted turn after the provider
  accepts it instead of compacting the same prefix again. Cancelling compaction now applies
  deterministic trimming before the turn stops.
- Enforce `provider.response_timeout` as a real caller-side deadline even when an SDK's synchronous
  request ignores cross-thread `close()`, and cap compaction requests at 60 seconds before falling
  back to deterministic trimming. Timed-out or cancelled request threads can no longer publish late
  stream previews, completed output, or provider-side tool activity into a later turn.


## 0.21.2 - 2026-08-06

### Fixed
- Compact long active turns around the latest follow-up instead of retaining the entire prefix before
  that user boundary, which could leave a session permanently over budget and make every resume enter
  compaction again. The compacting divider now includes elapsed time, preserves its phase around model
  retry waits, and reports the provider error when compaction falls back to deterministic trimming.


## 0.21.1 - 2026-08-05

### Fixed
- Suppress the MCP client transport loggers (`mcp.client.streamable_http`, `mcp.client.sse`,
  `mcp.client.stdio`). They log expected, already-surfaced failures (an `httpx.ReadTimeout` on a
  slow server, dropped SSE/stdio frames, JSON-RPC parse errors) at `ERROR` with full tracebacks via
  `logging.lastResort`, which dumped them onto the TUI mid-render (e.g. the `Error in post_writer`
  traceback that appeared behind `/provider` while background discovery timed out).
  `MCPManager` already captures these same failures into `server_errors` and the status bar, so the
  library's own transport traceback was pure noise.
- Retry streaming transport errors that the provider SDKs raise unwrapped. The SDKs' `Stream` iterator
  re-raises httpx failures directly instead of wrapping them as `APIConnectionError`, so a server dropping
  the connection mid-stream (`httpx.ReadError`) or closing before the chunked body completes
  (`httpx.RemoteProtocolError` — "peer closed connection without sending complete message body
  (incomplete chunked read)") reached `retryable_error` with an httpx cause. httpx transport errors don't
  inherit `OSError`, so the existing `ConnectionError`/`TimeoutError` isinstance checks missed them and the
  first failure surfaced as `Error: ...` instead of retrying. httpx `TransportError` is now retryable, the
  same class of transient failure as the SDK connection/timeout errors already handled.


## 0.21.0 - 2026-08-04

### Added
- A `[runtime] language` key and a `/language [NAME]` command force the reply language for the
  session. `auto` (the default) injects nothing, keeping requests byte-identical to before; a
  language name (e.g. `Chinese` or `简体中文`) appends one fixed `LANGUAGE OVERRIDE` block to the
  tail of the system prompt. The block is a pure function of the value — no timestamps or session
  state — so the cacheable system prefix is unchanged, and workers inherit the setting from the
  parent on every `Delegate` send.

### Changed
- Confirm a `Delegate` send even under `yolo`. `yolo` means "I trust you to edit files and run commands
  without asking", and those mistakes are visible in the diff or the command output at once; a
  delegation's mistake is the order text, which only surfaces a whole worker round later. The
  prompt shows the order, so it stays the one cheap check on a spec the model wrote for itself.
  `Delegate` status and reset still follow `yolo`, as does every other tool.
- A successful `Edit` now refunds the fresh anchors for the region it changed (the same
  `anchor=line:hash` lines `Read` shows), so consecutive edits to the same file can keep going
  without a re-Read; `create` and `replace_all`, which have no per-hunk region, skip the refund to
  keep output bounded. Anchor validation is unchanged: a stale anchor is still refused with the
  current line echoed.
- Model request retries now back off exponentially with jitter and honor the provider's
  `Retry-After` header (clamped to a 30s ceiling) instead of a fixed linear 0.5/1.0/1.5/2.0/2.5s
  ramp; the total budget goes from about 8s to about a minute. The wait is interruptible (observed
  in ~0.1s slices, so Ctrl-C aborts promptly) and shows a live countdown in the status bar. Only the
  pacing changed: which failures are retryable is untouched, and a single aberrant header cannot
  stall the CLI.
- A retry backoff wait now shows as its own live phase (`retrying`) instead of claiming the agent is
  `working`, mirroring the existing compaction phase. While the wait lasts, the divider shows the
  `retrying` label with the attempt count, reason, and a live countdown for its whole duration, no
  longer capped by the two-second notice window; the notice still lingers for two seconds after the
  wait ends, and the status bar keeps `attempt N/6`.
- Reword the Edit `start`/`end` anchor parameter descriptions: the anchor must be the exact current
  `line:hash` value copied verbatim from Read, Search, or InspectCode, never invented or calculated,
  and re-read after any file change or stale-anchor error. The old wording only named the format and
  the inclusive range, giving the model no instruction against deriving anchors itself.
- The status bar shows the worker only while a delegation is actually in flight: it leads with a
  yellow `[worker]` marker and reads the worker's provider, model, reasoning, and context/cache
  usage, returning to the parent's values the moment the worker answers (the engine clears the
  in-flight turn in `finish_turn`). An idle worker no longer shadows the parent's row. The working
  divider carries the same yellow `[worker]` prefix while a delegation runs. `/status` is now
  sectioned — common workspace/session/goal/runtime first, then the parent's model/context/cache/
  activity/usage, then the worker's model/context/state rows whenever a worker session exists (or
  the configured `[worker] provider` line when none does).
- The worker delegation bracket is now visibly closed on both ends: the finish block of a
  successful `Delegate` send or reset carries the same yellow `[worker]` identity as the start
  marker, and a reset states in the log exactly what it cleared and what survives (file changes
  and merged diffs stay). `/worker reset` answers in plain terms too, instead of echoing the raw
  envelope. `/status` sections lost the spare blank lines between a heading and its table.
- The delegation bracket is now a pair of full-width rule dividers in the same family as the
  turn-end line: the start marker and the finish of a successful send or reset each render as a
  gray rule carrying a yellow label (`worker · provider/model · order summary` and
  `worker done · steps · elapsed · tokens in/out · files`, or `worker reset · context cleared`),
  with the order summary capped at 60 characters and the file list at 48; when the worker touched
  no files the files segment is omitted from the done label instead of reading `(none)`. Without a
  wired UI the
  old `[worker] ▶` / `[worker] ◀` log lines remain, and the finish block still previews the
  worker's answer and its stored key.
- The `Delegate` description now leads with when to delegate: bounded, verifiable work you can
  spec in one order, bought for context hygiene and the worker's model, never speed; small work,
  open exploration, and the heart of the current request stay in the parent session, since
  writing the order and reviewing the result cost about as much as doing the work.
- `Delegate` send confirmation now shows an approval brief under the prompt: the title, an
  excerpt of the first 12 order lines (with a trailing "… N more lines" when the order is longer),
  any explicit `language`/`max_steps`, and a `worker:` line reading the effective
  provider/model/effort/api. The send prompt accepts `c` (or the whole word `config`) to open a
  small configuration loop — provider/model/effort/api with `off`/`default` clearing, each change
  live-applied to a running worker — that returns to a redrawn brief before confirming. A
  `c`-prefixed reason sentence never opens the loop: only a whole-line exact `c`/`config` does.
- The `[worker]` section and `/worker api [API]` now set a `worker.api` protocol override
  (`auto|chat|responses|anthropic`, empty = inherit the provider entry's own protocol); the live
  worker's entry is refreshed the same way `/worker model|reason` already did. The three
  copy-on-write refresh blocks behind `/worker provider|model|reason|api` are now one shared
  helper, `refresh_worker_entry`.
- The one-time per-task Delegate authorization is retired: the confirm prompt's `a` key and
  `/worker auto on|off` are gone, so every `Delegate` send asks again (still even under `yolo`),
  and the task-boundary reset hooks that cleared the flag are removed with it.
- The system prompt's SCOPE now gates action on intent: discussion -- questions, opinions,
  proposals -- is answered only, action does exactly what was instructed, ambiguity between the
  two resolves to discussion, and a reply that rejects or narrows a proposal approves only what
  it explicitly accepts.
- The `Delegate` send finish block now shows what that delegation cost: the envelope records the
  worker's prompt/completion token delta for the send (the program subtracts `worker.usage` before
  and after the run, never the model's word), and the summary line renders it as e.g.
  `8.2K in / 1.3K out`. Envelopes written before the `tokens` attribute still parse and simply
  omit the token part.
- The delegation start divider now reads `worker start · provider/model · order summary` instead
  of `worker · provider/model · order summary`.
- `/status` is back to markdown table rendering: the common workspace/session/goal/runtime rows
  lead with no `Common` heading, then the `### Parent` and `### Worker` headings introduce the two
  sections. The dense borderless-rendering experiment is removed.
- `/status` renders compact: the command path passes a new `compact` flag to `emit_answer` that
  drops every invisible line — the blank line rich markdown pads after each heading plus the
  whitespace rows above and below each table box — so each heading sits tight against its table.
- `Delegate` reset is no longer a full-width `worker reset · context cleared` rule. It is a
  one-shot tool call, so it keeps its ordinary tool root and a plain `done` child stating what it
  cleared and what survives; the delegation bracket (start + done dividers) applies only to sends.
- The `Delegate` start and done dividers now show a human-readable title: `send` accepts an
  optional `title` parameter, and when given it replaces the order-first-line summary on the start
  divider and leads the `worker done · steps · …` label; without one the dividers fall back to
  the order's first line as before.

### Added
- New `[worker]` config section and the `Delegate` tool: the model can hand a bounded task to a
  second in-process minacode session (the worker) that runs on its own configured provider, with its
  own system prompt and a reduced tool set, keeping context across delegations until reset. Enable
  with `[worker] provider = "..."` plus `runtime.worker = true` (or `/worker on`); `[worker]
  provider` defaults to disabled, and the spec suggests a different vendor than `provider.active` so
  the worker's reviews cross-validate the parent's. `Delegate` actions: `send` (with a free-text
  order and optional `max_steps`), `reset`, `status`. `runtime.worker` gates the tool block and thus
  the prompt-cache scope; turning it off does not clear the worker's context. `/worker` shows status
  and `/worker reset` clears the worker and tells the parent model via a session event.
- `/worker` now switches the worker's provider, model, and reasoning effort at runtime, mirroring
  the parent's `/provider` `/model` `/reason` temporary switches: `/worker provider NAME` re-targets
  the worker (`/worker provider off` clears the entry), `/worker model MODEL` and
  `/worker reason EFFORT` override the entry's model and reasoning effort, and `default` clears a
  model/reasoning override back to inheriting the entry's value. A change applies to a live worker
  immediately and to future spawns, and is session-scoped like `/provider`; the same defaults can
  be set persistently with the new `[worker] model` and `[worker] reasoning` keys. The `Delegate`
  tool block is fixed at session start, so a runtime switch tunes an already-enabled delegation but
  never flips the tool block mid-session, and enabling delegation from scratch (no `[worker]
  provider` at start) takes effect after a restart. The worker's active provider entry is now always
  a detached copy, so a switch can never leak into the parent's provider.
- A finished `Delegate send` now renders as a proper log block: the root line reads `Delegate
  send` (no argument blob), a summary line reports steps, elapsed, files, and a stopped-at-max-steps
  flag, and the worker's answer is previewed below like a Bash transcript — the raw envelope tags
  stay out of the log.
- The log now marks where a delegation starts: a yellow `[worker] ▶ provider/model · order` line
  is emitted just before the worker runs (the order collapsed to one line, 200 chars), so the
  scrollback has a boundary until the finish block closes it.
- `/worker` (no arguments) now answers in readable lines instead of the raw `Delegate` status
  envelope the model sees: one line per fact (provider/model, reasoning, state, rounds, context
  percent), or `worker: no active session` plus the configured `[worker] provider` when no worker
  exists. The model-facing `Delegate status` envelope is unchanged.
- `/worker provider`, `/worker model`, and `/worker reason` without an argument now open the same
  pickers as `/provider` `/model` `/reason` (provider entries plus `off`, the worker entry's
  models plus `default`, reasoning efforts plus `default`), and `/worker` tab-completes its
  subcommands and their values.
- `/worker provider` without an argument now flows on into the worker model and then the
  reasoning pickers after the provider is set, mirroring the parent's `/provider` chain, so a
  fresh worker setup is one selection flow instead of three commands; backing out of any stage
  keeps the stages already set and the returned message says what landed. The typed
  `provider NAME` still sets only the provider, and the worker keeps the provider entry's `api`
  — there is no `/worker api`.
- `Delegate` send accepts an optional `language`: the user watches the worker's live stream and
  reads its report, so the tool appends an explicit reply-language directive to the order covering
  everything the worker outputs, not just the final answer. The tool description tells the parent
  to pass the user's language unless the user works in English.

- `/status` is now one flat table instead of three stacked ones. The `### Parent` / `### Worker`
  headings and the two repeated header rows are gone; the worker's facts join the same table under
  `worker`, `worker ctx`, and `worker cache` labels, collapsing to a single `worker` row naming the
  configured `[worker] provider` when no worker session exists. The header column is `field` rather
  than `status`, and the `cache` row reports read ratios (`last 99.9% (w 1.2K); session 83.4%`)
  instead of the raw token pairs that made it the one row long enough to wrap.

### Fixed
- Fix `chat_reasoning = "enable_thinking"` sending a `thinking_budget` that a configured
  `provider.max_tokens` cannot cover: these hosts fold `max_tokens` into `max_completion_tokens`
  and reject a budget that is not strictly below it (`max_completion_tokens [16384] must be
  greater than thinking_budget [16384]`), so `max_tokens = 16384` with `reasoning = "xhigh"` or
  `"max"` failed every request. The budget now stays under the cap, the same clamp the Anthropic
  wire already applied; an unset `max_tokens` still leaves the budget to the host.
- Fix running-mode Ctrl-P/Up being a no-op while the input box holds text: the old handler
  returned after `cursor_up()`, which does nothing on a single-line buffer, so history recall and
  queued follow-up recall both silently failed whenever the draft was non-empty. The key now
  behaves like it does with an empty box: recall the latest queued follow-up first, otherwise
  walk history.
- Fix a worker crash when the worker model emitted text beside a tool call: the worker's output
  wrapper assumed every Agent emission was a LogLine and built `LogBlock([str])`, which
  `LogBlock.walk` rejects with `'str' object has no attribute 'walk'` at render time. Bare strings
  are now wrapped into LogLine items before reaching the parent's log stream.
- Wire the worker's model stream into the parent's live display: `ModelClient` only streams when
  `on_stream` is set, so the worker ran unstreamed and its reasoning and response progress never
  reached the CLI. Delegations now reuse the parent's stream preview.
- Make `/worker reset` finish the worker runtime it discards: stop and clean up its background
  jobs, remove a disk-only worker after the parent is resumed, and keep the live worker reachable
  with an actionable error if its snapshot cannot be deleted. This prevents orphaned processes and
  prevents a reported reset from silently restoring the old context on the next delegation.
- Ignore `/resend` during an automatic retry countdown, when no model request is in flight, and
  preserve a resend that races the start of that wait as a retry instead of cancelling the turn.
- Expire a worker in the same retention sweep as its expired parent regardless of directory scan
  order, so a fresh subordinate snapshot cannot survive as a hidden orphan.
- Reject non-integer or non-positive `Delegate.max_steps` values at the tool boundary instead of
  running an empty worker turn with an invalid budget.
- Estimate request tokens from UTF-8 bytes instead of characters (4 bytes/token), so CJK-heavy
  sessions are no longer undercounted about 3x: the status bar could show 100% while the next request
  was still estimated under budget and auto-compaction never fired. ASCII payloads estimate exactly
  as before. The tool-output trimmer keeps its chars/4 measure, and a last line of defense forces
  compaction when the previous request filled >=99% of its budget even if the estimate still fits.
  Compaction clears the recorded last-* usage (the compaction request's own fill would otherwise be
  mistaken for a full ordinary context and double-compact the just-shrunk history); the status bar
  falls back to the local estimate until the next ordinary request reports real usage.
- The delegation finish block's root line is now a pure closing divider `[worker] ◀` for both
  `Delegate` send and reset: the action name no longer crowds the boundary, and the details — the
  steps/elapsed/files summary or the reset notice — live in the child lines.
- Worker mid-turn text no longer repeats in the TUI: the worker's model stream
  now forwards to the parent's live display with `output_done` downgraded to a
  preview clear, because the worker's own output already lands in the parent
  scrollback and the parent loop's promote would write it a second time (the
  promoted-text marker is consumed only by the parent's own agent output path).
  Final reports and non-TUI output are untouched.


## 0.20.0 - 2026-08-03

### Fixed
- Show the thinking preview on Responses hosts that stream the raw reasoning chain rather than a
  summary. Only `response.reasoning_summary_text.delta` was recognized, so a provider that emits
  `response.reasoning_text.delta` — DeepSeek documents that it generates no summary at all — ran a
  high-effort step with nothing on screen until the answer arrived. Stored history was already
  complete; only the live preview was missing.
- Report a generation the provider cut off at the output cap instead of failing the turn with
  `empty final response`. Reasoning counts against `provider.max_tokens` on the Responses and
  Anthropic wires, so a high-effort step could spend the whole budget and return neither text nor a
  tool call; the error now names the cap, the tokens spent, and how much of that was reasoning. A
  truncation that still produced text keeps its partial answer. The failure is deterministic and no
  longer consumes a retry.
- Treat `finish_reason=length` on the Chat wire as output truncation only when the reported output
  tokens actually reached the configured cap. OpenAI-compatible providers also return `length` when
  the input exceeds the model's context window; the error now names both `provider.max_tokens` and
  `runtime.max_context_tokens` instead of pushing the output cap blindly.

### Changed
- Default `provider.max_tokens` back to `0`, so Chat and Responses omit the cap and let the provider
  apply its own default instead of sending a hard-coded 16K that some models reject (Claude 3.5
  Haiku on Bedrock, for one, caps at 8K). The Anthropic wire still requires the parameter and sends
  a conservative 8K when unset. The 16K output reserve taken out of the input budget is unchanged.
- Raise the default `runtime.max_context_tokens` from 240K to 256K. Both defaults appear in the
  generated config with a note that they trade against each other and that the context budget is how
  much of a model's window to use, not the window's size.
- Show the context fill in the status bar and `/status` from the provider-reported prompt tokens of
  the last request instead of the local estimate. The estimate still triggers compaction, which
  decides for a request that has not been sent yet; the reported usage describes the one that was.
- Show that fill against the budget the last request was prepared against
  (`session.usage.last_prompt_budget`) instead of recomputing the denominator from the current
  configuration, so `/set provider.max_tokens`, `/set runtime.max_context_tokens`, or a provider
  switch no longer moves the recorded percentage before the next request.


## 0.19.1 - 2026-08-03

### Fixed
- Never empty the tool list to force a live-follow-up acknowledgement. The extra request that
  carried `tools=[]` discarded the cached prefix and moved the request into another cache scope, and
  the model read the missing schemas as a broken tool set: it reported Bash, Edit, Search, and MCP
  as unavailable, pasted patches instead of editing files, and asked the user to restore them. The
  follow-up marker now just asks for a text acknowledgement in the same message as the next tool
  calls, and a tools-only response continues the turn as usual.
- Commit a live follow-up with the marker it was sent with. History stored the bare text while the
  request carried `[Live follow-up received while you were working]`, so the next request replayed
  different bytes at that position and ended the shared cache prefix there. The restored transcript
  hides the marker, so the scrollback still shows what the user typed.
- Commit textual tool-call corrections to history instead of sending them request-locally. What
  reached the provider is now what the next request replays, corrections stack rather than replace
  each other, and an aborted turn keeps the ones it already sent.

### Changed
- Write output for the terminal instead of a markdown renderer: reference local files as bare
  `path:line` rather than as markdown links, and drop banner headings, tables for short answers, and
  paste-back of file contents or command output the user already saw.


## 0.19.0 - 2026-08-02

### Added
- Ship a builtin `minacode-help` skill with an offline manual, troubleshooting guidance, a
  matching-version source inspection fallback, and an idle hint inviting minacode questions.
  Builtin skills use the same discovery and loading path as ordinary skills; user and project
  skills can override them by name.
- Add `max` reasoning effort and map normalized effort levels to each documented provider/model
  family, including OpenAI GPT-5 and o-series generations, Anthropic, DeepSeek, Qwen, Kimi, `Z.AI`,
  OpenRouter, and OpenCode Zen. Unknown providers and future model names retain generic
  pass-through behavior, and `/config` now shows the resolved effort sent to the active model.
- Add a `provider.builtin_tools` option (a list of tables, default empty) appended verbatim to the
  `tools` array of whichever protocol the provider speaks, so a provider's own server-side tools can
  be offered to the model. This is how provider web search is enabled: `{ type = "web_search" }` for
  OpenAI and Qwen on the Responses API, `{ type = "web_search_20250305", name = "web_search" }` for
  Anthropic, `{ type = "web_search", web_search = { enable = "True" } }` for `Z.AI`, and
  `{ type = "openrouter:web_search" }` for OpenRouter. Entries are passed through unchanged after a
  check for a non-empty `type` and against the provider's documented wire. Enabling an active entry
  changes the prompt cache key, and `/config` distinguishes configured entries from the active
  projection. One provider configures search through the request body instead — Qwen Chat's
  `enable_search` — and continues to use `provider.extra_body`.
- Scope known `provider.builtin_tools` entries to their documented request wire. Responses-only
  entries (OpenAI, Qwen) are omitted on Chat or Anthropic, Anthropic server tools only travel over
  Messages, and `Z.AI/Kimi` Chat entries only over Chat. Incompatible entries remain configured and
  become active again after switching back, so a shared provider configuration works across models
  without destructive edits; `/api` and `/config` report when they are inactive. On an active wire,
  known providers still reject unsupported entries (for example Qwen `code_interpreter` or
  Anthropic `web_fetch_*`) before SDK I/O, so approval, file, container, and client-callback
  lifecycles cannot leak through. OpenRouter server tools (`openrouter:web_search`,
  `openrouter:web_fetch`, `openrouter:datetime`) are supported; DeepSeek, Kimi Code, and OpenCode
  activate none. Unknown hosts and future tool shapes on unknown hosts keep generic pass-through.
- Log each provider-side tool call the provider reports
  line, and show it as a running status phase while it happens. The line is written from the parsed
  response, so it appears with streaming on or off.
- Answer a provider's own builtin function calls, so Kimi's `$web_search` (declared as
  `{ type = "builtin_function", function = { name = "$web_search" } }`) completes instead of
  failing as an unknown tool. The declared call is answered with its arguments verbatim, as that
  provider's protocol requires, without confirmation or result storage.
- Resume an Anthropic turn the provider paused mid-search (`stop_reason: "pause_turn"`) by sending
  the message back unchanged, instead of ending the turn on what looks like a complete answer.
  Each resumption counts as one agent step, so it stays bounded by `max_agent_steps`.
- List the sources a provider-side search reported under the answer. Sources are display only: the
  stored answer stays exactly what the model wrote, and nothing extra replays to the provider.

### Fixed
- Prevent a forced live-follow-up acknowledgement from retaining local tool calls returned despite
  tools being disabled, which could leave invalid history and make the next answer look duplicated.
  Provider builtin functions that were still offered remain available.
- Promote completed streamed text before provider-side tool output, so Responses web search and
  Anthropic server-tool activity no longer make the answer preview disappear before final output.
- Publish only the text written after a provider-side tool instead of repeating the promoted
  opening: a search runs inside one response, so the model keeps writing after it and the answer
  previously appeared twice in the transcript.
- Keep token estimation working when a configured `provider.builtin_tools` entry is unsupported on
  the active wire. `/status`, the status bar, and resuming a session only measure the payload, and
  raising there took down the frontend over config no request had tried to send; the request that
  would send it still fails with the same error.
- Keep the Anthropic extended-thinking budget under the request's own `max_tokens`, which the API
  requires. With the default `max_tokens = 8192`, `/reason high` and above sent a budget the model
  could not accept, so every request to a pre-4.6 Claude model failed until `max_tokens` was raised.
- Send the resolved reasoning effort on OpenRouter's top-level `reasoning` object, like every other
  reasoning control, so a documented per-model effort scale applies there too.


## 0.18.1 - 2026-07-31

### Changed
- Close each turn with a single full-width gray rule carrying its duration
  (`done in 1m05s ─────…`) instead of the plain `[done in 1m5s]` line. The label sits at the
  left of the rule rather than centered, and a blank line lifts the rule off the answer. It is
  the turn's only horizontal rule: the full-width rule that used to open the answer is gone, so
  a turn no longer shows two rules. The duration reuses the working divider's `elapsed_since`
  format, so the rule reads like the divider's final frame (`5s`, `1m05s`) instead of
  `0m5s` / `1m5s`.


## 0.18.0 - 2026-07-31

### Added
- Add test-only OpenAI behavior emulation for Chat Completions and Responses, including implicit
  user/tool breakpoints, exact-prefix cache reads, and cache-write accounting.
- Let `Note` view selected working-state fields, and let `RecallContext` list compacted history
  segments with pagination in addition to retrieving and searching them.
- Gate idle input tips on feature availability (`$skill` only with skills installed, `@server.tool`
  only with an MCP server connected), add `Paste an image path to attach it` and a `/ps lists
  background jobs` tip that appears while a job runs, and re-roll the tip each turn instead of
  pinning one for the whole session.
- Let `Shift-Tab` reverse-cycle the offered next-step chips at the idle prompt, mirroring `Tab`.

### Changed
- Make model context append-only between compactions: remove the rebuilt per-request Memory and
  history-index blocks, retain Note changes as tool history, and emit one complete working-state
  checkpoint when compaction starts a new cache epoch.
- Record session start and resume times in the user's local time with an explicit numeric timezone
  offset. Resume is now a durable user-role lifecycle event shared by Chat Completions and
  Responses; old snapshots migrate through one complete state checkpoint.
- Show provider-reported cache-write tokens alongside cache reads in `/status`.

### Fixed
- Keep Ctrl-C silent at an idle prompt: it clears a draft when present and otherwise does nothing,
  while an actual in-flight request interruption still reports `Cancelled`.
- Keep compaction, migrated-state, and resume lifecycle checkpoints out of resumed transcripts;
  they remain available to the model without exposing internal context bookkeeping in the CLI,
  and a checkpoint-only resume still confirms which session was restored.
- Treat strict-schema nulls, empty selector arrays, and irrelevant default arguments as omitted in
  `Note` and `RecallContext`, including `Note`'s user-facing call summary.
- Show empty `Note` goal and check updates as explicit clears instead of rendering them as `{}`.
- Keep exactly one blank line between assistant progress and tool calls, and between consecutive
  tool calls, without splitting a tool's header from its result.
- Keep large stale code indexes off the interactive startup path, so repository scanning and
  re-parsing cannot delay the first text a user types. Existing availability is read cheaply at
  startup; bounded freshness checks continue after turns, and large syncs remain explicit.
- Keep a session's name once it is latched from a goal: revising the goal no longer overwrites the
  name, matching how user-set names already behave.


## 0.17.0 - 2026-07-29

### Added
- Name sessions and find them again. Every session takes a name from its opening message, then
  from the agent's goal once there is one, and `/name TEXT` sets your own that nothing overwrites.
- `/sessions` browses saved sessions and re-enters one, with search, ages, and turn counts;
  `/sessions all` widens past the current project. Choosing a session ends the current run and
  starts the next one on it.
- `--resume` accepts a name or a uid prefix, not just a full uid. An ambiguous query lists the
  sessions it matched instead of guessing between them.
- The model can offer 2–3 short next-step prompts after its answer with `NextHints`; they show as
  selectable chips at the idle prompt (Tab cycles, Enter submits), `/hints` toggles them off, and an
  all-`NextHints` batch ends the turn in a single model call so the answer appears once.

### Changed
- Cap model output at 8,192 tokens per request by default instead of leaving compatible providers
  unbounded; set `provider.max_tokens = 0` to use the provider default.
- A successful `NextHints` call prints no call/result log line: its effect shows as chips at the idle
  prompt, so the line was noise. Failed calls still surface their error.

### Fixed
- Keep terminal `NextHints` turns from replaying the same provider response twice, which made the
  next Responses API request fail with a duplicate message ID.
- Stop replaying a streamed answer at the end of a terminal `NextHints` turn. The batch promotes its
  answer into scrollback like any tool batch, but the post-turn emit then printed the same text again.


## 0.16.0 - 2026-07-29

### Added
- Show the latest request's prompt-cache hit ratio live in the status bar, beside the context
  fill as `ctx 23% · cache 98%`, refreshing with each model response so a drop in prefix reuse
  is visible without running `/status`. The ratio stays hidden until the first request.

### Changed
- Declutter the status bar. The cache ratio folds into the context segment instead of taking its
  own field, and the `step N/M` counter now appears only once the turn reaches the final fifth of
  `max_agent_steps`, since a count like `step 1/200` carries no information
  far from the cap and only matters as the turn approaches the cutoff that ends it.
- Smooth and lighten the working divider. Its comet head advanced 6.8 cells between redraws, more
  than the width of its own glow, so the motion read as a dash blinking at scattered positions; the
  animation now runs on a 30fps ticker while the running region is up, at one cell per frame, and
  fades between cells so a late frame does not snap. The head is a softer cyan fading into a muted
  rule instead of bold bright cyan over the terminal's default gray.
- Lighten the status bar. Its working sweep is a softer crest over a quieter gradient, quantized
  into bands so neighbouring cells share a color: the terminal receives about a third of the escape
  sequences per frame and the set of styles stays bounded instead of growing for the life of the
  process. The sweep now follows the theme, so a light terminal gets a dark crest rather than a
  washed-out bright one, and the idle line sits a shade below full white.

### Fixed
- Keep a streamed response's promotion below the live follow-up it answers. Follow-ups reach
  scrollback only once the request that carried them returns, so promoting mid-stream printed the
  answer above the message that prompted it.
- Count Anthropic's cache tokens in the prompt total. `input_tokens` excludes what was read from or
  written to the cache, so a warm request reported a cache ratio far above 100% and a token total
  covering only the uncached remainder.
- Promote completed streamed response text from the dim `responding` preview into terminal
  scrollback before tool output begins. Responses and Anthropic use their explicit text/tool block
  boundaries without assuming which arrives first; Chat Completions promotes at its tool-call
  finish boundary.
- Label the working divider `compacting context` during automatic context compaction, then restore
  `working` before the normal model request, instead of presenting a long compaction call as generic
  work while the context meter remains full.


## 0.15.0 - 2026-07-28

### Added
- Report when retention deletes saved sessions, naming the count, the inactivity window, and the
  setting that governs it, instead of removing unrecoverable work silently.

### Changed
- Start roughly four times faster, so a fresh prompt accepts and echoes input immediately instead
  of lagging behind the first characters typed. The Anthropic and OpenAI SDKs cost about 0.8s to
  import and are not needed until the first request; they now load lazily and are warmed in the
  background while the prompt is already accepting input.
- Sweep expired sessions on a background thread. The scan reads every session file's timestamp,
  which is negligible on a local disk but can take seconds on a network-mounted home directory,
  where it previously delayed the prompt.

### Fixed
- Cap the input history file at 512 KB instead of letting it grow for the life of the install.
  prompt-toolkit only appends to it, so every line ever typed was kept. The newest entries are
  retained and older ones dropped at an entry boundary, so recall keeps working and the file stays
  loadable.


## 0.14.1 - 2026-07-28

### Added
- Add a `ViewImage` tool that lets the agent proactively inspect local PNG, JPEG, WebP, and single-frame GIF files through the active model's image input, with confirmation for paths outside the workspace and durable provider-neutral replay across Chat Completions, Responses, and Anthropic Messages.

### Changed
- Drop the per-request response-language reminder added in 0.14.0 and state the contract once, in the system prompt. The reminder sat immediately before the user's request as a block of English prose, which reinforced the English drift it was meant to prevent; every visible message and reasoning step must use the dominant language of the user's recent substantive messages from its first token, without translating at the end or switching after tool results.
- Reduce fixed model-context overhead by consolidating repeated system instructions and removing tool signatures and redundant examples already expressed by each tool's JSON Schema, while retaining concise system guidance for tool selection and lifecycle. Tool names, parameters, and execution behavior remain unchanged. Keep visible model output concise by default without limiting detail when requested or necessary.

### Fixed
- Recover up to five times when a model prints a known tool's `<invoke>` markup as final text: discard each malformed response, clear its live preview, and retry with a request-local native-call correction. A sixth textual call stops without execution, session-history pollution, or a misleading done marker.
- Let Edit safely relocate a stale anchor when the same line hash is unique across the file and moved by at most 50 lines; ambiguous, distant, and content-changing edits remain rejected.
- Accept a model's harmless duplicate of Edit's top-level path inside an operation while continuing to reject conflicting nested paths and other unexpected fields.


## 0.14.0 - 2026-07-27

### Changed
- Keep agent action within the phase the user actually requested: preparation, investigation, and plans no longer imply permission to implement, and live follow-ups can pause or narrow work immediately. Calibrate reasoning depth to task risk so routine, reversible work proceeds once the next step is clear while ambiguous or high-risk decisions retain deeper analysis.
- Preserve Chat reasoning in session history while replaying only the history each provider documents: full preserved thinking where required or explicitly enabled, tool-call reasoning within active tool loops for providers that clear older thinking, and DeepSeek tool-call reasoning across later requests. Context and compaction estimates now follow the effective Chat, Responses, or Anthropic wire payload instead of counting normalized fields that will not be sent.
- Add a generic 10-minute `provider.response_timeout` hard limit across streaming and non-streaming model calls, separate from the 120-second transport inactivity timeout. The hard limit closes the active request without automatic retries; set it to `0` to disable it.

### Fixed
- Reinforce the user's response language at the latest-user request boundary on every model step, including visible reasoning and progress, without rewriting stored session history.
- Preserve OpenRouter's `reasoning`, `reasoning_content`, and structured `reasoning_details` in both streaming and non-streaming Chat responses, and retain nested `thinking.keep` or `thinking.clear_thinking` settings when minacode adds its managed thinking mode.
- Allow `Edit create` in an existing directory outside the workspace while still refusing to create missing external parent directories implicitly. The error now tells the agent to create such a directory with an approved `Bash mkdir`, and Bash's tool description clarifies that commands start in the workspace rather than being confined to it.
- Move follow-ups already sent in the active request above the working divider while retaining them internally for safe retry, keep only unsent input in the queue region, and let recalled-input retries return to the normal `retrying`, `thinking`, and `responding` states instead of leaving a stale `revising queued input` label.


## 0.13.0 - 2026-07-26

### Added
- Add top-level `minacode update` and `minacode upgrade` commands that check PyPI and use the detected installer (`uv tool`, `pipx`, or `pip`) when an upgrade is available; startup hints now show the same command.
- Stream reasoning and answer text by default over OpenAI-compatible Chat Completions, OpenAI Responses, and Anthropic Messages. The running TUI now distinguishes `thinking` from `responding` while preserving the completed Rich transcript, tool-call replay, usage accounting, cancellation, and retry behavior.
- Add the generic `provider.stream` setting, enabled by default and switchable for the current session with `/set provider.stream on|off`, so compatible endpoints that reject streaming or Chat `stream_options` retain a non-streaming path without provider specialization.
- Accept local image paths directly in the interactive prompt, render recognized files as editable inline labels, preserve them across queued follow-ups and resumed sessions, and send the corresponding standard image content through OpenAI-compatible Chat, OpenAI Responses, and Anthropic Messages without provider- or model-name specialization.
- Allow the generic `provider.image_input` setting to force or disable image input, and learn support conservatively from successful requests or explicit modality errors so later known-unsupported submissions keep their draft.
- Add `/api` to select or set the request protocol during a session, and confirm it as a step in the `/provider` and `/model` selection chains. An endpoint serving several model families rarely serves them all over one protocol, and an OpenAI-compatible `/models` listing does not say which serves what, so a discovered model could be selected and then rejected as unsupported with no in-session way to change the wire. The reply names the protocol that took effect rather than echoing `auto` back.
- Support OpenAI's Responses protocol with `provider.api = "responses"` (also inferred from a `/responses` URL), including standardized reasoning effort, flattened function tools, tool-result round trips, cached-token usage, stateless requests, and replay of opaque reasoning items across turns.
- Document provider setup alongside the rest of the configuration, keeping the user-facing guide focused on settings rather than internal compatibility profiles.

### Changed
- Render `/help` as structured Rich Markdown in interactive terminals while retaining a clean plain-text fallback for redirected and non-TTY output.
- Keep long-running model requests visually neutral instead of turning the status bar red or prompting `/resend` based only on elapsed time; real retries and their causes remain visible.
- Require exposed reasoning and thinking summaries to follow the user's current language, alongside progress updates and final answers.
- Consolidate image recognition, asset storage, protocol payloads, token estimates, and learned capability into one session-owned `ImageInputs` component instead of module-level helpers and Session forwarding methods.
- Expand CLI and command validation coverage, remove an import-order dependency from resume tests, and replace fixed integration-test waits with event-driven synchronization; the full suite now finishes substantially faster while retaining real tmux, subprocess, and signal coverage.
- Split the oversized tool test module into focused core, edit, and Bash/Job suites, and expand shared validation, strict-schema, optional-capability, and background-job coverage.
- Add Pyright to CI and tighten internal annotations around tool arguments, lifecycle resources, MCP operations, and validated dynamic input while preserving `Any` at JSON and third-party SDK boundaries where the value is intentionally open-ended.
- Consolidate the small protocol-adaptation helpers into `provider_compat`, removing a thin module boundary while keeping provider-specific behavior isolated.
- Split the terminal frontend into focused `loop`, `tui`, and `render` modules without adding forwarding layers or changing behavior.
- Keep up to three lines from each completed Bash output stream in the transcript, with `Ctrl-O` offering a larger 24-line preview for recent commands.
- Replace wildcard imports between minacode modules with explicit imports, and have Ruff reject any future `import *` usage.
- Make `/status` context and prompt-cache usage immediately readable with compact progress bars, token counts, and percentages, while retaining its Rich table and hiding empty detail.
- Lower the default provider request timeout from 180 seconds to 120 seconds; explicitly configured values are unchanged.
- Increase transient model retries from two to five, and show attempts with concise reasons in the running TUI, such as `retrying 2/6 · timeout`, then keep `attempt 2/6` visible while the replacement request continues.
- Resolve explicit settings and necessary provider compatibility overrides once into a typed request policy, keeping the default OpenAI-compatible path generic and protocol request builders independent from host-profile storage. Domain overrides match real subdomains, and OpenCode routing uses broad Claude, Qwen, and GPT families because its single base URL multiplexes different wire protocols by model.
- Detect OpenAI reasoning through the `o` and `gpt-5` model families instead of enumerating releases. Aliyun endpoints under `aliyuncs.com` use `reasoning_effort` only for the documented Qwen3.8 family, including `none` for `/reason off`, so existing `chat_reasoning = "auto"` configurations work without per-model overrides.
- Stop imposing the obsolete 32K output cap on DeepSeek; current models use the server-side default unless `provider.max_tokens` is explicitly configured.
- Raise the normalized `minimal` manual-thinking budget to the provider-supported 1,024-token floor instead of sending an invalid 256-token Anthropic budget.
- Add documented compatibility overrides for Kimi and `Z.AI` across their international and China endpoints. Kimi Code remains distinct from the open platform; both `Z.AI` regions share GLM-5.2+ `reasoning_effort`, Kimi keeps its documented `prompt_cache_key`, and `Z.AI` relies on automatic context caching.

### Fixed
- Prevent models from narrating assumed tool failures and issuing speculative retries before minacode has executed their pending calls by making the tool-result boundary explicit in the system prompt.
- Clear the transient thinking preview before printing `Cancelled`, so interrupted reasoning does not remain in terminal scrollback. Protocol-required reasoning data is still retained in the session.
- Preserve image references when the simple CLI combines resumed queued inputs; joining their text previously left internal image markers without attachments and crashed before dispatch.
- Reject an ambiguous streamed Chat tool-call fragment when a compatibility endpoint omits both `index` and `id`, instead of silently appending its arguments to the wrong parallel tool call.
- Keep configured `temperature` on unversioned Anthropic aliases when no thinking mode was inferred, and report a clean CLI error if `update`/`upgrade` cannot launch its detected package manager.
- Omit the optional Responses `reasoning` object for documented non-reasoning models such as OpenAI GPT-4.1; it supports the Responses endpoint but rejects reasoning controls, so both the default effort and `/reason off` now produce a valid request.
- Keep streamed Chat tool calls distinct when a compatible endpoint omits their indexes, and treat a repeated function name as the complete name instead of concatenating it into an unknown tool.
- Return partial Responses output when a streamed response ends as `incomplete`, matching the non-streaming path, while rejecting genuinely `failed` responses consistently on both paths.
- Keep automatic and manual compaction output out of the live `responding` preview; only the user-facing model request streams into the TUI.
- Learn image support only from requests that actually transmit images and resolve to an eligible 400, 415, or 422 response, preserving permission and unrelated provider errors; omit image-tile estimates when historical images are sent as labels.
- Keep internal image-capability state out of the prompt, and render actionable image-input errors in their own TUI row instead of exposing the newline as `^J` inside the editable input. Text-only schema rejections such as `unknown variant image_url, expected text` are recognized explicitly and name the active provider/model.
- Render Edit diff previews as solid red and green bands across every visual row, including line-number gutters and wrapped continuations, instead of dropping all padding whenever one changed line exceeded the terminal width.
- Send each known Claude generation the thinking configuration it accepts. Extended thinking (`thinking.type = "enabled"` with `budget_tokens`) is rejected with a 400 from Claude 4.7 onward, so those models and the 4.6 generation now use adaptive thinking with `output_config.effort`, while Claude 4.5 and earlier keep their token budget; Opus 4.5 also receives its documented effort field. Unversioned gateway aliases remain unconfigured rather than being guessed as current-generation models. `/reason off` disables thinking where that is allowed and is omitted on the always-thinking families, which reject it.
- Echo Anthropic assistant turns back verbatim, thinking blocks and signatures included, as the Messages API requires for multi-turn and tool-use conversations; they were previously rebuilt from text and tool calls.
- Route OpenCode's GPT models to their documented Responses endpoint, alongside the existing Claude and Qwen routing to Messages.
- Apply the resolved provider policy to Responses requests too: reasoning effort uses the active protocol's documented mapping, GPT-5.1+ Responses models receive `none` for `/reason off`, unsupported off settings report an error instead of silently using the model default, and `temperature` is omitted for OpenAI reasoning models that reject it. Sibling chat models such as `gpt-4o` keep their sampling control.
- Accept every reasoning wire format `auto` can select as an explicit `chat_reasoning` value, so gateways and unrecognized model names keep a manual override; `thinking_toggle`, `thinking_effort`, and `mandatory_thinking` were previously resolvable but rejected by config validation.
- Avoid double-counting visible assistant text and raw ciphertext/signature bytes in replayed provider items, while retaining readable Responses summaries and Anthropic thinking in the context estimate.
- Drop saved reasoning items that carry no encrypted payload or readable summary instead of replaying an empty shell that a stateless request cannot resolve.
- Omit `temperature` from Anthropic requests while thinking is enabled, which the Messages API rejects.
- Preserve the CLI transcript after closing `/diff` by using the terminal's alternate screen; when tmux has alternate-screen support disabled, render the diff inline instead. The probe now reads the resolved window option, so the usual global `set -wg alternate-screen off` is detected and not just a per-window override.
- Restore both cache scopes in `/status`: the visual bar is now explicitly labeled `last`, while `last` and `session` each show cached tokens, prompt tokens, and hit rate.

## 0.12.0 - 2026-07-21

### Changed
- Ctrl-C while the agent runs now distinguishes a *retract* from an *interrupt*. If the agent has not produced any output yet, the turn is retracted — the message is discarded and leaves no trace in the model context or the persisted session (the input history still recalls it via Ctrl-P), as if it was never sent. Once the agent has spoken or called a tool, the partial turn stands and an interrupt marker is appended (with a cancelled result for any tool call the interrupt left unanswered), so the context stays valid and the model knows the turn ended early.
- Fold completed Bash output into a quiet gray row instead of leaving its live preview in scrollback. `Ctrl-O` opens a newest-first browser for the ten most recent Bash previews, including persisted history after resume; select with `j`/`k` or arrows, open with `Enter`, return with `Esc`, and close with `Ctrl-O` or `q`. Blank rows separate live output from the working divider and each viewer's subdued labeled rule from scrollback. Full results remain stored under their `tr.N` keys.

### Fixed
- Background jobs (`Job(start)`) now capture every stage of a compound command such as `configure && make && pytest`; previously only the final stage was logged and earlier output leaked to the terminal.
- Opening the external editor no longer truncates a draft at a git-style scissors line you typed yourself — only the reference context minacode appends below its own marker is stripped.


## 0.11.0 - 2026-07-20

### Added
- Ctrl-X Ctrl-E (and Ctrl-G) now opens the external editor with the agent's most recent reply appended below a git-style scissors line, so you can read what you are answering while you compose; everything from the scissors line down is stripped before the message is sent. Long replies are capped to their most recent lines.
- Show a placeholder tip in the idle input while it is empty, picked at random for the session from a few shortcuts (`Ctrl-X Ctrl-E opens $EDITOR`, `Type / for commands`, `Ctrl-U clears the line`) so they stay discoverable without cluttering the prompt; it disappears as soon as you start typing and gives way to the queue hints while the agent runs.

### Changed
- Renamed the project from `nanocode` to `minacode`: the package and import (`import minacode`), the `minacode` console script and `python -m minacode`, and the PyPI distribution (`minacode`, formerly `nanocode-cli`). The data dir moved from `~/.nanocode` to `~/.minacode`; when the new directory is absent the old one is still read, so existing sessions, skills, and config keep working without a migration step.
- Split the single `nanocode.py` module into a `minacode` package of focused submodules — `base`, `session`, `skill`, `mcp`, `tools`, `engine`, `tui` — plus a `__main__` entry module (so `python -m minacode` now works alongside the `minacode` console script), all re-exported through `minacode/__init__.py` so `import minacode` keeps exposing the same namespace as before. Module-internal helpers (`_read_and_release`, `_validate_edit_target`, `_resolved_tool_schemas`) are underscore-prefixed and gathered at the bottom of `tools.py`. Behavior, the CLI entry point, and the public API are otherwise unchanged; this reorganizes the source so it is easier to navigate and to trim optional features later.
- Ask previews now ask for concrete graphic content (snippets, directory trees, file shapes) instead of restating the choice label, with the built-in example updated to match.

### Fixed
- Run background job commands directly under `bash -lc` instead of `exec {command}`. The `exec` prefix failed for shell builtins and compound commands (`cd dir && cmd`) with `exec: cd: not found`; `start_new_session` already makes the shell the process-group leader, so `killpg` still reaches the command and its children.


## 0.10.0 - 2026-07-19

### Added
- Clear the whole input line with `Ctrl-U`, in the idle prompt, the follow-up editor, and approval prompts alike. `Ctrl-C` keeps its existing meanings, including interrupting a running turn, so clearing a draft never competes with stopping the agent.
- Replay the diff each `Edit` made when a session is resumed, instead of only the call line. Long diffs are trimmed to a readable window with a pointer to `/diff`.
- Add a `provider.extra_body` option (a JSON object, default empty) merged verbatim into the outgoing chat request's `extra_body`, so provider-specific extensions reach the wire without nanocode knowing about them. This is how a server-side built-in capability such as Qianwen/DashScope web search is enabled (`{"enable_search": true, "search_options": {...}}` against `https://dashscope.aliyuncs.com/compatible-mode/v1`). nanocode's own reasoning fields are layered on top, so they stay authoritative on key conflicts, and the configured value is surfaced in `/config`.

### Changed
- Let `Ctrl-C` discard a draft in the follow-up editor. It previously interrupted the turn and left the draft in place, so there was no way to clear a half-typed follow-up while the agent worked. With the editor empty it still interrupts, which is exactly when the `Ctrl-C interrupts` hint is shown.
- Apply the `/diff` snapshot size limit to each file snapshot rather than to the before-and-after pair. Snapshots are stored once per unique content, so the pair no longer costs twice a file's size, and the old test held the ceiling at half the file size it could afford — files up to roughly 1 MB are now tracked.
- Store sessions per project under `<data_dir>/projects/<project>/`, each with its own `latest` pointer, instead of a single flat `<data_dir>/sessions/` directory shared by every project. Resolving this project's latest session is now a pointer read rather than a scan that opens every stored session, and a project directory is pruned once its last session expires.
- Scope `--resume latest` and `--resume last` to the current project. They previously resolved a single global pointer and could resume a session belonging to a different directory; they now behave like `-c`.
- Begin each session log with a header line recording the format version, UID, working directory, and creation time. The version gate turns an unreadable log into a clear error instead of a misparse, and the log describes itself when read by hand.
- Store the file snapshots behind `/diff` by content hash instead of inline in every edit record. Editing one file repeatedly stored each version twice, once as an edit's `after` and again as the next edit's `before`; each version is now written to the log once. Past 100 edits, where every save rewrites the whole retained window, a save rewrote every snapshot again — it now rewrites only the references.

### Fixed
- Persist the session immediately after `/compact`. The compacted history lived only in memory until the next turn, so leaving the session right after compacting resumed from the pre-compaction log.
- Recompute the context percentage when a session is resumed. The value is derived rather than persisted, so the status bar read `ctx 0%` on a restored session until its first turn — a session actually sitting at 100% looked empty.
- Summarize the work that followed the current request during `/compact`, not just the conversation before it. One request can drive dozens of tool calls, and keeping that tail whole left the context nearly as large as it started — compacting a 93-message session summarized 47 messages holding 101k characters while keeping 46 messages holding 209k, so the context stayed at 43%.
- Show one diff per file in `/diff` when a file's edits are only partly covered by stored snapshots, which happens once a file grows past the snapshot size limit mid-session. The snapshot-derived diff and the reconstructed one were both emitted, repeating the file's changes under two headers.
- Isolate `HOME` in the test suite. Tests that built a config without an explicit `paths.data_dir` and then saved a session wrote into the developer's real `~/.nanocode`.
- Describe `Ctrl-C` in the running follow-up hint as interrupting the turn rather than "sending immediately". `Ctrl-C` interrupts the current task and keeps any draft in the editor; it does not submit the draft, so the previous hint advertised an action that does not exist.

### Removed
- Drop reading of pre-existing session files. Sessions written by earlier versions are not migrated and are no longer loadable; `<data_dir>/sessions/` can be deleted. This also removes the recovery of edit diffs from legacy `Edit` tool records and the tolerance for retired cache-prefix state keys.


## 0.9.11 - 2026-07-19

### Fixed
- Discover `auto_connect` MCP servers in the background at startup, so an unreachable server no longer holds up the prompt for the discovery timeout.

### Removed
- Remove the undocumented `runtime.max_steps` configuration alias; set `runtime.max_agent_steps` instead.


## 0.9.10 - 2026-07-18

### Added
- Add `g`/`G` jumps to the top/bottom in the `/diff` viewer and interactive selectors, matching the `less` pager convention.
- Add an interactive `/mcp` connection manager and per-server `auto_connect` configuration for explicit startup connections.

### Changed
- Replace the in-flight model retry keybinding with `/resend` in the running follow-up input, and show a `/resend` hint during long requests.
- Open the current input in `$VISUAL`/`$EDITOR` with `Ctrl-X Ctrl-E` or `Ctrl-G`; a lone `Ctrl-X` no longer opens the editor.
- Make `Ctrl-P` mirror the Up arrow when recalling the latest queued message for editing.
- Connect MCP servers on demand through targeted mentions or `/mcp connect NAME`, keeping disconnected servers entirely out of model context. Connecting handles OAuth authorization when required; disconnecting unloads the server and clears its saved OAuth authentication.
- Let `/mcp connect` accept multiple server names, connect them concurrently, and complete each successive server argument.
- Align `/mcp` status columns and add compact connection-state markers for faster scanning.
- Render multi-server connection results as a compact vertical list instead of one wrapped paragraph.
- Color a consistent solid-circle MCP status marker green when connected, yellow when disconnected, red on error, and dim when skipped.
- Show the number of connected MCP servers in the `/status` context row.
- Make each selection in `/mcp` toggle that server in place, showing connection progress without closing the selector, and limit `/mcp tools` listings and completion to connected servers.

### Removed
- Remove the unreachable non-TUI `Ctrl-\` model-request retry; interactive retries use `/resend`, while transient errors still retry automatically.
- Remove `/debug` cache-prefix diagnostics and persisted prefix fingerprints; prompt caching and `/status` cache-hit metrics remain unchanged.
- Remove startup tips and the built-in `nanocode-help` skill; `/help` and project/user skills remain available.
- Remove the `--mcp` per-run selector and per-server `enabled` behavior; comment out unwanted server sections instead. Legacy `enabled` keys are ignored.

### Fixed
- Retry OpenAI and Anthropic connection/timeout failures, retryable client statuses, and every provider `5xx` response.
- Bound MCP calls by timeout, recreate failed async loops, and close the MCP loop safely during shutdown.
- Keep concurrent MCP discovery status accurate, serialize OAuth authorization in the manager, and expose resource-only servers through `@server` mentions.
- Let explicit OAuth MCP connections validate cached credentials and reauthorize interactively when the server rejects them.
- Treat redirects to paths that merely start with `/dev/null` as writes requiring confirmation; only the exact device remains auto-approved.
- Prevent the plain CLI Bash live preview from accessing a nonexistent command-loop reference.


## 0.9.9 - 2026-07-16

### Added
- Show a one-line notice at startup when a newer version is available on PyPI, pointing to `uv tool upgrade nanocode-cli`. The update check stays cached and rate-limited to once per 24h, and never blocks startup.

### Fixed
- README pointed at a `nanocode upgrade` command that no longer exists; the upgrade instructions now use `uv tool upgrade nanocode-cli`.


## 0.9.8 - 2026-07-16

### Added
- `Ctrl-X` at the input prompt opens the current input in `$VISUAL`/`$EDITOR` (falling back to vim) and loads the edited text back; a non-zero exit or a failed launch leaves the input untouched.


## 0.9.7 - 2026-07-16

### Added
- Resume this project's latest session with `-c` (also `--last`, `--latest`), matched by working directory. Mutually exclusive with `--resume`, which still takes a UID or resolves the most recent session across all projects.

### Changed
- Rebuild the interactive REPL around a single persistent prompt-toolkit application: live activity, input, approval prompts, selectors, the `/diff` viewer, and the status bar now share one app, while completed output prints above it into native terminal scrollback.
- Define `Ctrl-C` uniformly across that app: cancel an approval prompt, clear the current input at an idle prompt, or interrupt a running turn. Exit stays on `Ctrl-D` at an empty prompt and `/exit`.

## 0.9.6 - 2026-07-14

### Changed
- Revert unified diff background fill: the live `/diff` viewer spans the pane width, while scrollback output keeps its natural width.

## 0.9.5 - 2026-07-14

### Changed
- Disable Rich Markdown hyperlinks so OSC 8 terminal escape sequences don't leak as visible text.
- Strip Markdown link URLs from assistant output for cleaner scrollback; revert and re-apply with narrower scope.
- Defensively strip unknown terminal escapes from assistant output and add a tip about hyperlink behavior.
- Update the system prompt to note that Markdown links won't render as hyperlinks in this terminal.

### Fixed
- Prevent OSC 8 payload from leaking as visible text in assistant Markdown output.

## 0.9.4 - 2026-07-14

### Added
- Auto-promote long-running Bash commands to background jobs after a configurable timeout.

### Changed
- Show sub-second precision in the Bash live-preview `running…` label.
- Color the `·N` batch counter gray via `LogLine.meta` instead of syntax-highlighted arguments.
- Pad diff backgrounds to pane width everywhere for solid Edit previews and a full-width live `/diff` viewer.
- Add breathing room to queued-message flushes so they match interactive echo spacing.

### Fixed
- Size diff background bands to the width after the outer log prefix so indented diffs render cleanly.
- Preserve background-colored padding when stripping Rich's line-fill whitespace on pane resize.

## 0.9.3 - 2026-07-13

### Changed
- Unify the live input prompt prefix to `>` and historical/queued user message prefix to `•`, extracting both as `UiPrinter` constants.
- Add a `Theme.user.log` color key with a muted desert-orange palette and resolve it through `UiPrinter.user_log_style()`.
- Render resumed user messages with a `•` prefix and theme-aware orange text, add a blank line above each user message, and drop the `user:` role label.
- Render resumed assistant messages without a role label or prefix.
- Keep nanocode's interactive theme active when the parent environment exports `NO_COLOR`; redirected and headless output remains plain.
- Add a separate `replay_prefix` to `read_input` so the main user message is echoed with `•` while the live prompt stays `>`.
- Drop the gray `inflight` styling from queued follow-up messages so they keep a consistent user color while pending.
- Update the system prompt to first acknowledge or briefly answer follow-ups before carrying out the request.

## 0.9.2 - 2026-07-13

### Added
- Add `/diff`, a read-only viewer for edits made by nanocode. It shows the latest user round and the overall net session diff in `Latest` and `Session` tabs, supports keyboard navigation and paging, and persists edit snapshots across `--resume`.
- Add a standalone animated progress row for manual `/compact` operations while keeping the normal status line visible.
- Show a bounded Bash stdout/stderr preview in finished tool summaries when no live preview was already kept visible.
- Add light-terminal theme support with automatic `COLORFGBG` detection and a `runtime.theme` override.

### Changed
- Use recursive internal layout boxes for turn hierarchy both live and after resume: user and final-assistant messages stay at column zero, intermediate-assistant messages and tool calls use two spaces, and tool details use four. Restored role labels share their message indentation, with a blank separator between turns.
- Add two-level hierarchy to CLI tool logs: indented tool calls are root events, approvals/output/errors/stored results use dim tree guides, Bash commands use a background-neutral GitHub Dark syntax palette, commands are shown once, compact tools remain single-line leaves, and flushed follow-ups use a `nano+` prompt.
- Replace the `--debug`/`runtime.debug` mode and `/debug on|off` toggle with an always-on, read-only `/debug` report. It keeps only the latest three cache-prefix mismatch records in memory, including changed region hashes and sizes, without retaining raw prompt content.
- Replace blank Enter as the queued-input send-now shortcut with Ctrl-C. Enter queues a follow-up for the next model request; when follow-ups are queued, Ctrl-C interrupts the active request so they can be sent immediately. Queue guidance now appears as a contextual `+>` placeholder and disappears while typing.
- Remove the synthesized FILE STATE context projection. `Read` and `Edit` outputs now remain as bounded, recallable conversation messages, and `/context` shows only Environment and Memory.
- Show each Bash command once before execution, keep live output visible without repeating the command, and use a compact dimmed `stored tr.N` result marker after live runs.
- Render changed diff code with equal-width saturated dark green/red backgrounds while leaving line-number gutters and context rows unfilled.
- Keep multiline queue pastes as one message; Up recalls the newest queued follow-up for editing or deletion before the model claims it.
- Render `/ps` as a Markdown table and print resume commands on their own line for easier copying.
- Show per-file addition/deletion counts in `/diff`, restore the previous CLI view after closing it, and syntax-highlight non-Bash tool arguments as well as Bash and Job commands.
- Refine the agent prompt around read-first engineering judgment, concise progress updates, review behavior, follow-up responsiveness, and using background jobs for potentially long-running work.

### Fixed
- Keep transient approval state (`approval required` and `[Y/n or reason]`) out of terminal scrollback after a tool decision.
- Follow edited files across unambiguous Bash moves so `/diff` reports their final paths and logical net changes.
- Prevent transient working dividers from leaking into CLI history during Bash previews and approval handoffs.
- Prioritize queued follow-ups when building the next model request.
- Keep Bash preview colors consistent, preserve literal closing-tag text, show bounded output when no live frame is available, and avoid repeating commands or output around live runs and failures.
- Coalesce capped fallback edits into one `/diff` entry per file, count only real hunk changes, and report `No changes` when a round has no net effect.
- Reject directory paths passed to `Edit` with a tool error instead of crashing during edit planning.
- Persist the safe prefix of an active turn before each model request so completed messages and tool calls survive interruption and `--resume`.
- Drain background-job output continuously so pipe buffers cannot deadlock verbose commands, refresh completed job state before capacity and `/ps` checks, and keep bounded output tails correct for small limits.
- Clip Bash previews and status lines by terminal display width, preventing CJK and emoji output from wrapping unexpectedly and desynchronizing live rendering.
- Keep assistant tool-call messages together with their tool results when compacting context, preventing invalid orphaned `tool` messages from reaching model APIs.
- Correct dark-terminal detection for ANSI bright-black backgrounds and keep the status line stable during manual compaction.
- Stop dropping files with oversized edit snapshots from the `/diff` Session view; fall back to the recorded per-turn diffs so the file still appears.
- Raise `TurnDiff.SNAPSHOT_CHAR_LIMIT` from 200KB to 1MB so typical large source files (including nanocode.py itself) keep their before/after snapshots and render as one clean unified diff instead of the concatenated per-Edit fallback.
- Reconstruct legacy `/diff` for oversized-snapshot files by reverse-applying stored per-Edit hunks against the on-disk content; produces one clean unified diff when disk still matches the last tracked edit, otherwise falls back to the raw hunks.
- Close the raw file descriptor if `os.fdopen` fails inside `MCPFileTokenStore.save`, preventing a descriptor leak on token-store write errors.

## 0.9.1 - 2026-07-09

### Added
- Add immediate queued-input flushing while the agent is waiting on a model request. Pressing Enter in the `+>` queue still records text for the next LLM request; pressing blank Enter with newly queued text interrupts the active model request and retries it with that queued input included.

### Fixed
- Harden queued-input flushing so stale SIGINT delivery cannot cancel the whole turn after a model request already returned.
- Avoid needless duplicate retries when the active model request already contains the queued input, prevent repeated blank Enter from stacking retry counts, and clear whitespace-only queue input on Enter.

## 0.9.0 - 2026-07-07

### Added
- Syntax-highlight inline Edit diff previews with Pygments. Added and context lines (the "new" file version) are lexed together as one code block so multiline strings and indentation-sensitive languages highlight correctly; removed lines stay plain diff-red. Degrades gracefully to plain diff coloring when Pygments is unavailable, the file extension is unknown, or the lexer fails.
- Add background jobs via a `Job` tool (`start`/`status`/`wait`/`list`/`kill`). Long-running or non-blocking work — dev servers, watchers, long builds and test suites — runs in its own process group without blocking the agent; output is buffered (capped per stream), drained non-blockingly on `status`, and returned as a tail. Concurrency is capped (`MAX_JOBS`), running jobs surface in the status bar (`jobs N`), the `/status` row, and a new read-only `/ps` command; `kill` sends SIGTERM then SIGKILL to the group. Jobs run until they exit or are killed (no foreground `shell_timeout` cap), and `Job(action="wait")` with no/zero timeout blocks until the process exits.

### Changed

- Rename the interactive `Question` tool to `Ask`; the old `Question` tool name is no longer registered.
- Replace the status-bar `+N` queued-message counter with a live queue region. While the agent works, messages you type sit below the +> input under a dim "── queued" divider with a left→right sweep animation, instead of being echoed into the scrollback log; when the turn flushes them they move up into the log as `+ <text>` lines and the region shrinks (the divider disappears once nothing is queued).
- Cache the Anthropic request's stable `tools`+`system` prefix with a `cache_control` breakpoint on the system block, so each turn no longer reprocesses the system prompt and tool schemas from scratch (Anthropic prompt caching only activates at an explicit breakpoint).
- Strengthen the system prompt for faster convergence: batch independent reads/searches into one parallel request by default, and drive each Bash call to complete in a single pass (chain known steps, split only on genuinely unpredictable dependencies).
- Replace the dedicated `LineCount`, `List`, `Find`, and `Git` tools with Bash-driven equivalents. The model now sees available shell commands near the top of Environment, read-only Bash commands (including safe `git status`/`diff`/`log` style commands) auto-run without confirmation, and mutating shell/git commands still require approval.
- Bash output is no longer erased from the terminal after the command finishes; the live preview output stays in the scrollback history.
- Edit diff previews and approve messages remain visible in the CLI history instead of being transiently cleared.
- Remove the Ctrl-A expanded Edit preview and fixed-height transient preview window. Edit approvals still show the full inline diff preview in the CLI history, with Pygments highlighting preserved.
- Git branch is no longer shown in the environment context sent to the model.
- Removed branch-change detection protection that prevented `git commit` after an external branch switch.

### Fixed
- Close a hole in Bash read-only auto-approval: `&&`/`||` were flattened to spaces and only the first command in a chain was validated, so `git log && rm -rf x` auto-ran without confirmation. Every stage of a `&& || | ; newline` chain is now validated independently, a lone background `&` is rejected, and a `cd` prefix (a benign builtin the model routinely adds) no longer forces a prompt on an otherwise read-only command.
- Follow common sense in the Bash read-only classifier rather than blanket strictness: `sort`, `uniq`, `sed`, and `tree` auto-run (with guards only on their real file-writing forms — `sort -o`, `uniq IN OUT`, `sed -i`, `tree -o`), and the ubiquitous harmless redirections (`2>/dev/null`, `>/dev/null`, `< /dev/null`, `2>&1`, `>&2`) no longer trigger a prompt. Real writes to a file and command substitution still ask.
- Suppress prompt-toolkit's CPR warning in terminals that do not answer cursor position requests by disabling CPR probing for nanocode's prompt applications.
- Stop closed stdin from escaping as an unhandled exception in the queue-input background thread.
- Stable Edit anchors, eliminating spurious "stale anchor" errors. `line_hash` no longer folds the trailing newline into the hash, so a line's anchor stays valid when only its final newline changes (e.g. the last line gaining or losing its `\n`) and Read/Search/Edit anchors now agree with InspectCode's. Edit also numbers lines exactly like Read — both split on `\n` only via a shared `ReadTool.split_lines`, replacing `str.splitlines(True)`, which additionally breaks on `\r`, `\v`, `\f`, `\x1c`–`\x1e`, `\x85`, `\u2028`, `\u2029` and desynced line numbering (and thus anchors) for any file containing those characters.
- Bump `code-symbol-index` to `>=0.3.5`, fixing a tree-sitter range extraction crash that could terminate `nanocode --resume last` with SIGSEGV and leave a core dump during background index refresh, while using the package's validated tree-sitter dependency floor.
- Render Ask choice previews with escaped `\n` sequences expanded into real lines, allow the choice window to use available vertical space above the status bar, and stop repeating the full raw question text when switching to "Type freely...".
- Render intermediate assistant markdown correctly while the `+>` queue prompt is active; Rich ANSI output is now replayed through prompt-toolkit instead of leaking as `?[90m`-style text.
- Move the active-turn elapsed timer into the animated `working` divider; the sweep now scans only the horizontal rule so the label stays readable.
- Show saved tool summaries on `--resume` even when the compacted transcript no longer contains the original assistant tool-call messages.
- Add regression coverage for Ask preview/free-text rendering, whole-second working timers, and restored saved tool summaries.

## 0.8.2 - 2026-07-03

### Fixed
- Auto-submit queued input you already pressed Enter on. While the agent is working, typing in the `+>` queue and pressing Enter records the line into `pending_user_inputs`; if the turn ended before a step consumed it, that already-entered text was pre-filled back into the prompt and sat there waiting for a *second* Enter (and could feel like a very delayed send). Now, in interactive mode, Enter-committed queue input auto-submits as the next turn with no second keypress, while text you were still typing (never Enter'd) is still pre-filled into the prompt for review. Mid-turn injection and the headless combined-submit path are unchanged.

## 0.8.1 - 2026-07-02

### Added
- Add a `nanocode update` (alias `upgrade`) command that upgrades nanocode to the latest PyPI release. It first checks the latest version, reports and exits when already current, then detects how nanocode was installed and runs the matching upgrade command: `uv tool upgrade nanocode-cli` for uv-tool installs, `pipx upgrade nanocode-cli` for pipx, and `python -m pip install --upgrade nanocode-cli` otherwise. Editable/dev installs are detected (via the distribution's `direct_url.json`) and refused with a hint to update the source instead.

## 0.8.0 - 2026-07-01

### Added
- Add Skills: reusable instruction packs discovered from `.nanocode/skills/<name>/SKILL.md` (project) and `~/.nanocode/skills/<name>/SKILL.md` (user; project wins on name clash). Each skill is a Markdown file with `name`/`description` frontmatter and may bundle helper scripts in its folder. A compact `SKILLS` index (name + description) rides the cache-stable request prefix so the model knows what exists, and the full body is pulled into the conversation only on demand — either when the model calls the new `Skill(name)` tool or when the user references a skill inline with `$name` (Tab-completed). A `{skill_dir}` / `${SKILL_DIR}` placeholder in the body expands to the skill's absolute folder path when loaded, so bundled scripts are runnable via `Bash` without hardcoded paths. Repeated `Skill(name)` loads collapse to a one-line pointer (like MCP `describe`), keeping the first full body cached and not re-billing the instructions. A built-in `nanocode-help` skill ships by default (out of the box, no install) that answers questions about nanocode itself — how to use it, its features, and common problems — from a self-contained manual, so it does not read the source per question. The body is an authored manual (concepts, workflows, troubleshooting) plus lists assembled at load time from the same in-code constants the app uses (the `/help` text, each tool's description, the settable config keys), so the prose is rich while the lists cannot drift from the running version; the raw source is named only as a last-resort fallback. A project/user skill of the same name overrides it. Because a built-in is always present, the `SKILLS` section and `Skill` tool ride every request; only a session with its skills explicitly cleared drops back to the prior byte-identical prefix. Installed skills are surfaced in the status bar (`skills N`), the `/status` context row (`skills N`), a read-only `/skills` command, and `/help` now documents the `$skill` inline mention.
- Replace `/memory` with `/context`, a viewer for the model's synthesized context frame — the Environment, Memory (goal, plan, known facts, check, code-index status), and File State sections that wrap the conversation on every request. At an interactive prompt, bare `/context` opens a tabbed viewer (`←/→` or `h/l` switch Environment/Memory/File State, `↑/↓` or `j/k` scroll, `1`–`3` jump, `Esc`/`q` close) with the sections rendered as Markdown tables/headings and a scroll-position indicator; while the agent is working or without a TTY it prints the same content as a static Markdown dump. `/context <path>` shows one in-context file's current anchored lines (matched by exact path, basename, or suffix) inside a fenced block. Unlike the old `/memory`, the Memory section now renders exactly as the model receives it (so it includes `check` and code-index status and omits the transcript-only summary).
- Detect prompt-prefix drift via fingerprinting. The cache-stable request prefix (system prompt + environment + MCP tool index + sorted tool schemas) is fingerprinted (SHA-256) each turn against a baseline pinned to the first one seen; a healthy session keeps a single fingerprint start to finish, while a second one means every token from the change onward is a cache miss that previously failed silently and only surfaced on the bill. `/status` shows a `⚠ prefix churn` warning with the distinct-fingerprint count when more than one is seen, `--debug` writes a unified diff of exactly what changed (`cache-prefix-drift`), and the fingerprints persist across `--resume`.

### Changed
- Track and show the context compaction count in `/status`. `AgentState` gains a persisted `compaction_count` incremented on every compaction path (history, turn, and their fallbacks), surfaced in the `/status` context row and preserved across session save/load.

## 0.7.2 - 2026-06-26

### Added
- Show a short, context-aware tip on startup (a curated set covering sessions, context, model/reasoning, tools, and config). Tips whose feature is unavailable are filtered out (e.g. `/strict` only shows when the provider supports strict tools; MCP tips only when MCP is configured). The line is styled as a muted hint with highlighted `code` spans and can be disabled with `runtime.tips = false` (or `/set runtime.tips off`).
- Add per-property `description` fields to every built-in tool's JSON Schema (`Read`, `LineCount`, `List`, `Find`, `Search`, `InspectCode`, `Edit`, `Bash`, `Git`, `Recall`, `Note`). The argument contract now lives in the schema the model parses natively rather than only in the prose signature, improving tool-call argument accuracy across providers.
- Add a `/strict` command to toggle `provider.strict_tools` for the active provider, reporting when it is enabled but inactive (the provider does not support strict tool calling).
- Add a `provider.strict_tools` option (default `false`, also settable via `/set`) that constrains tool-call arguments to each tool's JSON Schema. When enabled it is only active on hosts whose profile supports strict mode (`api.openai.com`, `api.deepseek.com`) and on the chat path; every other provider is unaffected. Activating it emits a strict-compliant schema (all properties required, genuine optionals made nullable, `additionalProperties: false`, and the unsupported `minItems`/`maxItems`/`minLength`/`maxLength` keywords dropped) with `strict: true` per function, and on DeepSeek it routes the client to the `/beta` endpoint where the feature lives. Because optionals become nullable, the model may send explicit `null` for an omitted argument; nanocode now strips `null` values (recursively) before dispatching to a tool, where `null` has always meant "absent".

### Changed
- Add a `provider.max_tokens` option (also settable via `/set`) capping chat-completion output. It defaults to `0` (unset) for generic OpenAI-compatible providers, so their requests are unchanged, and resolves to a profile default of `32768` on `api.deepseek.com` so DeepSeek thinking mode has enough room for `reasoning_content` plus the answer.
- Map `reasoning = "high"` to DeepSeek's agent-recommended `max` effort in the thinking effort table (was `high`); `xhigh` still maps to `max` and the default `medium` still maps to `high`. Only the DeepSeek/Qwen `thinking` style is affected.
- Skip the OpenAI-only `prompt_cache_key` parameter for `api.deepseek.com`, which caches by prefix automatically and ignores the key. All other hosts keep emitting it as before.

### Fixed
- Keep the Bash live preview alive during blocking commands so the terminal no longer looks frozen. A command that buffers its output (e.g. a quiet long-runner, or `... | tail` that emits nothing until EOF) previously left the screen completely static — the status bar is stopped while a command runs and the preview only drew when output arrived. The preview now renders an immediate frame showing the command being executed (`$ <command>`) and a live elapsed timer (`running… 12.3s`), ticked by a daemon heartbeat so the timer advances even with no output, and switches to `output · 12.3s` once output streams. On finish the frame is now erased (it previously lingered as dimmed ghost lines), and a full-width line can no longer auto-wrap and desync the redraw cursor math.
- Set an explicit output ceiling for DeepSeek thinking mode so long `reasoning_content` no longer exhausts the server-side default and truncates the response or drops the tool call.
- Stop sending `temperature` on the chat path when a native thinking style (`thinking`/`enable_thinking`) is enabled, since DeepSeek and Qwen reject or ignore it in that mode. Other reasoning styles and providers are untouched.
- Drain the MCP background event loop on exit. MCP work runs on a daemon loop thread; at interpreter shutdown the `concurrent.futures` atexit hook tore down the default executors before that thread was joined, so an in-flight client cleanup (HTTP session termination, DNS via `run_in_executor`) raced the teardown and printed `cannot schedule new futures after shutdown` / `Session termination failed`. `MCPManager.close()` now cancels pending tasks, stops the loop, and joins the thread (5s timeout) from `main()`'s `finally` before the interpreter tears down its executors.

## 0.7.1 - 2026-06-25

### Changed
- Run the code-symbol-index working-tree check off the UI critical path. `/status` previously forced a synchronous `csi.status(check=True)` full-tree scan (~1.5s on large repos) before rendering, and the same scan ran inline at the end of every turn, delaying the answer. `/status` now reads the last-known status instantly (`check=False`) and triggers the scan asynchronously, and the per-turn `update_pending()` runs in a guarded one-shot daemon thread, so neither blocks the UI. A new transient `code_index_checking` guard prevents overlapping scans.

## 0.7.0 - 2026-06-24

### Added
- Execute a model's batch of read-only tool calls concurrently. Within one assistant turn, maximal runs of auto-approved, non-interactive read tools (`Read`, `Search`, `Find`, `List`, `Recall`, `InspectCode`, read-only `Git`, read-only `MCP`) now run in a bounded thread pool, while results are finalized on the main thread in the exact order the model issued them. Mutating tools, `Edit` batches, confirmations, `Bash` (live output), and `Question` (interactive) stay serial. Bounded by the new `runtime.max_parallel_tools` setting (default `4`; set to `1` to disable and restore fully serial execution).
- Store `Note` plan items as explicit objects with `status` (`todo`, `doing`, `done`, `blocked`) and `text`, while preserving old string plans as `todo` during load/compaction. Model-visible memory renders readable status text; CLI memory and Note previews keep compact `[ ]`, `[~]`, `[x]`, `[-]` symbols.
- Added a concise system-prompt `GUIDE` section covering `THINK BEFORE CODING`, `SIMPLICITY FIRST`, `SURGICAL CHANGES`, and `GOAL-DRIVEN EXECUTION`.

### Changed
- Stop auto-approved (yolo) tool calls from logging two near-identical CLI lines. The pre-execution `auto …` line duplicated the result line's header for tools without a preview (MCP, Bash, Git); it is now suppressed in that case and the auto-approval is recorded by an `[auto]` tag on the single result line. `Edit` still shows its `auto …` pre-line because it carries the diff preview the result line omits.
- Depend on `fastmcp-slim[client]` instead of the full `fastmcp` meta-package, since only the MCP client is used. This drops the unused server/CLI stack (cyclopts, griffelib, uvicorn-side bits, openapi/jsonschema tooling, the keyring/py-key-value cluster, etc.) for a much smaller install.
- Switched Read/Search/Edit/InspectCode anchors to the explicit `anchor=line:hash | text` format and documented the hash as `hash(line_content)` in FILE STATE.
- Bumped the `code-symbol-index` dependency floor to `>=0.3.1` and use its upstream explicit InspectCode anchor formatter instead of local normalization.
- Reduced `nanocode.py` by removing dead code, thin wrappers, and over-complex helper paths without changing behavior.
- Render Note success output with user-facing `goal:`, `check:`, `plan:`, and `known:` labels, plus status-aware colors in the CLI, instead of exposing internal field names like `set_goal` and `replace_plan`.

### Fixed
- Keep every MCP server visible in the tools index. It previously built full per-tool schemas and then hard-truncated the whole block at the size cap, so one verbose server could consume the entire budget and silently drop every later server, leaving the model blind to them. The index now degrades by shedding detail rather than entities — inline schemas, then schemas-dropped, then name-only — keeping all servers and tool names visible (the model can re-fetch a schema via `describe`); only at thousands of tools does it truncate, and that case now warns via a post-discovery notice and a `!` status-line marker.
- Declare `websockets` as a direct dependency. The MCP client transport imports it at runtime, but `fastmcp` only declared it under its server extra, so trimming that extra broke MCP connections (including OAuth login).
- Render structured plan items in `/memory` and Note previews as status symbols/text instead of leaking `PlanItem(...)` repr strings.
- Resume sessions whose transcript contains tool calls with multi-line argument strings (e.g. a multi-line `git commit` message). Strict `json.loads` rejected the literal newlines and dropped the args to `{}`, which then failed the tool's own validation (`Git requires a non-empty 'argv' list`) and aborted `--resume`. Tool-call arguments now parse with `strict=False`, and a malformed historical call renders without args instead of crashing.
- Stop malformed live tool-call arguments from aborting the whole turn. Argument validation that ran while parsing a model response (e.g. `Git` with an empty `argv`) raised outside the tool-execution boundary, ending the turn with an error the model never saw. The error is now captured on the call and surfaced as a normal failed tool result, so the model can self-correct.
- Decode `Bash` output with a per-stream incremental UTF-8 decoder. Each 4096-byte read was decoded independently, so a multibyte character split across a read boundary (common with CJK output) was mangled into replacement characters.

## 0.6.6 - 2026-06-23

### Changed
- Report the prompt-cache hit rate in `/status` against input/prompt tokens instead of total tokens (which folded in completion/reasoning output and structurally understated the rate), and show the `cached/prompt` split for both the cumulative and last-call figures.
- Land MCP `describe` results inline in the conversation history (bounded and recallable like any tool output) instead of stripping them into a recomputed `MCP TOOL DETAILS` tail block. The schema is now written once and cached as the conversation grows, rather than re-billed uncached on every request; the tools index tells the model not to describe a tool again once its schema is already shown.
- Collapse repeated MCP `describe` results for the same tool to a one-line pointer when building the request, keeping only the first full schema per `(server, tool)`. This is a stateless send-time transform that never mutates stored history and only ever collapses the newer occurrence, so it reclaims duplicate-schema tokens without disturbing the cached prefix.

- Tighten the final-answer style guidance: default to a few lines, scale length to the task, lead with the result, and drop preamble / request-restatement / step-by-step recaps, replacing the soft "be concise" wording that models tended to ignore.
- Add Tab/Shift-Tab completion (commands, paths, mentions) and a completion menu to the queue-input area, matching the main prompt.
- Run read-only slash commands (`/help`, `/status`, `/memory`, read-only `/mcp`) plus `/yolo` typed in the queue-input area while the agent is working, instead of sending them to the model as literal text. `/yolo` is allowed because it is a single atomic flag the agent reads at the next approval; other state-mutating or control commands are refused with a hint to interrupt first, since they would race the in-flight turn.
- Pre-fill leftover queued input into the prompt for review/edit when a turn finishes, instead of auto-submitting it as the next turn. Input typed while the agent is working is still injected into the running task; only input left over at the turn boundary now waits at an editable prompt (non-interactive/piped input keeps auto-submitting, since there is no one to confirm).

- Collapse argument/usage tool-call rejections to a quiet dim one-liner (`tool X · rejected: <reason>`) in non-debug mode, instead of the full red `[failed]` + error block. These are usually self-corrected on retry; the full error still goes to the model and shows in debug, and real execution failures stay fully visible.

### Fixed
- Moved the volatile `code_index` status out of the early `Environment` block (which sits ahead of the conversation history) into the late `Memory` section, so its `synced ↔ stale` churn no longer invalidates the cached conversation prefix every time files change or the indexer runs.

## 0.6.5 - 2026-06-22

### Added
- Surface each MCP tool's input JSON Schema to the model: the tools index now carries a compact (capped) schema per tool and `MCP(action="describe")` emits the full schema, so the model can build nested arguments correctly instead of guessing.
- Added MCP resource support: discover resources alongside tools (concurrently, best-effort), list them in the tools index, and read them via `MCP(action="list_resources")` and `MCP(action="read_resource", uri=...)`. Resource reads flow through the normal tool-result path so they land in cached conversation history.
- Surface resource-like URIs referenced in MCP tool descriptions on the tool's index line, so doc references survive description truncation and stay discoverable.
- List a server's resources in `@server` mention blocks, matching the tools index.
- Auto-read a resource referenced by an MCP tool's description on the first call to that tool (once per uri per session, best-effort, web links skipped), attaching it to the call result on success and to the error on failure so argument-grammar docs reach the model on the first attempt and land in cached history.

### Changed
- Default the `MCP` tool `action` to `"call"` when omitted but `tool`/`arguments` are present, and make the unknown-action error actionable (lists valid actions and shows how to invoke a remote tool by name).

### Fixed
- Render connected MCP servers that advertise resources but no tools in the tools index (previously gated on having ≥1 tool, so resource-only servers were mislabeled "not connected" and their resources hidden); `_pending_status` now distinguishes a connected-but-empty server from a disconnected one.

## 0.6.4 - 2026-06-21

### Added
- Added JSONL session persistence with append-only deltas for the normal save path and `nanocode --resume [UID]` for restoring saved sessions.
- Added a `latest` session pointer and a resume command hint printed when nanocode exits.
- Added `nanocode --resume last` as an alias for resuming the latest saved session.
- Render restored session history once on resume without re-running tools or commands.
- Show the active session id in `/status`.
- Added session persistence coverage for snapshot/delta save/load, `latest`, usage/state/tool record roundtrips, missing snapshots, and exit resume hints.

### Changed
- Kept persisted session snapshots focused on necessary recovery data only, deriving runtime tool-result lookup state from saved tool records instead of storing config, settings, timestamps, git branch, or other rebuildable runtime data.
- Skip first-save persistence for sessions with no recoverable content, avoiding empty snapshots and stale `latest` pointers.
- Delete session files older than the configurable `runtime.session_retention_days` retention window on startup; the default is `7` days.
- Treat retention and update-check interval settings as config-file policy rather than `/set` runtime mutations.
- Split session snapshot encoding/merging from JSONL file storage, keeping `Session` as a thin owner of runtime state.
- Resume now applies the current config/runtime flags (`--config`, `--yolo`, `--debug`, `--mcp`) while loading only conversation state from the snapshot.

### Fixed
- Fixed command dispatch so ordinary non-command input reaches the agent instead of being treated as an unknown slash command.
- Changed the resume marker to an internal system message so it does not affect latest-user compaction behavior.
- Report resume/load errors as CLI errors instead of uncaught exceptions.

- Auto-submit queued input at round end: `queue_input_text` (typed but not confirmed) and unconsumed `pending_user_inputs` both get drained and submitted as the next user input; extracted as `CommandLoop.drain_queued_input()`.
- Clear blank `queue_input_text` even when it contains only whitespace, preventing stale whitespace from lingering in the input area.

- Record auto-submitted queued input in CLI history via `FileHistory.append_string()`, so previously-queued text appears in prompt history like any manually typed input.

## 0.6.3 - 2026-06-21

### Added
- Added `Question` tool that pauses the agent to ask the user one or more questions (asked in sequence) when intent is genuinely ambiguous, a design choice affects the external shape (module structure, public API, naming), or prioritization is needed. Each question supports structured choices with optional dynamic previews and an optional `recommended` choice index (pre-selected and marked).
- Wired `Question` tool into the shared interactive selector with `j`/`k` navigation, dynamic per-choice preview, Rich markdown rendering of the question text, and a free-text fallback.
- Echo the selected choice in the CLI after the interactive selector closes, giving clear feedback on what was chosen.
- Prefix a position indicator (e.g. `(1/3)`) onto each question when several are asked in one `Question` call.

### Changed
- Refined `Question` tool DESCRIPTION, SYSTEM_PROMPT usage guidance, and README with clear principles: question only when intent is truly ambiguous, design affects external shape, or prioritization is needed. Skip internal details, contextually-determinable items, and already-specified matters.

### Fixed
- GitTool: validate non-empty `argv` in `payload_args` with a clear error message when the model sends an empty command list.
- Update the tracked commit-target branch (renamed `initial_git_branch` → `expected_git_branch`) after a tool-driven branch switch so commits on a branch nanocode switched to are not rejected, while still guarding against unexpected external switches.

## 0.6.2 - 2026-06-21

### Changed
- Simplified the agent system prompt while preserving the main `TOOLS`, `FLOW`, `FILE STATE`, and `FINAL` sections.
- Simplified MCP command handling and server config parsing by sharing command metadata and config field parsing helpers.

### Fixed
- Reused one manager-owned MCP event loop for async MCP operations, including calls made while another event loop is already running.
- Made MCP OAuth token storage reuse one store and shared path lock to avoid concurrent token file writes racing each other.
- Allowed `env_http_headers.Authorization` when it is the only configured authorization source.
- Rejected extra `/mcp` subcommand arguments instead of silently ignoring them.

## 0.6.1 - 2026-06-19

### Changed
- Renamed `Note` fields: `goal` → `set_goal`, `plan` → `replace_plan`, `known` → `append_known`, `check` → `set_check`; added `replace_known` for full replacement of known facts.
- Aligned `Note` schemas and prompt guidance so `replace_plan`, `append_known`, and `replace_known` are always arrays and can be empty when replacing state.
- Split FILE STATE into its own prompt section and clarified it as the working view for visible file content and Edit anchors, while noting it may be partial.
- MCP tool calls now confirm by default, auto-approving only tools explicitly marked read-only (`readOnlyHint`) or non-destructive (`destructiveHint: false`); undiscovered tools also require confirmation.

### Added
- New Note field `replace_known` that completely replaces the known list, with test coverage.

### Fixed
- Made `Note` updates transactional, so invalid fields no longer partially mutate session notes before returning a tool error.
- Guarded MCP `login`/discovery error paths with the manager lock so they no longer race background discovery while mutating server tool/error state.

## 0.6.0 - 2026-06-18

### Added
- Added MCP client router support through a single `MCP` tool, with URL-based server configuration, bearer-token environment variable support, OAuth login/logout with persistent tokens, asynchronous tool discovery, compact model-visible MCP tool indexes, on-demand tool details, and `/mcp` inspection/refresh commands.
- Added local stdio MCP servers via `command`/`args`/`env`, alongside the existing streamable-HTTP transport; each server is `url` or `command`, and stdio servers reject the HTTP-only auth options.
- Added `@server` and `@server.tool` mentions in user input that inject a server's tool list or a tool's details into the turn, force discovery of undiscovered servers, report login/errors inline, and tab-complete server and tool names.
- Added MCP coverage for result normalization, successful tool calls, context pruning, `/mcp tools NAME`, missing-server refresh handling, and stdio config parsing/validation.
- Added bounded MCP connection timeouts, concise MCP connection-failure logs, a `--debug` flag for starting with debug mode enabled, and a `--mcp` selector for choosing MCP servers by name glob.

### Changed
- Refined the status bar with lowercase `mcp`, right-side loading animation, and semantic per-section colors.
- Show MCP discovery progress in the status bar while servers are loading.
- Render `/mcp` server and tool listings as Markdown tables for clearer terminal display.
- Load configured MCP servers in parallel during discovery.
- Include the MCP endpoint URL when OAuth login fails before an authorization URL is available.
- Suppress duplicate FastMCP OAuth URL logs and omit OAuth-login-required notices from startup MCP error logs.
- Skip MCP server discovery without startup error logs when bearer-token environment variables are missing.
- List all enabled MCP servers in the model-visible index, adding a "not yet available" section for servers that are still discovering, need login, or errored, so the model never assumes a configured server is absent.
- Enriched MCP tool-call logging with compact `key=value` arguments in the header (also shown in the approval preview), a success result summary (shape and payload size), and round-trip latency.
- Steered the agent toward `InspectCode` for symbol navigation with a `SEARCH/NAV` prompt section, and surfaced the code-index status in the Environment context so it knows when the tool is usable.
- Separated each round with a blank line after the user input and a rule before the agent's answer.
- Clarified prompt guidance around FILE STATE snapshots, automatic Read/Edit refreshes, stale-anchor retries, and avoiding unchanged failed tool-call retries.
- Tightened the system prompt's tool-choice guidance to prefer `Edit` for file changes, `Read` for known file ranges, `Search` for text lookup, and `InspectCode` for symbol navigation.
- Added current git branch to Environment context while keeping branch-specific data out of the stable system prompt.

### Fixed
- Expanded `Edit` no-op errors with current target-range content when anchored edits produce no changes, so agents can distinguish already-applied edits from wrong replacement content.
- Guarded git branch safety so yolo mode cannot auto-approve branch-changing commands, and git commits refuse to run after the branch changes from the session start.

## 0.5.12 - 2026-06-15

### Fixed
- Preserved queued input typed during a running turn and prefilled it into the next prompt instead of dropping it at turn completion.

## 0.5.11 - 2026-06-12

### Added
- `InspectCode` gained `refs`, `impls`, `callers`, and `callees` modes, backed by `code-symbol-index` 0.3.0: behavior-classified references, implementor listing, and transitive call-chain walks (`depth`). `refs` hides import/attribute noise by default, with `ref_kind` to filter to an explicit subset or `all_kinds` to show everything; `callees` takes `loose` to include ambiguous cross-module matches; `refs`/`impls` page with `offset`.

### Changed
- Bumped `code-symbol-index` floor to `>=0.3.0`.

## 0.5.10 - 2026-06-02

### Added
- Cached gitignore patterns across tool calls with mtime-based invalidation.

### Changed
- Clarified Bash tool output limits in its description.
- Strengthened assistant language prompt to reduce unnecessary `cd` commands.
- Clarified Git tool description so the model sees it defaults to the cwd from Environment.

## 0.5.9 - 2026-06-01

### Changed
- Colorized `Ctrl-A` full edit previews when shown through `less`.

## 0.5.8 - 2026-06-01

### Added
- Added `Ctrl-A` full edit preview in an external pager during approval.

### Changed
- Compact oversized current turns instead of only prior history.
- Removed thin internal wrappers without changing behavior.

### Fixed
- Rejected broad `git add` commands unless explicit file paths are supplied.
- Stopped exposing the output-language sentinel inside model-visible file state.

## 0.5.7 - 2026-05-30

### Changed
- Encouraged early `Note` usage for multi-step work with goal and plan updates.
- Added lightweight empty-memory guidance when goal or plan has not been set.
- Ordered `FILE STATE` files by most recent visible Read/Edit source before stable path fallback.

## 0.5.6 - 2026-05-30

### Added
- Added a visible purple approval wait indicator.
- Expanded compact logic tests around latest-turn retention, recent-message windows, fallback trimming, and tool-result preservation.

### Changed
- Clarified prompt rules for user-visible interim output and final answers.
- Simplified core flow by removing thin wrappers and duplicate tool-schema name extraction.

### Fixed
- Preserved raw tool results referenced from compact summaries so `tr.N` keys remain recallable.
- Improved transient model error retry detection and final retry reporting.

## 0.5.5 - 2026-05-30

### Changed
- Tightened the system prompt around FILE STATE, anchored edits, and final-answer flow.

### Fixed
- Added limited automatic retries for transient model request failures such as 5xx, rate limits, and timeouts.

## 0.5.4 - 2026-05-29

### Changed
- Reworked running-turn context as a current turn conversation, preserving mid-turn assistant text and appended user input.
- Made running-turn appended input visible through a `+>` prompt and compact `+N` status indicator.

### Fixed
- Started Bash live preview before command output so the `+>` prompt cannot cover it.
- Avoided extra approval prompt line clearing after confirmation.

## 0.5.3 - 2026-05-29

### Added
- Added support for additional user input during running agent turns.
- Added multiline approval input for pasted refusal reasons.

### Changed
- Simplified approval handling so direct non-yes input is treated as a refusal reason.

### Fixed
- Fixed CreateFile escaped-newline handling so preview and written content stay multiline.
- Made CreateFile/Edit code-index updates use the tool call path as a fallback.

## 0.5.2 - 2026-05-29

### Changed
- Animated the statusbar code-index refresh indicator while keeping `/status` semantic.

### Fixed
- Switched startup code-index refresh to the `code-symbol-index` async refresh API to avoid parser thread ownership errors.

## 0.5.1 - 2026-05-29

### Added
- Added the current date to context immediately before the current user request.

### Fixed
- Sanitized context/debug/model-request text so surrogate characters from terminal input cannot break UTF-8 encoding.
- Made Search ignore hidden paths and `.gitignore` paths consistently across ripgrep and Python fallback paths.

## 0.5.0 - 2026-05-28

### Added
- Added cached system information to the top of model context: cwd, OS, arch, shell timeout, and detected commands.
- Added configurable `runtime.max_context_tokens`.
- Added key behavior tests for tools, agent loop, context management, provider adaptation, and code index integration.

### Changed
- Replaced the legacy implementation with the smaller v1 core in `nanocode.py`.
- Rebuilt README around the current command set, context design, and screenshot.
- Simplified tool schemas for broader OpenAI-compatible provider support.
- Improved tool-call display for Search and Recall, and surfaced intermediate assistant progress before tool calls.

### Fixed
- Fixed Moonshot/Kimi-compatible tool schemas by avoiding unsupported schema forms.
- Fixed repeated command probing in context rendering by caching system command detection per session.

## 0.4.11 - 2026-05-24

### Changed
- Bumped version from 0.4.10 to 0.4.11.
## 0.4.8 - 2026-05-23

### Changed
- Renamed the `EditFile` tool to `Edit` across the codebase and tests.

## 0.4.5 - 2026-05-21

### Changed
- Updated the built-in code index integration for `code-symbol-index` 0.1.7.
- Added indexed symbol filters for kind, path, and exact matching.
- Added file-local symbol outlines and bounded pending-index details in `/status`.

## 0.4.4 - 2026-05-20

### Added
- Added built-in indexed code navigation backed by project data and `/index` for manual init/sync.

### Changed
- Replaced the external code-navigation CLI integration with the bundled code index API.
- Hid code navigation tools until an index exists, while lightly updating existing indexes at startup.
- Updated status/docs to describe code index availability without exposing dependency-install wording.

## 0.4.3 - 2026-05-20

### Changed
- Removed stable knowledge state while keeping current-task known facts.
- Extracted shared numbered-content and line-range helpers for tool output/range handling.
- Trimmed thin helper wrappers in List and indexed code-inspection tools.

## 0.4.2 - 2026-05-19

### Added
- Added indexed code inspection tools for symbol lookup, symbol investigation, and file outlines when the local index is available.
- Added queued user feedback during long-running turns.
- Added `PatchFile` for multi-location file edits.

### Changed
- Moved model calls to the OpenAI SDK and function-tool protocol.
- Reworked task-shape prompts for chat, one-shot tasks, and tracked tasks.
- Prioritized indexed code inspection for structural lookup while keeping Search/Read for exact literals and edit ranges.
- Improved terminal UX with persistent status, queued-input handling, Bash live preview, and terminal-friendly assistant output rules.
- Renamed `ListDir` to `List`.
- Improved `Read`, `Edit`, `ReplaceRange`, `PatchFile`, `Bash`, and `Git` tool guidance.
- Simplified gate behavior so only deterministic, correctable model errors are refused.

### Fixed
- Fixed duplicate final replies for goal-only text answers.
- Fixed repeated recall loops and several format/tool-name compatibility issues.
- Fixed PatchFile diagnostics and empty-hunk handling.
- Fixed queued feedback delivery, Ctrl-C/Ctrl-D handling, and Bash interrupt reporting.

## 0.3.35 - 2026-05-16

### Added
- Added batched `ReplaceRange` edits for multiple independent ranges in the same file.
- Added a design document covering agent state, context construction, tool-result storage, observe policy, and verification.

### Changed
- Aligned tool-result context layout with the design document.
- Refined tool-result context reduction around unreduced raw results, retained results, and checkpoint-based pruning.
- Compressed ACT and OBSERVE system prompts.
- Reduced routine OBSERVE triggers by raising the pending-result threshold and keeping ordinary tool failures in ACT for repair.
- Simplified agent gate and feedback handling, including single active plan item normalization.
- Added soft feedback for state-update-only ACT turns so models continue with frontier tools, verification, or completion.
- Highlighted recognized slash commands and reported unknown slash commands directly.

### Fixed
- Accepted harmless model output variants including trailing progress text, action type casing, and `message` action aliases.
- Ignored pending verification requests instead of treating them as blocking model output.

## 0.3.34 - 2026-05-16

### Changed
- Trigger observe by unresolved pending tool-result count only, instead of consecutive tool batch count.

## 0.3.33 - 2026-05-16

### Fixed
- Keep unresolved pending tool results visible as raw ACT context until observe mode explicitly keeps or forgets them.

## 0.3.32 - 2026-05-16

### Changed
- Removed the pending tool-result character threshold for observe mode; observe now triggers from failures, pending result count, or consecutive tool turns.

## 0.3.31 - 2026-05-16

### Changed
- Require a new user turn with retained task context to align via `start`, `goal`, or `plan` before running more tools.

### Fixed
- Removed an unused pytest import from bash tool tests.

## 0.3.30 - 2026-05-16

### Changed
- Status bar now shows compact token totals and model stream rate, including `turn:` duration labeling.
- Stream rate uses live character-based estimation and completion-token usage when available.

## 0.3.29 - 2026-05-16

### Fixed
- Ignored code-fence-only text when converting interleaved model output into progress actions.

## 0.3.28 - 2026-05-16

### Changed
- Status bar now labels model calls as `working`, `observing`, or `compacting`.
- Removed the hard gate for multiple `doing` plan items while keeping prompt guidance to prefer a single active item.

### Fixed
- Preserved specific retry notices such as `err:format` instead of overwriting them with generic gate notices.
- Accepted unmarked action streams with interleaved progress text between JSON actions.
- Normalized tool-name action types such as `Search` or `ListDir` into tool actions.

## 0.3.27 - 2026-05-16

### Fixed
- Added a nanocode `User-Agent` header to provider requests so OpenAI-compatible gateways that reject Python urllib defaults can accept chat and model-list requests.

## 0.3.26 - 2026-05-16

### Changed
- Generated configs now leave `reasoning_payload` unset by default for broader provider compatibility.
- Documented when to enable `reasoning_payload`, including OpenRouter-style reasoning providers.

## 0.3.25 - 2026-05-16

### Changed
- Added Vim-style selector search with `/keyword`, `j`/`k` navigation, and step-back Esc behavior.
- Made `/model` reasoning selection transactional so Esc returns to model selection instead of applying a partial change.

## 0.3.24 - 2026-05-16

### Changed
- `/model` now groups configured `available_models` first and appends deduplicated models discovered from the provider.
- Default generated config now documents `available_models` without writing an empty setting.
- Split latest/recent and kept tool-result context budgets for steadier context growth.
- Compacted tool-result CLI output while keeping result keys visible.
- Removed `ApplyPatch`; editing now uses `Edit` for tiny literal changes and `ReplaceRange` for read-backed focused ranges.
- Refined editing prompts to prefer minimal new-file skeletons followed by focused `ReplaceRange` chunks.

### Fixed
- Stopped executing later tool calls after the first failed tool call in a batch.
- Reported Ctrl-C interrupted Bash runs as explicit interrupted tool results.

## 0.3.23 - 2026-05-16

### Changed
- Reworked tool-result context around latest, recent, pending, and kept results.
- Increased default provider and plan-mode response timeouts.
- Simplified result keep/forget handling and removed stale evidence naming from agent context.

### Fixed
- Kept non-argument tool failures visible to observe mode while treating argument errors as immediate feedback.
- Preserved Recall access to stored tool logs while allowing noisy context entries to be forgotten.

## 0.3.22 - 2026-05-16

### Fixed
- Preserved tool-result store entries referenced by Known, Hypotheses, and Evidence.
- Aligned plan-mode verify guidance with the implemented verify action shape.

### Changed
- Generated hypothesis status prompt schema from the enum to avoid prompt drift.

## 0.3.21 - 2026-05-16

### Added
- Added investigation hypotheses, including `dropped` for branches that are no longer worth tracking.
- Added evidence forgetting so ruled-out or dropped branches can release old tool results from context.

### Changed
- Tightened completion gates, verification blockers, and compact state update grouping.
- Simplified Search argument parsing and removed legacy knowledge-update behavior.
- Made provider reasoning payload shape configurable.

## 0.3.20 - 2026-05-15

### Changed
- Clarified Search tool guidance so models use at most one `glob=` per Search action and split multiple globs into multiple actions.

## 0.3.19 - 2026-05-15

### Changed
- Observe mode now requires every latest tool-result key to be covered by either `evidence` or `discard`.
- Verification pass/fail/block tool results are treated as decision-changing evidence until verification is recorded.

### Fixed
- Prevented partial observe checkpoints from silently dropping unhandled tool results.

## 0.3.18 - 2026-05-15

### Added
- Added `provider.<name>.available_models` and an interactive `/model` selector.
- Added an interactive `/provider` selector.
- Added reasoning effort selection after changing models.

### Changed
- Selection prompts now use a subtle selected-item background, support `j`/`k`, and clear after completion.
- Removed the redundant `keep current` option from selection prompts; current values are marked inline.
- Removed the startup status snapshot now that the persistent status bar is always shown.

### Fixed
- Disallowed no-value `/set provider.key` and `/set provider.url` queries.

## 0.3.17 - 2026-05-15

### Added
- Added plan mode with readonly tool limits, plan-specific timeouts, and stricter plan-mode completion formatting.
- Added persistent prompt status display while keeping active status updates during agent work.
- Added completion pressure gates so settled tasks finish by default unless the plan is reopened with context.
- Added global runtime data storage under `~/.nanocode`, with per-session debug/tool-result logs and per-project user rules.

### Changed
- Replaced project-local `[paths].nanocode_dir` with `[paths].data_dir`.
- Moved prompt history to global `~/.nanocode/history`.
- Replaced `/clean-logs` with `/clean`, which removes tool-result logs across all stored sessions.
- Compact observed tool-call results after they have been digested into agent state.

### Fixed
- Rejected chat actions in plan mode.
- Tightened blocked verification completion so it requires explicit user/manual confirmation context.
- Kept unbounded tool-result logs available on disk while bounded results stay in model context.
