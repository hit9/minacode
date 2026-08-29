# Compatibility catalog

minacode keeps documented compatibility adjustments in a versioned catalog. It covers only
well-known provider and model behavior that needs an exception to the generic API path: protocol
routing, reasoning controls and levels, reasoning replay, strict schemas, prompt caching,
temperature constraints, provider-side tools, and documented text-only models.

The catalog is not a model allowlist. An unknown provider or model still uses the ordinary
OpenAI-compatible path, and an unknown image route is tried on the active model first. Explicit
[provider settings](configuration.md#optional-provider-settings) take precedence when an endpoint
needs a different choice.

## Which copy is active

Every installation includes a bundled catalog. minacode may also have a complete copy previously
synced from [the minacode repository](https://github.com/hit9/minacode/blob/master/minacode/providers/catalog.json).
At startup it validates both copies and selects the one with the higher numeric version. The two
documents are never merged, and publication dates do not decide which one wins.

An invalid cached copy is ignored, leaving the bundled copy active. If two different documents
claim the same version, minacode keeps the bundled copy and reports the conflict in `/catalog`.
An invalid bundled copy means the installation is damaged and stops startup rather than hiding
the problem behind hard-coded compatibility behavior.

Run `/catalog` to see the active version, publication date, source, bundled and cached versions,
maintenance scope, remote URL, and the result of the latest sync attempt.

## Updates

After startup, minacode checks GitHub at most once every 72 hours. This is one short background
request and does not delay the first prompt. A newer document is saved for the next session; the
running session keeps its current catalog so one turn cannot change compatibility policy midway.

Use `/catalog sync` to check immediately. A newer valid version is saved and activated at that
command boundary for the current session. An older, unchanged, invalid, or unreachable remote does
not replace the active copy. `/catalog` keeps the latest result visible.

## Known models, gateways, and overrides

Model facts follow a documented model family even when a different compatible endpoint serves it.
For catalog-declared canonical `vendor/model` IDs, the model suffix participates in the same
matching; an unrecognized vendor prefix or a custom alias stays unknown and uses the generic path.
Endpoint facts such as protocol routing still come from the endpoint.

If the catalog and an endpoint disagree, prefer the narrowest explicit setting that describes the
endpoint you actually use:

- set `api` or `chat_reasoning` when automatic wire selection is wrong;
- set `reasoning_history` when a gateway needs a different replay policy;
- declare `[provider.NAME.models]` when a model offers a different reasoning scale;
- use `omit_body` or `extra_body` for request fields the endpoint rejects or requires.

See [Configuration](configuration.md) for these settings. `/config` shows both configured and
resolved values, while `/reason` shows the evidence behind a shortened reasoning menu.

## When a sync or rule looks wrong

The current active copy stays usable when a network request or cached document fails. Check
`/catalog` first: it distinguishes a network failure, invalid cache, same-version conflict, and an
already-current remote. Then use `/config` to compare the resolved protocol and reasoning policy
with the endpoint's documentation.

If the endpoint is compatible but absent from the catalog, configure the needed override rather
than renaming it to resemble a known provider. Unknown names deliberately stay on the generic path.
