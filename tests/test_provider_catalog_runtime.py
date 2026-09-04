import asyncio
import json
import os
import threading
from pathlib import Path

import httpx2
import pytest

from wizolt.base import ConfigError
from wizolt.config import Config, ConfigFile, ProviderConfig
from wizolt.providers.catalog import CatalogCodec, decode_bundled
from wizolt.providers.compat import ProviderPolicy
from wizolt.providers.schema import CatalogFormatError, CatalogSyncError, CatalogVersionConflict
import wizolt.providers.sync as sync_module
from wizolt.providers.sync import CATALOG_URL, MAX_REMOTE_BYTES, CatalogRepository, CatalogRuntime
from wizolt.session import Session

CATALOG_PATH = Path(__file__).parents[1] / "wizolt" / "providers" / "catalog.json"


def catalog_data() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def catalog_payload(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False).encode()


# `sync.py` looks `AsyncClient` up on the shared httpx2 module, so patching it patches that module
# for everyone. Hold the real class from before any patch, or a second patch wraps the first.
REAL_ASYNC_CLIENT = httpx2.AsyncClient


def mock_catalog_http(monkeypatch, handler, requests: list | None = None):
    """Route the catalog runtime's async client at `handler`, recording the requests it made."""
    real = REAL_ASYNC_CLIENT

    def client(**kwargs):
        def record(request):
            if requests is not None:
                requests.append(request)
            return handler(request)

        return real(**kwargs, transport=httpx2.MockTransport(record))

    monkeypatch.setattr(sync_module.httpx2, "AsyncClient", client)


def serving(payload: bytes, etag: str = '"catalog-test"'):
    """A handler that answers every request with `payload` and a 200."""
    return lambda _request: httpx2.Response(200, content=payload, headers={"ETag": etag})


def test_compiled_catalog_snapshot_is_deeply_immutable():
    snapshot = decode_bundled()

    with pytest.raises(TypeError):
        snapshot.defaults.provider_policy["api"] = "responses"
    with pytest.raises(TypeError):
        snapshot.request_recipes["new"] = snapshot.request_recipes["off"]
    with pytest.raises(TypeError):
        snapshot.model_rules[0].set["image.input"] = "auto"
    with pytest.raises(TypeError):
        snapshot.providers[0].defaults["api"] = "responses"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_id_forms", [1]),
        ("model_rules", [1]),
        ("providers", [1]),
    ),
)
def test_codec_reports_malformed_catalog_collections_as_format_errors(field, value):
    data = catalog_data()
    data[field] = value

    with pytest.raises(CatalogFormatError):
        CatalogCodec().decode(catalog_payload(data), "cached")


def test_codec_requires_both_named_version_groups():
    data = catalog_data()
    data["model_rules"][0]["match"] = {
        "version": {
            "pattern": "(?P<major>[0-9]+)",
            "min_inclusive": [1, 0],
        }
    }

    with pytest.raises(CatalogFormatError, match="major/minor"):
        CatalogCodec().decode(catalog_payload(data), "cached")


@pytest.mark.parametrize("value", (None, "all_provider_and_model_facts"))
def test_codec_enforces_the_narrow_maintenance_scope(value):
    data = catalog_data()
    if value is None:
        data.pop("maintenance_scope")
    else:
        data["maintenance_scope"] = value

    with pytest.raises(CatalogFormatError, match="maintenance_scope"):
        CatalogCodec().decode(catalog_payload(data), "cached")


@pytest.mark.parametrize(
    "remove",
    (
        lambda data: data["model_rules"][0].pop("why"),
        lambda data: data["providers"][0].pop("evidence"),
        lambda data: data["request_recipes"]["off"].pop("description"),
    ),
)
def test_codec_requires_knowledge_provenance(remove):
    data = catalog_data()
    remove(data)

    with pytest.raises(CatalogFormatError):
        CatalogCodec().decode(catalog_payload(data), "cached")


def test_codec_rejects_policy_references_to_unknown_reasoning_dialects():
    data = catalog_data()
    rule = next(rule for rule in data["model_rules"] if "reasoning.dialect" in rule["set"])
    rule["set"]["reasoning.dialect"] = "removed-by-new-catalog"

    with pytest.raises(CatalogFormatError, match="unknown dialect"):
        CatalogCodec().decode(catalog_payload(data), "cached")


def test_codec_requires_every_recipe_case_to_have_a_result():
    data = catalog_data()
    value = data["request_recipes"]["messages.manual-opus-45"]["steps"][0]["set"][2]["value"]
    value["case"][0].pop("then")

    with pytest.raises(CatalogFormatError, match="need when and then"):
        CatalogCodec().decode(catalog_payload(data), "cached")


def test_image_auto_is_not_text_only():
    data = catalog_data()
    rule = next(rule for rule in data["model_rules"] if rule["id"] == "model.text-only-00")
    rule["set"]["image.input"] = "auto"
    policy = ProviderPolicy(CatalogCodec().decode(catalog_payload(data), "cached"))

    assert policy.text_only(ProviderConfig(model="deepseek-chat")) is False


def test_provider_image_rules_and_ignore_mode_take_part_in_resolution():
    data = catalog_data()
    provider = next(provider for provider in data["providers"] if provider["id"] == "provider.openai")
    provider["model_rules"] = [
        {
            "id": "provider.openai.image-test",
            "match": {"prefixes": ["provider-text-model"]},
            "set": {"image.input": "text_only"},
            "why": "Exercises provider-local image policy.",
            "evidence": ["https://example.test/provider-image-policy"],
        }
    ]
    policy = ProviderPolicy(CatalogCodec().decode(catalog_payload(data), "cached"))
    assert policy.text_only(ProviderConfig(url="https://api.openai.com/v1", model="provider-text-model")) is True

    provider["model_rule_modes"] = {"image": "ignore"}
    provider["defaults"]["image.input"] = "auto"
    policy = ProviderPolicy(CatalogCodec().decode(catalog_payload(data), "cached"))
    assert policy.text_only(ProviderConfig(url="https://api.openai.com/v1", model="deepseek-chat")) is False


def test_stale_explicit_dialect_is_a_config_error_not_a_key_error():
    policy = ProviderPolicy(decode_bundled())
    config = ProviderConfig(model="future-model", chat_reasoning="removed-by-new-catalog")

    with pytest.raises(ConfigError, match="is not supported by catalog"):
        policy.resolve(config)


def test_snapshot_default_load_validates_config_against_the_active_cached_catalog(tmp_path, monkeypatch):
    data = catalog_data()
    data["version"] += 1
    data["defaults"]["reasoning_dialects"]["future-dialect"] = "off"
    repository = CatalogRepository(str(tmp_path))
    os.makedirs(repository.catalog_dir)
    Path(repository.cache_path).write_bytes(catalog_payload(data))

    saved = Session(cwd=str(tmp_path), config=Config(data_dir=str(tmp_path)))
    saved.messages.append({"role": "user", "content": "before catalog update"})
    saved.save_snapshot()
    raw_config = {
        "paths": {"data_dir": str(tmp_path)},
        "provider": {"model": "future-model", "chat_reasoning": "future-dialect"},
    }
    monkeypatch.setattr(ConfigFile, "load", classmethod(lambda _cls, _path=None: raw_config))

    resumed = Session.load_snapshot(saved.uid, cwd=str(tmp_path))

    assert resumed.config.provider.chat_reasoning == "future-dialect"
    assert resumed.catalog is not None
    assert resumed.catalog.source == "cached"
    assert resumed.catalog.snapshot.version == data["version"]


def test_provider_default_reasoning_levels_keep_their_provenance():
    policy = ProviderPolicy(decode_bundled())
    config = ProviderConfig(url="https://api.deepseek.com/v1", model="future-model")

    why, evidence = policy.effort_source(config)

    assert why
    assert evidence.startswith("https://")


def test_repository_reports_an_invalid_cached_copy(tmp_path):
    repository = CatalogRepository(str(tmp_path))
    os.makedirs(repository.catalog_dir)
    Path(repository.cache_path).write_text("not json", encoding="utf-8")

    snapshot, source, note = repository.select()

    assert snapshot.version == decode_bundled().version
    assert source == "bundled"
    assert "invalid" in note


def test_selected_catalog_defines_config_vocabulary(tmp_path):
    data = catalog_data()
    data["version"] += 1
    data["defaults"]["effort_order"].append("ultra")
    data["defaults"]["thinking_budgets"]["ultra"] = 65_536
    data["defaults"]["reasoning_dialects"]["custom"] = "off"
    repository = CatalogRepository(str(tmp_path))
    os.makedirs(repository.catalog_dir)
    Path(repository.cache_path).write_bytes(catalog_payload(data))

    runtime = CatalogRuntime(str(tmp_path))
    config = Config.from_dict(
        {"provider": {"default": {"reasoning": "ultra", "chat_reasoning": "custom"}}},
        policy=runtime.policy,
    )

    assert runtime.source == "cached"
    assert config.provider.reasoning == "ultra"
    assert config.provider.chat_reasoning == "custom"


async def test_fetch_rejects_same_version_with_different_content(tmp_path, monkeypatch):
    data = catalog_data()
    data["defaults"]["provider_policy"]["cache.prompt_key"] = False
    runtime = CatalogRuntime(str(tmp_path))
    mock_catalog_http(monkeypatch, serving(catalog_payload(data)))

    with pytest.raises(CatalogVersionConflict, match="same version"):
        await runtime.fetch()

    assert not os.path.exists(runtime.repository.cache_path)


async def test_fetch_wraps_transport_failures_without_a_ui_prefix(tmp_path, monkeypatch):
    runtime = CatalogRuntime(str(tmp_path))

    def unavailable(request):
        raise httpx2.ConnectError("offline", request=request)

    mock_catalog_http(monkeypatch, unavailable)

    with pytest.raises(CatalogSyncError) as raised:
        await runtime.fetch()

    assert str(raised.value) == "offline"


async def test_fetch_wraps_a_timeout_as_a_sync_error(tmp_path, monkeypatch):
    runtime = CatalogRuntime(str(tmp_path))

    def times_out(request):
        raise httpx2.ReadTimeout("timed out", request=request)

    mock_catalog_http(monkeypatch, times_out)

    with pytest.raises(CatalogSyncError, match="timed out"):
        await runtime.fetch()


async def test_fetch_rejects_a_malformed_remote_document(tmp_path, monkeypatch):
    runtime = CatalogRuntime(str(tmp_path))
    mock_catalog_http(monkeypatch, serving(b"<html>not a catalog</html>"))

    with pytest.raises(CatalogSyncError):
        await runtime.fetch()

    assert not os.path.exists(runtime.repository.cache_path)


async def test_fetch_rejects_a_body_over_the_limit_while_it_streams(tmp_path, monkeypatch):
    """The bound is enforced against the stream, so an oversize remote never becomes an oversize
    object in memory first."""
    runtime = CatalogRuntime(str(tmp_path))
    chunks = 0

    def flooding(_request):
        async def stream():
            nonlocal chunks
            while True:
                chunks += 1
                yield b"x" * (256 * 1024)

        return httpx2.Response(200, content=stream())

    mock_catalog_http(monkeypatch, flooding)

    with pytest.raises(CatalogSyncError, match="exceeds"):
        await runtime.fetch()

    assert chunks <= MAX_REMOTE_BYTES // (256 * 1024) + 2  # stopped at the bound, not after the body
    assert not os.path.exists(runtime.repository.cache_path)


async def test_fetch_sends_the_cached_etag_and_treats_304_as_current(tmp_path, monkeypatch):
    data = catalog_data()
    data["version"] += 1
    runtime = CatalogRuntime(str(tmp_path))
    requests: list = []
    mock_catalog_http(monkeypatch, serving(catalog_payload(data), etag='"v2"'), requests)

    assert (await runtime.fetch()).version == data["version"]
    assert requests[0].headers.get("if-none-match") is None  # no cache yet, so no condition

    mock_catalog_http(monkeypatch, lambda _request: httpx2.Response(304), requests)

    assert (await runtime.fetch()).version == data["version"]
    assert requests[-1].headers["if-none-match"] == '"v2"'


async def test_a_304_about_a_removed_cache_reprobes_exactly_once(tmp_path, monkeypatch):
    """Another process removed the cache between the probe and the answer, so this 304 is about a
    document that is no longer here. Ask again -- once, never in a loop."""
    data = catalog_data()
    data["version"] += 1
    runtime = CatalogRuntime(str(tmp_path))
    mock_catalog_http(monkeypatch, serving(catalog_payload(data), etag='"v2"'))
    await runtime.fetch()

    answers = []

    def answer(_request):
        answers.append(True)
        if len(answers) == 1:
            # Simulate a concurrent removal landing while this request was in flight.
            os.unlink(runtime.repository.cache_path)
        return httpx2.Response(304)

    mock_catalog_http(monkeypatch, answer)

    assert (await runtime.fetch()).version == decode_bundled().version
    assert len(answers) == 2


async def test_a_304_beside_a_newer_concurrent_cache_is_accepted_directly(tmp_path, monkeypatch):
    """The answer is stale but the cache that replaced it is newer and valid, so it is the answer:
    nothing is gained by asking again about a document already superseded on disk."""
    data = catalog_data()
    data["version"] += 1
    runtime = CatalogRuntime(str(tmp_path))
    mock_catalog_http(monkeypatch, serving(catalog_payload(data), etag='"v2"'))
    await runtime.fetch()

    replacement = catalog_data()
    replacement["version"] += 5
    answers = []

    def answer(_request):
        answers.append(True)
        Path(runtime.repository.cache_path).write_bytes(catalog_payload(replacement))
        return httpx2.Response(304)

    mock_catalog_http(monkeypatch, answer)

    assert (await runtime.fetch()).version == replacement["version"]
    assert len(answers) == 1


async def test_fetch_does_not_cache_a_remote_older_than_bundled(tmp_path, monkeypatch):
    data = catalog_data()
    data["version"] -= 1
    runtime = CatalogRuntime(str(tmp_path))
    mock_catalog_http(monkeypatch, serving(catalog_payload(data)))

    assert (await runtime.fetch()).version == decode_bundled().version
    assert not os.path.exists(runtime.repository.cache_path)


async def test_a_second_runtime_cannot_downgrade_a_newer_cache(tmp_path, monkeypatch):
    """Two runtimes over one data dir: the one holding the older document loses at commit."""
    newer = catalog_data()
    newer["version"] += 5
    older = catalog_data()
    older["version"] += 1

    first = CatalogRuntime(str(tmp_path))
    mock_catalog_http(monkeypatch, serving(catalog_payload(newer), etag='"newer"'))
    assert (await first.fetch()).version == newer["version"]

    second = CatalogRuntime(str(tmp_path))
    mock_catalog_http(monkeypatch, serving(catalog_payload(older), etag='"older"'))

    assert (await second.fetch()).version == newer["version"]
    assert CatalogRepository(str(tmp_path)).cached().version == newer["version"]


async def test_cancelling_a_blocked_request_writes_no_cache_and_closes_the_client(tmp_path, monkeypatch):
    runtime = CatalogRuntime(str(tmp_path))
    entered = asyncio.Event()
    closed = []
    class TrackedClient(REAL_ASYNC_CLIENT):
        def stream(self, *args, **kwargs):
            outer = self

            class Blocked:
                async def __aenter__(self):
                    entered.set()
                    await asyncio.Event().wait()

                async def __aexit__(self, *_args):
                    return False

            del outer
            return Blocked()

        async def __aexit__(self, *args):
            closed.append(True)
            return await super().__aexit__(*args)

    monkeypatch.setattr(sync_module.httpx2, "AsyncClient", TrackedClient)

    fetch = asyncio.ensure_future(runtime.fetch())
    await entered.wait()
    fetch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await fetch

    assert closed == [True]
    assert not os.path.exists(runtime.repository.cache_path)


async def test_cancelling_during_the_commit_waits_for_the_atomic_write(tmp_path, monkeypatch):
    """The commit holds a cross-process lock around an atomic replace; abandoning it mid-write is
    how a cache becomes a half file every later process has to reject."""
    data = catalog_data()
    data["version"] += 1
    runtime = CatalogRuntime(str(tmp_path))
    mock_catalog_http(monkeypatch, serving(catalog_payload(data)))
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    real_commit = runtime.repository.commit

    def slow_commit(response):
        entered.set()
        release.wait(5)
        try:
            return real_commit(response)
        finally:
            finished.set()

    monkeypatch.setattr(runtime.repository, "commit", slow_commit)

    fetch = asyncio.ensure_future(runtime.fetch())
    await asyncio.to_thread(entered.wait, 5)
    fetch.cancel()
    await asyncio.sleep(0.05)
    assert not finished.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await fetch

    assert finished.is_set()
    assert CatalogRepository(str(tmp_path)).cached().version == data["version"]


async def test_the_loop_advances_while_the_catalog_response_is_blocked(tmp_path, monkeypatch):
    runtime = CatalogRuntime(str(tmp_path))
    data = catalog_data()
    data["version"] += 1
    release = asyncio.Event()

    def slow(_request):
        return httpx2.Response(200, content=catalog_payload(data))

    async def gated(request):
        await release.wait()
        return slow(request)

    monkeypatch.setattr(sync_module.httpx2, "AsyncClient", lambda **kwargs: REAL_ASYNC_CLIENT(**kwargs, transport=httpx2.MockTransport(gated)))
    beats = 0

    async def heartbeat():
        nonlocal beats
        while True:
            beats += 1
            await asyncio.sleep(0.001)

    pulse = asyncio.ensure_future(heartbeat())
    fetch = asyncio.ensure_future(runtime.fetch())
    await asyncio.sleep(0.05)
    assert beats > 5
    release.set()
    assert (await fetch).version == data["version"]
    pulse.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pulse


def test_catalog_lock_keeps_one_inode_across_users(tmp_path):
    repository = CatalogRepository(str(tmp_path))

    with repository._locked():
        first = os.stat(Path(repository.catalog_dir) / "catalog.lock").st_ino
    assert (Path(repository.catalog_dir) / "catalog.lock").exists()
    with repository._locked():
        second = os.stat(Path(repository.catalog_dir) / "catalog.lock").st_ino

    assert first == second


async def test_refresh_updates_the_cache_without_activating_it(tmp_path, monkeypatch):
    """The automatic refresh is cache-only: a long turn's requests keep one catalog version, and
    the newer document is picked up at the next startup."""
    data = catalog_data()
    data["version"] += 1
    runtime = CatalogRuntime(str(tmp_path))
    active = runtime.snapshot.version
    mock_catalog_http(monkeypatch, serving(catalog_payload(data)))

    assert runtime.refresh_due() is True
    await runtime.refresh()

    assert runtime.snapshot.version == active  # not hot-swapped
    assert runtime.sync_state.last_version == data["version"]
    assert runtime.sync_state.error == ""
    assert runtime.sync_state.checking is False
    assert runtime.refresh_due() is False  # the 72h gate now holds


async def test_refresh_records_a_failure_as_status_and_leaves_the_policy_alone(tmp_path, monkeypatch):
    runtime = CatalogRuntime(str(tmp_path))
    policy = runtime.policy

    def unavailable(request):
        raise httpx2.ConnectError("offline", request=request)

    mock_catalog_http(monkeypatch, unavailable)

    assert runtime.refresh_due() is True
    await runtime.refresh()

    assert "offline" in runtime.sync_state.error
    assert runtime.policy is policy
    assert runtime.sync_state.checking is False


async def test_cancelling_a_refresh_waits_for_the_critical_section(tmp_path, monkeypatch):
    """The commit holds a cross-process lock around an atomic cache write. Cancelling the awaiter
    may not return while that section is mid-write, and the 72h gate stays where it was: nothing
    was checked."""
    runtime = CatalogRuntime(str(tmp_path))
    mock_catalog_http(monkeypatch, serving(catalog_payload(catalog_data())))
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def slow_commit(_response):
        entered.set()
        release.wait(5)
        finished.set()
        return runtime.snapshot

    monkeypatch.setattr(runtime.repository, "commit", slow_commit)
    assert runtime.refresh_due() is True

    task = asyncio.ensure_future(runtime.refresh())
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0.05)
    assert not finished.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished.is_set()
    assert runtime.sync_state.checked_at == 0.0  # the attempt did not complete, so the gate is untouched
    assert runtime.sync_state.checking is False


async def test_manual_sync_activates_a_newer_catalog_at_the_command_boundary(tmp_path, monkeypatch):
    data = catalog_data()
    data["version"] += 1
    runtime = CatalogRuntime(str(tmp_path))
    mock_catalog_http(monkeypatch, serving(catalog_payload(data)))

    snapshot = await runtime.sync()

    assert snapshot.version == data["version"]
    assert runtime.snapshot.version == data["version"]
    assert runtime.source == "cached"


async def test_manual_sync_persists_state_off_the_event_loop(tmp_path, monkeypatch):
    runtime = CatalogRuntime(str(tmp_path))
    loop_thread = threading.get_ident()
    save_threads = []
    real_save = runtime._save_state

    mock_catalog_http(monkeypatch, serving(catalog_payload(catalog_data())))
    monkeypatch.setattr(runtime, "_save_state", lambda: save_threads.append(threading.get_ident()) or real_save())

    await runtime.sync()

    assert save_threads and save_threads[0] != loop_thread
