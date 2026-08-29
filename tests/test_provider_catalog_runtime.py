import json
import os
from pathlib import Path
from typing import ClassVar
from urllib.error import HTTPError

import pytest

from minacode.config import Config
from minacode.providers.catalog import CatalogCodec, decode_bundled
from minacode.providers.schema import CatalogFormatError, CatalogSyncError
from minacode.providers.sync import CATALOG_URL, CatalogRepository, CatalogRuntime

CATALOG_PATH = Path(__file__).parents[1] / "minacode" / "providers" / "catalog.json"


def catalog_data() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def catalog_payload(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False).encode()


class Response:
    status = 200
    headers: ClassVar[dict[str, str]] = {"ETag": '"catalog-test"'}

    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit: int) -> bytes:
        return self.payload


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


def test_fetch_rejects_same_version_with_different_content(tmp_path, monkeypatch):
    data = catalog_data()
    data["defaults"]["provider_policy"]["cache.prompt_key"] = False
    repository = CatalogRepository(str(tmp_path))
    monkeypatch.setattr("minacode.providers.sync.urlopen", lambda *_args, **_kwargs: Response(catalog_payload(data)))

    with pytest.raises(CatalogSyncError, match="same version"):
        repository.fetch()

    assert not os.path.exists(repository.cache_path)


def test_fetch_treats_http_304_as_current(tmp_path, monkeypatch):
    repository = CatalogRepository(str(tmp_path))

    def not_modified(*_args, **_kwargs):
        raise HTTPError(CATALOG_URL, 304, "Not Modified", {}, None)

    monkeypatch.setattr("minacode.providers.sync.urlopen", not_modified)

    assert repository.fetch().version == decode_bundled().version


def test_fetch_does_not_cache_a_remote_older_than_bundled(tmp_path, monkeypatch):
    data = catalog_data()
    data["version"] -= 1
    repository = CatalogRepository(str(tmp_path))
    monkeypatch.setattr("minacode.providers.sync.urlopen", lambda *_args, **_kwargs: Response(catalog_payload(data)))

    assert repository.fetch().version == decode_bundled().version
    assert not os.path.exists(repository.cache_path)


def test_catalog_lock_keeps_one_inode_across_users(tmp_path):
    repository = CatalogRepository(str(tmp_path))

    with repository._locked():
        first = os.stat(Path(repository.catalog_dir) / "catalog.lock").st_ino
    assert (Path(repository.catalog_dir) / "catalog.lock").exists()
    with repository._locked():
        second = os.stat(Path(repository.catalog_dir) / "catalog.lock").st_ino

    assert first == second
