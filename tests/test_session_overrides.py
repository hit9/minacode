"""session overrides (split from tests/test_session_persistence.py)."""
import json
import os

from test_session_persistence import log_path, read_jsonl, read_lines, session_with_data_dir

from minacode.cli import CommandLoop
from minacode.cli.commands import provider, set_model
from minacode.config import (
    ProviderConfig,
)
from minacode.engine import Agent
from minacode.session import Session


def test_provider_overrides_persist_and_restore(tmp_path):
    """Runtime /provider /model /reason /api switches survive a save/load round trip, and an
    unchanged override is not rewritten by the next save."""
    s = session_with_data_dir(tmp_path)
    s.config.providers["other"] = ProviderConfig(model="m", api="chat", reasoning="low")
    s.provider_overrides = {
        "active_provider": "other",
        "providers": {"other": {"model": "model-x", "reasoning": "high", "api": "responses"}},
    }
    s.messages.append({"role": "user", "content": "hi"})
    s.save_snapshot()

    lines = read_jsonl(log_path(s))
    assert lines[0]["provider_overrides"] == s.provider_overrides

    restored = Session.load_snapshot(s.uid, config=s.config)
    assert restored.config.active_provider == "other"
    entry = restored.config.providers["other"]
    assert (entry.model, entry.reasoning, entry.api) == ("model-x", "high", "responses")

    restored.save_snapshot()
    assert "provider_overrides" not in read_jsonl(log_path(s))[-1]

def test_provider_overrides_stale_values_are_skipped(tmp_path):
    """A resume applies overrides best-effort: a removed entry or a renamed choice falls back to
    the config value, while a free-string model still applies."""
    s = session_with_data_dir(tmp_path)
    s.config.providers["other"] = ProviderConfig(model="m", api="chat", reasoning="low")
    s.provider_overrides = {
        "active_provider": "gone",
        "providers": {"other": {"model": "model-x", "reasoning": "bogus", "api": "bogus"}},
    }
    s.messages.append({"role": "user", "content": "hi"})
    s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config)
    assert restored.config.active_provider == "default"
    entry = restored.config.providers["other"]
    assert entry.model == "model-x"
    assert entry.reasoning == "low"
    assert entry.api == "chat"

def test_legacy_snapshot_without_provider_overrides_loads(tmp_path):
    """Snapshots written before this feature carry no provider_overrides key and load unchanged."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hi"})
    s.save_snapshot()
    path = log_path(s)
    lines = read_lines(path)
    for line in lines:
        line.pop("provider_overrides", None)
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    restored = Session.load_snapshot(s.uid, config=s.config)
    assert restored.provider_overrides == {}
    assert restored.config.active_provider == s.config.active_provider

def test_provider_overrides_survive_delta_saves(tmp_path):
    """A session that already has a snapshot continues through delta writes; the override added
    later must be merged back on load, not dropped with the rest of the un-listed delta keys."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hi"})
    s.save_snapshot()

    s.provider_overrides = {"providers": {"default": {"model": "model-y"}}}
    s.messages.append({"role": "user", "content": "more"})
    s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config)
    assert restored.config.provider.model == "model-y"

def test_provider_overrides_alone_do_not_force_a_save(tmp_path):
    """A fresh session whose only content is a provider switch has no value and must not persist:
    an empty session file would show up in /sessions and claim the latest pointer. The switch is
    remembered only once the session actually has content."""
    s = session_with_data_dir(tmp_path)
    s.provider_overrides = {"providers": {"default": {"model": "model-z"}}}

    s.save_snapshot()

    assert not os.path.exists(log_path(s))

def test_provider_switch_chain_round_trips_through_commands(tmp_path):
    """End to end through the real slash-command handlers: /model on default, /provider a,
    /model on a, /provider b — save and resume keeps each switch keyed to the entry it was made
    on, with the last active provider winning."""
    s = session_with_data_dir(tmp_path)
    s.config.providers["a"] = ProviderConfig(model="ma", api="chat", reasoning="low")
    s.config.providers["b"] = ProviderConfig(model="mb", api="chat", reasoning="low")
    loop = CommandLoop(Agent(s, output_fn=lambda _text: None), output_fn=lambda _text: None)
    loop.interactive_input = False

    set_model(loop, "m-on-default")
    provider(loop, "a")
    set_model(loop, "m-on-a")
    provider(loop, "b")
    s.messages.append({"role": "user", "content": "hi"})
    s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config)
    assert restored.config.active_provider == "b"
    assert restored.config.providers["default"].model == "m-on-default"
    assert restored.config.providers["a"].model == "m-on-a"
    assert restored.config.providers["b"].model == "mb"

def test_resumed_session_switch_writes_a_new_delta(tmp_path):
    """After a resume restores an override, switching again writes a delta for the new value and a
    second resume reads it back — the marker/delta chain stays aligned across a full round trip."""
    s = session_with_data_dir(tmp_path)
    s.provider_overrides = {"providers": {"default": {"model": "model-1"}}}
    s.messages.append({"role": "user", "content": "hi"})
    s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config)
    assert restored.config.provider.model == "model-1"

    restored.provider_overrides = {"providers": {"default": {"model": "model-2"}}}
    restored.messages.append({"role": "user", "content": "more"})
    restored.save_snapshot()

    again = Session.load_snapshot(s.uid, config=s.config)
    assert again.config.provider.model == "model-2"

def test_switch_then_first_message_carries_the_override(tmp_path):
    """A switch made while the session is still empty is dropped by the empty-content early return;
    once the session gains its first message, the full snapshot is written with the override intact."""
    s = session_with_data_dir(tmp_path)
    s.provider_overrides = {"providers": {"default": {"model": "model-z"}}}
    s.save_snapshot()
    assert not os.path.exists(log_path(s))

    s.messages.append({"role": "user", "content": "hi"})
    s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config)
    assert restored.config.provider.model == "model-z"

def test_pending_user_inputs_persist_and_restore(tmp_path):
    s = session_with_data_dir(tmp_path)
    s.enqueue_user_input("queued one")
    s.enqueue_user_input("queued two")

    s.save_snapshot()

    lines = read_jsonl(log_path(s))
    assert lines[0]["pending_user_inputs"] == ["queued one", "queued two"]
    restored = Session.load_snapshot(s.uid, config=s.config)
    assert [item.text for item in restored.pending_user_inputs] == ["queued one", "queued two"]
    assert all(not item.inflight for item in restored.pending_user_inputs)

def test_pending_user_input_delta_replaces_queue_state(tmp_path):
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "active"})
    s.save_snapshot()
    s.enqueue_user_input("queued")
    s.save_snapshot()
    s.pending_user_inputs.clear()
    s.save_snapshot()

    lines = read_jsonl(log_path(s))
    assert lines[1]["pending_user_inputs"] == ["queued"]
    assert lines[2]["pending_user_inputs"] == []
    restored = Session.load_snapshot(s.uid, config=s.config)
    assert restored.pending_user_inputs == []
