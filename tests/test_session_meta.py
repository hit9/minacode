"""session meta (split from tests/test_session_persistence.py)."""
import json
import os
import time

import pytest
from test_session_persistence import session_with_data_dir

from wizolt.base import WizoltError
from wizolt.config import (
    Config,
)
from wizolt.session import HistorySegment, Session, SessionSnapshotCodec, SessionSnapshotStore


def test_snapshot_messages_strips_non_persistable_roles(tmp_path):
    s = Session(cwd=str(tmp_path))
    s.messages = [
        {"role": "system", "content": "[Session resumed: old-session-id]"},
        {"role": "user", "content": "hello"},
    ]
    messages = SessionSnapshotCodec.snapshot_messages(s)
    # Internal resume marker is stripped; user message is kept.
    roles = [m["role"] for m in messages]
    assert "system" not in roles
    assert "user" in roles
    assert len(messages) == 1

async def test_session_name_latches_then_follows_the_goal(tmp_path):
    s = session_with_data_dir(tmp_path)
    assert s.name == ""

    s.messages.append({"role": "user", "content": "fix the fd leak in MCPFileTokenStore\nsecond line"})
    await s.save_snapshot()
    # Nothing to derive from until there is a message, then the opening line names the session.
    assert (s.name, s.state.name_source) == ("fix the fd leak in MCPFileTokenStore", "input")

    s.state.goal = "close every descriptor opened by the token store"
    await s.save_snapshot()
    # A goal is a better description of the same work, so it takes over from the opening line.
    assert (s.name, s.state.name_source) == ("close every descriptor opened by the token store", "goal")

    s.rename("token store cleanup")
    await s.save_snapshot()
    s.state.goal = "something else entirely"
    await s.save_snapshot()
    # A name the user chose is never replaced by a derived one.
    assert (s.name, s.state.name_source) == ("token store cleanup", "user")
    assert Session.load_snapshot(s.uid, config=s.config).name == "token store cleanup"

async def test_session_name_does_not_change_when_goal_changes(tmp_path):
    """Once the name is derived from a goal, later goal changes do not overwrite it."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "fix the parser"})
    await s.save_snapshot()
    assert (s.name, s.state.name_source) == ("fix the parser", "input")

    s.state.goal = "rewrite the tokenizer"
    await s.save_snapshot()
    assert (s.name, s.state.name_source) == ("rewrite the tokenizer", "goal")

    s.state.goal = "add error recovery to the parser"
    await s.save_snapshot()
    # Goal changed, but the name was already latched from the first goal — stays put.
    assert (s.name, s.state.name_source) == ("rewrite the tokenizer", "goal")
    assert Session.load_snapshot(s.uid, config=s.config).name == "rewrite the tokenizer"

async def test_session_name_survives_compaction_dropping_the_opening_message(tmp_path):
    from wizolt.prompts import COMPACTION_SUMMARY_TITLE

    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "add a session picker"})
    await s.save_snapshot()

    s.messages = [
        {"role": "user", "content": COMPACTION_SUMMARY_TITLE + "\nearlier work"},
        {"role": "user", "content": "now also sort them by date"},
    ]
    await s.save_snapshot()

    # Compaction replaces the opening message; a name derived afresh here would silently rewrite
    # what the session has been listed as since it started.
    assert s.name == "add a session picker"
    assert s.opening_text() == "now also sort them by date"

async def test_listing_sessions_reads_no_logs(tmp_path, monkeypatch):
    config = Config(data_dir=str(tmp_path / "data"))
    project = tmp_path / "project"
    project.mkdir()
    first = Session(cwd=str(project), config=config)
    first.messages.append({"role": "user", "content": "older session"})
    await first.save_snapshot()
    second = Session(cwd=str(project), config=config)
    second.messages.append({"role": "user", "content": "newer session"})
    second.state.round_count = 3
    await second.save_snapshot()
    os.utime(SessionSnapshotStore.session_path(config.data_dir, str(project), first.uid), (1, 1))

    real_open = open

    def guard(file, *args, **kwargs):
        assert not str(file).endswith(".jsonl"), f"listing opened a session log: {file}"
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guard)
    entries = SessionSnapshotStore.list_sessions(config.data_dir, str(project))

    assert [entry.uid for entry in entries] == [second.uid, first.uid]
    assert [entry.name for entry in entries] == ["newer session", "older session"]
    assert entries[0].rounds == 3
    assert entries[0].cwd == str(project)

async def test_listing_survives_a_missing_sidecar(tmp_path):
    config = Config(data_dir=str(tmp_path / "data"))
    s = Session(cwd=str(tmp_path), config=config)
    s.messages.append({"role": "user", "content": "unlabelled"})
    await s.save_snapshot()
    os.unlink(SessionSnapshotStore.meta_path(config.data_dir, s.cwd, s.uid))

    entry = SessionSnapshotStore.list_sessions(config.data_dir, s.cwd)[0]

    # The log is what makes a session real; the sidecar only labels it.
    assert (entry.uid, entry.name) == (s.uid, "")
    assert entry.label() == s.uid

async def test_expired_sessions_take_their_sidecar_with_them(tmp_path):
    config = Config(data_dir=str(tmp_path / "data"))
    s = Session(cwd=str(tmp_path), config=config)
    s.messages.append({"role": "user", "content": "old"})
    await s.save_snapshot()
    stale = Session(cwd=str(tmp_path), config=config)
    stale.messages.append({"role": "user", "content": "stale"})
    await stale.save_snapshot()
    meta = SessionSnapshotStore.meta_path(config.data_dir, stale.cwd, stale.uid)
    old = time.time() - 40 * 86400
    os.utime(SessionSnapshotStore.session_path(config.data_dir, stale.cwd, stale.uid), (old, old))
    s.settings.session_retention_days = 30

    assert SessionSnapshotStore.clean_expired(s.config.data_dir, s.uid, s.settings.session_retention_days) == 1
    assert not os.path.exists(meta)

async def test_resume_accepts_a_name_or_uid_prefix(tmp_path):
    config = Config(data_dir=str(tmp_path / "data"))
    project = tmp_path / "project"
    project.mkdir()
    s = Session(cwd=str(project), config=config)
    s.messages.append({"role": "user", "content": "teach the status bar to breathe"})
    await s.save_snapshot()

    for query in ("status bar", "TEACH the status", s.uid[:8]):
        assert SessionSnapshotStore.resolve_uid(query, config.data_dir, str(project)) == s.uid

    # A search from another directory still finds it: the user moved, the session did not.
    assert SessionSnapshotStore.resolve_uid("status bar", config.data_dir, str(tmp_path)) == s.uid
    assert Session.load_snapshot("status bar", config=config, cwd=str(project)).uid == s.uid

async def test_ambiguous_resume_names_its_candidates(tmp_path):
    config = Config(data_dir=str(tmp_path / "data"))
    first = Session(cwd=str(tmp_path), config=config)
    first.messages.append({"role": "user", "content": "rename the sweep constants"})
    await first.save_snapshot()
    second = Session(cwd=str(tmp_path), config=config)
    second.messages.append({"role": "user", "content": "rename the glow styles"})
    await second.save_snapshot()

    with pytest.raises(WizoltError) as error:
        SessionSnapshotStore.resolve_uid("rename the", config.data_dir, str(tmp_path))

    # Guessing between them would resume the wrong work silently.
    assert "2 sessions match" in str(error.value)
    assert first.uid in str(error.value) and second.uid in str(error.value)

async def test_listing_survives_a_malformed_sidecar(tmp_path):
    config = Config(data_dir=str(tmp_path / "data"))
    s = Session(cwd=str(tmp_path), config=config)
    s.messages.append({"role": "user", "content": "labelled session"})
    await s.save_snapshot()
    # A hand-edited or torn sidecar: valid JSON, but the turn count is not a number.
    with open(SessionSnapshotStore.meta_path(config.data_dir, s.cwd, s.uid), "w", encoding="utf-8") as file:
        json.dump({"name": "kept", "opening": "labelled session", "rounds": "many", "cwd": s.cwd}, file)

    entry = SessionSnapshotStore.list_sessions(config.data_dir, s.cwd)[0]

    # The bad turn count is dropped, not the session: one corrupt cache must not break the picker.
    assert (entry.uid, entry.name, entry.rounds) == (s.uid, "kept", 0)

async def test_search_widens_only_after_a_miss(tmp_path, monkeypatch):
    config = Config(data_dir=str(tmp_path / "data"))
    here = tmp_path / "here"
    here.mkdir()
    local = Session(cwd=str(here), config=config)
    local.messages.append({"role": "user", "content": "a local session"})
    await local.save_snapshot()
    calls = []
    real = SessionSnapshotStore.list_sessions

    def spy(cls, data_dir, cwd="", *, all_projects=False):
        calls.append(all_projects)
        return real(data_dir, cwd, all_projects=all_projects)

    monkeypatch.setattr(SessionSnapshotStore, "list_sessions", classmethod(spy))

    # A hit in the current project never scans the rest.
    assert SessionSnapshotStore.search_sessions("local", config.data_dir, str(here))
    assert calls == [False]

    # Only a miss widens to every project.
    calls.clear()
    assert SessionSnapshotStore.search_sessions("local", config.data_dir, str(tmp_path / "elsewhere"))
    assert calls == [False, True]

async def test_history_segment_persists_effective_model(tmp_path):
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.history.append(HistorySegment(key="seg.1", title="earlier task", text="user:\nfind the bug", model="compactor-x"))
    await s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))

    assert restored.history[0].model == "compactor-x"

def test_history_segment_missing_model_field_restores_empty():
    """Snapshots written before the model field existed decode with an empty model, not a crash."""
    data = [
        {
            "key": "seg.1",
            "title": "old",
            "blob": "",
            "created_at": "",
            "scope": "history",
            "trigger": "auto",
            "fallback": False,
            "messages": 2,
            "summary": "",
        }
    ]
    segments = SessionSnapshotCodec.history(data, {})
    assert segments[0].model == ""
    assert segments[0].key == "seg.1"
