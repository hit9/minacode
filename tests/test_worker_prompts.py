"""worker prompts (split from tests/test_worker_handoff.py)."""
import os
import time

import pytest
from agent_harness import session
from test_worker_handoff import _requested_system

from minacode.context import ContextManager
from minacode.prompts import SYSTEM_PROMPT
from minacode.tools import TOOL_REGISTRY, Tool


def test_tool_names_filter_resolved_schemas_and_keep_registry_order(tmp_path):
    s = session(tmp_path)
    names = ("Read", "Edit", "Bash")
    s.tool_names = names
    resolved = [schema["function"]["name"] for schema in Tool.resolved_schemas(s)]
    assert resolved == [name for name in TOOL_REGISTRY if name in names]

    # Empty tuple = no filtering; identical to a session that never set tool_names.
    s.tool_names = ()
    unfiltered = [schema["function"]["name"] for schema in Tool.resolved_schemas(s)]
    plain = [schema["function"]["name"] for schema in Tool.resolved_schemas(session(tmp_path))]
    assert unfiltered == plain
    assert "Read" in unfiltered and "Edit" in unfiltered and "Bash" in unfiltered

def test_system_prompt_comes_from_session(tmp_path):
    _, system = _requested_system(tmp_path, custom="CUSTOM WORKER ROLE")
    assert system == "CUSTOM WORKER ROLE"

    _, parent_system = _requested_system(tmp_path)
    assert parent_system == SYSTEM_PROMPT.strip()

def test_system_prompt_default_matches_prompts_module(tmp_path):
    _, system = _requested_system(tmp_path)
    assert system == SYSTEM_PROMPT.strip()
    assert ContextManager(session(tmp_path)).model_messages(SYSTEM_PROMPT)[0]["content"] == SYSTEM_PROMPT.strip()

def test_worker_snapshot_hidden_from_listing_and_latest(tmp_path):
    from minacode.session import Session, SessionSnapshotStore

    parent = session(tmp_path)
    parent.messages.append({"role": "user", "content": "parent request"})
    parent.save_snapshot()  # latest -> parent.uid
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    worker.messages.append({"role": "user", "content": "worker request"})
    worker.save_snapshot()

    assert worker.uid.endswith(".w")
    assert SessionSnapshotStore.read_latest(SessionSnapshotStore.project_dir(parent.config.data_dir, str(tmp_path))) == parent.uid
    assert not os.path.exists(SessionSnapshotStore.meta_path(parent.config.data_dir, str(tmp_path), worker.uid))
    entries = SessionSnapshotStore.list_sessions(parent.config.data_dir, cwd=str(tmp_path))
    assert all(entry.uid != worker.uid for entry in entries)
    assert any(entry.uid == parent.uid for entry in entries)
    # `-c` still resolves to the parent even though the worker log is newer on disk.
    assert SessionSnapshotStore.latest_uid(parent.config.data_dir, cwd=str(tmp_path)) == parent.uid

def test_clean_expired_removes_worker_when_parent_expires_later_in_scan(tmp_path, monkeypatch):
    from minacode.session import Session, SessionSnapshotStore

    parent = session(tmp_path)
    parent.settings.session_retention_days = 1
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    worker.messages.append({"role": "user", "content": "worker request"})
    worker.save_snapshot()  # create first: the worker is visited before its parent below
    parent.messages.append({"role": "user", "content": "parent request"})
    parent.save_snapshot()
    directory = SessionSnapshotStore.project_dir(parent.config.data_dir, str(tmp_path))
    parent_path = os.path.join(directory, parent.uid + ".jsonl")
    worker_path = os.path.join(directory, worker.uid + ".jsonl")
    old = time.time() - 3 * 86400
    os.utime(parent_path, (old, old))  # parent expired; worker remains fresh

    real_scandir = os.scandir

    def worker_first(path):
        entries = list(real_scandir(path))
        return iter(sorted(entries, key=lambda entry: (not entry.name.endswith(".w.jsonl"), entry.name)))

    monkeypatch.setattr("minacode.session.os.scandir", worker_first)
    cleaner = session(tmp_path)
    cleaner.settings.session_retention_days = 1

    assert SessionSnapshotStore.clean_expired(cleaner) >= 2
    assert not os.path.isfile(parent_path)
    assert not os.path.isfile(worker_path)

def test_delegate_registration_gates(tmp_path):
    from minacode.session import Session

    def names(s):
        return {schema["function"]["name"] for schema in Tool.resolved_schemas(s)}

    # Gate off at session start (no [worker] provider): runtime.worker alone cannot register it,
    # and setting the provider mid-session is frozen out.
    off = session(tmp_path)
    off.settings.worker = True
    assert "Delegate" not in names(off)
    off.config.worker_provider = "default"
    assert "Delegate" not in names(off)

    # Gate on at session start: both halves are required, and a runtime provider change never
    # flips the tool block.
    on = Session(cwd=str(tmp_path), config=off.config)
    assert "Delegate" not in names(on)  # runtime.worker off
    on.settings.worker = True
    assert "Delegate" in names(on)
    on.config.worker_provider = ""
    assert "Delegate" in names(on)  # the frozen half is unchanged by runtime changes
    on.settings.worker = False
    assert "Delegate" not in names(on)  # the live half still drops the schema
    on.settings.worker = True
    assert "Delegate" in names(on)

def test_worker_config_parsing_and_validation(tmp_path):
    from minacode.base import ConfigError
    from minacode.config import (
        Config,
        RuntimeSettings,
    )

    config = Config.from_dict({"worker": {"provider": "fast"}, "provider": {"active": "default", "default": {"model": "d"}, "fast": {"model": "m"}}})
    assert config.worker_provider == "fast"
    assert RuntimeSettings.from_dict({"runtime": {"worker": True}}).worker is True
    assert RuntimeSettings.from_dict({}).worker is False
    with pytest.raises(ConfigError, match="worker.provider"):
        Config.from_dict({"worker": {"provider": "nope"}})

def test_resolve_uid_prefix_skips_worker_snapshot(tmp_path):
    from minacode.session import Session, SessionSnapshotStore

    parent = session(tmp_path)
    parent.messages.append({"role": "user", "content": "parent request"})
    parent.save_snapshot()
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    worker.messages.append({"role": "user", "content": "worker request"})
    worker.save_snapshot()

    resolved = SessionSnapshotStore.resolve_uid(parent.uid[:12], parent.config.data_dir, str(tmp_path))
    assert resolved == parent.uid

    # And an exact worker uid is not reachable through the prefix path either: it would require
    # typing the full uid, which the user never sees in listings.
    resolved_worker = SessionSnapshotStore.resolve_uid(worker.uid, parent.config.data_dir, str(tmp_path))
    assert resolved_worker == worker.uid  # explicit full uid still works, by design

def test_worker_prompt_shares_language_and_secret_rules_with_parent():
    from minacode.prompts import LANGUAGE_RULES, SECRET_RULES, SYSTEM_PROMPT, WORKER_PROMPT

    assert LANGUAGE_RULES in SYSTEM_PROMPT
    assert LANGUAGE_RULES in WORKER_PROMPT
    assert SECRET_RULES in SYSTEM_PROMPT
    assert SECRET_RULES in WORKER_PROMPT

def test_worker_prompt_does_not_inherit_parent_review_or_terminal_output():
    from minacode.prompts import SYSTEM_PROMPT, WORKER_PROMPT

    assert "REVIEW:" in SYSTEM_PROMPT and "REVIEW:" not in WORKER_PROMPT
    assert "terminal scrollback" in SYSTEM_PROMPT and "terminal scrollback" not in WORKER_PROMPT
    assert "You write for the delegator" in WORKER_PROMPT and "You write for the delegator" not in SYSTEM_PROMPT
    for unavailable in ("Ask", "NextHints"):
        assert unavailable not in WORKER_PROMPT

def test_prompts_never_name_tools_outside_their_toolset():
    import re

    from minacode.prompts import WORKER_PROMPT
    from minacode.tools import TOOL_REGISTRY
    from minacode.tools.delegate import WORKER_TOOLS

    def mentioned(prompt):
        return {name for name in TOOL_REGISTRY if re.search(rf"\b{re.escape(name)}\b", prompt)}

    worker_mentioned = mentioned(WORKER_PROMPT)
    assert worker_mentioned <= set(WORKER_TOOLS), worker_mentioned - set(WORKER_TOOLS)


def test_worker_toolset_includes_image_and_script_tools():
    from minacode.tools.delegate import WORKER_TOOLS

    assert "ViewImage" in WORKER_TOOLS
    assert "ToolScript" in WORKER_TOOLS
    for excluded in ("Ask", "NextHints", "Delegate"):
        assert excluded not in WORKER_TOOLS


def test_worker_schemas_include_viewimage_and_toolscript(tmp_path):
    from minacode.tools.delegate import WORKER_TOOLS

    s = session(tmp_path)
    s.tool_names = WORKER_TOOLS
    resolved = [schema["function"]["name"] for schema in Tool.resolved_schemas(s)]
    assert "ViewImage" in resolved
    assert "ToolScript" in resolved
    for excluded in ("Ask", "NextHints", "Delegate"):
        assert excluded not in resolved

def test_system_prompt_stable_across_refactors():
    import hashlib

    from minacode.prompts import SYSTEM_PROMPT

    assert hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest() == "83db5525a584bba0efffcaafae3497289d7cc14abdaee608278f139baaa7ef92"
