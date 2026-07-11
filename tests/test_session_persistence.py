import json
import os
import time

import pytest

import nanocode as n


def session_with_data_dir(tmp_path):
    """Session targeting tmp_path as data_dir (avoids touching ~/.nanocode)."""
    return n.Session(
        cwd=str(tmp_path),
        config=n.Config(data_dir=str(tmp_path)),
    )


def read_jsonl(path) -> list[dict]:
    """Read all JSON lines from a JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_first_save_writes_init_line(tmp_path):
    """First save writes a single init line with full snapshot data."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.store_tool_result("Read", ["foo.py"], "# content")
    s.save_snapshot()

    lines = read_jsonl(tmp_path / "sessions" / f"{s.uid}.jsonl")
    assert len(lines) == 1
    init = lines[0]
    assert init["uid"] == s.uid
    assert init["messages"] == [{"role": "user", "content": "hello"}]
    assert init["tool_counter"] == 1
    assert init["tool_records"][0]["output"] == "# content"
    assert "usage" in init
    assert "state" in init
    # Runtime/config and derivable data are NOT stored in the snapshot
    assert "config" not in init
    assert "settings" not in init
    assert "tool_results" not in init

def test_latest_pointer_created_on_first_save(tmp_path):
    """First save creates the latest pointer file."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()

    latest_path = tmp_path / "latest"
    assert latest_path.exists()
    assert latest_path.read_text().strip() == s.uid


def test_second_save_writes_delta_with_only_new_data(tmp_path):
    """Second save appends a delta line containing only new messages and tool records."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "first"})
    s.store_tool_result("Read", ["a.py"], "# a")
    s.save_snapshot()  # init

    s.messages.append({"role": "assistant", "content": "reply"})
    s.store_tool_result("Search", ["pat"], "result")
    s.save_snapshot()  # delta

    lines = read_jsonl(tmp_path / "sessions" / f"{s.uid}.jsonl")
    assert len(lines) == 2
    delta = lines[1]
    # Only new data in delta
    assert delta["messages"] == [{"role": "assistant", "content": "reply"}]
    assert [record["key"] for record in delta["tool_records"]] == ["tr.2"]
    assert delta["tool_records"][0]["output"] == "result"
    assert delta["tool_counter"] == 2


def test_delta_omits_messages_when_nothing_new(tmp_path):
    """Delta line omits the messages key when no new messages."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hi"})
    s.save_snapshot()  # init

    # No new messages
    s.save_snapshot()  # delta

    lines = read_jsonl(tmp_path / "sessions" / f"{s.uid}.jsonl")
    delta = lines[1]
    assert "messages" not in delta


def test_delta_omits_tool_records_when_nothing_new(tmp_path):
    """Delta line omits tool_records when no new tool calls."""
    s = session_with_data_dir(tmp_path)
    s.store_tool_result("Bash", ["pwd"], "/home")
    s.save_snapshot()  # init

    s.messages.append({"role": "user", "content": "more"})
    s.save_snapshot()  # delta

    lines = read_jsonl(tmp_path / "sessions" / f"{s.uid}.jsonl")
    delta = lines[1]
    assert "messages" in delta
    assert "tool_records" not in delta  # No new tool calls since init
    assert "tool_results" not in delta


def test_delta_omits_unchanged_turn_diffs_without_serializing_payload(tmp_path, monkeypatch):
    s = session_with_data_dir(tmp_path)
    s.store_turn_diff("tr.1", 1, "large.py", "-old\n+new\n", before="old\n" * 1000, after="new\n" * 1000, round=1)
    s.save_snapshot()  # init

    def fail_turn_diff(_diff):
        raise AssertionError("unchanged turn diffs should not be serialized")

    monkeypatch.setattr(n.SessionSnapshotCodec, "turn_diff", fail_turn_diff)
    s.messages.append({"role": "user", "content": "next"})
    s.save_snapshot()  # delta

    lines = read_jsonl(tmp_path / "sessions" / f"{s.uid}.jsonl")
    assert "turn_diffs" not in lines[1]
    assert "turn_diffs_replace" not in lines[1]


def test_load_merges_init_and_deltas(tmp_path):
    """load_snapshot reads and merges all lines, returning the full session state."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "q1"})
    s.store_tool_result("Read", ["f.py"], "# f")
    s.save_snapshot()  # init

    s.messages.append({"role": "assistant", "content": "a1"})
    s.store_tool_result("Search", ["pat"], "found")
    s.save_snapshot()  # delta

    s.messages.append({"role": "user", "content": "q2"})
    s.save_snapshot()  # delta (no new tool results)

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
    # All messages across all lines
    assert [m["content"] for m in s2.messages[:3]] == ["q1", "a1", "q2"]
    # Fourth message is resume marker
    assert s2.messages[3]["content"].startswith("[Session resumed:")
    # All tool results
    assert s2.tool_results["tr.1"] == "# f"
    assert s2.tool_results["tr.2"] == "found"
    assert s2.tool_counter == 2


def test_load_preserves_uid(tmp_path):
    """load_snapshot preserves the original uid."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
    assert s2.uid == s.uid


def test_load_with_latest_alias(tmp_path):
    """load_snapshot with uid='latest' resolves from the latest pointer file."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()

    s2 = n.Session.load_snapshot("latest", config=s.config)
    assert s2.uid == s.uid


def test_load_with_last_alias(tmp_path):
    """load_snapshot with uid='last' resolves from the latest pointer file."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()

    s2 = n.Session.load_snapshot("last", config=s.config)
    assert s2.uid == s.uid


def test_load_appends_resume_marker(tmp_path):
    """After load, the session has a resume marker message at the end."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
    assert len(s2.messages) == 2  # hello + resume marker
    assert s2.messages[-1]["content"].startswith(f"[Session resumed: uid={s.uid}]")


def test_save_after_load_produces_a_delta(tmp_path):
    """Save after load appends a delta (not re-init)."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()  # init (line 1)

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
    # s2 now has messages = [hello, resume_marker]
    s2.messages.append({"role": "assistant", "content": "post-resume"})
    s2.save_snapshot()  # delta (line 2)

    lines = read_jsonl(tmp_path / "sessions" / f"{s.uid}.jsonl")
    assert len(lines) == 2
    delta = lines[1]
    # The delta should contain the post-resume message, NOT the resume marker
    # (resume marker was already in s2 when _snapshot_saved was set by load)
    assert delta["messages"] == [{"role": "assistant", "content": "post-resume"}]


def test_repeated_resume_preserves_history(tmp_path):
    """Repeated resume/save cycles keep appending new messages to the same history."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "m1"})
    s.save_snapshot()

    expected = ["m1"]
    for role, content in (("assistant", "a1"), ("user", "m2"), ("assistant", "a2")):
        s = n.Session.load_snapshot(s.uid, config=s.config)
        assert [m["content"] for m in n.SessionSnapshotCodec.persistable_messages(s.messages)] == expected
        s.messages.append({"role": role, "content": content})
        s.save_snapshot()
        expected.append(content)

    loaded = n.Session.load_snapshot(s.uid, config=s.config)
    assert [m["content"] for m in n.SessionSnapshotCodec.persistable_messages(loaded.messages)] == expected


def test_resume_marker_is_never_persisted(tmp_path):
    """Resume markers are runtime-only, including when a message rewrite forces replace."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "m1"})
    s.save_snapshot()

    resumed = n.Session.load_snapshot(s.uid, config=s.config)
    resumed.messages = [
        {"role": "system", "content": f"[Session resumed: uid={s.uid}]"},
        {"role": "user", "content": "rewritten"},
    ]
    resumed.save_snapshot()

    lines = read_jsonl(tmp_path / "sessions" / f"{s.uid}.jsonl")
    assert lines[-1]["messages_replace"] == [{"role": "user", "content": "rewritten"}]
    assert "[Session resumed:" not in json.dumps(lines)


def test_load_discards_persisted_resume_markers(tmp_path):
    """Older or malformed snapshots may contain resume markers; load should not keep them."""
    s = session_with_data_dir(tmp_path)
    path = tmp_path / "sessions" / f"{s.uid}.jsonl"
    path.parent.mkdir(parents=True)
    marker = {"role": "system", "content": f"[Session resumed: uid={s.uid}]"}
    n.SessionSnapshotStore.write_jsonl(
        str(path),
        {
            "uid": s.uid,
            "cwd": str(tmp_path),
            "messages": [{"role": "user", "content": "m1"}, marker, {"role": "assistant", "content": "a1"}],
        },
        mode="w",
    )

    loaded = n.Session.load_snapshot(s.uid, config=s.config)

    assert [m["content"] for m in n.SessionSnapshotCodec.persistable_messages(loaded.messages)] == ["m1", "a1"]
    assert sum(1 for m in loaded.messages if n.SessionSnapshotCodec.is_internal_message(m)) == 1


def test_empty_session_first_save_is_skipped(tmp_path):
    """A session with no recoverable content is not persisted."""
    s = session_with_data_dir(tmp_path)
    assert s.save_snapshot() == ""

    assert not (tmp_path / "latest").exists()
    assert not (tmp_path / "sessions" / f"{s.uid}.jsonl").exists()


def test_tool_results_roundtrip(tmp_path):
    """Tool results survive save/load."""
    s = session_with_data_dir(tmp_path)
    s.store_tool_result("Bash", ["echo hi"], "hi")
    s.store_tool_result("Read", ["f.py"], "code")
    s.save_snapshot()

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
    assert s2.tool_results["tr.1"] == "hi"
    assert s2.tool_results["tr.2"] == "code"


def test_tool_records_roundtrip(tmp_path):
    """Tool records survive save/load."""
    s = session_with_data_dir(tmp_path)
    s.store_tool_result("Bash", ["pwd"], "/tmp")
    s.store_tool_result("Search", ["x"], "match")
    s.save_snapshot()

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
    assert len(s2.tool_records) == 2
    assert s2.tool_records[0].key == "tr.1"
    assert s2.tool_records[0].name == "Bash"
    assert s2.tool_records[0].output == "/tmp"
    assert s2.tool_records[1].key == "tr.2"
    assert s2.tool_records[1].name == "Search"


def test_tool_errors_roundtrip(tmp_path):
    """Tool errors survive save/load."""
    s = session_with_data_dir(tmp_path)
    s.record_tool_error("tr.1", "Bash", ["bad"], "command not found")
    s.save_snapshot()

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
    assert len(s2.tool_errors) == 1
    assert s2.tool_errors[0].key == "tr.1"
    assert s2.tool_errors[0].error == "command not found"


def test_usage_roundtrip_with_prompt_and_completion_tokens(tmp_path):
    """All usage fields (including prompt_tokens/completion_tokens) survive save/load."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.usage.calls = 3
    s.usage.prompt_tokens = 100
    s.usage.completion_tokens = 50
    s.usage.total_tokens = 150
    s.usage.cached_prompt_tokens = 20
    s.usage.last_cached_prompt_tokens = 5
    s.usage.last_total_tokens = 60
    s.save_snapshot()

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
    assert s2.usage.calls == 3
    assert s2.usage.prompt_tokens == 100
    assert s2.usage.completion_tokens == 50
    assert s2.usage.total_tokens == 150
    assert s2.usage.cached_prompt_tokens == 20
    assert s2.usage.last_cached_prompt_tokens == 5
    assert s2.usage.last_total_tokens == 60


def test_agent_state_roundtrip(tmp_path):
    """Agent state (goal, plan, known, check, summary) survives save/load."""
    s = session_with_data_dir(tmp_path)
    s.state.goal = "fix bug"
    s.state.plan = ["step 1", "step 2"]
    s.state.known = ["file at src/a.py"]
    s.state.check = "assert x == 1"
    s.state.summary = "working on it"
    s.state.round_count = 7
    s.save_snapshot()

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
    assert s2.state.goal == "fix bug"
    assert [item.to_json() for item in s2.state.plan] == [{"status": "todo", "text": "step 1"}, {"status": "todo", "text": "step 2"}]
    assert s2.state.known == ["file at src/a.py"]
    assert s2.state.check == "assert x == 1"
    assert s2.state.summary == "working on it"
    assert s2.state.round_count == 7

def test_multiple_deltas_accumulate_correctly(tmp_path):
    """Multiple delta saves accumulate data correctly."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "m1"})
    s.save_snapshot()  # init
    s.messages.append({"role": "assistant", "content": "a1"})
    s.save_snapshot()  # delta 1
    s.messages.append({"role": "user", "content": "m2"})
    s.save_snapshot()  # delta 2
    s.messages.append({"role": "assistant", "content": "a2"})
    s.save_snapshot()  # delta 3

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
    contents = [m["content"] for m in s2.messages if not m["content"].startswith("[Session resumed:")]
    assert contents == ["m1", "a1", "m2", "a2"]


def test_multiple_deltas_with_tool_calls(tmp_path):
    """Tool calls across multiple deltas accumulate correctly."""
    s = session_with_data_dir(tmp_path)
    s.store_tool_result("Read", ["a.py"], "# a")
    s.save_snapshot()  # init: tr.1
    s.store_tool_result("Search", ["pat"], "hit")
    s.save_snapshot()  # delta 1: tr.2
    s.store_tool_result("Bash", ["pwd"], "/tmp")
    s.save_snapshot()  # delta 2: tr.3

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
    assert s2.tool_results["tr.1"] == "# a"
    assert s2.tool_results["tr.2"] == "hit"
    assert s2.tool_results["tr.3"] == "/tmp"
    assert s2.tool_counter == 3
    assert len(s2.tool_records) == 3


def test_load_missing_snapshot_raises_error(tmp_path):
    """Loading a non-existent session raises NanocodeError."""
    with pytest.raises(n.NanocodeError, match="Session snapshot not found"):
        n.Session.load_snapshot("nonexistent-uid", config=n.Config(data_dir=str(tmp_path)))


def test_resolve_uid_missing_latest_file(tmp_path):
    """Resolving 'latest' when no latest file exists raises NanocodeError."""
    with pytest.raises(n.NanocodeError, match="No latest session to resume"):
        n.SessionSnapshotStore.resolve_uid("latest", data_dir=str(tmp_path))


def test_resolve_uid_missing_last_file(tmp_path):
    """Resolving 'last' when no latest file exists raises NanocodeError."""
    with pytest.raises(n.NanocodeError, match="No latest session to resume"):
        n.SessionSnapshotStore.resolve_uid("last", data_dir=str(tmp_path))


def test_resolve_uid_passthrough_normal_uid(tmp_path):
    """Resolving a normal uid (not 'latest') returns it as-is."""
    result = n.SessionSnapshotStore.resolve_uid("my-uid", data_dir=str(tmp_path))
    assert result == "my-uid"


def test_jsonl_file_is_append_only(tmp_path):
    """Multiple saves only add lines, never rewrite the file."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()  # l1
    s.save_snapshot()  # l2
    s.save_snapshot()  # l3
    s.save_snapshot()  # l4

    lines = read_jsonl(tmp_path / "sessions" / f"{s.uid}.jsonl")
    assert len(lines) == 4
    # First line has all fields (init)
    assert "uid" in lines[0]
    assert "messages" not in lines[1]
    assert "tool_records" not in lines[2]
    assert "tool_results" not in lines[3]


def test_runtime_session_retention_defaults_to_seven_days():
    settings = n.RuntimeSettings.from_dict({})

    assert settings.session_retention_days == 7


def test_clean_expired_sessions_removes_old_files_and_latest(tmp_path):
    s = session_with_data_dir(tmp_path)
    old = session_with_data_dir(tmp_path)
    old.messages.append({"role": "user", "content": "old"})
    old.save_snapshot()
    old_path = tmp_path / "sessions" / f"{old.uid}.jsonl"
    stale_time = time.time() - 8 * 86400
    os.utime(old_path, (stale_time, stale_time))

    assert n.SessionSnapshotStore.clean_expired(s) == 1

    assert not old_path.exists()
    assert not (tmp_path / "latest").exists()


def test_clean_expired_sessions_skips_current_session(tmp_path):
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "current"})
    s.save_snapshot()
    path = tmp_path / "sessions" / f"{s.uid}.jsonl"
    stale_time = time.time() - 8 * 86400
    os.utime(path, (stale_time, stale_time))

    assert n.SessionSnapshotStore.clean_expired(s) == 0

    assert path.exists()


# ---------------------------------------------------------------------------
# Transcript replay resilience (regression: multi-line tool arguments such as
# a Bash command with embedded newlines must not crash --resume rendering)
# ---------------------------------------------------------------------------

def _bash_raw_call(arguments: str) -> dict:
    return {"id": "c1", "type": "function", "function": {"name": "Bash", "arguments": arguments}}


def test_transcript_tool_call_parses_multiline_arguments():
    """Argument strings with literal newlines (invalid strict JSON) still parse, so the
    Bash command survives instead of being dropped to {}."""
    raw = _bash_raw_call("{\"command\": \"printf 'line one\nline two'\"}")
    call = n.CommandLoop.transcript_tool_call(raw)
    assert call is not None
    assert call.args == ["printf 'line one\nline two'"]


def test_transcript_tool_call_does_not_crash_on_unparseable_args():
    """A historical Bash call whose payload fails validation must render, not raise."""
    raw = _bash_raw_call("{not valid json at all")
    call = n.CommandLoop.transcript_tool_call(raw)  # must not raise ToolError
    assert call is not None
    assert call.name == "Bash"


def test_chat_tool_calls_parse_multiline_commit_message():
    """The live chat path recovers args from a multi-line Bash command too."""
    class _Fn:
        name = "Bash"
        arguments = "{\"command\": \"printf 'subject\n\nbody line'\"}"
    class _Raw:
        id = "x1"
        function = _Fn()
    class _Msg:
        tool_calls = [_Raw()]
    s = n.Session(cwd="/tmp")
    calls = n.ModelClient(s).tool_calls(_Msg())
    assert len(calls) == 1
    assert calls[0].args == ["printf 'subject\n\nbody line'"]


def test_snapshot_messages_strips_non_persistable_roles(tmp_path):
    s = n.Session(cwd=str(tmp_path))
    s.messages = [
        {"role": "system", "content": "[Session resumed: old-session-id]"},
        {"role": "user", "content": "hello"},
    ]
    messages = n.SessionSnapshotCodec.snapshot_messages(s)
    # Internal resume marker is stripped; user message is kept.
    roles = [m["role"] for m in messages]
    assert "system" not in roles
    assert "user" in roles
    assert len(messages) == 1
