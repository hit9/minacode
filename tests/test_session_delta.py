"""session delta (split from tests/test_session_persistence.py)."""
import os

from test_session_persistence import log_path, project_dir, read_jsonl, session_with_data_dir

from wizolt.base import SESSION_EVENT_KEY
from wizolt.context import ContextManager
from wizolt.session import Session


async def test_latest_pointer_created_on_first_save(tmp_path):
    """First save creates the latest pointer file."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    await s.save_snapshot()

    latest_path = os.path.join(project_dir(s), "latest")
    assert os.path.exists(latest_path)
    with open(latest_path) as file:
        assert file.read().strip() == s.uid

async def test_second_save_writes_delta_with_only_new_data(tmp_path):
    """Second save appends a delta line containing only new messages and tool records."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "first"})
    s.store_tool_result("Read", ["a.py"], "# a")
    await s.save_snapshot()  # init

    s.messages.append({"role": "assistant", "content": "reply"})
    s.store_tool_result("Search", ["pat"], "result")
    await s.save_snapshot()  # delta

    lines = read_jsonl(log_path(s))
    assert len(lines) == 2
    delta = lines[1]
    # Only new data in delta
    assert delta["messages"] == [{"role": "assistant", "content": "reply"}]
    assert [record["key"] for record in delta["tool_records"]] == ["tr.2"]
    assert delta["tool_records"][0]["output"] == "result"
    assert delta["tool_counter"] == 2

async def test_transcript_appends_when_model_messages_are_replaced(tmp_path):
    s = session_with_data_dir(tmp_path)
    first = {"role": "user", "content": "original request"}
    s.messages.append(first)
    s.transcript_messages.append(first)
    await s.save_snapshot()

    s.messages[:] = [{"role": "user", "content": "compacted context", SESSION_EVENT_KEY: "compaction_checkpoint"}]
    s.transcript_messages.append({"role": "assistant", "content": "original answer"})
    await s.save_snapshot()

    delta = read_jsonl(log_path(s))[1]
    assert delta["messages_replace"] == s.messages
    assert delta["transcript_messages"] == [{"role": "assistant", "content": "original answer"}]

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
    assert restored.messages[0]["content"] == "compacted context"
    assert [message["content"] for message in restored.transcript_messages] == ["original request", "original answer"]

async def test_transcript_tool_metadata_does_not_duplicate_retained_output_or_file_snapshots(tmp_path):
    s = session_with_data_dir(tmp_path)
    key = s.store_tool_result("Edit", ["x.py"], "full retained output")
    s.store_turn_diff(key, 1, "x.py", "-old\n+new\n", before="old\n", after="new\n")
    await s.save_snapshot()

    snapshot = next(line for line in read_jsonl(log_path(s)) if "uid" in line)
    assert "transcript_tool_records" not in snapshot
    assert snapshot["transcript_turn_diffs"] == [{"key": key, "turn": 1, "path": "x.py", "diff": "-old\n+new\n", "round": 0}]
    assert "before_blob" not in snapshot["transcript_turn_diffs"][0]

async def test_materialized_tool_output_survives_the_asset_collector(tmp_path):
    """The reported failure: the asset collector kept only image refs, so the file a truncated tool
    result was materialized to was deleted on the very next save -- while the marker in the
    conversation went on advertising its path, sending the model hunting for a file that was gone."""
    s = session_with_data_dir(tmp_path)
    large = "\n".join(f"line {index}" for index in range(20000))
    key = s.store_tool_result("Bash", ["big"], large)
    marker = ContextManager(s).bound_output(large, key, path=await ContextManager(s).materialize_output(key, large))
    path = os.path.join(s.images.assets_dir(), key + ".txt")
    assert f'file="{path}"' in marker

    s.messages.append({"role": "user", "content": marker})
    await s.save_snapshot()
    await s.save_snapshot()  # the collector runs on every save, not only the first

    with open(path, encoding="utf-8") as file:
        assert file.read() == large

def test_owns_asset_covers_the_assets_directory_and_nothing_else(tmp_path):
    """The predicate that waives the out-of-workspace approval, so its edges are a permission
    boundary: a sibling whose name merely starts the same way is not inside it, and neither is a
    parent reached back through it."""
    s = session_with_data_dir(tmp_path)
    assets = s.images.assets_dir()
    os.makedirs(assets, exist_ok=True)
    parent = os.path.dirname(assets)

    assert s.owns_asset(os.path.join(assets, "tr.1.txt")) is True
    assert s.owns_asset(os.path.join(assets, "nested", "deep.txt")) is True
    assert s.owns_asset(assets) is True
    assert s.owns_asset(assets + "-sibling/secret.txt") is False  # prefix match is not containment
    assert s.owns_asset(os.path.join(assets, "..", "other.jsonl")) is False  # normalized, not textual
    assert s.owns_asset(parent) is False
    assert s.owns_asset(str(tmp_path / "elsewhere.txt")) is False

async def test_materialized_tool_output_expires_with_its_tool_result(tmp_path):
    """It is retained by its result, not forever: once the result is pruned the file is collected,
    so a long session cannot accumulate an asset for every large output it ever saw."""
    s = session_with_data_dir(tmp_path)
    large = "\n".join(f"line {index}" for index in range(20000))
    key = s.store_tool_result("Bash", ["big"], large)
    await ContextManager(s).materialize_output(key, large)
    path = os.path.join(s.images.assets_dir(), key + ".txt")
    assert os.path.exists(path)

    s.tool_results.pop(key)
    await s.save_snapshot()

    assert not os.path.exists(path)
