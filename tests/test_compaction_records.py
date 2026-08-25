"""compaction records (split from tests/test_context.py)."""


class _StubModel:
    """Compactor requires a model; planning-only tests never touch it."""


from agent_harness import session

from minacode import compaction
from minacode.context import ContextManager
from minacode.tools import EditTool, ReadTool


def test_compaction_keeps_assistant_with_tool_results(tmp_path):
    context = ContextManager(session(tmp_path))
    messages = [
        *({"role": "user", "content": f"old {index}"} for index in range(3)),
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc.1", "type": "function", "function": {"name": "Read", "arguments": "{}"}},
                {"id": "tc.2", "type": "function", "function": {"name": "Read", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "tc.1", "content": "one"},
        {"role": "tool", "tool_call_id": "tc.2", "content": "two"},
        *({"role": "user", "content": f"recent {index}"} for index in range(6)),
    ]

    compacted, keep = compaction.Compactor(context, _StubModel()).parts_for(messages)

    assert compacted == messages[:3]
    assert keep == messages[3:]
    assert keep[0]["role"] == "assistant"
    assert [message["role"] for message in keep[1:3]] == ["tool", "tool"]


def test_compaction_keeps_tool_records_referenced_from_summary(tmp_path):
    s = session(tmp_path)
    context = ContextManager(s)
    kept = s.store_tool_result("Bash", ["kept"], "kept output")
    dropped = s.store_tool_result("Bash", ["dropped"], "dropped output")

    context.apply_compaction({"summary": f"Continue from {kept}."}, [])

    assert kept in s.tool_results
    assert dropped not in s.tool_results
    assert [record.key for record in s.tool_records] == [kept]


def test_compaction_prunes_unreferenced_tool_records(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("one\n", encoding="utf-8")
    s = session(tmp_path)
    context = ContextManager(s)
    old_key = s.store_tool_result("Bash", ["old"], "old output")
    read_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 0]]}], ReadTool(s, [{"path": "a.txt", "ranges": [[0, 0]]}]).call())
    current_key = s.store_tool_result("Bash", ["current"], "current output")

    context.apply_compaction({"summary": "summary"}, [{"role": "tool", "content": f"tool {current_key} Bash current"}])

    assert old_key not in s.tool_results
    assert read_key not in s.tool_results
    assert {record.key for record in s.tool_records} == {current_key}
    assert set(s.tool_results) == {current_key}


def test_compaction_keeps_current_turn_tool_records(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    s.messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "old answer"}, {"role": "user", "content": "latest"}]
    old_key = s.store_tool_result("Bash", ["old"], "old output")
    current_key = s.store_tool_result("Bash", ["current"], "current output")

    class FakeModel:
        last_compaction_model = ""
        def compact(self, text, *_args, **_kwargs):
            return {"summary": "summary"}

    ContextManager(s).prepare_messages(FakeModel(), "system", [{"role": "tool", "content": f"tool {current_key} Bash current"}])

    assert old_key not in s.tool_results
    assert current_key in s.tool_results
    assert [record.key for record in s.tool_records] == [current_key]


def test_compaction_drops_unreferenced_read_edit_records(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    s = session(tmp_path)
    context = ContextManager(s)
    read_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 0]]}], ReadTool(s, [{"path": "a.txt", "ranges": [[0, 0]]}]).call())
    edit_key = s.store_tool_result(
        "Edit",
        ["a.txt"],
        EditTool(s, ["a.txt", [{"op": "delete", "start": "1:" + ReadTool.line_hash("b\n"), "end": "1:" + ReadTool.line_hash("b\n")}]]).call(),
    )

    context.apply_compaction({"summary": "summary"}, [])

    assert read_key not in s.tool_results
    assert edit_key not in s.tool_results
    assert s.tool_records == []
