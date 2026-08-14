"""`/compact log`: the compaction record, its persistence, and the two ways to review it."""

from minacode.base import LogBlock
from minacode.cli import QUEUE_SAFE_COMMANDS, CommandLoop
from minacode.cli.commands import compact
from minacode.cli.modals import compaction_log_viewer
from minacode.config import Config
from minacode.context import ContextManager
from minacode.engine import Agent
from minacode.session import HistorySegment, Session, SessionSnapshotCodec


def session(tmp_path):
    config = Config()
    config.data_dir = str(tmp_path / "data")
    return Session(cwd=str(tmp_path), config=config)


def loop(session):
    return CommandLoop(Agent(session, output_fn=lambda text: None), output_fn=lambda text: None)


def block_text(block) -> str:
    assert isinstance(block, LogBlock), block
    return "\n".join(f"{item.label} {item.text}" for item in block.items)


def store(session, **fields) -> HistorySegment:
    segment = HistorySegment(
        key=f"seg.{len(session.history) + 1}",
        title=fields.pop("title", "a title"),
        text=fields.pop("text", "user: do the thing\nassistant: done"),
        created_at=fields.pop("created_at", "2026-08-13T14:02:11+08:00"),
        scope=fields.pop("scope", "history"),
        trigger=fields.pop("trigger", "auto"),
        messages=fields.pop("messages", 12),
        summary=fields.pop("summary", "worked on the parser"),
        **fields,
    )
    session.history.append(segment)
    return segment


# --- the record -------------------------------------------------------------------------------


def test_compaction_records_what_it_evicted(tmp_path):
    s = session(tmp_path)
    context = ContextManager(s)
    compacted = [
        {"role": "user", "content": "find the parser bug"},
        {"role": "assistant", "content": "looking into it"},
    ]

    context.apply_compaction({"summary": "found the off-by-one"}, [], compacted=compacted)

    segment = s.history[0]
    assert segment.scope == "history"
    assert segment.trigger == "auto"
    assert segment.fallback is False
    assert segment.messages == 2
    assert segment.created_at  # local wall-clock stamp, format owned by local_timestamp
    # The summary this compaction produced, not the one it replaced.
    assert segment.summary == "found the off-by-one"


def test_turn_compaction_and_summarizer_failure_are_distinguishable(tmp_path):
    s = session(tmp_path)
    context = ContextManager(s)
    turn = [{"role": "user", "content": "keep"}]

    context.apply_compaction(
        None,
        [{"role": "user", "content": "keep"}],
        turn_messages=turn,
        compacted=[{"role": "assistant", "content": "evicted"}],
        fallback_note="prior context trimmed",
    )

    segment = s.history[0]
    assert segment.scope == "turn"  # the running turn, not prior conversation
    assert segment.fallback is True  # no summary data: deterministic trim, the lossiest case


def test_manual_compaction_is_recorded_as_manual(tmp_path):
    s = session(tmp_path)
    context = ContextManager(s)

    context.apply_compaction({"summary": "s"}, [], compacted=[{"role": "user", "content": "x"}], trigger="manual")

    assert s.history[0].trigger == "manual"


def test_earlier_summaries_survive_later_compactions(tmp_path):
    # The live checkpoint carries only the newest summary; the segments are what make the earlier
    # ones reviewable at all.
    s = session(tmp_path)
    context = ContextManager(s)

    context.apply_compaction({"summary": "first phase"}, [], compacted=[{"role": "user", "content": "one"}])
    context.apply_compaction({"summary": "second phase"}, [], compacted=[{"role": "user", "content": "two"}])

    assert [segment.summary for segment in s.history] == ["first phase", "second phase"]


# --- persistence ------------------------------------------------------------------------------


def test_segment_metadata_survives_a_snapshot_round_trip(tmp_path):
    s = session(tmp_path)
    original = store(s, fallback=True, scope="turn", trigger="manual")
    blobs: dict[str, str] = {}

    encoded = SessionSnapshotCodec.history_segment(original, blobs)
    restored = SessionSnapshotCodec.history([encoded], blobs)[0]

    assert restored == original


def test_a_snapshot_written_before_the_metadata_still_loads():
    legacy = {"key": "seg.1", "title": "older session", "blob": "b1"}

    restored = SessionSnapshotCodec.history([legacy], {"b1": "evicted text"})[0]

    assert restored.key == "seg.1"
    assert restored.text == "evicted text"
    assert (restored.created_at, restored.scope, restored.trigger, restored.messages) == ("", "", "", 0)
    assert restored.fallback is False


# --- headless review --------------------------------------------------------------------------


def test_compact_log_lists_stored_segments(tmp_path):
    s = session(tmp_path)
    s.state.compaction_count = 3
    store(s, title="first task", created_at="2026-08-13T13:12:00+08:00")
    store(s, title="second task", scope="turn", trigger="manual")

    text = block_text(compact(loop(s), "log"))

    assert "3 compactions · 2 stored segments" in text  # a pass with nothing to evict stores none
    assert text.index("seg.2") < text.index("seg.1")  # newest first, like RecallContext(list)
    assert "08-13 13:12" in text
    assert "manual · this turn" in text  # the reader's words, not the stored scope/trigger
    assert "first task" in text and "second task" in text


def test_compact_log_dumps_one_segment_in_full(tmp_path):
    s = session(tmp_path)
    store(s, text="user: line one\nassistant: line two", summary="what survived")

    text = block_text(compact(loop(s), "log seg.1"))

    assert "what survived" in text
    assert "line one" in text and "line two" in text  # every line, nothing folded or clipped


def test_compact_log_rejects_an_unknown_segment(tmp_path):
    s = session(tmp_path)
    store(s)

    assert compact(loop(s), "log seg.9") == "No stored segment seg.9"
    assert compact(loop(s), "log nonsense") == "Usage: /compact log [seg.N]"


def test_compact_log_says_so_when_nothing_is_stored(tmp_path):
    assert compact(loop(session(tmp_path)), "log") == "No compaction has stored a segment yet"


def test_compact_still_rejects_other_arguments(tmp_path):
    assert compact(loop(session(tmp_path)), "now") == "Usage: /compact [log [seg.N]]"


# --- the viewer -------------------------------------------------------------------------------


class Modal:
    """Captures what show_modal was given so a test can render and drive the viewer."""

    def __init__(self):
        self.fragments = None
        self.key = None
        self.exclusive = False

    def show_modal(self, fragments_fn, key_fn, **kwargs):
        self.fragments, self.key = fragments_fn, key_fn
        self.exclusive = kwargs.get("exclusive", False)

    def text(self) -> str:
        return "".join(fragment for _style, fragment in self.fragments())


def viewer(tmp_path, count=3):
    s = session(tmp_path)
    s.state.compaction_count = count
    for index in range(count):
        store(s, title=f"task {index + 1}", text=f"user: request {index + 1}\nassistant: reply", summary=f"summary {index + 1}")
    lp = loop(s)
    modal = Modal()
    lp.tui = modal
    compaction_log_viewer(lp)
    return modal


def test_viewer_lists_segments_newest_first(tmp_path):
    modal = viewer(tmp_path)

    text = modal.text()
    assert modal.exclusive is True  # full-screen, like the diff viewer
    assert "3 compactions · 3 stored segments" in text
    assert text.index("seg.3") < text.index("seg.1")
    assert "task 3" in text
    assert "[list]" in text and "[1/3]" in text


def test_viewer_opens_a_segment_and_scrolls_back_to_the_list(tmp_path):
    modal = viewer(tmp_path)

    modal.key("down", "")  # select seg.2
    modal.key("enter", "")
    detail = modal.text()
    assert "[detail]" in detail
    assert "summary 2" in detail
    assert "request 2" in detail  # the verbatim excerpt, not just the summary

    modal.key("escape", "")
    assert "[list]" in modal.text()
    assert "summary 2" not in modal.text()


def open_first(tmp_path, **fields):
    s = session(tmp_path)
    s.state.compaction_count = 1
    store(s, **fields)
    lp = loop(s)
    modal = Modal()
    lp.tui = modal
    compaction_log_viewer(lp)
    modal.key("enter", "")
    return modal.text()


def test_viewer_detail_says_what_happened_in_plain_words(tmp_path):
    text = open_first(tmp_path, messages=96, summary="reviewed the approval flow")

    assert "Compacted automatically, dropping earlier conversation · 96 messages" in text
    assert "What the agent kept" in text
    assert "The conversation it replaced" in text
    # The stored words are for the code, not the reader.
    for jargon in ("evicted", "verbatim", "scope", "trigger"):
        assert jargon not in text


def test_viewer_detail_warns_when_the_summarizer_failed(tmp_path):
    text = open_first(tmp_path, fallback=True, summary="")

    assert "Summarizing failed" in text
    assert "the excerpt below is all that was kept" in text
    assert "(none recorded)" in text


def test_viewer_detail_explains_a_segment_older_than_the_log(tmp_path):
    text = open_first(tmp_path, created_at="", scope="", trigger="", messages=0, summary="")

    # Missing detail is the record's age, not a failure of this compaction.
    assert "Compacted before minacode kept these details" in text
    assert "predates the log" in text


def test_viewer_detail_only_mentions_a_missing_middle_when_there_is_one(tmp_path):
    whole = open_first(tmp_path, text="user: short request")
    assert "saved as written" in whole
    assert "too long to keep whole" not in whole

    bounded = open_first(tmp_path, text='head\n<bounded_output omitted="middle" max_tokens="8000" />\ntail')
    assert "too long to keep whole, the middle is marked below" in bounded


def test_viewer_closes_from_the_list(tmp_path):
    modal = viewer(tmp_path)

    assert modal.key("escape", "") is None
    assert modal.key("q", "") is None


def test_viewer_detail_keeps_the_evicted_text_line_for_line(tmp_path):
    s = session(tmp_path)
    s.state.compaction_count = 1
    store(s, text="Touch these files:\nminacode/loop.py\n<Config> stays\n\n    indented line")
    lp = loop(s)
    modal = Modal()
    lp.tui = modal
    compaction_log_viewer(lp)

    modal.key("enter", "")
    rows = [row.strip() for row in modal.text().splitlines()]
    for line in ("Touch these files:", "minacode/loop.py", "<Config> stays", "indented line"):
        assert line in rows, f"{line!r} did not survive the viewer"


def test_viewer_reports_an_empty_store(tmp_path):
    s = session(tmp_path)
    lp = loop(s)
    modal = Modal()
    lp.tui = modal
    compaction_log_viewer(lp)

    assert "No compaction has stored a segment yet" in modal.text()
    assert modal.key("enter", "") is not None  # nothing to open, and it does not crash
    assert modal.key("q", "") is None


# --- discoverability --------------------------------------------------------------------------


def test_compact_log_is_documented_and_not_queue_safe():
    assert "/compact log" in CommandLoop.HELP
    # Reading the log is harmless, but /compact itself rewrites context: the registry is per
    # command, so the whole command stays out of the mid-turn allowlist.
    assert "/compact" not in QUEUE_SAFE_COMMANDS
