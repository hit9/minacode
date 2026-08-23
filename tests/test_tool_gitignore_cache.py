"""tool gitignore cache (split from tests/test_tools.py)."""
import json
import os
import shutil
import code_symbol_index as csi
import pytest
import minacode
from minacode.base import (
    LogBlock,
    LogEdge,
    LogLine,
    LogRole,
    ToolCall,
    ToolError,
)
from minacode.config import (
    Config,
    ConfigFile,
    RuntimeSettings,
)
from minacode.context import ContextManager
from minacode.model import ModelClient
from minacode.render import UiPrinter
from minacode.runner import ToolRunner
from minacode.session import HistorySegment, Session, SessionSnapshotCodec
from minacode.tools import (
    TOOL_REGISTRY,
    TOOLS,
    AskTool,
    BashTool,
    CodeIndex,
    EditTool,
    InspectCodeTool,
    MCPTool,
    NextHintsTool,
    NoteTool,
    ReadTool,
    RecallContextTool,
    RecallTool,
    SearchTool,
    SkillTool,
    Tool,
    ViewImageTool,
)
from test_tools import session

def test_gitignore_cache_cleanup_on_file_delete(tmp_path):
    """Cache entry is removed when .gitignore is deleted."""
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("delete_me.txt\n", encoding="utf-8")
    s = session(tmp_path)
    tool = SearchTool(s, [{"pattern": "x"}])

    tool.gitignore_patterns(str(tmp_path))
    ws_gitignore = str(gitignore)
    assert ws_gitignore in s._gitignore_cache

    # Delete the .gitignore file
    gitignore.unlink()
    patterns = tool.gitignore_patterns(str(tmp_path))
    assert patterns == []
    assert ws_gitignore not in s._gitignore_cache

def test_gitignore_cache_invalidates_on_file_change(tmp_path):
    """Cache re-reads .gitignore when mtime changes."""
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("old.txt\n", encoding="utf-8")
    s = session(tmp_path)
    tool = SearchTool(s, [{"pattern": "x"}])

    patterns1 = tool.gitignore_patterns(str(tmp_path))
    assert patterns1 == ["old.txt"]

    ws_gitignore = str(gitignore)
    old_mtime = s._gitignore_cache[ws_gitignore][0]

    # Modify the .gitignore file
    gitignore.write_text("new.txt\n", encoding="utf-8")
    patterns2 = tool.gitignore_patterns(str(tmp_path))
    assert patterns2 == ["new.txt"]

    new_mtime = s._gitignore_cache[ws_gitignore][0]
    assert new_mtime != old_mtime

def test_gitignore_cache_keyed_by_root(tmp_path):
    """Different root directories cache independently."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / ".gitignore").write_text("root_ignored.txt\n", encoding="utf-8")
    (sub / ".gitignore").write_text("sub_ignored.txt\n", encoding="utf-8")

    s = session(tmp_path)
    tool = SearchTool(s, [{"pattern": "x"}])

    # Root patterns include only workspace .gitignore
    root_patterns = tool.gitignore_patterns(str(tmp_path))
    assert "root_ignored.txt" in root_patterns
    assert "sub_ignored.txt" not in root_patterns

    # Sub patterns include workspace + sub .gitignore
    sub_patterns = tool.gitignore_patterns(str(sub))
    assert "root_ignored.txt" in sub_patterns  # workspace always included
    assert "sub_ignored.txt" in sub_patterns

    # Two separate cache entries
    assert len(s._gitignore_cache) == 2

def test_gitignore_cache_noop_when_no_gitignore(tmp_path):
    """When no .gitignore exists, returns empty list and cache stays empty."""
    s = session(tmp_path)
    tool = SearchTool(s, [{"pattern": "x"}])

    patterns = tool.gitignore_patterns(str(tmp_path))
    assert patterns == []
    assert len(s._gitignore_cache) == 0

def test_gitignore_cache_populated_and_reused(tmp_path):
    """Cache stores parsed patterns and reuses them on subsequent calls."""
    (tmp_path / ".gitignore").write_text("ignored.txt\nbuild/\n", encoding="utf-8")
    s = session(tmp_path)
    tool = SearchTool(s, [{"pattern": "x"}])

    # First call populates the cache
    patterns1 = tool.gitignore_patterns(str(tmp_path))
    assert "ignored.txt" in patterns1
    assert "build/" in patterns1

    # Cache should exist for the workspace .gitignore
    ws_gitignore = str(tmp_path / ".gitignore")
    assert ws_gitignore in s._gitignore_cache
    cached_mtime, cached_patterns = s._gitignore_cache[ws_gitignore]
    assert cached_patterns == patterns1

    # Second call reuses cache (mtime unchanged)
    patterns2 = tool.gitignore_patterns(str(tmp_path))
    assert patterns2 == patterns1
    # Cache entry unchanged
    assert s._gitignore_cache[ws_gitignore][0] == cached_mtime

def test_gitignore_cache_preserves_order(tmp_path):
    """After a no-op stat (no change), patterns come from cache unchanged."""
    (tmp_path / ".gitignore").write_text("a.txt\nb.txt\n", encoding="utf-8")
    s = session(tmp_path)
    tool = SearchTool(s, [{"pattern": "x"}])

    p1 = tool.gitignore_patterns(str(tmp_path))
    p2 = tool.gitignore_patterns(str(tmp_path))

    # Same object identity isn't required, but content must match
    assert p1 == p2 == ["a.txt", "b.txt"]

def test_gitignore_cache_shared_across_tools(tmp_path):
    """SearchTool instances share the same gitignore cache via Session."""
    (tmp_path / ".gitignore").write_text("secret.log\n", encoding="utf-8")
    s = session(tmp_path)

    find = SearchTool(s, [{"pattern": "x"}])
    search = SearchTool(s, [{"pattern": "needle", "path": "."}])

    # Find populates the cache
    find_patterns = find.gitignore_patterns(str(tmp_path))
    assert find_patterns == ["secret.log"]

    # Search reuses the same cache entry
    search_patterns = search.gitignore_patterns(str(tmp_path))
    assert search_patterns == find_patterns

    ws_key = str(tmp_path / ".gitignore")
    assert ws_key in s._gitignore_cache
    # Only one cache entry, not duplicated
    assert len(s._gitignore_cache) == 1

def test_gitignore_line_filtering_unchanged(tmp_path):
    """Cache still filters blank lines, comments, and negation patterns."""
    (tmp_path / ".gitignore").write_text("keep.txt\n\n  # comment\n!negated.txt\n  \n", encoding="utf-8")
    s = session(tmp_path)
    tool = SearchTool(s, [{"pattern": "x"}])

    patterns = tool.gitignore_patterns(str(tmp_path))
    assert patterns == ["keep.txt"]

    assert s.tool_errors == []
