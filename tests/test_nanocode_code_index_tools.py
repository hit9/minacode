from types import SimpleNamespace

import nanocode
import pytest

from nanocode import Agent, FindCodeSymbolTool, InspectCodeSymbolTool, OutlineCodeFileTool, Session, ToolCallArgError, ToolCallError


class FakeRepository:
    events = []
    status = "ready"
    refresh_status = None

    def __init__(self, root, *, db_path=None, create_index=False):
        self.root = root
        self.db_path = db_path
        self.create_index = create_index
        self.events.append(("repo", root, db_path, create_index))

    def refresh(self, *, progress=None):
        self.events.append(("refresh", self.root, self.db_path, progress is not None))
        if progress is not None:
            progress("scan")
            progress("start", done=0, total=2)
            progress("file", done=1, total=2, path="code.py")
        if self.refresh_status is not None:
            type(self).status = self.refresh_status
        return self

    def update(self, paths=None, *, progress=None):
        self.events.append(("update", tuple(paths or ()), self.root, self.db_path, progress is not None))
        if progress is not None:
            progress("scan")
            progress("finish", done=1, total=1)
        return self

    def search_text(self, query, *, kind=None, path=None, exact_only=False, limit=20):
        self.events.append(("search_text", query, kind, path, exact_only, limit, self.root, self.db_path))
        return "query: " + query + "\ncount: 1\nsymbol Tool nanocode.py:10:20"

    def inspect_text(self, symbol, *, kind=None, path=None, exact_only=False, anchors=False):
        self.events.append(("inspect_text", symbol, kind, path, exact_only, anchors, self.root, self.db_path))
        return "symbol:\n  name: " + symbol + "\nsource:\n  status: full"

    def outline_text(self, filepath, *, symbol=None):
        self.events.append(("outline_text", filepath, symbol, self.root, self.db_path))
        return "file: " + filepath + "\noutline:\n  class Tool 0:2 class Tool:"


def fake_code_index_module(status="ready", *, refresh_status=None):
    FakeRepository.status = status
    FakeRepository.refresh_status = refresh_status

    def status_fn(root, *, db_path=None, check=False, max_pending_files=50, format="object"):
        status = FakeRepository.status
        FakeRepository.events.append(("status", root, db_path, check, max_pending_files, format))
        return SimpleNamespace(status=status, reason="index not initialized" if status == "missing" else "", message="")

    def refresh_async(root, *, db_path=None, progress=None, **kwargs):
        FakeRepository.events.append(("refresh_async", root, db_path, progress is not None, kwargs))
        if progress is not None:
            progress("scan")
            progress("finish", done=1, total=1)
        return SimpleNamespace()

    return SimpleNamespace(Repository=FakeRepository, refresh_async=refresh_async, status=status_fn)


@pytest.fixture(autouse=True)
def reset_fake_repository():
    FakeRepository.events = []
    FakeRepository.status = "ready"
    FakeRepository.refresh_status = None


def test_inspect_code_requires_code_index(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode, "_code_index_module", lambda: None)

    with pytest.raises(ToolCallError, match="code index is not available"):
        InspectCodeSymbolTool.make(Session(cwd=str(tmp_path)), ["Tool"])


def test_code_index_schema_accepts_expected_args():
    for tool in (FindCodeSymbolTool, InspectCodeSymbolTool, OutlineCodeFileTool):
        args_schema = tool.tool_schema()["function"]["parameters"]["properties"]["args"]
        assert args_schema["minItems"] == 1
        assert args_schema["maxItems"] == 2


def test_inspect_code_rejects_natural_language(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode, "_code_index_module", lambda: fake_code_index_module())

    with pytest.raises(ToolCallArgError, match="do not pass natural language"):
        InspectCodeSymbolTool.make(Session(cwd=str(tmp_path)), ["Tool class callers"])
    with pytest.raises(ToolCallArgError, match="do not pass natural language"):
        FindCodeSymbolTool.make(Session(cwd=str(tmp_path)), ["Tool class"])


def test_code_index_missing_is_not_initialized_implicitly(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path), config=nanocode.Config(data_dir=str(tmp_path / "data")))
    monkeypatch.setattr(nanocode, "_code_index_module", lambda: fake_code_index_module("missing"))

    with pytest.raises(ToolCallError, match="code index is not available"):
        FindCodeSymbolTool.make(session, ["Tool"])

    assert not [event for event in FakeRepository.events if event[0] in {"repo", "refresh"}]


def test_code_index_status_formats_checked_pending_files(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path), config=nanocode.Config(data_dir=str(tmp_path / "data")))

    def status_fn(root, *, db_path=None, check=False, max_pending_files=50, format="object"):
        return SimpleNamespace(status="stale", reason="", message="", pending_changes=5, pending_files=("a.py", "b.py", "c.py", "d.py"))

    monkeypatch.setattr(nanocode, "_code_index_module", lambda: SimpleNamespace(status=status_fn))

    assert nanocode._code_index_status(session, check=True) == ("stale", "pending 5 (a.py, b.py, c.py...)")


def test_code_index_sync_initializes_missing_index_in_project_data(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path), config=nanocode.Config(data_dir=str(tmp_path / "data")))
    module = fake_code_index_module("missing", refresh_status="ready")
    monkeypatch.setattr(nanocode, "_code_index_module", lambda: module)

    result = nanocode._code_index_sync(session)

    db_path = str(tmp_path / "data" / "projects" / session.project_key() / "code-symbol-index" / "index.sqlite")
    assert ("repo", str(tmp_path), db_path, True) in FakeRepository.events
    assert ("refresh", str(tmp_path), db_path, True) in FakeRepository.events
    assert session.state.status_notice == "index:done"
    assert result == "code_index: initialized\nstatus: ready\npath: " + db_path


def test_code_index_force_rebuild_removes_project_index_dir(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path), config=nanocode.Config(data_dir=str(tmp_path / "data")))
    module = fake_code_index_module("ready")
    monkeypatch.setattr(nanocode, "_code_index_module", lambda: module)
    index_dir = tmp_path / "data" / "projects" / session.project_key() / "code-symbol-index"
    index_dir.mkdir(parents=True)
    (index_dir / "old.sqlite").write_text("old", encoding="utf-8")

    result = nanocode._code_index_sync(session, force=True)

    assert not (index_dir / "old.sqlite").exists()
    assert ("repo", str(tmp_path), nanocode._code_index_db_path(session), True) in FakeRepository.events
    assert ("refresh", str(tmp_path), nanocode._code_index_db_path(session), True) in FakeRepository.events
    assert result == "code_index: rebuilt\nstatus: ready\npath: " + nanocode._code_index_db_path(session)


def test_code_index_refresh_existing_async_starts_for_ready_index(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path), config=nanocode.Config(data_dir=str(tmp_path / "data")))
    monkeypatch.setattr(nanocode, "_code_index_module", lambda: fake_code_index_module("ready"))

    assert nanocode._code_index_refresh_existing_async(session) is True

    assert ("refresh_async", str(tmp_path), nanocode._code_index_db_path(session), True, {}) in FakeRepository.events
    assert session.code_index_repository is None
    assert session.state.status_notice == "index:done 1/1"
    assert session.state.code_index_refreshing is False
    assert session.state.code_index_reload_needed is True

    nanocode._code_index_reload_if_ready(session)

    assert isinstance(session.code_index_repository, FakeRepository)
    assert session.state.code_index_reload_needed is False


def test_find_code_symbol_uses_search_text(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path), config=nanocode.Config(data_dir=str(tmp_path / "data")))
    monkeypatch.setattr(nanocode, "_code_index_module", lambda: fake_code_index_module())

    result = FindCodeSymbolTool.make(session, ["Tool", {"limit": 12, "kind": "class", "path": "nanocode.py", "exact_only": True}]).call()

    db_path = str(tmp_path / "data" / "projects" / session.project_key() / "code-symbol-index" / "index.sqlite")
    assert ("search_text", "Tool", "class", "nanocode.py", True, 12, str(tmp_path), db_path) in FakeRepository.events
    assert result == "<FindCodeSymbolToolResult>\nquery: Tool\ncount: 1\nsymbol Tool nanocode.py:10:20\n</FindCodeSymbolToolResult>"


def test_find_code_symbol_clamps_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode, "_code_index_module", lambda: fake_code_index_module())
    assert FindCodeSymbolTool.make(Session(cwd=str(tmp_path)), ["Tool", {"limit": 999}]).limit == 80
    assert FindCodeSymbolTool.make(Session(cwd=str(tmp_path)), ["Tool", {"limit": 0}]).limit == 1
    with pytest.raises(ToolCallArgError, match="limit must be an integer"):
        FindCodeSymbolTool.make(Session(cwd=str(tmp_path)), ["Tool", {"limit": "many"}])


def test_inspect_code_symbol_rejects_files_directories_and_dotted_module_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode, "_code_index_module", lambda: fake_code_index_module())
    (tmp_path / "orion" / "biz" / "handlers" / "syftpp").mkdir(parents=True)
    (tmp_path / "code.py").write_text("class Tool:\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallArgError, match="file or directory"):
        InspectCodeSymbolTool.make(session, ["code.py"])
    with pytest.raises(ToolCallArgError, match="file or directory"):
        InspectCodeSymbolTool.make(session, ["orion.biz.handlers.syftpp"])
    with pytest.raises(ToolCallArgError, match="module path"):
        InspectCodeSymbolTool.make(session, ["pkg.module.symbol"])


def test_inspect_code_symbol_uses_inspect_text(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode, "_code_index_module", lambda: fake_code_index_module())

    result = InspectCodeSymbolTool.make(session, ["Tool", {"path": "nanocode.py", "exact_only": True}]).call()

    assert ("inspect_text", "Tool", None, "nanocode.py", True, True, str(tmp_path), nanocode._code_index_db_path(session)) in FakeRepository.events
    assert result == "<InspectCodeSymbolToolResult>\nsymbol:\n  name: Tool\nsource:\n  status: full\n</InspectCodeSymbolToolResult>"


def test_agent_tool_call_preserves_code_index_options_object(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode, "_code_index_module", lambda: fake_code_index_module())

    Agent(session).execute_tool_calls(
        [
            {
                "name": "InspectCodeSymbol",
                "intention": "inspect exact symbol",
                "args": ["Tool", {"path": "nanocode.py", "exact_only": True}],
            }
        ]
    )

    assert ("inspect_text", "Tool", None, "nanocode.py", True, True, str(tmp_path), nanocode._code_index_db_path(session)) in FakeRepository.events


def test_outline_code_file_uses_outline_text(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path))
    filepath = tmp_path / "code.py"
    filepath.write_text("class Tool:\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(nanocode, "_code_index_module", lambda: fake_code_index_module())

    result = OutlineCodeFileTool.make(session, ["code.py", "Tool"]).call()

    assert ("outline_text", str(filepath), "Tool", str(tmp_path), nanocode._code_index_db_path(session)) in FakeRepository.events
    assert result == "<OutlineCodeFileToolResult>\nfile: " + str(filepath) + "\noutline:\n  class Tool 0:2 class Tool:\n</OutlineCodeFileToolResult>"


def test_outline_code_file_rejects_directories_and_symbols(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode, "_code_index_module", lambda: fake_code_index_module())
    (tmp_path / "pkg").mkdir()
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallArgError, match="existing file"):
        OutlineCodeFileTool.make(session, ["pkg"])
    with pytest.raises(ToolCallArgError, match="existing file"):
        OutlineCodeFileTool.make(session, ["Tool"])
