from types import SimpleNamespace

import nanocode
import pytest

from nanocode import Agent, InspectCodeTool, Session, ToolCallArgError, ToolCallError


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


def fake_code_index_module(status="ready", *, refresh_status=None, pending_changes=None, pending_files=()):
    FakeRepository.status = status
    FakeRepository.refresh_status = refresh_status

    def status_fn(root, *, db_path=None, check=False, max_pending_files=50, format="object"):
        status = FakeRepository.status
        FakeRepository.events.append(("status", root, db_path, check, max_pending_files, format))
        files = tuple(pending_files[:max_pending_files])
        return SimpleNamespace(
            status=status,
            reason="index not initialized" if status == "missing" else "",
            message="",
            pending_changes=len(pending_files) if pending_changes is None else pending_changes,
            pending_files=files,
        )

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
    monkeypatch.setattr(nanocode.CodeIndex, "module", staticmethod(lambda: None))

    with pytest.raises(ToolCallError, match="code index is not available"):
        InspectCodeTool.make(Session(cwd=str(tmp_path)), ["inspect", "Tool"])


def test_code_index_schema_accepts_expected_args():
    args_schema = InspectCodeTool.tool_schema()["function"]["parameters"]["properties"]["args"]
    assert args_schema["minItems"] == 2
    assert args_schema["maxItems"] == 3


def test_inspect_code_rejects_natural_language(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode.CodeIndex, "module", staticmethod(lambda: fake_code_index_module()))

    with pytest.raises(ToolCallArgError, match="do not pass natural language"):
        InspectCodeTool.make(Session(cwd=str(tmp_path)), ["inspect", "Tool class callers"])
    with pytest.raises(ToolCallArgError, match="do not pass natural language"):
        InspectCodeTool.make(Session(cwd=str(tmp_path)), ["find", "Tool class"])


def test_inspect_code_rejects_invalid_mode_and_options(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode.CodeIndex, "module", staticmethod(lambda: fake_code_index_module()))

    with pytest.raises(ToolCallArgError, match="mode must be find, inspect, or outline"):
        InspectCodeTool.make(Session(cwd=str(tmp_path)), ["search", "Tool"])
    with pytest.raises(ToolCallArgError, match="options must be an object"):
        InspectCodeTool.make(Session(cwd=str(tmp_path)), ["find", "Tool", "limit=10"])


def test_code_index_missing_is_not_initialized_implicitly(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path), config=nanocode.Config(data_dir=str(tmp_path / "data")))
    monkeypatch.setattr(nanocode.CodeIndex, "module", staticmethod(lambda: fake_code_index_module("missing")))

    with pytest.raises(ToolCallError, match="code index is not available"):
        InspectCodeTool.make(session, ["find", "Tool"])

    assert not [event for event in FakeRepository.events if event[0] in {"repo", "refresh"}]


def test_code_index_status_formats_checked_pending_files(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path), config=nanocode.Config(data_dir=str(tmp_path / "data")))

    def status_fn(root, *, db_path=None, check=False, max_pending_files=50, format="object"):
        return SimpleNamespace(status="stale", reason="", message="", pending_changes=5, pending_files=("a.py", "b.py", "c.py", "d.py"))

    monkeypatch.setattr(nanocode.CodeIndex, "module", staticmethod(lambda: SimpleNamespace(status=status_fn)))

    assert nanocode.CodeIndex(session).status(check=True) == ("stale", "pending 5 (a.py, b.py, c.py...)")


def test_code_index_sync_initializes_missing_index_in_project_data(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path), config=nanocode.Config(data_dir=str(tmp_path / "data")))
    module = fake_code_index_module("missing", refresh_status="ready")
    monkeypatch.setattr(nanocode.CodeIndex, "module", staticmethod(lambda: module))

    result = nanocode.CodeIndex(session).sync()

    db_path = str(tmp_path / "data" / "projects" / session.project_key() / "code-symbol-index" / "index.sqlite")
    assert ("repo", str(tmp_path), db_path, True) in FakeRepository.events
    assert ("refresh", str(tmp_path), db_path, True) in FakeRepository.events
    assert session.state.status_notice == "index:done"
    assert result == "code_index: initialized\nstatus: ready\npath: " + db_path


def test_code_index_force_rebuild_removes_project_index_dir(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path), config=nanocode.Config(data_dir=str(tmp_path / "data")))
    module = fake_code_index_module("ready")
    monkeypatch.setattr(nanocode.CodeIndex, "module", staticmethod(lambda: module))
    index_dir = tmp_path / "data" / "projects" / session.project_key() / "code-symbol-index"
    index_dir.mkdir(parents=True)
    (index_dir / "old.sqlite").write_text("old", encoding="utf-8")

    code_index = nanocode.CodeIndex(session)
    result = code_index.sync(force=True)

    assert not (index_dir / "old.sqlite").exists()
    assert ("repo", str(tmp_path), code_index.db_path(), True) in FakeRepository.events
    assert ("refresh", str(tmp_path), code_index.db_path(), True) in FakeRepository.events
    assert result == "code_index: rebuilt\nstatus: ready\npath: " + code_index.db_path()


def test_code_index_refresh_existing_async_starts_for_ready_index(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path), config=nanocode.Config(data_dir=str(tmp_path / "data")))
    code_index = nanocode.CodeIndex(session)
    monkeypatch.setattr(nanocode.CodeIndex, "module", staticmethod(lambda: fake_code_index_module("ready")))

    assert code_index.refresh_existing_async() is True

    assert ("refresh_async", str(tmp_path), code_index.db_path(), True, {}) in FakeRepository.events
    assert session.code_index_repository is None
    assert session.state.status_notice == "index:done 1/1"
    assert session.state.code_index_refreshing is False
    assert session.state.code_index_reload_needed is True

    code_index.reload_if_ready()

    assert isinstance(session.code_index_repository, FakeRepository)
    assert session.state.code_index_reload_needed is False


def test_code_index_update_pending_updates_small_stale_file_set(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path), config=nanocode.Config(data_dir=str(tmp_path / "data")))
    code_index = nanocode.CodeIndex(session)
    monkeypatch.setattr(nanocode.CodeIndex, "module", staticmethod(lambda: fake_code_index_module("stale", pending_files=("a.py", "pkg/b.py"))))

    code_index.update_pending(limit=3)

    assert ("status", str(tmp_path), code_index.db_path(), True, 4, "object") in FakeRepository.events
    assert ("update", (str(tmp_path / "a.py"), str(tmp_path / "pkg" / "b.py")), str(tmp_path), code_index.db_path(), False) in FakeRepository.events


def test_code_index_update_pending_skips_large_stale_file_set(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path), config=nanocode.Config(data_dir=str(tmp_path / "data")))
    monkeypatch.setattr(nanocode.CodeIndex, "module", staticmethod(lambda: fake_code_index_module("stale", pending_changes=4, pending_files=("a.py", "b.py", "c.py"))))

    nanocode.CodeIndex(session).update_pending(limit=3)

    assert not [event for event in FakeRepository.events if event[0] == "update"]


def test_inspect_code_find_uses_search_text(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path), config=nanocode.Config(data_dir=str(tmp_path / "data")))
    monkeypatch.setattr(nanocode.CodeIndex, "module", staticmethod(lambda: fake_code_index_module()))

    result = InspectCodeTool.make(session, ["find", "Tool", {"limit": 12, "kind": "class", "path": "nanocode.py", "exact_only": True}]).call()

    db_path = str(tmp_path / "data" / "projects" / session.project_key() / "code-symbol-index" / "index.sqlite")
    assert ("search_text", "Tool", "class", "nanocode.py", True, 12, str(tmp_path), db_path) in FakeRepository.events
    assert result == "<InspectCodeToolResult>\nmode: find\nquery: Tool\ncount: 1\nsymbol Tool nanocode.py:10:20\n</InspectCodeToolResult>"


def test_inspect_code_find_clamps_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode.CodeIndex, "module", staticmethod(lambda: fake_code_index_module()))
    assert InspectCodeTool.make(Session(cwd=str(tmp_path)), ["find", "Tool", {"limit": 999}]).limit == 80
    assert InspectCodeTool.make(Session(cwd=str(tmp_path)), ["find", "Tool", {"limit": 0}]).limit == 1
    with pytest.raises(ToolCallArgError, match="limit must be an integer"):
        InspectCodeTool.make(Session(cwd=str(tmp_path)), ["find", "Tool", {"limit": "many"}])


def test_inspect_code_symbol_rejects_files_directories_and_dotted_module_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode.CodeIndex, "module", staticmethod(lambda: fake_code_index_module()))
    (tmp_path / "orion" / "biz" / "handlers" / "syftpp").mkdir(parents=True)
    (tmp_path / "code.py").write_text("class Tool:\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallArgError, match="file or directory"):
        InspectCodeTool.make(session, ["inspect", "code.py"])
    with pytest.raises(ToolCallArgError, match="file or directory"):
        InspectCodeTool.make(session, ["inspect", "orion.biz.handlers.syftpp"])
    with pytest.raises(ToolCallArgError, match="module path"):
        InspectCodeTool.make(session, ["inspect", "pkg.module.symbol"])


def test_inspect_code_inspect_uses_inspect_text(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path))
    code_index = nanocode.CodeIndex(session)
    monkeypatch.setattr(nanocode.CodeIndex, "module", staticmethod(lambda: fake_code_index_module()))

    result = InspectCodeTool.make(session, ["inspect", "Tool", {"path": "nanocode.py", "exact_only": True}]).call()

    assert ("inspect_text", "Tool", None, "nanocode.py", True, True, str(tmp_path), code_index.db_path()) in FakeRepository.events
    assert result == "<InspectCodeToolResult>\nmode: inspect\nsymbol:\n  name: Tool\nsource:\n  status: full\n</InspectCodeToolResult>"


def test_agent_tool_call_preserves_code_index_options_object(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path))
    code_index = nanocode.CodeIndex(session)
    monkeypatch.setattr(nanocode.CodeIndex, "module", staticmethod(lambda: fake_code_index_module()))

    Agent(session).execute_tool_calls(
        [
            {
                "name": "InspectCode",
                "intention": "inspect exact symbol",
                "args": ["inspect", "Tool", {"path": "nanocode.py", "exact_only": True}],
            }
        ]
    )

    assert ("inspect_text", "Tool", None, "nanocode.py", True, True, str(tmp_path), code_index.db_path()) in FakeRepository.events


def test_inspect_code_outline_uses_outline_text(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path))
    filepath = tmp_path / "code.py"
    filepath.write_text("class Tool:\n    pass\n", encoding="utf-8")
    code_index = nanocode.CodeIndex(session)
    monkeypatch.setattr(nanocode.CodeIndex, "module", staticmethod(lambda: fake_code_index_module()))

    result = InspectCodeTool.make(session, ["outline", "code.py", {"symbol": "Tool"}]).call()

    assert ("outline_text", str(filepath), "Tool", str(tmp_path), code_index.db_path()) in FakeRepository.events
    assert result == "<InspectCodeToolResult>\nmode: outline\nfile: " + str(filepath) + "\noutline:\n  class Tool 0:2 class Tool:\n</InspectCodeToolResult>"


def test_outline_code_file_rejects_directories_and_symbols(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode.CodeIndex, "module", staticmethod(lambda: fake_code_index_module()))
    (tmp_path / "pkg").mkdir()
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallArgError, match="existing file"):
        InspectCodeTool.make(session, ["outline", "pkg"])
    with pytest.raises(ToolCallArgError, match="existing file"):
        InspectCodeTool.make(session, ["outline", "Tool"])
