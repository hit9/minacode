from types import SimpleNamespace

import nanocode
import pytest

from nanocode import FindCodeSymbolTool, InspectCodeSymbolTool, OutlineCodeFileTool, Session, ToolCallArgError, ToolCallError


class FakeRepository:
    events = []
    status = "ready"
    refresh_status = None

    def __init__(self, root, *, db_path=None, create_index=False):
        self.root = root
        self.db_path = db_path
        self.create_index = create_index
        self.events.append(("repo", root, db_path, create_index))

    def refresh(self):
        self.events.append(("refresh", self.root, self.db_path))
        if self.refresh_status is not None:
            type(self).status = self.refresh_status
        return self

    def update(self, paths=None):
        self.events.append(("update", tuple(paths or ()), self.root, self.db_path))
        return self

    def search_text(self, query, *, limit):
        self.events.append(("search_text", query, limit, self.root, self.db_path))
        return "query: " + query + "\ncount: 1\nsymbol Tool nanocode.py:10:20"

    def inspect_text(self, symbol):
        self.events.append(("inspect_text", symbol, self.root, self.db_path))
        return "symbol:\n  name: " + symbol + "\nsource:\n  status: full"

    def outline_text(self, filepath):
        self.events.append(("outline_text", filepath, self.root, self.db_path))
        return "file: " + filepath + "\noutline:\n  class Tool 0:2 class Tool:"


def fake_code_index_module(status="ready", *, refresh_status=None):
    FakeRepository.status = status
    FakeRepository.refresh_status = refresh_status

    def status_fn(root, *, db_path=None, check=False, format="object"):
        status = FakeRepository.status
        FakeRepository.events.append(("status", root, db_path, check, format))
        return SimpleNamespace(status=status, reason="index not initialized" if status == "missing" else "", message="")

    return SimpleNamespace(Repository=FakeRepository, status=status_fn)


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
    for tool in (InspectCodeSymbolTool, OutlineCodeFileTool):
        args_schema = tool.tool_schema()["function"]["parameters"]["properties"]["args"]
        assert args_schema["minItems"] == 1
        assert args_schema["maxItems"] == 1
        assert args_schema["items"]["type"] == "string"
    args_schema = FindCodeSymbolTool.tool_schema()["function"]["parameters"]["properties"]["args"]
    assert args_schema["minItems"] == 1
    assert args_schema["maxItems"] == 2
    assert args_schema["items"]["type"] == ["string", "number"]


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


def test_code_index_sync_initializes_missing_index_in_project_data(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path), config=nanocode.Config(data_dir=str(tmp_path / "data")))
    module = fake_code_index_module("missing", refresh_status="ready")
    monkeypatch.setattr(nanocode, "_code_index_module", lambda: module)

    result = nanocode._code_index_sync(session)

    db_path = str(tmp_path / "data" / "projects" / session.project_key() / "code-symbol-index" / "index.sqlite")
    assert ("repo", str(tmp_path), db_path, True) in FakeRepository.events
    assert ("refresh", str(tmp_path), db_path) in FakeRepository.events
    assert result == "code_index: initialized\nstatus: ready\npath: " + db_path


def test_code_index_update_existing_syncs_ready_index_only(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path), config=nanocode.Config(data_dir=str(tmp_path / "data")))
    monkeypatch.setattr(nanocode, "_code_index_module", lambda: fake_code_index_module("ready"))

    nanocode._code_index_update_existing(session)

    assert ("update", tuple(), str(tmp_path), nanocode._code_index_db_path(session)) in FakeRepository.events


def test_find_code_symbol_uses_search_text(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path), config=nanocode.Config(data_dir=str(tmp_path / "data")))
    monkeypatch.setattr(nanocode, "_code_index_module", lambda: fake_code_index_module())

    result = FindCodeSymbolTool.make(session, ["Tool", 12]).call()

    db_path = str(tmp_path / "data" / "projects" / session.project_key() / "code-symbol-index" / "index.sqlite")
    assert ("search_text", "Tool", 12, str(tmp_path), db_path) in FakeRepository.events
    assert result == "<FindCodeSymbolToolResult>\nquery: Tool\ncount: 1\nsymbol Tool nanocode.py:10:20\n</FindCodeSymbolToolResult>"


def test_find_code_symbol_clamps_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode, "_code_index_module", lambda: fake_code_index_module())
    assert FindCodeSymbolTool.make(Session(cwd=str(tmp_path)), ["Tool", 999]).limit == 80
    assert FindCodeSymbolTool.make(Session(cwd=str(tmp_path)), ["Tool", 0]).limit == 1
    with pytest.raises(ToolCallArgError, match="limit must be an integer"):
        FindCodeSymbolTool.make(Session(cwd=str(tmp_path)), ["Tool", "many"])


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

    result = InspectCodeSymbolTool.make(session, ["Tool"]).call()

    assert ("inspect_text", "Tool", str(tmp_path), nanocode._code_index_db_path(session)) in FakeRepository.events
    assert result == "<InspectCodeSymbolToolResult>\nsymbol:\n  name: Tool\nsource:\n  status: full\n</InspectCodeSymbolToolResult>"


def test_outline_code_file_uses_outline_text(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path))
    filepath = tmp_path / "code.py"
    filepath.write_text("class Tool:\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(nanocode, "_code_index_module", lambda: fake_code_index_module())

    result = OutlineCodeFileTool.make(session, ["code.py"]).call()

    assert ("outline_text", str(filepath), str(tmp_path), nanocode._code_index_db_path(session)) in FakeRepository.events
    assert result == "<OutlineCodeFileToolResult>\nfile: " + str(filepath) + "\noutline:\n  class Tool 0:2 class Tool:\n</OutlineCodeFileToolResult>"


def test_outline_code_file_rejects_directories_and_symbols(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode, "_code_index_module", lambda: fake_code_index_module())
    (tmp_path / "pkg").mkdir()
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallArgError, match="existing file"):
        OutlineCodeFileTool.make(session, ["pkg"])
    with pytest.raises(ToolCallArgError, match="existing file"):
        OutlineCodeFileTool.make(session, ["Tool"])
