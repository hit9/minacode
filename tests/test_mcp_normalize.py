"""mcp normalize (split from tests/test_mcp_tools.py)."""
import asyncio
from types import SimpleNamespace

from mcp_harness import mcp_cfg, mcp_tool_info, session

from minacode.config import (
    Config,
)
from minacode.session import Session, bootstrap_features


class TestNormalizeResult:
    def test_string_content(self):
        """String content is passed through."""
        s = session("/tmp")
        result = s.mcp.normalize_result("hello world")
        assert result == "hello world"

    def test_dict_text_type(self):
        """Dict with type='text' extracts text field."""
        s = session("/tmp")
        result = s.mcp.normalize_result({"type": "text", "text": "hello from mcp"})
        assert result == "hello from mcp"

    def test_dict_resource_type(self):
        """Dict with type='resource' dumps resource field."""
        s = session("/tmp")
        resource = {"uri": "file:///tmp/test.txt", "text": "contents"}
        result = s.mcp.normalize_result({"type": "resource", "resource": resource})
        assert "file:///tmp/test.txt" in result
        assert "contents" in result

    def test_dict_other_type(self):
        """Dict with unknown type is dumped as JSON."""
        s = session("/tmp")
        result = s.mcp.normalize_result({"type": "image", "data": "...", "mimeType": "image/png"})
        assert "image/png" in result

    def test_object_text_type(self):
        """Object with type='text' extracts text attribute."""
        s = session("/tmp")
        item = SimpleNamespace(type="text", text="object text")
        result = s.mcp.normalize_result(item)
        assert result == "object text"

    def test_object_resource_type(self):
        """Object with type='resource' converts resource to string."""
        s = session("/tmp")
        item = SimpleNamespace(type="resource", resource={"uri": "test://uri"})
        result = s.mcp.normalize_result(item)
        assert "test://uri" in result

    def test_list_of_items(self):
        """List of content items is joined."""
        s = session("/tmp")
        result = s.mcp.normalize_result(
            [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ]
        )
        assert "first" in result
        assert "second" in result

    def test_object_model_dump(self):
        """Object with model_dump is serialized."""
        s = session("/tmp")
        obj = SimpleNamespace(model_dump=lambda mode="json": {"result": "ok", "value": 42})
        result = s.mcp.normalize_result(obj)
        assert "ok" in result
        assert "42" in result

    def test_structured_content_stands_in_for_missing_text(self):
        """A tool that declares an outputSchema returns `structuredContent` and only *should* also
        repeat it as text. Without the repeat the result would arrive empty — which the model reads
        as a query that matched nothing, not as a client that dropped the payload."""
        s = session("/tmp")
        result = SimpleNamespace(content=[], structuredContent={"total": 3, "items": ["a", "b"]})

        text = s.mcp.normalize_result(result)

        assert '"total": 3' in text and '"items"' in text

    def test_structured_content_is_not_repeated_when_text_is_present(self):
        """Servers that honor the compatibility repeat send the same payload twice; printing both
        would double the size of every result they return."""
        s = session("/tmp")
        result = SimpleNamespace(content=[{"type": "text", "text": '{"total": 3}'}], structuredContent={"total": 3})

        text = s.mcp.normalize_result(result)

        assert text.count("total") == 1

    def test_snake_case_structured_content_is_read_too(self):
        """The wire field is camelCase, but a hand-built server object may use the Python spelling."""
        s = session("/tmp")

        assert '"ok"' in s.mcp.normalize_result(SimpleNamespace(content=[], structured_content={"status": "ok"}))

    def test_empty_result_with_no_structured_content_stays_empty(self):
        s = session("/tmp")

        assert s.mcp.normalize_result(SimpleNamespace(content=[])) == ""

    def test_long_output_truncation(self, monkeypatch):
        """Output exceeding RAW_OUTPUT_LIMIT is truncated."""
        s = session("/tmp")
        monkeypatch.setattr(s.mcp, "RAW_OUTPUT_LIMIT", 100)
        long_result = "x" * 200
        text = s.mcp.normalize_result(long_result)
        assert len(text) <= 150  # 100 + truncated marker
        assert "<MCPOutputTruncated" in text
        assert "200" in text

class TestCallToolSuccess:
    def test_call_success_mocked(self, monkeypatch):
        """call_tool returns wrapped output on success."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)

        async def fake_call(url, headers, name, arguments):
            return {"type": "text", "text": f"called {name} with {arguments}"}

        monkeypatch.setattr(s.mcp, "_call_tool", fake_call)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]

        result = s.mcp.call_tool("test", "echo", {"text": "hi"})
        assert "<MCPCall server=" in result
        assert 'tool="echo"' in result
        assert "called echo" in result
        assert "</MCPCall>" in result

    def test_call_content_list(self, monkeypatch):
        """call_tool with multi-item content list."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)

        async def fake_call(url, headers, name, arguments):
            return {
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                ]
            }

        monkeypatch.setattr(s.mcp, "_call_tool", fake_call)
        s.mcp.tools["test"] = [mcp_tool_info("test", "multi")]

        result = s.mcp.call_tool("test", "multi", {})
        assert "part one" in result
        assert "part two" in result

    def test_call_from_running_event_loop(self, monkeypatch):
        """Synchronous call_tool still works when the caller already has an event loop."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)

        async def fake_call(config, headers, name, arguments):
            return {"type": "text", "text": "ok"}

        async def run_call():
            return s.mcp.call_tool("test", "echo", {})

        monkeypatch.setattr(s.mcp, "_call_tool", fake_call)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]

        assert "ok" in asyncio.run(run_call())
