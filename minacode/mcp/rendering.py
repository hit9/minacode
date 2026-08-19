"""MCP display models and pure formatting/normalization of tools, resources, and results."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from minacode.base import Json, Text
from minacode.mcp.config import MCPServerConfig

if TYPE_CHECKING:
    from mcp.types import Resource


@dataclass
class MCPToolInfo:
    server: str
    name: str
    description: str
    input_schema: Json
    # The tool's declared result shape (MCP `outputSchema`, 2025-06-18). Empty when the server
    # declares none, which is most of them: it is rendered only when present.
    output_schema: Json = field(default_factory=dict)
    annotations: Json = field(default_factory=dict)


@dataclass
class MCPResourceInfo:
    server: str
    uri: str
    name: str
    description: str
    mime_type: str = ""


URI_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s'\"<>)\]}]+")


def render_describe(
    server: str,
    info: MCPToolInfo,
    *,
    description_limit: int,
    argument_limit: int,
    argument_description_limit: int,
) -> str:
    """Render the full `<MCPDescribe>` block for one tool: description, arguments, returns, and raw schemas."""
    from minacode.tools import Tool  # local import: tools is built on top of mcp

    schema = info.input_schema or {}
    returns = info.output_schema or {}
    lines = [f"<MCPDescribe server={json.dumps(server)} tool={json.dumps(info.name)}>"]
    if info.description:
        lines.append("<description>")
        lines.append(Tool.compact(info.description, description_limit))
        lines.append("</description>")
    lines.append("<arguments>")
    lines.extend(describe_properties(schema, "arguments", argument_limit=argument_limit, argument_description_limit=argument_description_limit))
    lines.append("</arguments>")
    # The result shape, on the servers that declare one. Without it the only way to learn what
    # a tool returns is to call it, so an exploratory call is spent on every unfamiliar tool.
    # Rendered in the same shape as the arguments above, and omitted entirely when undeclared.
    if returns:
        lines.append("<returns>")
        lines.extend(
            describe_properties(returns, "fields", bare="returns_schema", argument_limit=argument_limit, argument_description_limit=argument_description_limit)
        )
        lines.append("</returns>")
    if isinstance(schema, dict) and schema:
        lines.append("<schema>")
        lines.append(json.dumps(schema, ensure_ascii=False, indent=2))
        lines.append("</schema>")
    if returns:
        lines.append("<returns_schema>")
        lines.append(json.dumps(returns, ensure_ascii=False, indent=2))
        lines.append("</returns_schema>")
    lines.append("</MCPDescribe>")
    return "\n".join(lines)


def describe_properties(
    schema: Json,
    label: str,
    bare: str = "",
    *,
    argument_limit: int,
    argument_description_limit: int,
) -> list[str]:
    """One `- name required type: description` line per property, bounded like the block it
    serves. Shared by arguments and returns so a reader learns one rendering, not two.

    `bare` is what to say for a schema with no properties at all -- a bare array or scalar
    result. Arguments pass nothing and keep rendering an empty block, as they always have."""
    from minacode.tools import Tool  # local import: tools is built on top of mcp

    if not isinstance(schema, dict):
        return []
    props, required = schema_props_required(schema)
    if not props:
        kind = schema.get("type")
        return [f"({kind}; see {bare} below)"] if bare and isinstance(kind, str) and kind else []
    lines: list[str] = []
    for index, (name, prop) in enumerate(props.items()):
        if index >= argument_limit:
            lines.append(f"... {len(props) - argument_limit} more {label} omitted")
            break
        req = "required" if name in required else "optional"
        prop = prop if isinstance(prop, dict) else {}
        typ = prop.get("type", "any")
        desc = Tool.compact(str(prop.get("description", "") or ""), argument_description_limit)
        lines.append(f"- {name} {req} {typ}: {desc}")
    return lines


def resources_block(server: str, resources: list[MCPResourceInfo]) -> list[str]:
    """The 'resources (N) — read with ...' header plus one line per resource, or [] if none."""
    if not resources:
        return []
    header = f'resources ({len(resources)}) — read with MCP(action="read_resource", server={json.dumps(server)}, uri=...):'
    return [header, *(format_resource_line(res) for res in resources)]


def server_lines(
    server: str,
    tools: list[MCPToolInfo],
    resources: list[MCPResourceInfo],
    *,
    include_schema: bool = True,
    schema_limit: int,
) -> list[str]:
    """A server's header, tool lines, and resources block — shared by the tools index and mentions."""
    lines = [f"[{server}] {server.capitalize()}"]
    lines.extend(line for info in tools if (line := format_tool_line(server, info, include_schema=include_schema, schema_limit=schema_limit)))
    lines.extend(resources_block(server, resources))
    return lines


def index_body(
    configs: list[MCPServerConfig],
    *,
    detail: str = "schema",
    tools: dict[str, list[MCPToolInfo]],
    resources: dict[str, list[MCPResourceInfo]],
    pending_status: Callable[[str], str],
    schema_limit: int,
) -> list[str]:
    """Render the per-server body lines of the tools index at one detail level.

    `tools` and `resources` are the connected server catalogs (name -> rows); `pending_status`
    classifies a connected-but-empty server for the pending list.

    detail controls how much of each tool is emitted (richest to cheapest):
        "schema" — full line via format_tool_line, including the inline JSON schema
        "args"   — same line without the schema (name + arg summary + description)
        "names"  — one "tools: a, b, c" line per server, names only

    Every connected server is represented regardless of detail.
    """
    lines: list[str] = []
    pending: list[str] = []
    for config in configs:
        server_tools = tools.get(config.name, [])
        server_resources = resources.get(config.name, [])
        if not server_tools and not server_resources:
            pending.append(f"- {config.name}: {pending_status(config.name)}")
            continue
        if detail == "names":
            lines.append(f"[{config.name}] {config.name.capitalize()}")
            if server_tools:
                lines.append("tools: " + ", ".join(tool.name for tool in server_tools))
            lines.extend(resources_block(config.name, server_resources))
        else:
            lines.extend(server_lines(config.name, server_tools, server_resources, include_schema=detail == "schema", schema_limit=schema_limit))
        lines.append("")

    if pending:
        lines.append("Configured servers not yet available (they exist — do not assume otherwise):")
        lines.extend(pending)
        lines.append("")
    return lines


def format_tool_line(server: str, info: MCPToolInfo, *, include_schema: bool = True, schema_limit: int) -> str:
    """One tool's index/mention line: `server.name(args) - description`, schema block optional."""
    args_str = tool_args_summary(info)
    desc = (info.description or "").split("\n")[0].strip()
    desc = " ".join(desc.split())
    if len(desc) > 80:
        desc = desc[:77] + "..."

    line = f"{server}.{info.name}{args_str} - {desc}"
    if len(line) > 200:
        line = line[:197] + "..."
    # The full description (often naming a resource doc with the argument grammar) is
    # truncated above, so surface any resource-like URIs it mentions explicitly.
    uris = extract_uris(info.description)
    if uris:
        line += '\n  refs (read with MCP action="read_resource"): ' + ", ".join(uris)
    if include_schema:
        schema = schema_json(info.input_schema, schema_limit)
        if schema:
            line += f"\n  schema: {schema}"
    return line


def format_resource_line(info: MCPResourceInfo) -> str:
    """One `- uri [mime] - description` line for the resources index."""
    desc = " ".join((info.description or "").split())
    if len(desc) > 100:
        desc = desc[:97] + "..."
    mime = f" [{info.mime_type}]" if info.mime_type else ""
    label = f"{info.uri}{mime}"
    return f"- {label} - {desc}" if desc else f"- {label}"


def join_bounded(parts: list[str], *, raw_output_limit: int) -> str:
    """Join non-empty parts and clip to RAW_OUTPUT_LIMIT with a truncation marker."""
    text = "\n".join(part for part in parts if part).strip()
    if len(text) > raw_output_limit:
        text = text[:raw_output_limit] + f"\n<MCPOutputTruncated chars={json.dumps(len(text))}/>"
    return text


def schema_json(schema: Json, limit: int) -> str:
    """Render a remote tool's input schema as compact JSON, capped at `limit` chars (0 = no cap)."""
    if not isinstance(schema, dict) or not schema:
        return ""
    text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    if limit and len(text) > limit:
        text = text[: limit - 1].rstrip() + "… (truncated; MCP describe for full schema)"
    return text


def tool_args_summary(info: MCPToolInfo) -> str:
    """The `(req: type; opt: type)` argument summary for one tool."""
    schema = info.input_schema or {}
    props, required = schema_props_required(schema)

    def _fmt(name: str) -> str:
        t = props.get(name, {}).get("type", "")
        return f"{name}: {t}" if t else name

    req_args = [_fmt(k) for k in required if k in props]
    opt_args = [_fmt(k) for k in props if k not in required]

    if len(req_args) > 8:
        req_args = req_args[:8] + ["..."]
    if len(opt_args) > 8:
        opt_args = opt_args[:8] + ["..."]

    parts = []
    if req_args:
        parts.append("(" + ", ".join(req_args))
    else:
        parts.append("(")
    if opt_args:
        parts.append("; " + ", ".join(opt_args))
    parts.append(")")
    return "".join(parts)


def structured_content(result: Any) -> str:
    """The result's `structuredContent` as JSON text, or "" when it carries none."""
    for attribute in ("structuredContent", "structured_content"):
        structured = getattr(result, attribute, None)
        if isinstance(structured, (dict, list)) and structured:
            return json.dumps(structured, ensure_ascii=False, indent=2)
        if structured is not None and not isinstance(structured, (dict, list)):
            return dump_object(structured)
    return ""


def dump_object(item: Any) -> str:
    """Render a non-str/dict MCP item: pydantic-style model_dump as JSON, else str()."""
    if hasattr(item, "model_dump"):
        return json.dumps(item.model_dump(mode="json"), ensure_ascii=False, indent=2)
    return str(item)


def normalize_resource(result: Any, *, raw_output_limit: int) -> str:
    """Render a resource read result as text: text blobs verbatim, binaries as a size marker."""
    items = result if isinstance(result, list) else [result]
    parts: list[str] = []
    for item in items:
        text = getattr(item, "text", None)
        if text:
            parts.append(str(text))
            continue
        blob = getattr(item, "blob", None)
        if blob is not None:
            mime = str(getattr(item, "mimeType", "") or "application/octet-stream")
            parts.append(f"<binary mimeType={json.dumps(mime)} bytes={len(blob)}/>")
            continue
        parts.append(dump_object(item))
    return join_bounded(parts, raw_output_limit=raw_output_limit)


def normalize_result(result: Any, *, raw_output_limit: int) -> str:
    """Render a tool result as the text the model reads.

    A tool that declares an `outputSchema` returns its payload as `structuredContent`, and only
    *should* also repeat it as text for older clients. A server that skips the repeat would
    otherwise arrive here as an empty result -- indistinguishable, to the model, from a query
    that matched nothing -- so the structured payload stands in when the content blocks are
    empty. It is not appended when they are not: servers that honor the repeat send the same
    payload twice, and printing both would double every result.
    """
    parts: list[str] = []
    content = getattr(result, "content", result)
    items = content if isinstance(content, list) else [content]
    for item in items:
        if isinstance(item, str):
            parts.append(item)
            continue
        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "text":
                parts.append(str(item.get("text") or ""))
            elif item_type == "resource":
                parts.append(json.dumps(item.get("resource"), ensure_ascii=False, indent=2))
            else:
                parts.append(json.dumps(item, ensure_ascii=False, indent=2))
            continue
        item_type = getattr(item, "type", "")
        if item_type == "text":
            parts.append(str(getattr(item, "text", "") or ""))
        elif item_type == "resource":
            parts.append(str(getattr(item, "resource", "") or ""))
        else:
            parts.append(dump_object(item))
    text = join_bounded(parts, raw_output_limit=raw_output_limit)
    return text or join_bounded([structured_content(result)], raw_output_limit=raw_output_limit)


def schema_props_required(schema: Json) -> tuple[Json, list[Any]]:
    """Extract a JSON-Schema object's `properties` dict and `required` list, tolerant of bad types."""
    props = schema.get("properties", {})
    required = schema.get("required", [])
    return (props if isinstance(props, dict) else {}, required if isinstance(required, list) else [])


def extract_uris(text: str, limit: int = 5) -> list[str]:
    """Pull resource-like URIs out of free text, deduped and lightly de-punctuated."""
    seen: list[str] = []
    for match in URI_PATTERN.findall(text or ""):
        uri = match.rstrip(".,;:")
        if uri not in seen:
            seen.append(uri)
        if len(seen) >= limit:
            break
    return seen


def resources_info(server: str, resources: list[Resource]) -> list[MCPResourceInfo]:
    """Snapshot discovered resources into MCPResourceInfo rows, dropping uri-less entries."""
    infos: list[MCPResourceInfo] = []
    for r in resources or []:
        uri = str(getattr(r, "uri", "") or "")
        if not uri:
            continue
        infos.append(
            MCPResourceInfo(
                server=server,
                uri=uri,
                name=str(getattr(r, "name", "") or ""),
                description=str(getattr(r, "description", "") or ""),
                mime_type=str(getattr(r, "mimeType", "") or ""),
            )
        )
    return infos


def markdown_cell(text: str) -> str:
    """Escape one table cell: single line, pipes backslash-escaped."""
    return Text.clean(str(text)).replace("\n", " ").replace("|", "\\|")
