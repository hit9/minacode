"""Write `llms.txt` and `llms-full.txt` into the HTML output.

Read the Docs serves both from the root of a project's default version but generates neither.
They are built from the Markdown sources rather than the rendered HTML: the sources are already
the plain text an LLM wants, and they carry no navigation, theme, or search markup to strip.

The one thing that does get rewritten is a `term-shot`, the raw HTML block the docs use for
terminal illustrations. Its span soup is noise, but its `aria-label` is a written-out description
of the same picture, so the label replaces the block.

Written for the English build only: a model does not need the translated copy, and Read the Docs
serves these files from a single version.

See https://docs.readthedocs.com/platform/latest/reference/llms-txt.html and https://llmstxt.org.
"""

from __future__ import annotations

import re
from pathlib import Path

from sphinx.util import logging

logger = logging.getLogger(__name__)

TERM_SHOT_RE = re.compile(r'<div class="term-shot"[^>]*aria-label="(?P<label>[^"]*)"[^>]*>.*?</div>', re.DOTALL)
FIGURE_RE = re.compile(r"```\{figure\}.*?\n```\n", re.DOTALL)
COMMENT_RE = re.compile(r"^\s*<!--.*?-->\s*$", re.MULTILINE | re.DOTALL)
INCLUDE_RE = re.compile(r"^```\{include\}\s*(?P<path>\S+)\s*\n```\s*$", re.MULTILINE)
INLINE_TAG_RE = re.compile(r"</?(?:span|em|strong|br|sup|sub)\b[^>]*>")
ANCHOR_RE = re.compile(r"^\(\S+\)=\s*$", re.MULTILINE)


def _page_order(app) -> list[str]:
    """Doc names in navigation order: the root, then whatever its toctrees include."""
    root = app.config.root_doc
    ordered = [root]
    for docname in app.env.toctree_includes.get(root, []):
        if docname not in ordered:
            ordered.append(docname)
    ordered += sorted(name for name in app.env.found_docs if name not in ordered)
    return ordered


def _source_text(app, docname: str) -> str:
    path = Path(app.env.doc2path(docname))
    if not path.exists():
        return ""
    text = path.read_text("utf-8")
    # An included file is the page: docs/changelog.md is one `include` of the project changelog.
    text = INCLUDE_RE.sub(lambda match: _included(path, match.group("path")), text)
    text = TERM_SHOT_RE.sub(lambda match: f"[terminal illustration] {match.group('label')}\n", text)
    text = FIGURE_RE.sub("", text)
    text = INLINE_TAG_RE.sub("", text)  # decorative spans; the words inside them are the content
    text = ANCHOR_RE.sub("", text)  # MyST target anchors address HTML, not readers
    return COMMENT_RE.sub("", text).strip()


def _included(source: Path, target: str) -> str:
    included = (source.parent / target).resolve()
    return included.read_text("utf-8") if included.exists() else ""


def _summary(text: str) -> str:
    """The page's first real sentence, for the index listing.

    Only the intro above the first section counts: a page that opens straight into a heading gets
    no summary rather than a sentence lifted from the middle of its first section."""
    intro = text.split("\n## ", 1)[0]
    for block in intro.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(("#", "```", "<", ":", "|", "-", "*", "(")):
            continue
        sentence = " ".join(block.split()).split(". ")[0].rstrip(".")
        return re.sub(r"<[^>]+>|[*`]", "", sentence)
    return ""


def _write(app, name: str, body: str) -> None:
    (Path(app.outdir) / name).write_text(body, "utf-8")
    logger.info("wrote %s (%d KB)", name, len(body.encode("utf-8")) // 1024)


def _generate(app, exception) -> None:
    # English only: these files exist for models, which do not need the translation, and Read the
    # Docs serves them from one version anyway.
    if exception is not None or app.builder.name != "html" or app.config.language not in ("en", None, ""):
        return
    base = (app.config.html_baseurl or "").rstrip("/")
    pages = [(name, _source_text(app, name)) for name in _page_order(app)]
    pages = [(name, text) for name, text in pages if text]

    def url(name: str, suffix: str = ".html") -> str:
        page = f"{name}{suffix}"
        return f"{base}/{page}" if base else page

    index = [f"# {app.config.project}", "", f"> {_summary(dict(pages).get(app.config.root_doc, ''))}.", "", "## Pages", ""]
    for name, text in pages:
        title = app.env.titles[name].astext() if name in app.env.titles else name
        summary = _summary(text)
        index.append(f"- [{title}]({url(name)})" + (f": {summary}" if summary else ""))
    index += ["", "## Optional", "", f"- [Full documentation as one file]({url('llms-full', '.txt')})", ""]
    _write(app, "llms.txt", "\n".join(index))

    full = [f"# {app.config.project} {app.config.release} — full documentation", ""]
    for name, text in pages:
        full += [f"<!-- source: {name}.md · {url(name)} -->", "", text, ""]
    _write(app, "llms-full.txt", "\n".join(full))


def setup(app):
    app.connect("build-finished", _generate)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
