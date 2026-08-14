"""Sphinx configuration for the minacode documentation."""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))

_pyproject = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text("utf-8"))
_meta = _pyproject["project"]

# -- Project information -----------------------------------------------------

project = "minacode"
author = _meta["authors"][0]["name"]
copyright = f"{author}"
release = _meta["version"]
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    # Local: writes llms.txt and llms-full.txt beside the HTML, which Read the Docs serves at the
    # domain root of the default version.
    "llms_txt",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
locale_dirs = ["locale/"]
gettext_compact = False

myst_enable_extensions = ["colon_fence", "deflist", "linkify", "substitution"]
myst_heading_anchors = 3

# -- HTML output -------------------------------------------------------------

import better  # noqa: E402

html_theme = "better"
html_theme_path = [better.better_theme_path]
html_theme_options = {
    "showheader": False,
    "showrelbartop": False,
    "showrelbarbottom": True,
    "sidebarwidth": "16rem",
    # Show every page's major sections while keeping third-level headings out
    # of the global navigation.
    "globaltoc_collapse": False,
    "globaltoc_includehidden": True,
    "globaltoc_maxdepth": 2,
}
html_sidebars = {
    # The global TOC shows the current page's sections, so a separate local TOC
    # would duplicate them.
    "**": ["globaltoc.html", "searchbox.html"],
}
# Read the Docs publishes the canonical URL; it makes the canonical <link> correct and gives
# llms.txt absolute links to the pages it lists.
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
html_title = f"{project} {release}"
html_short_title = project
html_static_path = ["_static"]
# Loaded via html_css_files (not the theme's cssfiles option) so Sphinx appends a
# ?v=<checksum> cache buster; without it browsers keep a stale custom.css.
html_css_files = ["custom.css"]
html_js_files = ["scrollspy.js"]
html_show_sphinx = False
html_copy_source = False
html_show_sourcelink = False
pygments_style = "one-dark"

# -- Extension tuning --------------------------------------------------------

copybutton_prompt_text = r"\$ |>>> "

copybutton_prompt_is_regexp = True


def _add_language_context(app, _pagename, _templatename, context, _doctree):
    context["doc_language"] = app.config.language.replace("-", "_")


def setup(app):
    app.connect("html-page-context", _add_language_context)
