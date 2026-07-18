"""Sphinx configuration for the nanocode documentation."""

from __future__ import annotations

import tomllib
from pathlib import Path

_pyproject = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text("utf-8"))
_meta = _pyproject["project"]

# -- Project information -----------------------------------------------------

project = "nanocode"
author = _meta["authors"][0]["name"]
copyright = f"{author}"
release = _meta["version"]
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]

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
    "cssfiles": ["_static/custom.css"],
    # Expand the whole navigation tree by default instead of collapsing sections.
    "globaltoc_collapse": False,
    "globaltoc_includehidden": True,
}
html_sidebars = {
    # globaltoc is expanded (globaltoc_collapse above), so it already shows the current
    # page's sub-sections — a separate localtoc would just duplicate the heading.
    "**": ["globaltoc.html", "nc_links.html", "searchbox.html"],
}
html_title = f"{project} {release}"
html_short_title = project
html_static_path = ["_static"]
html_js_files = ["scrollspy.js"]
html_show_sphinx = False
html_copy_source = False
html_show_sourcelink = False
pygments_style = "one-dark"

# -- Extension tuning --------------------------------------------------------

copybutton_prompt_text = r"\$ |>>> "
copybutton_prompt_is_regexp = True
