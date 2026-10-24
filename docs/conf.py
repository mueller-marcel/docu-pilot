"""Sphinx configuration of the DocuPilot technical documentation."""

from __future__ import annotations

import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PROJECT = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

# ── Project ──────────────────────────────────────────────────────────────────

project = "DocuPilot"
author = "Marcel Müller"
copyright = f"2026, {author}"
release = _PROJECT["version"]
version = release
language = "de"

# ── General ──────────────────────────────────────────────────────────────────

extensions = [
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3
myst_fence_as_directive = ["mermaid"]

# ── API reference (sphinx-autoapi) ───────────────────────────────────────────
#
# AutoAPI reads the sources statically. The package is never imported, so the
# reference builds without Qt, ffmpeg, Whisper or an API key.

autoapi_type = "python"
autoapi_dirs = [str(_ROOT / "src" / "docupilot")]
autoapi_root = "api"
autoapi_add_toctree_entry = False
autoapi_keep_files = False
autoapi_member_order = "bysource"
autoapi_python_class_content = "both"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autodoc_typehints = "description"

# The docstrings are prose written for readers of the code: their emphasis
# markers and indented blocks are not reST. Their rendering is kept as is
# rather than letting docstring style break the build.
suppress_warnings = ["autoapi.python_import_resolution", "docutils"]

# ── Cross references ─────────────────────────────────────────────────────────

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# ── HTML ─────────────────────────────────────────────────────────────────────

html_theme = "furo"
html_title = f"DocuPilot {release}"
html_static_path: list[str] = []
html_theme_options = {
    "source_repository": "https://github.com/mueller-marcel/docu-pilot/",
    "source_branch": "main",
    "source_directory": "docs/",
}
copybutton_prompt_text = r"^\$ |^> "
copybutton_prompt_is_regexp = True
