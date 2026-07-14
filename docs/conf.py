"""Sphinx configuration for the ITASC documentation.

The narrative pages are authored in Markdown (MyST); the API reference is
generated from the package's own docstrings via autosummary, so it cannot drift
from the code. See ``development/maintaining-docs.md`` for the maintenance model.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

# Prefer the installed (editable) package, but fall back to the source tree so
# the docs can build from a bare checkout.
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information ------------------------------------------------------
project = "ITASC"
author = "Artur Ruppel"
copyright = f"{datetime.now():%Y}, {author}"

try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("itasc")
except Exception:  # pragma: no cover - best effort during local builds
    release = "0.2.0"
version = release

# -- General configuration ----------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "superpowers", "plans"]

# -- MyST (Markdown) ----------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# -- autodoc / autosummary ----------------------------------------------------
# The whole point of the maintenance model: the API tree is *generated* from
# docstrings at build time, never hand-written.
autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Heavy / optional / GUI dependencies are mocked so the docs build without a
# display or the [all] extras. The API reference intentionally covers the
# programmatic subpackages, not the napari UI layer.
autodoc_mock_imports = [
    "napari",
    "qtpy",
    "pyqtgraph",
    "cellpose",
    "torch",
    "torchvision",
    "ultrack",
    "maxflow",
    "pymaxflow",
    "numba",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Intersphinx --------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# -- HTML output --------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "ITASC"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/ArturRuppel/ITASC",
    "navbar_align": "content",
    "show_prev_next": True,
    "navigation_with_keys": True,
}

# Cross-references that don't resolve should be visible but not (yet) fatal.
# Flip ``nitpicky`` on once the baseline is clean; the CI ``-W`` gate is what
# turns documentation drift into a failing check.
nitpicky = False
