# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sphinx configuration for the generated Isaac Lab Arena API reference (static analysis, no imports)."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

project = "isaaclab_arena API reference"
copyright = "NVIDIA"
author = "NVIDIA"

PACKAGES = [
    "isaaclab_arena",
    "isaaclab_arena_benchmarking",
    "isaaclab_arena_cosmos",
    "isaaclab_arena_curobo",
    "isaaclab_arena_dreamzero",
    "isaaclab_arena_environments",
    "isaaclab_arena_examples",
    "isaaclab_arena_g1",
    "isaaclab_arena_gr00t",
    "isaaclab_arena_openpi",
    "isaaclab_arena_vlm_agent",
]

extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "source_links",
]

# sphinx-autoapi parses the sources statically and never imports them, so Isaac Sim is not needed.
autoapi_type = "python"
# Implicit namespaces keep directories without __init__.py in the package tree.
autoapi_dirs = [os.path.join(REPO_ROOT, p) for p in PACKAGES]
autoapi_python_use_implicit_namespaces = True
autoapi_root = "api"
autoapi_add_toctree_entry = True
autoapi_keep_files = False
autoapi_python_class_content = "both"  # class docstring + __init__ docstring (where the Args live)
autoapi_member_order = "bysource"
autoapi_ignore = ["*/tests/*", "*/test/*", "*/conftest.py", "*/__pycache__/*"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-module-summary",
]
suppress_warnings = ["autoapi.python_import_resolution", "ref.python", "toc.not_included"]

# Source links appended to every documented object (see source_links.py).
source_link_repo_root = REPO_ROOT
source_link_base_url = os.environ.get("API_DOCS_SOURCE_URL", "https://github.com/isaac-sim/IsaacLab-Arena/blob/main")

html_theme = "nvidia_sphinx_theme"
html_title = project
html_show_sphinx = False
exclude_patterns = ["_build"]
