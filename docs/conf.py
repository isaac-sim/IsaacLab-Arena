# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# pylint: disable=redefined-builtin

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import datetime
from typing import List
import os
import sys

# TODO(alexmillane, 2025-10-03): Currently we can't import the version number.
# Modify PYTHONPATH so we can obtain the version data from setup module.
# pylint: disable=wrong-import-position
# path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from setup import ISAACLAB_ARENA_VERSION_NUMBER

# TODO(alexmillane, 2025-10-03): Get this programmatically, as above.
ISAACLAB_ARENA_VERSION_NUMBER = "0.1"


# Modify PYTHONPATH so we can import the helpers module.
sys.path.insert(0, os.path.abspath("."))
from helpers import TemporaryLinkcheckIgnore, to_datetime, is_expired

# NOTE(alexmillane, 2025-04-24): This file is in a separate folder to avoid
# duplicate configuration errors coming from mypy. The only way I could find
# to solve this was to add this new folder.

# -- Project information -----------------------------------------------------

project = "isaaclab_arena"
copyright = "2025, NVIDIA"
author = "NVIDIA"
released = False  # Indicates if this is a public or internal version of the repo.

# -- General configuration ---------------------------------------------------

sys.path.append(os.path.abspath("_ext"))

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "sphinx_tabs.tabs",
    "sphinx_design",
    "sphinx_copybutton",
    "isaaclab_arena_doc_tools",
]

# put type hints inside the description instead of the signature (easier to read)
# autodoc_typehints = 'description'
# document class *and* __init__ methods
# autoclass_content = 'both'    #

todo_include_todos = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "venv_docs"]

# Be picky about missing references
nitpicky = True  # warns on broken references
nitpick_ignore: list[str] = []  # can exclude known bad refs

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "nvidia_sphinx_theme"
html_title = f"isaaclab_arena {ISAACLAB_ARENA_VERSION_NUMBER}"
html_show_sphinx = False
html_theme_options = {
    "copyright_override": {"start": 2023},
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "footer_links": {},
    "github_url": "https://github.com/isaac-sim/IsaacLab-Arena",
    # TODO(alexmillane, 2025-04-24): Try re-enabling this once we have a pypi page.
    # "icon_links": [
    #     {
    #         "name": "PyPI",
    #         "url": "https://pypi.org/project/isaaclab_arena",
    #         "icon": "fa-brands fa-python",
    #         "type": "fontawesome",
    #     },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = []
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Todos
todo_include_todos = True

# Linkcheck
# NOTE(alexmillane, 2025-05-09): The links in the main example page are relative links
# which are only valid post-build. linkcheck doesn't like this. So here we ignore
# links to the example pages via html.
linkcheck_ignore = [
    # r'pages/torch_examples_.*\.html',    # Ignore all pages/torch_examples_*.html links
]

temporary_linkcheck_ignore = [
    # TemporaryLinkcheckIgnore(
    #     url='https://3dmatch.cs.princeton.edu/',
    #     start_date=to_datetime('09.07.2025'),
    #     days=14,
    # ),
]

for ignore in temporary_linkcheck_ignore:
    if not is_expired(ignore.start_date, ignore.days):
        print(f"Ignoring {ignore.url} until {ignore.start_date + datetime.timedelta(days=ignore.days)}")
        linkcheck_ignore.append(ignore.url)

#####################################
#  Macros dependent on release state
#####################################

isaaclab_arena_docs_config = {
    "released": released,
    "internal_git_url": "git@github.com:isaac-sim/IsaacLab-Arena.git",
    "external_git_url": "UNDECIDED",
    "internal_code_link_base_url": "https://github.com/isaac-sim/IsaacLab-Arena",
    "external_code_link_base_url": "UNDECIDED",
}
