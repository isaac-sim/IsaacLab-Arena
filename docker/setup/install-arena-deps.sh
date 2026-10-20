#!/bin/bash
set -euo pipefail

# Install the Arena package and developer dependencies from its frozen lock. The
# default dependency groups are disabled because Isaac Lab was already installed
# from its own lock; --inexact retains that editable workspace and its dependencies.
uv sync --project /tmp/arena-project --frozen --inexact \
    --no-default-groups --extra dev --no-install-project
# A legacy transitive dependency installs the Python 2 backport over Python
# 3.12's standard-library argparse and breaks pytest during parser creation.
uv pip uninstall argparse

# HuggingFace CLI for downloading datasets and models.
# Use pipx so the hf binary gets an isolated venv with all its deps (e.g. requests),
# without touching system Python packages.
apt-get update
apt-get install -y pipx
PIPX_HOME=/opt/pipx PIPX_BIN_DIR=/usr/local/bin pipx install "huggingface-hub[cli]"
