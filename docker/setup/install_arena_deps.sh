#!/bin/bash
set -euo pipefail
# Preserve group write access inherited from the shared Isaac Lab venv.
umask 0002

# Install the Arena package and developer dependencies from its frozen lock. The
# default dependency groups are disabled because Isaac Lab was already installed
# from its own lock; --inexact retains that editable workspace and its dependencies.
uv sync --project /tmp/arena-project --frozen --inexact \
    --no-default-groups --group openpi --extra dev --no-install-project
# uv preserves permissions from cached wheels, so repair cached files after installation.
find "${VIRTUAL_ENV}" \( -type f -o -type d \) \
    \( ! -group isaac-sim -o ! -perm -g+w \) \
    -exec chgrp isaac-sim {} + -exec chmod g+rwX {} +

# HuggingFace CLI for downloading datasets and models.
# Use pipx so the hf binary gets an isolated venv with all its deps (e.g. requests),
# without touching system Python packages.
apt-get update
apt-get install -y pipx
PIPX_HOME=/opt/pipx PIPX_BIN_DIR=/usr/local/bin pipx install "huggingface-hub[cli]"
