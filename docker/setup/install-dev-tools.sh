#!/bin/bash
# Install developer tools.
set -euo pipefail

# Keep pre-commit in its own Python environment so its dependencies do not affect Isaac Sim.
PIPX_HOME=/opt/pipx PIPX_BIN_DIR=/usr/local/bin pipx install pre-commit

# Ensure the downloader is available before fetching GitHub's package signing key.
if ! command -v wget >/dev/null; then
    apt-get update
    apt-get install -y wget
fi
# Store the signing key and make it readable so apt can verify GitHub CLI packages.
mkdir -p -m 755 /etc/apt/keyrings
key_file=$(mktemp)
wget -nv -O "$key_file" https://cli.github.com/packages/githubcli-archive-keyring.gpg
cat "$key_file" > /etc/apt/keyrings/githubcli-archive-keyring.gpg
rm "$key_file"
chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
# Register GitHub's apt repository for this architecture, then install the gh command.
mkdir -p -m 755 /etc/apt/sources.list.d
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list
apt-get update
apt-get install -y gh

# Add a shortcut that waits for a debugger to connect on port 5678.
# The quoted SHELL marker writes the following lines literally into the shell configuration.
cat >> /etc/bash.bashrc <<'SHELL'
alias debugpy='python -Xfrozen_modules=off -m debugpy --listen localhost:5678 --wait-for-client'
SHELL
cp /etc/bash.bashrc /root/.bashrc
