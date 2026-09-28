#!/bin/bash
set -euo pipefail

# Make interactive Python, pip, and pytest commands use Isaac Sim's Python environment.
# The quoted SHELL marker preserves the aliases and prompt text until a shell reads them.
cat >> /etc/bash.bashrc <<'SHELL'
alias python='/isaac-sim/python.sh'
alias pip3='/isaac-sim/python.sh -m pip'
alias pytest='/isaac-sim/python.sh -m pytest'
PS1='[IsaacLab Arena] \[\e[0;32m\]~\u \[\e[0;34m\]\w\[\e[0m\] \$ '
alias ll='ls -alF --color=auto'
alias ..='cd ..'
SHELL
# Apply the same defaults to root; the entrypoint also copies them to the host-matched user.
cp /etc/bash.bashrc /root/.bashrc
