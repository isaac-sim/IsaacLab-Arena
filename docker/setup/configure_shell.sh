#!/bin/bash
set -euo pipefail

# Keep the Lab launcher available when su or sudo resets PATH. Execute the original
# script: copying or symlinking it would break its BASH_SOURCE-based repo discovery.
# Bake in the image's paths as fallbacks for shells that clear the environment.
cat > /usr/local/bin/isaaclab.sh <<LAUNCHER
#!/bin/sh
set -eu
export VIRTUAL_ENV="\${VIRTUAL_ENV:-${VIRTUAL_ENV}}"
export PATH="\$VIRTUAL_ENV/bin:\$PATH"
exec "\${ISAACLAB_PATH:-${ISAACLAB_PATH}}/isaaclab.sh" "\$@"
LAUNCHER
chmod 755 /usr/local/bin/isaaclab.sh

# Make interactive Python and pytest commands initialize Isaac Sim around the Lab venv.
# The quoted SHELL marker preserves the aliases and prompt text until a shell reads them.
cat >> /etc/bash.bashrc <<'SHELL'
alias python='isaaclab.sh -p'
alias python3='isaaclab.sh -p'
alias pip='uv pip'
alias pip3='uv pip'
alias pytest='isaaclab.sh -p -m pytest'
PS1='[IsaacLab Arena] \[\e[0;32m\]~\u \[\e[0;34m\]\w\[\e[0m\] \$ '
alias ll='ls -alF --color=auto'
alias ..='cd ..'
SHELL
# Apply the same defaults to root; the entrypoint also copies them to the host-matched user.
cp /etc/bash.bashrc /root/.bashrc
