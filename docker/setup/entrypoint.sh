#!/bin/bash

# This script is used as entrypoint for the docker container.
# It will setup an user account for the host user inside the docker
# s.t. created files will have correct ownership.
set -euo pipefail

# Make sure that all shared libs are found. This should normally not be needed, but resolves a
# problem with the opencv installation. For unknown reasons, the command doesn't bite if placed
# at the end of the dockerfile
ldconfig

# Add the group of the user. User/group ID of the host user are set through env variables when calling docker run further down.
groupadd --force --gid "$DOCKER_RUN_GROUP_ID" "$DOCKER_RUN_GROUP_NAME"

# Re-add the user
userdel "$DOCKER_RUN_USER_NAME" 2>/dev/null || true
userdel ubuntu || true
useradd --no-log-init \
        --uid "$DOCKER_RUN_USER_ID" \
        --gid "$DOCKER_RUN_GROUP_NAME" \
        --groups sudo \
        --shell /bin/bash \
        $DOCKER_RUN_USER_NAME
chown $DOCKER_RUN_USER_NAME /home/$DOCKER_RUN_USER_NAME

# Change the root user password (so we can su root)
echo 'root:root' | chpasswd
echo "$DOCKER_RUN_USER_NAME:root" | chpasswd

# Allow sudo without password
echo "$DOCKER_RUN_USER_NAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Re-install isaaclab (note that the deps have been installed in the Dockerfile)
echo "Re-installing isaaclab packages from mounted repo..."
for DIR in /workspaces/isaac_arena/submodules/IsaacLab/source/isaaclab*/; do
    echo "Installing $DIR"
    pip install --root-user-action=ignore --quiet --no-deps -e "$DIR"
done
# Re-doing symlink (we do this in the Dockerfile, but we overwrite with the mapping).
if [ ! -d "/workspaces/isaac_arena/submodules/IsaacLab/_isaac_sim" ]; then
    ln -s /isaac-sim/ /workspaces/isaac_arena/submodules/IsaacLab/_isaac_sim
fi

# change prompt so it's obvious we're inside the arena container
echo "PS1='[Isaac Arena] \[\e[0;32m\]~\u \[\e[0;34m\]\w\[\e[0m\] \$ '" >> /home/$DOCKER_RUN_USER_NAME/.bashrc

# useful aliases:
# list files as a table, with colors, show hidden, append indicators (/ for dirs, * for executables, @ for symlinks)
echo "alias ll='ls -alF --color=auto'" >> /home/$DOCKER_RUN_USER_NAME/.bashrc
# go one level up
echo "alias ..='cd ..'" >> /home/$DOCKER_RUN_USER_NAME/.bashrc

set +x

# Suppress sudo hint message
touch /home/$DOCKER_RUN_USER_NAME/.sudo_as_admin_successful

su $DOCKER_RUN_USER_NAME

exit
