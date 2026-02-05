#!/bin/bash

# This script is used as entrypoint for the docker container.
# It will setup an user account for the host user inside the docker
# s.t. created files will have correct ownership.

# Exit on error
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
        --groups sudo,isaac-sim \
        --shell /bin/bash \
        $DOCKER_RUN_USER_NAME
chown $DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME /home/$DOCKER_RUN_USER_NAME
chown $DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME $WORKDIR

# Change the root user password (so we can su root)
echo 'root:root' | chpasswd
echo "$DOCKER_RUN_USER_NAME:root" | chpasswd

# Allow sudo without password
echo "$DOCKER_RUN_USER_NAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Suppress sudo hint message
touch /home/$DOCKER_RUN_USER_NAME/.sudo_as_admin_successful

cp /etc/bash.bashrc /home/$DOCKER_RUN_USER_NAME/.bashrc
chown $DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME /home/$DOCKER_RUN_USER_NAME/.bashrc

# Add the models, datasets, and eval folders if they don't exist
mkdir -p /datasets /models /eval
chown $DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME /datasets /models /eval

# Create the _isaac_sim symlink if it doesn't exist
if [ ! -e "$WORKDIR/submodules/IsaacLab/_isaac_sim" ]; then
    ln -s /isaac-sim/ "$WORKDIR/submodules/IsaacLab/_isaac_sim"
fi

# Run the passed command or just start the shell as the created user
if [ $# -ge 1 ]; then
    echo "alias pytest='/isaac-sim/python.sh -m pytest'" >> /etc/aliasess.bashrc
    # -i makes bash to expand aliases
    # -c makes bash to run a command
    exec sudo --preserve-env -u $DOCKER_RUN_USER_NAME \
        -- env HOME=/home/$DOCKER_RUN_USER_NAME bash -ic "$@"
else
    su $DOCKER_RUN_USER_NAME
fi

exit
