#!/bin/bash
set -e
DOCKER_IMAGE_NAME='isaac_arena'

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Default mount directory on the host machine for the datasets
DATASETS_HOST_MOUNT_DIRECTORY="$HOME/datasets"
# Default mount directory on the host machine for the models
MODELS_HOST_MOUNT_DIRECTORY="$HOME/models"
# Default mount directory on the host machine for the evaluation directory
EVAL_HOST_MOUNT_DIRECTORY="$HOME/eval"

while getopts ":d:m:e:hn:" OPTION; do
    case $OPTION in

        d)
            DATASETS_HOST_MOUNT_DIRECTORY=$OPTARG
            ;;
        m)
            MODELS_HOST_MOUNT_DIRECTORY=$OPTARG
            ;;
        e)
            EVAL_HOST_MOUNT_DIRECTORY=$OPTARG
            ;;
        n)
            DOCKER_IMAGE_NAME=${OPTARG}
            ;;
        h)
            echo "Helper script to build $DOCKER_IMAGE_NAME (default)"
            echo "Usage:"
            echo "$script_name -h"
            echo "$script_name -d <datasets directory>"
            echo "$script_name -m <models directory>"
            echo "$script_name -e <evaluation directory>"
            echo "$script_name -n <docker name>"
            echo ""
            echo "  -d <datasets directory> (default is $DATASETS_HOST_MOUNT_DIRECTORY)"
            echo "  -m <models directory> (default is $MODELS_HOST_MOUNT_DIRECTORY)"
            echo "  -e <evaluation directory> (default is $EVAL_HOST_MOUNT_DIRECTORY)"
            echo "  -n <docker name> (default is $DOCKER_IMAGE_NAME)"
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done

# Display the values being used
echo "Using Docker name: $DOCKER_IMAGE_NAME"

# This portion of the script will only be executed *inside* the docker when
# this script is used as entrypoint further down. It will setup an user account for
# the host user inside the docker s.t. created files will have correct ownership.
if [ -f /.dockerenv ]
then
    set -euo pipefail

    # Make sure that all shared libs are found. This should normally not be needed, but resolves a
    # problem with the opencv installation. For unknown reasons, the command doesn't bite if placed
    # at the end of the dockerfile
    ldconfig

    # Add the group of the user. User/group ID of the host user are set through env variables when calling docker run further down.
    groupadd --force --gid "$DOCKER_RUN_GROUP_ID" "$DOCKER_RUN_GROUP_NAME"

    # Re-add the user
    userdel "$DOCKER_RUN_USER_NAME" || true
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
    echo "Re-installing isaaclab packages from mounted repo"
    for DIR in /workspaces/isaac_arena/submodules/IsaacLab/source/isaaclab*/; do
        echo "Installing $DIR"
        pip install --no-deps -e "$DIR"
    done
    # Re-doing symlink (we do this in the Dockerfile, but we overwrite with the mapping).
    if [ ! -d "/workspaces/isaac_arena/submodules/IsaacLab/_isaac_sim" ]; then
        ln -s /isaac-sim/ /workspaces/isaac_arena/submodules/IsaacLab/_isaac_sim
    fi

    set +x

    su $DOCKER_RUN_USER_NAME

    exit
fi

# Build the Docker image with the specified or default name
docker build --pull -t ${DOCKER_IMAGE_NAME} --file $SCRIPT_DIR/Dockerfile.isaac_arena $SCRIPT_DIR/..

# Remove any exited containers
if [ "$(docker ps -a --quiet --filter status=exited --filter name=$DOCKER_IMAGE_NAME)" ]; then
    docker rm $DOCKER_IMAGE_NAME > /dev/null
fi

# If container is running, attach to it, otherwise start
if [ "$( docker container inspect -f '{{.State.Running}}' $DOCKER_IMAGE_NAME 2>/dev/null)" = "true" ]; then
  echo "Container already running. Attaching."
  docker exec -it $DOCKER_IMAGE_NAME su $(id -un)
else
    DOCKER_RUN_ARGS=("--name" "$DOCKER_IMAGE_NAME"
                    "--privileged"
                    "--ulimit" "memlock=-1"
                    "--ulimit" "stack=-1"
                    "--ipc=host"
                    "--net=host"
                    "--runtime=nvidia"
                    "--gpus=all"
                    "-v" ".:/workspaces/isaac_arena"
                    "-v" "$DATASETS_HOST_MOUNT_DIRECTORY:/datasets"
                    "-v" "$MODELS_HOST_MOUNT_DIRECTORY:/models"
                    "-v" "$EVAL_HOST_MOUNT_DIRECTORY:/eval"
                    "-v" "$HOME/.bash_history:/home/$(id -un)/.bash_history"
                    "-v" "$HOME/.config/osmo:/home/$(id -un)/.config/osmo"
                    "-v" "/tmp:/tmp"
                    "-v" "/tmp/.X11-unix:/tmp/.X11-unix:rw"
                    "-v" "/var/run/docker.sock:/var/run/docker.sock"
                    "-v" "$HOME/.Xauthority:/root/.Xauthority"
                    "--env" "DISPLAY"
                    "--env" "ACCEPT_EULA=Y"
                    "--env" "PRIVACY_CONSENT=Y"
                    "--env" "DOCKER_RUN_USER_ID=$(id -u)"
                    "--env" "DOCKER_RUN_USER_NAME=$(id -un)"
                    "--env" "DOCKER_RUN_GROUP_ID=$(id -g)"
                    "--env" "DOCKER_RUN_GROUP_NAME=$(id -gn)"
                    "--env" "OMNI_USER=\$omni-api-token"
                    "--env" "OMNI_PASS=$OMNI_PASS"
                    # Setting envs for XR: https://isaac-sim.github.io/IsaacLab/v2.1.0/source/how-to/cloudxr_teleoperation.html#run-isaac-lab-with-the-cloudxr-runtime
                    "--env" "XDG_RUNTIME_DIR=/workspaces/isaac_arena/submodules/IsaacLab/openxr/run"
                    "--env" "XR_RUNTIME_JSON=/workspaces/isaac_arena/submodules/IsaacLab/openxr/share/openxr/1/openxr_cloudxr.json"
                    # NOTE(alexmillane, 2025.07.23): This looks a bit suspect to me. We should be running
                    # as a user inside the container, not root. I've left it in for now, but we should
                    # remove it, if indeed it's not needed.
                    # "--env" "OMNI_KIT_ALLOW_ROOT=1"
                    "--entrypoint" "/workspaces/isaac_arena/docker/run_docker.sh"
                    )

    # Allow X11 connections
    xhost +local:docker

    docker run "${DOCKER_RUN_ARGS[@]}" --interactive --rm --tty ${DOCKER_IMAGE_NAME}
fi
