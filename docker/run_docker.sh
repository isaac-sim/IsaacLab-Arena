#!/bin/bash
set -e
DOCKER_IMAGE_NAME='isaaclab_arena'
DOCKER_VERSION_TAG='latest'

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

WORKDIR="/workspaces/isaaclab_arena"

# Default mount directory on the host machine for the datasets
DATASETS_HOST_MOUNT_DIRECTORY="$HOME/datasets"
# Default mount directory on the host machine for the models
MODELS_HOST_MOUNT_DIRECTORY="$HOME/models"
# Default mount directory on the host machine for the evaluation directory
EVAL_HOST_MOUNT_DIRECTORY="$HOME/eval"
# Default GR00T installation settings (false means no GR00T installation)
INSTALL_GROOT="false"
GROOT_DEPS_GROUP="base"
# Whether to forcefully rebuild the docker image
# (it takes a while to re-build, but for testing is not really necessary)
FORCE_REBUILD=false

while getopts ":d:m:e:hn:rn:Rn:vn:gn:G:" OPTION; do
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
        r)
            FORCE_REBUILD=true
            ;;

        R)
            FORCE_REBUILD=true
            NO_CACHE="--no-cache"
            ;;
        v)
            set -x
            ;;
        g)
            INSTALL_GROOT="true"
            DOCKER_VERSION_TAG='cuda_gr00t'
            ;;
        G)
            GROOT_DEPS_GROUP=${OPTARG}
            INSTALL_GROOT="true"
            ;;
        h)
            script_name=$(basename "$0")
            echo "Helper script to build and IsaacLab Arena docker environment."
            echo ""
            echo "Usage:"
            echo "$script_name [options]"
            echo ""
            echo "Options:"
            echo "  -v (Verbose output)"
            echo "  -d <datasets directory> (Path to datasets on the host. Default is \"$DATASETS_HOST_MOUNT_DIRECTORY\".)"
            echo "  -m <models directory> (Path to models on the host. Default is \"$MODELS_HOST_MOUNT_DIRECTORY\".)"
            echo "  -e <evaluation directory> (Path to evaluation data on the host. Default is \"$EVAL_HOST_MOUNT_DIRECTORY\".)"
            echo "  -n <docker name> (Name of the docker image that will be built or used. Default is \"$DOCKER_IMAGE_NAME\".)"
            echo "  -r (Force rebuilding of the docker image.)"
            echo "  -R (Force rebuilding of the docker image, without cache.)"
            echo "  -g (Install GR00T with base dependencies.)"
            echo "  -G <deps_group> (Install GR00T with dependency group: base, dev, orin, thor, deploy.)"
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

# Shift off the processed options so that $@ has a command to pass to docker run
shift $((OPTIND-1))

# Display the values being used
echo "Using Docker image: $DOCKER_IMAGE_NAME:$DOCKER_VERSION_TAG"

# Build the Docker image with the specified or default name
echo "Building Docker image with GR00T installation: $INSTALL_GROOT"
if [ "$INSTALL_GROOT" = "true" ]; then
    echo "GR00T dependency group: $GROOT_DEPS_GROUP"
fi

if [ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_VERSION_TAG 2> /dev/null)" ] && \
    [ "$FORCE_REBUILD" = false ]; then
    echo "Docker image $DOCKER_IMAGE_NAME:$DOCKER_VERSION_TAG already exists. Not rebuilding."
    echo "Use -r option to force the rebuild."
else
    docker build --pull \
        $NO_CACHE \
        --build-arg WORKDIR="${WORKDIR}" \
        --build-arg INSTALL_GROOT=$INSTALL_GROOT \
        --build-arg GROOT_DEPS_GROUP=$GROOT_DEPS_GROUP \
        -t ${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG} \
        --file $SCRIPT_DIR/Dockerfile.isaaclab_arena \
        $SCRIPT_DIR/..
fi

# Remove any exited containers
if [ "$(docker ps -a --quiet --filter status=exited --filter name=$DOCKER_IMAGE_NAME-$DOCKER_VERSION_TAG)" ]; then
    docker rm $DOCKER_IMAGE_NAME-$DOCKER_VERSION_TAG > /dev/null
fi

# If container is running, attach to it, otherwise start
if [ "$( docker container inspect -f '{{.State.Running}}' $DOCKER_IMAGE_NAME'-'$DOCKER_VERSION_TAG 2>/dev/null)" = "true" ]; then
  echo "Container already running. Attaching."
  docker exec -it $DOCKER_IMAGE_NAME-$DOCKER_VERSION_TAG su $(id -un)
else
    DOCKER_RUN_ARGS=("--name" "$DOCKER_IMAGE_NAME-$DOCKER_VERSION_TAG"
                    "--privileged"
                    "--ulimit" "memlock=-1"
                    "--ulimit" "stack=-1"
                    "--ipc=host"
                    "--net=host"
                    "--runtime=nvidia"
                    "--gpus=all"
                    "-v" "./docs:${WORKDIR}/docs"
                    "-v" "./isaaclab_arena:${WORKDIR}/isaaclab_arena"
                    "-v" "./submodules/IsaacLab:${WORKDIR}/submodules/IsaacLab"
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
                    # Setting envs for XR: https://isaac-sim.github.io/IsaacLab/v2.1.0/source/how-to/cloudxr_teleoperation.html#run-isaac-lab-with-the-cloudxr-runtime
                    "--env" "XDG_RUNTIME_DIR=${WORKDIR}/submodules/IsaacLab/openxr/run"
                    "--env" "XR_RUNTIME_JSON=${WORKDIR}/submodules/IsaacLab/openxr/share/openxr/1/openxr_cloudxr.json"
                    # NOTE(alexmillane, 2025.07.23): This looks a bit suspect to me. We should be running
                    # as a user inside the container, not root. I've left it in for now, but we should
                    # remove it, if indeed it's not needed.
                    # "--env" "OMNI_KIT_ALLOW_ROOT=1"
                    "--env" "ISAACLAB_PATH=${WORKDIR}/submodules/IsaacLab"
                    )

    # map omniverse auth or config so we have connection to the dev nucleus
    if [ -n "$OMNI_PASS" ]; then
        DOCKER_RUN_ARGS+=("--env" "OMNI_USER=\$omni-api-token")
        DOCKER_RUN_ARGS+=("--env" "OMNI_PASS=$OMNI_PASS")
    else
        if [ -d "$HOME/.nvidia-omniverse" ]; then
            DOCKER_RUN_ARGS+=("-v" "$HOME/.nvidia-omniverse:/home/$(id -un)/.nvidia-omniverse")
        fi
    fi

    # if gr00t is installed, mount the gr00t directory in case anything needs to change there
    if [ "$INSTALL_GROOT" = "true" ]; then
        DOCKER_RUN_ARGS+=("-v" "./submodules/Isaac-GR00T:${WORKDIR}/submodules/Isaac-GR00T")
    fi
    # Allow X11 connections
    xhost +local:docker > /dev/null

    docker run "${DOCKER_RUN_ARGS[@]}" --interactive --rm --tty ${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG} "${@}"
fi
