---
name: arena-install
description: Scaffold a new external IsaacLab-Arena project — creates directory structure, pyproject.toml, Dockerfile, and run_docker.sh from a spec file.
---

## How to invoke

```
/arena-install /path/to/spec.json
```

The argument is the path to a filled-in spec file. Start from the template at
`.claude/skills/arena-spec-template.json` in the IsaacLab-Arena repo.

---

## Step 1 — Read the spec

Read the JSON file at the path given in the argument. Extract:

| Variable | JSON field | Derived if absent |
|---|---|---|
| `project_name` | `project_name` | — |
| `base_dir` | `base_dir` | — |
| `extra_mounts` | `extra_mounts` | `[]` |

Derived (never in spec):
- `PROJECT_DIR` = `<base_dir>/<project_name>`
- `PKG` = `<PROJECT_DIR>/<project_name>`
- `CONTAINER_NAME` = `<project_name>-latest`
- `CONTAINER_PROJECT_PATH` = `/workspaces/<project_name>`

---

## Step 2 — Create directory structure

```bash
mkdir -p <PKG>/{environments,embodiments,tasks,assets}
mkdir -p <PROJECT_DIR>/docker
cd <PROJECT_DIR> && git init
touch <PKG>/__init__.py
touch <PKG>/environments/__init__.py
touch <PKG>/embodiments/__init__.py
touch <PKG>/tasks/__init__.py
touch <PKG>/assets/__init__.py
touch <PKG>/assets/asset_paths.py
touch <PKG>/assets/object_library.py
touch <PKG>/assets/background_library.py
```

---

## Step 3 — Write `pyproject.toml`

Write to `<PROJECT_DIR>/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "<project_name>"
version = "0.1.0"
description = "<project_name> IsaacLab-Arena external environment"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["."]
include = ["<project_name>*"]

[tool.black]
line-length = 120

[tool.isort]
profile = "black"
line_length = 120

[tool.flake8]
max-line-length = 120
```

---

## Step 4 — Write `docker/Dockerfile`

Write to `<PROJECT_DIR>/docker/Dockerfile`.

This Dockerfile mirrors the franka-ultrasound pattern: it builds FROM the Isaac Sim base image,
installs IsaacLab and Arena from the submodule, then installs this package.

```dockerfile
ARG BASE_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.0-dev2

FROM ${BASE_IMAGE}

USER root

ARG WORKDIR="/workspaces/isaaclab_arena"
ENV WORKDIR=${WORKDIR}
WORKDIR "${WORKDIR}"

# Hide conflicting Vulkan files if present
RUN if [ -e "/usr/share/vulkan" ] && [ -e "/etc/vulkan" ]; then \
      mv /usr/share/vulkan /usr/share/vulkan_hidden; \
    fi

RUN apt-get update && apt-get install -y \
  git git-lfs cmake sudo python3-pip

################################
# Install Isaac Lab
################################
COPY submodules/IsaacLab-Arena/submodules/IsaacLab ${WORKDIR}/submodules/IsaacLab
ENV ISAACLAB_PATH=${WORKDIR}/submodules/IsaacLab
ENV TERM=xterm
RUN ln -s /isaac-sim/ ${WORKDIR}/submodules/IsaacLab/_isaac_sim
RUN for DIR in ${WORKDIR}/submodules/IsaacLab/source/isaaclab*/; do \
      /isaac-sim/python.sh -m pip install --no-deps -e "$DIR"; \
    done
RUN chmod 777 -R /isaac-sim/kit/ && chmod a+x /isaac-sim
RUN /isaac-sim/python.sh -m pip install --no-deps -e ${WORKDIR}/submodules/IsaacLab/source/isaaclab_visualizers
RUN /isaac-sim/python.sh -m pip install --no-deps -e ${WORKDIR}/submodules/IsaacLab/source/isaaclab_teleop
RUN ${ISAACLAB_PATH}/isaaclab.sh -i
RUN /isaac-sim/python.sh -m pip install isaacteleop~=1.0 --extra-index-url https://pypi.nvidia.com
RUN /isaac-sim/python.sh -m pip install --upgrade pip

################################
# Install IsaacLab Arena
################################
COPY submodules/IsaacLab-Arena ${WORKDIR}
RUN /isaac-sim/python.sh -m pip install -e "${WORKDIR}/[dev]"

################################
# Install <project_name>
################################
COPY pyproject.toml    /workspaces/<project_name>/pyproject.toml
COPY <project_name>    /workspaces/<project_name>/<project_name>
RUN /isaac-sim/python.sh -m pip install -e /workspaces/<project_name>

# Shell aliases
RUN echo "alias python='/isaac-sim/python.sh'"      >> /etc/bash.bashrc && \
    echo "alias pip3='/isaac-sim/python.sh -m pip'" >> /etc/bash.bashrc && \
    echo "alias pytest='/isaac-sim/python.sh -m pytest'" >> /etc/bash.bashrc && \
    echo "PS1='[<project_name>] \[\e[0;32m\]~\u \[\e[0;34m\]\w\[\e[0m\] \$ '" >> /etc/bash.bashrc && \
    cp /etc/bash.bashrc /root/.bashrc

COPY submodules/IsaacLab-Arena/docker/setup/entrypoint.sh /entrypoint.sh
RUN sed -i 's/useradd --no-log-init/useradd --no-log-init --non-unique/' /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
```

---

## Step 5 — Write `run_docker.sh`

Write to `<PROJECT_DIR>/run_docker.sh` and make it executable.

Build the `extra_mounts` block from the spec's `extra_mounts` array:
each entry becomes a `-v "<host>:<container>" \` line.

```bash
#!/bin/bash
# Build and run the <project_name> container.
#
# Usage:
#   ./run_docker.sh        # build (if needed) and start / attach
#   ./run_docker.sh -r     # force rebuild
#   ./run_docker.sh -R     # force rebuild without cache
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

IMAGE_NAME="<project_name>"
CONTAINER_NAME="<project_name>-latest"
WORKDIR="/workspaces/isaaclab_arena"
FORCE_REBUILD=false
NO_CACHE=""

while getopts ":rR" OPTION; do
    case $OPTION in
        r) FORCE_REBUILD=true ;;
        R) FORCE_REBUILD=true; NO_CACHE="--no-cache" ;;
        *) echo "Usage: $0 [-r|-R]"; exit 1 ;;
    esac
done

# Attach if already running.
if [ "$(docker container inspect -f '{{.State.Running}}' ${CONTAINER_NAME} 2>/dev/null)" = "true" ]; then
    echo "Container already running — attaching."
    docker exec -it ${CONTAINER_NAME} su "$(id -un)"
    exit 0
fi

# Build image if missing or rebuild requested.
if [ -z "$(docker images -q ${IMAGE_NAME}:latest 2>/dev/null)" ] || [ "${FORCE_REBUILD}" = true ]; then
    echo "Building ${IMAGE_NAME}:latest ..."
    docker build --pull \
        ${NO_CACHE} \
        --progress=plain \
        -t ${IMAGE_NAME}:latest \
        -f "${SCRIPT_DIR}/docker/Dockerfile" \
        "${SCRIPT_DIR}"
fi

xhost +local:docker > /dev/null 2>&1 || true

docker run \
    --name "${CONTAINER_NAME}" \
    --privileged \
    --ulimit memlock=-1 \
    --ulimit stack=-1 \
    --ipc=host \
    --net=host \
    --runtime=nvidia \
    --gpus=all \
    -v "${SCRIPT_DIR}/submodules/IsaacLab-Arena:${WORKDIR}" \
    -v "${SCRIPT_DIR}:/workspaces/<project_name>" \
    <EXTRA_MOUNTS>
    -v "$HOME/.bash_history:/home/$(id -un)/.bash_history" \
    -v "$HOME/.cache:/home/$(id -un)/.cache" \
    -v "$HOME/.nvidia-omniverse:/home/$(id -un)/.nvidia-omniverse" \
    -v "/tmp:/tmp" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "/var/run/docker.sock:/var/run/docker.sock" \
    -v "/etc/ssl/certs:/etc/ssl/certs:ro" \
    --env DISPLAY \
    --env ACCEPT_EULA=Y \
    --env PRIVACY_CONSENT=Y \
    --env DOCKER_RUN_USER_ID="$(id -u)" \
    --env DOCKER_RUN_USER_NAME="$(id -un)" \
    --env DOCKER_RUN_GROUP_ID="$(id -g)" \
    --env DOCKER_RUN_GROUP_NAME="$(id -gn)" \
    --env ISAACLAB_PATH="${WORKDIR}/submodules/IsaacLab" \
    --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --interactive --rm --tty \
    ${IMAGE_NAME}:latest
```

Replace `<EXTRA_MOUNTS>` with one `-v "<host>:<container>" \` line per entry in `extra_mounts`.
If `extra_mounts` is empty, remove the placeholder entirely.

After writing, make executable:
```bash
chmod +x <PROJECT_DIR>/run_docker.sh
```

---

## Step 6 — Add IsaacLab-Arena as a git submodule

```bash
cd <PROJECT_DIR>
git submodule add git@github.com:isaac-sim/IsaacLab-Arena.git submodules/IsaacLab-Arena
git submodule update --init --recursive
```

If this fails (no SSH key or network), skip it and tell the user — the submodule is required to build the Docker image.

---

## Verify

```bash
ls <PROJECT_DIR>/run_docker.sh \
   <PROJECT_DIR>/docker/Dockerfile \
   <PROJECT_DIR>/pyproject.toml \
   <PKG>/__init__.py
```

All four paths must exist. Report: `arena-install complete — <PROJECT_DIR> scaffolded.`
