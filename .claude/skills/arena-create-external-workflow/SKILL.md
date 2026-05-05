---
name: arena-create-external-workflow
description: Create a new external IsaacLab-Arena project — creates directory structure, pyproject.toml, Dockerfile, run_docker.sh, and entrypoint.sh, then launches the docker container. Use this skill whenever the user wants to create a new project, package, or workflow that builds on top of IsaacLab-Arena, even if they don't say "create" or "external workflow" — any request like "start a new Arena-based project", "set up my custom environment package", or "create a project using Arena" should trigger this skill.
---

## How to invoke

```
/arena-create-external-workflow <project_name>
```

`<project_name>` is both the project directory name and the Docker image name. Use `snake_case`.

---

## Prerequisites

The user must have a local IsaacLab-Arena clone with submodules initialized. It will be copied into the project at build time:

```bash
git clone git@github.com:isaac-sim/IsaacLab-Arena.git ~/IsaacLab-Arena
cd ~/IsaacLab-Arena && git submodule update --init --recursive
```

---

## Host project layout

```
<project_name>/
├── submodules/
│   └── IsaacLab-Arena/              ← cp -r from local clone (used at docker build time)
├── pyproject.toml                   ← at project root, defines the Python package
├── <project_name>/
│   ├── __init__.py
│   └── isaaclab_arena_environments/
│       ├── __init__.py
│       └── my_environment.py        ← inherits ExampleEnvironmentBase
├── docker/
│   ├── Dockerfile                   ← adapted from Arena's Dockerfile.isaaclab_arena
│   ├── run_docker.sh                ← adapted from Arena's run_docker.sh
│   └── entrypoint.sh                ← adapted from Arena's entrypoint.sh (one-line change)
```

---

## Container layout

```
/
├── opt/arena/                          ← IsaacLab-Arena (baked into image)
│   └── submodules/
│       ├── IsaacLab/                   ← ISAACLAB_PATH
│       └── Isaac-GR00T/                ← optional
├── isaac-sim/                          ← Isaac Sim runtime (base image)
├── workspaces/
│   └── <project_name>/                 ← WORKDIR (bind-mounted at runtime)
│       ├── submodules/IsaacLab-Arena/  ← present via bind-mount but unused by Python
│       ├── pyproject.toml
│       └── <project_name>/
│           ├── __init__.py
│           └── isaaclab_arena_environments/
│               ├── __init__.py
│               └── my_environment.py
├── datasets/                           ← optional mount
├── models/                             ← optional mount
└── eval/                               ← optional mount
```

`submodules/IsaacLab-Arena/` appears inside the container via the project bind-mount but is harmless — Python imports from `/opt/arena/` (where `pip install -e` registered it), not from the mounted copy.

**Key variables:**

| Variable | Value | Purpose |
|---|---|---|
| `ARENA_DIR` | `/opt/arena` (hardcoded) | Arena install path |
| `ISAACLAB_PATH` | `${ARENA_DIR}/submodules/IsaacLab` | decoupled from `WORKDIR` |
| `WORKDIR` | `/workspaces/<project_name>` (via `--build-arg`) | Docker working dir + user project path |

**Always use `/isaac-sim/python.sh`** for all pip and python commands — never system `python`.

---

## Step 1 — Create project directory and copy Arena

**Ask the user where to create the project** before proceeding. The project must be created **outside** the IsaacLab-Arena repo.

```bash
mkdir -p <parent_dir>/<project_name>/{submodules,<project_name>/isaaclab_arena_environments,docker}
cd <parent_dir>/<project_name>
git init

# Copy Arena into the project — used only at docker build time, not at runtime
cp -r ~/IsaacLab-Arena submodules/IsaacLab-Arena
```

Ask the user for the path to their local Arena clone if it is not at `~/IsaacLab-Arena`.

---

## Step 2 — Write `<project_name>/pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "<project_name>"
version = "0.1.0"
description = "Custom environments built on IsaacLab-Arena"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["."]
include = ["<project_name>*"]
exclude = ["submodules*"]   # skip submodules/IsaacLab-Arena/ during package discovery
```

---

## Step 3 — Write Python stubs

```python
# <project_name>/__init__.py  (empty)
```

```python
# <project_name>/isaaclab_arena_environments/__init__.py  (empty)
```

```python
# <project_name>/isaaclab_arena_environments/my_environment.py
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase
from isaaclab_arena.assets.register import register_environment

@register_environment
class MyEnvironment(ExampleEnvironmentBase):
    name: str = "<project_name>_env"
```

External environments are registered explicitly at runtime via the CLI — they are NOT auto-discovered by Arena:
```bash
/isaac-sim/python.sh policy_runner.py \
    --external_environment_class_path \
    <project_name>.isaaclab_arena_environments.my_environment:MyEnvironment \
    <project_name>_env
```

---

## Step 4 — Write `docker/Dockerfile`

**Adapted from `submodules/IsaacLab-Arena/docker/Dockerfile.isaaclab_arena`.** Read Arena's Dockerfile and apply the changes below. Do not write from scratch.

**Change 1 — Add `ARENA_DIR` and decouple `ISAACLAB_PATH` from `WORKDIR`** (after `ARG INSTALL_GROOT`):

```dockerfile
# Add:
ARG ARENA_DIR=/opt/arena
ENV ARENA_DIR=${ARENA_DIR}

# Change WORKDIR default to user project path:
ARG WORKDIR=/workspaces/<project_name>

# Change ISAACLAB_PATH to use ARENA_DIR instead of WORKDIR:
ENV ISAACLAB_PATH=${ARENA_DIR}/submodules/IsaacLab
```

**Change 2 — Replace all IsaacLab/Arena COPY and install steps** to use `${ARENA_DIR}`:

Replace Arena's individual COPY instructions and all `${WORKDIR}` references in install steps:
```dockerfile
# Replace all of Arena's individual COPY lines with one:
COPY submodules/IsaacLab-Arena ${ARENA_DIR}

# Replace all ${WORKDIR} in install steps with ${ARENA_DIR}, e.g.:
RUN ln -s /isaac-sim/ ${ARENA_DIR}/submodules/IsaacLab/_isaac_sim
RUN for DIR in ${ARENA_DIR}/submodules/IsaacLab/source/isaaclab*/; do ...
RUN ${ISAACLAB_PATH}/isaaclab.sh -i
# ... and so on for all install steps
```

**Change 3 — GR00T scripts** no longer need a separate COPY — they are already inside `${ARENA_DIR}/docker/setup/` from Change 2:

```dockerfile
# Replace:
COPY docker/setup/install_cuda.sh /tmp/install_cuda.sh
# With (no COPY needed, run directly):
RUN if [ "$INSTALL_GROOT" = "true" ]; then \
        chmod +x ${ARENA_DIR}/docker/setup/install_cuda.sh && \
        ${ARENA_DIR}/docker/setup/install_cuda.sh; ...

# Same for install_gr00t_deps.sh
```

**Change 4 — Arena `pip install -e`**: remove `[dev]` extras:

```dockerfile
# Replace:
RUN /isaac-sim/python.sh -m pip install -e "${WORKDIR}/[dev]"
# With:
RUN /isaac-sim/python.sh -m pip install -e "${ARENA_DIR}/"
```

**Change 5 — Add user package section** (after Arena install, before bash aliases):

```dockerfile
# COPY bakes the package into the image. The bind-mount at runtime overlays
# this with live source — no reinstall needed for daily edits.
COPY pyproject.toml ${WORKDIR}/pyproject.toml
COPY <project_name> ${WORKDIR}/<project_name>
RUN /isaac-sim/python.sh -m pip install -e "${WORKDIR}/"
```

**Change 6 — Update prompt and entrypoint**:

```dockerfile
# Update prompt:
RUN echo "PS1='[<project_name>] \[\e[0;32m\]~\u \[\e[0;34m\]\w\[\e[0m\] \$ '" >> /etc/bash.bashrc

# Point to custom entrypoint (Step 6):
COPY docker/entrypoint.sh /entrypoint.sh
```

---

## Step 5 — Write `docker/run_docker.sh`

**Adapted from `submodules/IsaacLab-Arena/docker/run_docker.sh`.** Read Arena's script and apply the three changes below. Everything else (flags, Omniverse auth, X11, SSH, datasets/models/eval mounts) is inherited unchanged.

**Change 1 — Update variables** (right after `SCRIPT_DIR=...`):

Replace:
```bash
DOCKER_IMAGE_NAME='isaaclab_arena'
...
WORKDIR="/workspaces/isaaclab_arena"
```
With:
```bash
DOCKER_IMAGE_NAME='<project_name>'
...
PROJECT_WORKDIR="/workspaces/<project_name>"
WORKDIR="$PROJECT_WORKDIR"
```

**Change 2 — Replace the `docker build` block** (single build, no two-stage):

```bash
if [ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_VERSION_TAG 2>/dev/null)" ] && \
    [ "$FORCE_REBUILD" = false ]; then
    echo "Docker image $DOCKER_IMAGE_NAME:$DOCKER_VERSION_TAG already exists. Not rebuilding."
else
    docker build --pull $NO_CACHE --progress=plain \
        --build-arg WORKDIR="${PROJECT_WORKDIR}" \
        --build-arg INSTALL_GROOT=$INSTALL_GROOT \
        -t ${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG} \
        --file $SCRIPT_DIR/Dockerfile \
        $SCRIPT_DIR/..
fi
```

**Change 3 — Replace the Arena volume mount** (in `DOCKER_RUN_ARGS`):

Replace:
```bash
"-v" "$SCRIPT_DIR/..:${WORKDIR}"
```
With:
```bash
"-v" "$SCRIPT_DIR/..:${PROJECT_WORKDIR}"
```

Arena is baked at `/opt/arena/` — it is **not** mounted at runtime. Also remove the GR00T volume mount (`./submodules/Isaac-GR00T`) — Isaac-GR00T is baked inside `/opt/arena/submodules/Isaac-GR00T/`.

---

## Step 6 — Write `docker/entrypoint.sh`

**Adapted from `submodules/IsaacLab-Arena/docker/setup/entrypoint.sh`.** Copy it verbatim and change one line:

```bash
# Before (Arena's original):
if [ ! -e "$WORKDIR/submodules/IsaacLab/_isaac_sim" ]; then
    ln -s /isaac-sim/ "$WORKDIR/submodules/IsaacLab/_isaac_sim"
fi

# After (Arena is baked at /opt/arena, not at $WORKDIR):
if [ ! -e "$ARENA_DIR/submodules/IsaacLab/_isaac_sim" ]; then
    ln -s /isaac-sim/ "$ARENA_DIR/submodules/IsaacLab/_isaac_sim"
fi
```

`ARENA_DIR` is set as `ENV` in the Dockerfile and is available inside the container without additional configuration.

---

## Key path reference

| What | Container path | Source |
|---|---|---|
| IsaacLab-Arena | `/opt/arena/` | baked at build time |
| IsaacLab | `/opt/arena/submodules/IsaacLab/` | baked at build time |
| Isaac-GR00T | `/opt/arena/submodules/Isaac-GR00T/` | baked (optional) |
| Isaac Sim runtime | `/isaac-sim/` | base image |
| User project | `/workspaces/<project_name>/` | bind-mounted at runtime |
| User package | `/workspaces/<project_name>/<project_name>/` | bind-mounted + `pip install -e` |
| Datasets | `/datasets` | `~/datasets` (optional) |
| Models | `/models` | `~/models` (optional) |

---

## Verify

```bash
ls <project_name>/docker/run_docker.sh \
   <project_name>/docker/Dockerfile \
   <project_name>/docker/entrypoint.sh \
   <project_name>/pyproject.toml \
   <project_name>/<project_name>/__init__.py
```

All five paths must exist. Then build and launch:

```bash
cd <project_name> && bash docker/run_docker.sh
```

Inside the container, run all three checks:

```bash
/isaac-sim/python.sh -c "import isaaclab; print(isaaclab.__file__)"
# Expected: /opt/arena/submodules/IsaacLab/source/isaaclab/isaaclab/__init__.py

/isaac-sim/python.sh -c "import isaaclab_arena; print(isaaclab_arena.__file__)"
# Expected: /opt/arena/isaaclab_arena/__init__.py

/isaac-sim/python.sh -c "import <project_name>; print('OK')"
# Expected: OK
```

Only report `arena-create-external-workflow complete — <project_name> created.` if the container starts and all three import checks pass.
