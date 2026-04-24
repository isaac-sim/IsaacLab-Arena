---
name: arena-scaffold-external-workflow
description: Scaffold a new external IsaacLab-Arena project — creates directory structure, pyproject.toml, Dockerfile, run_docker.sh and launches the docker container. Use this skill whenever the user wants to scaffold a new project, package, or workflow that builds on top of IsaacLab-Arena, even if they don't say "scaffold" or "external workflow" — any request like "start a new Arena-based project", "set up my custom environment package", or "create a project using Arena" should trigger this skill.
---

## How to invoke

```
/arena-create-external-workflow <project_name>
```

`<project_name>` is both the Python package name (e.g. `my_robot_tasks`) and the Docker image name. Use `snake_case`.

---

## Prerequisites

Before running this skill, make sure user must have the following in place:

**An existing local IsaacLab-Arena clone with submodules initialized**

The skill uses it as a `--reference` to avoid re-downloading git objects. The clone must have its nested submodules (`submodules/IsaacLab` and `submodules/Isaac-GR00T`) already initialized:

```bash
git clone git@github.com:isaac-sim/IsaacLab-Arena.git
cd IsaacLab-Arena
git submodule update --init --recursive
```

---

## Container workspace layout

The container uses `/workspaces/` as the root, with one directory per project:

```
/workspaces/
├── isaaclab_arena/        ← IsaacLab-Arena source (mounted from submodules/IsaacLab-Arena/)
└── <project_name>/        ← Your project (mounted from project root)
    ├── submodules/
    │   └── IsaacLab-Arena/
    ├── <project_name>/    ← Python package (pip install -e'd)
    │   ├── __init__.py
    │   └── isaaclab_arena_environments/
    └── docker/
```

**Why this matters:** Arena's `run_docker.sh` mounts the Arena repo at `/workspaces/isaaclab_arena`. Your project mounts alongside it at `/workspaces/<project_name>`. They are siblings, not nested.

**Always use `/isaac-sim/python.sh`** for all pip installs and python commands — never system `python`. Isaac Sim has its own embedded Python where Arena and IsaacLab are installed.

---

## Host project layout

```
<project_name>/
├── submodules/
│   └── IsaacLab-Arena/          ← git submodule (unmodified Arena)
├── <project_name>/
│   ├── __init__.py
│   └── isaaclab_arena_environments/
│       ├── __init__.py
│       └── my_environment.py    ← custom environment class
├── pyproject.toml               ← defines the Python package
├── docker/
│   ├── Dockerfile
│   └── run_docker.sh
└── .gitmodules
```

---

## Step 1 — Create project directory

**Ask the user where to create the project** before proceeding — do not assume a location. The project must be created **outside** the IsaacLab-Arena repo.

```bash
mkdir -p <parent_dir>/<project_name>/{submodules,<project_name>/isaaclab_arena_environments,docker}
cd <parent_dir>/<project_name>
git init
```

---

## Step 2 — Write `pyproject.toml`

Place at the project root (`<project_name>/pyproject.toml`):

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
include = ["<project_name>*", "isaaclab_arena_environments*"]
```

The `include` list must cover both your top-level package and `isaaclab_arena_environments` so they are both importable after `pip install -e`.

---

## Step 3 — Write `<project_name>/__init__.py` and environment stubs

```python
# <project_name>/__init__.py
```

```python
# <project_name>/isaaclab_arena_environments/__init__.py
```

```python
# <project_name>/isaaclab_arena_environments/my_environment.py
# Extend Arena environments here
```

---

## Step 4 — Write `docker/Dockerfile`

The project Dockerfile is intentionally thin — it builds `FROM isaaclab_arena:latest` rather than duplicating all of Arena's installation steps. The Arena base image is built first by `run_docker.sh` (Step 5).

```dockerfile
ARG ARENA_IMAGE=isaaclab_arena:latest
FROM ${ARENA_IMAGE}

# Install <project_name> on top of the Arena base image.
# Source must be present before pip install -e: setuptools needs the package directory
# to exist at install time to discover and register packages. Installing before the
# source copy would register an empty package list and break imports at runtime.
# At runtime, the volume mount overlays /workspaces/<project_name>/ with live source.
COPY pyproject.toml /workspaces/<project_name>/pyproject.toml
COPY <project_name> /workspaces/<project_name>/<project_name>
RUN /isaac-sim/python.sh -m pip install -e /workspaces/<project_name>

# Entrypoint is inherited from the Arena base image (/entrypoint.sh).
# It mirrors the host user, creates /datasets /models /eval, and
# re-creates the IsaacLab → /isaac-sim symlink after volume mounts.
```

---

## Step 5 — Write `docker/run_docker.sh`

**Do not write this from scratch.** Read `submodules/IsaacLab-Arena/docker/run_docker.sh` and apply the three changes below. Everything else — flags, env vars, volume mounts, GR00T handling, Omniverse auth — is inherited unchanged. This keeps the script in sync with Arena automatically.

**Change 1 — Replace image/name variables and add project variables** (right after `SCRIPT_DIR=...`):

Replace:
```bash
DOCKER_IMAGE_NAME='isaaclab_arena'
...
WORKDIR="/workspaces/isaaclab_arena"
```
With:
```bash
DOCKER_IMAGE_NAME='<project_name>'
ARENA_IMAGE_NAME='isaaclab_arena'
...
ARENA_DIR="$SCRIPT_DIR/../submodules/IsaacLab-Arena"
WORKDIR="/workspaces/isaaclab_arena"
PROJECT_WORKDIR="/workspaces/<project_name>"
```
`DOCKER_VERSION_TAG` is shared by both the Arena base image and the project image — no separate variable needed. The `-g` flag already sets it to `'cuda_gr00t_gn16'` in Arena's script, so both images automatically get the GR00T variant tag.

**Change 2 — Replace the single `docker build` block with two stages:**

```bash
# ── Stage 1: Arena base image ─────────────────────────────────────────────────
# Pass WORKDIR so the editable install path matches the volume mount below.
# Arena's Dockerfile defaults to /workspace (singular); we need the plural form
# /workspaces/isaaclab_arena so live source edits are picked up by Python.
if [ "$(docker images -q $ARENA_IMAGE_NAME:$DOCKER_VERSION_TAG 2>/dev/null)" ] && \
    [ "$FORCE_REBUILD" = false ]; then
    echo "Arena image $ARENA_IMAGE_NAME:$DOCKER_VERSION_TAG already exists. Skipping."
else
    docker build --pull $NO_CACHE --progress=plain \
        --build-arg WORKDIR="${WORKDIR}" \
        --build-arg INSTALL_GROOT=$INSTALL_GROOT \
        -t ${ARENA_IMAGE_NAME}:${DOCKER_VERSION_TAG} \
        --file $ARENA_DIR/docker/Dockerfile.isaaclab_arena \
        $ARENA_DIR
fi

# ── Stage 2: Project image ────────────────────────────────────────────────────
# Note: no --pull here — isaaclab_arena is a local image, not on Docker Hub.
# --pull would cause Docker to attempt a registry fetch and fail.
if [ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_VERSION_TAG 2>/dev/null)" ] && \
    [ "$FORCE_REBUILD" = false ]; then
    echo "Docker image $DOCKER_IMAGE_NAME:$DOCKER_VERSION_TAG already exists. Not rebuilding."
else
    docker build $NO_CACHE --progress=plain \
        --build-arg ARENA_IMAGE="${ARENA_IMAGE_NAME}:${DOCKER_VERSION_TAG}" \
        -t ${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG} \
        --file $SCRIPT_DIR/Dockerfile \
        $SCRIPT_DIR/..
fi
```

**Change 3 — Replace the Arena volume mount and add the project mount** (in `DOCKER_RUN_ARGS`):

Replace:
```bash
"-v" "$SCRIPT_DIR/..:${WORKDIR}"
```
With:
```bash
"-v" "$ARENA_DIR:${WORKDIR}"           # Arena submodule (live source)
"-v" "$SCRIPT_DIR/..:${PROJECT_WORKDIR}"  # Project root (live source)
```

Also update the GR00T conditional at the bottom:
```bash
# Change:
"-v" "$SCRIPT_DIR/../submodules/Isaac-GR00T:..."
# To:
"-v" "$ARENA_DIR/submodules/Isaac-GR00T:..."
```

---

## Step 6 — Add IsaacLab-Arena as a git submodule

**Ask the user once for the path to their existing IsaacLab-Arena clone.** Use it to derive `--reference` paths for the Arena submodule and each nested submodule — all from that single answer.

```bash
# Add IsaacLab-Arena, borrowing objects from the existing local clone
git submodule add --reference <existing-arena-clone> \
    git@github.com:isaac-sim/IsaacLab-Arena.git submodules/IsaacLab-Arena

# Initialize each nested submodule with its own reference to avoid network fetches
git -C submodules/IsaacLab-Arena submodule update --init \
    --reference <existing-arena-clone>/submodules/IsaacLab \
    submodules/IsaacLab

git -C submodules/IsaacLab-Arena submodule update --init \
    --reference <existing-arena-clone>/submodules/Isaac-GR00T \
    submodules/Isaac-GR00T
```

Use separate `submodule update --init` calls (not `--recursive`) so each nested submodule gets the right reference path — a single `--recursive` call can only take one `--reference`, which wouldn't match all submodules.

**Why `--reference`:** `run_docker.sh` mounts `submodules/IsaacLab-Arena` over `/workspaces/isaaclab_arena` at runtime. IsaacLab is installed as an editable install pointing to `/workspaces/isaaclab_arena/submodules/IsaacLab/source/isaaclab` — if that nested path is empty, `import isaaclab` fails at runtime even though it worked at build time.

**Note:** clones created with `--reference` have a hard dependency on the reference path via `.git/objects/info/alternates`. This is fine as long as the existing Arena clone stays in place.

If this fails (no SSH key or network), skip it and tell the user — the submodule is required to run the container.

---

## Key path reference

| What | Container path | Host source |
|------|---------------|-------------|
| Arena source | `/workspaces/isaaclab_arena` | `submodules/IsaacLab-Arena/` |
| Your project | `/workspaces/<project_name>` | project root |
| Your Python package | `/workspaces/<project_name>/<project_name>` | `<project_name>/` |
| Isaac Sim runtime | `/isaac-sim/` | (baked into image) |
| IsaacLab | `/workspaces/isaaclab_arena/submodules/IsaacLab` | (baked into image) |
| Datasets | `/datasets` | `~/datasets` |
| Models | `/models` | `~/models` |

---

## Verify

```bash
ls <project_name>/docker/run_docker.sh \
   <project_name>/docker/Dockerfile \
   <project_name>/pyproject.toml \
   <project_name>/<project_name>/__init__.py
```

All four paths must exist. Then launch the container:

```bash
cd <project_name> && bash docker/run_docker.sh
```

Watch the output. If the build or container start fails, report the error and stop.

Inside the container, run both checks:

```bash
/isaac-sim/python.sh -c "import isaaclab; print(isaaclab.__file__)"
/isaac-sim/python.sh -c "import <project_name>; print('OK')"
```

The first confirms IsaacLab is correctly installed in the base image. The second confirms the project package was registered by `pip install -e`.

Only report `arena-create-external-workflow complete — <project_name> scaffolded.` if the container starts and both import checks pass.
