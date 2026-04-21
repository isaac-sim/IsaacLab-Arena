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
| `scene_assets` | `scene_assets` | `[]` |
| `extra_mounts` | `extra_mounts` | `[]` |

Derived (never in spec):
- `PROJECT_DIR` = `<base_dir>/<project_name>`
- `PKG` = `<PROJECT_DIR>/<project_name>`
- `CONTAINER_NAME` = `<project_name>-latest`
- `CONTAINER_PROJECT_PATH` = `/workspaces/<project_name>`

### Derive `container_usd_path` for custom assets

For every entry in `scene_assets` where `"source": "custom"`:

1. Take the asset's `host_usd_path`.
2. Iterate through `extra_mounts`. Find the first entry where
   `host_usd_path == mount["host"]` OR `host_usd_path.startswith(mount["host"] + "/")`.
3. Substitute: `container_usd_path = mount["container"] + host_usd_path[len(mount["host"]):]`
4. If no matching mount found, warn:
   > WARNING: No extra_mounts entry covers host_usd_path "\<path\>" for asset "\<name\>".
   > Add a mount whose "host" is a prefix of this path.

This derived `container_usd_path` is used by downstream skills (e.g. `arena-create`) when generating `assets/asset_paths.py`.

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

Read `.claude/skills/arena-install/docker/Dockerfile` (the template in this repo).
Write it to `<PROJECT_DIR>/docker/Dockerfile`, substituting:

| Placeholder | Replace with |
|---|---|
| `<project_name>` | actual project name from spec |

---

## Step 5 — Write `docker/run_docker.sh`

Read `.claude/skills/arena-install/docker/run_docker.sh` (the template in this repo).
Write it to `<PROJECT_DIR>/docker/run_docker.sh`, substituting:

| Placeholder | Replace with |
|---|---|
| `<project_name>` | actual project name from spec |
| `<EXTRA_MOUNTS>` | one `"-v" "<host>:<container>"` array element per `extra_mounts` entry; omit entirely if empty |

After writing, make executable:
```bash
chmod +x <PROJECT_DIR>/docker/run_docker.sh
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
ls <PROJECT_DIR>/docker/run_docker.sh \
   <PROJECT_DIR>/docker/Dockerfile \
   <PROJECT_DIR>/pyproject.toml \
   <PKG>/__init__.py
```

All four paths must exist.

Then launch the container to catch build or runtime errors:

```bash
cd <PROJECT_DIR> && bash docker/run_docker.sh
```

Watch the output. If the build or container start fails, report the error to the user and stop.
Only report `arena-install complete — <PROJECT_DIR> scaffolded.` if the container starts successfully.
