---
name: arena-verify
description: Launch the Docker container for an Arena external project, install the package, run the smoke test, and self-fix errors until it passes.
---

## How to invoke

```
/arena-verify /path/to/spec.json
```

The argument is the path to a filled-in spec file (same one used with `/arena-install` and `/arena-create`).

---

## Step 1 — Read the spec

Read the JSON file at the path given in the argument. Extract:

| Variable | JSON field | Derived if absent |
|---|---|---|
| `project_name` | `project_name` | — |
| `base_dir` | `base_dir` | — |
| `extra_mounts` | `extra_mounts` | `[]` |

Derived:
- `PROJECT_DIR` = `<base_dir>/<project_name>`
- `CONTAINER_NAME` = `<project_name>-latest`
- `CONTAINER_PROJECT_PATH` = `/workspaces/<project_name>`

---

## Step 2 — Ensure the container is running

```bash
docker inspect --format '{{.State.Running}}' <CONTAINER_NAME> 2>/dev/null
```

- Output is `true` → skip to Step 3.
- Output is `false` or command errors → launch the container.

### Launch

The project has its own `run_docker.sh` that manages image build and startup.
However, it is interactive (`--interactive --rm --tty`) so it cannot be used as-is for
`docker exec`-based automation. Instead, run the equivalent detached:

```bash
cd <PROJECT_DIR>

# Build image if it doesn't exist yet
if [ -z "$(docker images -q <project_name>:latest 2>/dev/null)" ]; then
    docker build --pull --progress=plain \
        -t <project_name>:latest \
        -f docker/Dockerfile \
        .
fi

xhost +local:docker > /dev/null 2>&1 || true

docker run -d \
    --name <CONTAINER_NAME> \
    --privileged \
    --ulimit memlock=-1 \
    --ulimit stack=-1 \
    --ipc=host \
    --net=host \
    --runtime=nvidia \
    --gpus=all \
    -v "$(pwd)/submodules/IsaacLab-Arena:/workspaces/isaaclab_arena" \
    -v "$(pwd):<CONTAINER_PROJECT_PATH>" \
    <EXTRA_MOUNT_FLAGS> \
    -v "$HOME/.cache:$HOME/.cache" \
    -v "/tmp:/tmp" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "/etc/ssl/certs:/etc/ssl/certs:ro" \
    --env DISPLAY \
    --env ACCEPT_EULA=Y \
    --env PRIVACY_CONSENT=Y \
    --env ISAACLAB_PATH=/workspaces/isaaclab_arena/submodules/IsaacLab \
    --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    <project_name>:latest \
    bash -c "sleep infinity"
```

Replace `<EXTRA_MOUNT_FLAGS>` with `-v "<host>:<container>" \` for each entry in `extra_mounts`.

Verify launch succeeded:
```bash
docker inspect --format '{{.State.Running}}' <CONTAINER_NAME>
```
Must output `true`.

**Hard blockers — stop and ask the user:**
- Image build fails (missing submodule, base image unavailable, etc.)
- `--runtime=nvidia` fails (no GPU driver)

---

## Step 3 — Install the package

```bash
docker exec <CONTAINER_NAME> bash -c \
    "/isaac-sim/python.sh -m pip install -e <CONTAINER_PROJECT_PATH>"
```

---

## Step 4 — Smoke test loop

Run the smoke test. Use a **300 s timeout** — Isaac Sim takes 60–120 s to boot.

```bash
docker exec <CONTAINER_NAME> bash -c \
    "cd /workspaces/isaaclab_arena && \
     /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --num_steps 50 \
     --external_environment_class_path \
       <project_name>.environments.<project_name>_environment:<ProjectName>Environment \
     <project_name> 2>&1"
```

**Pass:** output ends without a Python traceback.
Report: `arena-verify complete — smoke test passed for <project_name>.`

**Fail:** diagnose and fix using the table below, then re-run. Repeat until it passes.

---

## Fix guide

| Error pattern | Root cause | Fix |
|---|---|---|
| `ModuleNotFoundError: <project_name>` | Package not installed | Re-run Step 3 |
| `KeyError` from `AssetRegistry` | `@register_asset` module not imported in `get_env()` | Add missing side-effect import |
| `AttributeError: rotation_wxyz` | Old field name | Rename to `rotation_xyzw` |
| Warp indexing / type error on `.data` | Field not converted to torch | Wrap with `wp.to_torch(...)` |
| Prim path collision | Two assets share the same `prim_path` | Make prim paths unique |
| `ImportError` at module level in environment file | Sim import at top level | Move import inside `get_env()` |
| USD not found / path error | Container path mismatch | Check `asset_paths.py` against the `-v` mount in Step 2 |
| Physics explodes / NaN on first step | Asset spawns inside another geometry | Adjust `init_state.pos` in library file |
| `AttributeError` on unknown method/field | API changed since code was written | Read the current Arena source file for the correct API, fix accordingly |

For any error not in this table: read the full traceback, find the source file, fix it, re-run.
Reinstall (Step 3) only if `pyproject.toml` or package structure changed.

Only ask the user if the fix requires information you cannot find — e.g. a missing USD file path
or Nucleus authentication.
