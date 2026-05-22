---
name: run-tests
description: Runs the Isaac Lab-Arena pytest suite across its three required phases (no-cameras, with-cameras, with-subprocess) inside the running isaaclab_arena Docker container. Use when the user asks to run the tests, test their changes, verify the test suite passes, check whether a change broke anything (e.g. "did I break anything", "do the tests still pass", "is everything still working"), run only one specific phase (e.g. "run Phase 1", "run only the no-camera tests", "skip the camera and subprocess tests"), or run coverage for a specific module.
argument-hint: "[phase1|phase2|phase3]"
allowed-tools: Bash(docker exec *)
---

# Run Tests

**Default (no argument):** run all three phases in order.
**`phase1`:** no-cameras, no-subprocess only.
**`phase2`:** with-cameras only.
**`phase3`:** with-subprocess only.

The three phases run mutually exclusive sets of tests (selected via different pytest marker filters) and must be invoked separately. For a full pre-PR run, every phase must pass before moving on to the next.

All commands run inside the already-running container via `docker exec`. If the container is not running, use the `dev-container` skill first.

## Phase 1 — no cameras, no subprocess

The fastest of the three. Useful on its own when iterating on changes that don't touch rendering or subprocess paths.

```bash
docker exec isaaclab_arena-latest bash -c \
  "cd /workspaces/isaaclab_arena && \
   /isaac-sim/python.sh -m pytest -sv \
   -m 'not with_cameras and not with_subprocess' isaaclab_arena/tests"
```

## Phase 2 — with cameras

```bash
docker exec isaaclab_arena-latest bash -c \
  "cd /workspaces/isaaclab_arena && \
   /isaac-sim/python.sh -m pytest -sv \
   -m 'with_cameras and not with_subprocess' isaaclab_arena/tests"
```

## Phase 3 — with subprocess

```bash
docker exec isaaclab_arena-latest bash -c \
  "cd /workspaces/isaaclab_arena && \
   /isaac-sim/python.sh -m pytest -sv \
   -m 'with_subprocess' isaaclab_arena/tests"
```

## Run a single test file or function

```bash
# single file
docker exec isaaclab_arena-latest bash -c \
  "cd /workspaces/isaaclab_arena && \
   /isaac-sim/python.sh -m pytest isaaclab_arena/tests/test_asset_registry.py"

# single function
docker exec isaaclab_arena-latest bash -c \
  "cd /workspaces/isaaclab_arena && \
   /isaac-sim/python.sh -m pytest isaaclab_arena/tests/test_asset_registry.py::test_default_assets_registered"
```

## On failure

1. Report the failing phase and the first failing test (file, function, and one-line error).
2. Stop. Do not run later phases until the failure is investigated.
3. If the failure looks unrelated to recent changes, run the same test in isolation to confirm.

A run is "all green" only when all three phases pass with no failures or errors.
