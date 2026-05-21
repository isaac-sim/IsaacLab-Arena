---
name: run-tests
description: Runs the Isaac Lab-Arena pytest suite across its three required phases (no-cameras, with-cameras, with-subprocess) inside the running isaaclab_arena Docker container. Use when the user asks to run the tests, verify the test suite passes, check whether a change broke anything, smoke-test the install, run smoke tests, or run coverage for a specific module.
argument-hint: "[smoke]"
allowed-tools: Bash(docker exec *)
---

# Run Tests

**Default (no argument):** run all three phases in order.
**`smoke`:** Phase 1 only (no-cameras, no-subprocess) — the fast install / sanity check used in the first-touch journey.

The three phases use mutually exclusive pytest markers and must be invoked separately. For a full pre-PR run, every phase must pass before moving on to the next.

All commands run inside the already-running container via `docker exec`. If the container is not running, use the `dev-container` skill first.

## Phase 1 — no cameras, no subprocess

The fastest phase. Suitable on its own as a smoke check after install or after small edits.

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
