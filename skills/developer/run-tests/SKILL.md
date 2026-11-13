---
name: run-tests
description: Runs the Isaac Lab-Arena pytest suite across its three required phases (no-cameras, with-cameras, with-subprocess) in the checkout's native uv environment or running isaaclab_arena Docker container. Use when the user asks to run the tests, test their changes, verify the test suite passes, check whether a change broke anything (e.g. "did I break anything", "do the tests still pass", "is everything still working"), run only one specific phase (e.g. "run Phase 1", "run only the no-camera tests", "skip the camera and subprocess tests"), or run coverage for a specific module.
argument-hint: "[phase1|phase2|phase3]"
allowed-tools: Bash(git rev-parse --show-toplevel) Bash(id --user --name) Bash(test -x .venv/bin/python) Bash(env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python *) Bash(docker exec *) Bash(docker ps *)
---

# Run Tests

**Default (no argument):** run all three phases in order.
**`phase1`:** no-cameras, no-subprocess only.
**`phase2`:** with-cameras only.
**`phase3`:** with-subprocess only.

The three phases run mutually exclusive sets of tests (selected via different pytest marker filters) and must be invoked separately. For a full pre-PR run, every phase must pass before moving on to the next.

## Select the runtime

At the start of a session, use the `arena-runtime` skill to select the checkout's ready native uv or Docker route. If neither route is ready, report it and use `setup-arena`; do not skip the tests.

Every command below starts with `<arena-pytest>`. Replace it with the selected route's launcher:

```bash
# Native uv, from the repository root
env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python -m pytest

# Docker, with the container that mounts this checkout
docker exec --user "$(id --user --name)" "$ARENA_CONTAINER" isaaclab.sh -p -m pytest
```

For Docker, run the phase commands in the shell that set `$ARENA_CONTAINER`, or substitute the literal name.

## Phase 1 — no cameras, no subprocess

The fastest of the three. Useful on its own when iterating on changes that don't touch rendering or subprocess paths.

```bash
<arena-pytest> -sv -m 'not with_cameras and not with_subprocess' isaaclab_arena/tests
```

## Phase 2 — with cameras

```bash
<arena-pytest> -sv -m 'with_cameras and not with_subprocess' isaaclab_arena/tests
```

## Phase 3 — with subprocess

```bash
<arena-pytest> -sv -m 'with_subprocess' isaaclab_arena/tests
```

## Run a single test file or function

```bash
# single file
<arena-pytest> isaaclab_arena/tests/test_asset_registry.py

# single function
<arena-pytest> isaaclab_arena/tests/test_asset_registry.py::test_default_assets_registered
```

## On failure

1. Report the failing phase and the first failing test (file, function, and one-line error).
2. Stop. Do not run later phases until the failure is investigated.
3. If the failure looks unrelated to recent changes, run the same test in isolation to confirm.

A run is "all green" only when all three phases pass with no failures or errors. Report which runtime route ran the tests.
