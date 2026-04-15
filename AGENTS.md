# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, OpenAI Codex, etc.) when working with code in this repository.

## Docker Environment

All commands (tests, linting, training scripts, etc.) must be run inside the Docker container. The default container is `isaaclab_arena-latest`, started via:

```bash
./docker/run_docker.sh          # build image (if needed) and start/attach
./docker/run_docker.sh -r       # force rebuild
./docker/run_docker.sh -g       # include GR00T N1.6 dependencies
./docker/run_docker.sh -d ~/datasets -m ~/models -e ~/eval  # custom mount dirs
```

The repo root is mounted at `/workspaces/isaaclab_arena` inside the container. To run a command in the already-running container:

```bash
docker exec isaaclab_arena-latest bash -c "cd /workspaces/isaaclab_arena && <command>"
```

**Important:** Inside the container, `python` is aliased to `/isaac-sim/python.sh`, so both forms work. Prefer `/isaac-sim/python.sh` for explicitness (e.g. in `docker exec` commands run from outside the container, where the alias is not active).

```bash
# Example: run kitchen_pick_and_place with zero_action policy
docker exec isaaclab_arena-latest bash -c "cd /workspaces/isaaclab_arena && \
  /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action \
  --num_steps 10 \
  kitchen_pick_and_place \
  --object cracker_box \
  --embodiment franka_ik"
```

## Common Commands

### Running Tests

Tests require Isaac Sim and run via pytest:

```bash
# Run all tests
# Our tests are separated into different phases of running. To run the full test suite requires three commands.
/isaac-sim/python.sh -m pytest -sv -m "not with_cameras and not with_subprocess" isaaclab_arena/tests
/isaac-sim/python.sh -m pytest  -sv -m "with_cameras and not with_subprocess" isaaclab_arena/tests
/isaac-sim/python.sh -m pytest  -sv -m "with_subprocess" isaaclab_arena/tests

# Run a single test file
/isaac-sim/python.sh -m pytest isaaclab_arena/tests/test_asset_registry.py

# Run a specific test function
/isaac-sim/python.sh -m pytest isaaclab_arena/tests/test_asset_registry.py::test_default_assets_registered

# Run tests that require cameras
/isaac-sim/python.sh -m pytest isaaclab_arena/tests/ -m with_cameras
```

### Linting, Formatting, and Coding Style

Pre-commit hooks enforce the style guide: black (line length 120), flake8, isort, pyupgrade (py310+), and codespell. Run checks **before** committing — not after:

```bash
# Install pre-commit hooks
pre-commit install

# Run all checks (if hooks modify files, stage them and re-run before committing)
pre-commit run --all-files
```

### Contributing

All commits must be signed off per DCO requirements:
```bash
git commit -s -m "Your commit message"
```

**Branch naming:** `<username>/feature-desc` (e.g. `cvolk/feature-video-recording`, `cvolk/refactor-no-unwrap`).

**Commit messages:**
- Subject: imperative mood, ~50 chars, no trailing period (e.g. "Fix attribute access on wrapped env")
- Separate subject from body with a blank line
- Body: explain *what* and *why* (not how — the diff shows that), wrap at 72 chars
- Do not include AI attribution lines (e.g. "Co-Authored-By: Claude...")

**PR iteration:** when addressing review feedback, add new commits rather than amending existing ones — this lets reviewers easily verify each change was addressed.

## Conventions

### Wrapped Environment

`ArenaEnvBuilder.make_registered()` returns the gym-wrapped env (not the base env). Use `env.unwrapped` explicitly to access Isaac Lab-specific attributes (`cfg`, `device`, `step_dt`, etc.) that are not forwarded by gymnasium's `OrderEnforcing` wrapper:

```python
env = arena_builder.make_registered()   # wrapped env
env.step(actions)                       # goes through OrderEnforcing
env.unwrapped.cfg                       # access Isaac Lab config
env.unwrapped.device                    # access Isaac Lab device
```

### Writing Tests

Simulation tests use an inner/outer function pattern to handle Isaac Sim's process lifecycle:

```python
def _test_foo(simulation_app):  # runs inside SimulationApp
    from isaaclab_arena.X import Y  # deferred imports after sim init
    ...
    return True  # indicates pass

def test_foo():  # pytest-visible outer function
    result = run_simulation_app_function(_test_foo)
    assert result
```
