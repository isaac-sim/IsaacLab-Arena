# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, OpenAI Codex, etc.) when working with code in this repository.

## Project

Isaac Lab-Arena is a composable environment-creation and policy-evaluation library for robotics simulation, built on Isaac Sim 5.1 and Isaac Lab 2.3. Status: alpha (`v0.2.x`); APIs are unstable. `develop` is the active branch; `main` is under SQA and receives only fixes.

## Skill library

Recurring multi-step workflows (container management, the three-phase test suite, commits and PRs) are captured as Agent Skills under `.agents/skills/`. When a task matches a skill, prefer invoking it over re-deriving the procedure from this file.

Fresh-clone setup (run once):

```bash
ln -s ../.agents/skills .claude/skills    # so Claude Code reads the skill library
pre-commit install                         # register git hooks
```

## Docker environment

All commands (tests, linting, training scripts) must run inside the `isaaclab_arena-latest` Docker container. The repo root is mounted at `/workspaces/isaaclab_arena`. Inside the container, `python` is aliased to `/isaac-sim/python.sh` — prefer the explicit path in `docker exec` invocations from outside the container, where the alias is not active.

Use the `dev-container` skill for build, start, attach, and exec.

## Repository layout

- `isaaclab_arena/` — core package: `tasks/`, `policy/`, `evaluation/`, `embodiments/`, `scene/`, `assets/`, `tests/`
- `isaaclab_arena_environments/`, `isaaclab_arena_examples/`, `isaaclab_arena_g1/`, `isaaclab_arena_gr00t/` — first-party extension packages
- `docker/` — container build and run scripts
- `submodules/` — vendored dependencies (IsaacLab, Isaac-GR00T, …)
- `osmo/` — OSMO policy-runner workflow
- `docs/` — Sphinx documentation

## Coding style

- Prefer `assert condition, "message"` over `if not condition: raise ValueError("message")` for internal invariant checks. (Formatting, imports, and typing are enforced by `pre-commit` — see `.pre-commit-config.yaml`.)

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

## Boundaries

- **Never** force-push to `main`, `develop`, or `release/*`. **Instead**, push to a `<username>/feature-desc` branch and open a PR against `develop`.
- **Never** add AI-attribution lines to commits (no `Co-Authored-By: Claude…`, no `Generated with…`). **Instead**, sign off with `git commit -s` — DCO is the only required trailer.
- **Never** commit models, datasets, or secrets. **Instead**, keep them on the host and mount them via `./docker/run_docker.sh -d <datasets> -m <models> -e <eval>`.
- **Ask first** before changing `docker/`, `.github/workflows/`, `.pre-commit-config.yaml`, or `submodules/` — these affect every contributor. **Instead** of pushing directly, open a draft PR or raise it in the relevant channel before merging.
