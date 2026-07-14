# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, OpenAI Codex, etc.) when working with code in this repository.

## Project

Isaac Lab-Arena is a composable environment-creation and policy-evaluation library for robotics simulation, built on Isaac Sim 6.0 and Isaac Lab 3.0 Beta. Status: alpha (`v0.2.x`); APIs are unstable. `main` is the active development branch.

## Skill library

Recurring multi-step workflows (container management, the three-phase test suite, commits and PRs) are captured as Agent Skills under `.agents/skills/`. When a task matches a skill, prefer invoking it over re-deriving the procedure from this file.

Claude Code reads the library via the committed `.claude/skills` symlink; Codex scans `.agents/skills/` directly.

Fresh-clone setup (run once):

```bash
pre-commit install    # on the host — registers git pre-commit hooks
```

## Docker environment

Commands that touch Isaac Sim or Arena's package code (tests, training, evaluation, runtime scripts) run inside the local repo clone's Docker container. The repo root is mounted at `/workspaces/isaaclab_arena`. Inside the container, `python` is aliased to `/isaac-sim/python.sh` — prefer the explicit path in `docker exec` invocations from outside the container, where the alias is not active.

Each clone gets its own container (shared image, per-clone name), so clones run in parallel. **Don't hardcode the container name** — use the `dev-container` skill to build, start, attach to, discover, or exec into the local clone's container.

Run as the host user, not root.

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c \
  "cd /workspaces/isaaclab_arena && <command>"
```

Lint and format tooling (`pre-commit` and the hooks it runs — black, flake8, isort, pyupgrade, codespell) runs **on the host**.

## Repository layout

- `isaaclab_arena/` — core package: `tasks/`, `policy/`, `evaluation/`, `embodiments/`, `scene/`, `assets/`, `tests/`
- `isaaclab_arena_environments/`, `isaaclab_arena_examples/`, `isaaclab_arena_g1/`, `isaaclab_arena_gr00t/` — first-party extension packages
- `docker/` — container build and run scripts
- `submodules/` — vendored dependencies (IsaacLab, Isaac-GR00T, …)
- `osmo/` — OSMO policy-runner workflow
- `docs/` — Sphinx documentation

## Coding style

- Prefer `assert condition, "message"` over `if not condition: raise ValueError("message")` for internal invariant checks. (Formatting, imports, and typing are enforced by `pre-commit` — see `.pre-commit-config.yaml`.)
- PR bodies follow `.github/pull_request_template.md` — a one-line Summary plus 2–5 detail bullets. Resist the agent default of long, multi-section descriptions.
- Attribute docstrings should be included below the attribute, rather than in the class-level docstring.
- Copyright headers: a newly created file uses the current year alone (e.g. `2026`); a file created earlier and edited this year uses a range (e.g. `2025-2026`). Don't copy a neighbouring file's year — the pre-commit hooks (`insert-license`, `fix-new-file-copyright-year`) set and enforce this, so you generally don't hand-edit it.

## Docstrings style

- Prefer one line; a 2–3 line paragraph may follow if needed.
- The docstring should describe the function’s calling syntax and its semantics, but generally not its
    implementation details, unless those details are relevant to how the function is to be used.
- Document `Args` and `Returns`, but **not** `Raises`. Omit `Returns` when it only returns None
    or the summary already covers it.
- Don't use Sphinx-style cross-references.

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

- **Never** force-push to `main` or `release/*`. **Instead**, push to a `<username>/<type>/<short-description>` branch (`<type>` ∈ `feature`, `fix`, `docs`, `refactor`, `chore`, `ci`) and open a PR against `main`.
- **Never** add AI-attribution lines to commits (no `Co-Authored-By: Claude…`, no `Generated with…`). **Instead**, sign off with `git commit -s` — DCO is the only required trailer.
- **Never** commit models, datasets, or secrets. **Instead**, keep them on the host and mount them via `./docker/run_docker.sh -d <datasets> -m <models> -e <eval>`.
- **Ask first** before changing `docker/`, `.github/workflows/`, `.pre-commit-config.yaml`, or `submodules/` — these affect every contributor. **Instead** of pushing directly, open a draft PR or raise it in the relevant channel before merging.
