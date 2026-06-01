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

Commands that touch Isaac Sim or Arena's package code (tests, training, evaluation, runtime scripts) run inside the `isaaclab_arena-latest` Docker container. The repo root is mounted at `/workspaces/isaaclab_arena`. Inside the container, `python` is aliased to `/isaac-sim/python.sh` — prefer the explicit path in `docker exec` invocations from outside the container, where the alias is not active.

Lint and format tooling (`pre-commit` and the hooks it runs — black, flake8, isort, pyupgrade, codespell) runs **on the host**.

Use the `dev-container` skill for build, start, attach, and exec inside the container.

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

## Coding Style

### Checks/Asserts

In Arena we prefer asserts over if...raises in most cases where such an check represents a coding error
(i.e) something that shouldn't be recovered from. Asserts are briefer than the exception based counterpart.
The raises-based approach is more verbose (for example it always uses multiple lines), and because
this type of error is not intended to be recovered from, it offers no advantage.

### Docstrings

Prefer short docstrings. Where a single line is not sufficient, a short (i.e. 2-3 line)
paragraph can follow. Document args and returns, but not raises.
We follow google-style for docstrings which is repeated below:

A docstring should give enough information to write a call to the function without reading the function’s code. The docstring should describe the function’s calling syntax and its semantics, but generally not its implementation details, unless those details are relevant to how the function is to be used. For example, a function that mutates one of its arguments as a side effect should note that in its docstring. Otherwise, subtle but important details of a function’s implementation that are not relevant to the caller are better expressed as comments alongside the code than within the function’s docstring.

#### Args
List each parameter by name. A description should follow the name, and be separated by a colon followed by either a space or newline. If the description is too long to fit on a single 80-character line, use a hanging indent of 2 or 4 spaces more than the parameter name (be consistent with the rest of the docstrings in the file). The description should include required type(s) if the code does not contain a corresponding type annotation. If a function accepts *foo (variable length argument lists) and/or **bar (arbitrary keyword arguments), they should be listed as *foo and **bar.

#### Returns: (or Yields: for generators)
Describe the semantics of the return value, including any type information that the type annotation does not provide. If the function only returns None, this section is not required. It may also be omitted if the docstring starts with “Return”, “Returns”, “Yield”, or “Yields” (e.g. """Returns row from Bigtable as a tuple of strings.""") and the opening sentence is sufficient to describe the return value. Do not imitate older ‘NumPy style’ (example), which frequently documented a tuple return value as if it were multiple return values with individual names (never mentioning the tuple). Instead, describe such a return value as: “Returns: A tuple (mat_a, mat_b), where mat_a is …, and …”. The auxiliary names in the docstring need not necessarily correspond to any internal names used in the function body (as those are not part of the API). If the function uses yield (is a generator), the Yields: section should document the object returned by next(), instead of the generator object itself that the call evaluates to.

#### Raises
We don't document raises in Arena

#### Configclasses
Attribute docstrings should be included below the attribute, rather than in the class-level docstring.

#### Example
Here is an example docstring

```python
def fetch_smalltable_rows(
    table_handle: smalltable.Table,
    keys: Sequence[bytes | str],
    require_all_keys: bool = False,
) -> Mapping[bytes, tuple[str, ...]]:
    """Fetches rows from a Smalltable.
    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.
    Args:
        table_handle: An open smalltable.Table instance.
        keys: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: If True only rows with values set for all keys will be
          returned.
    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings.
    """
```


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
