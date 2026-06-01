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

### Linting, Formatting

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

### Coding Style

- Prefer `assert` over `if-then-raise ValueError` for internal invariant checks. Use `assert condition, "message"` instead of `if not condition: raise ValueError("message")`.

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

The docstring may be descriptive-style ("""Fetches rows from a Bigtable.""") or imperative-style ("""Fetch rows from a Bigtable."""), but the style should be consistent within a file. The docstring for a @property data descriptor should use the same style as the docstring for an attribute or a function argument ("""The Bigtable path.""", rather than """Returns the Bigtable path.""").

Certain aspects of a function should be documented in special sections, listed below. Each section begins with a heading line, which ends with a colon. All sections other than the heading should maintain a hanging indent of two or four spaces (be consistent within a file). These sections can be omitted in cases where the function’s name and signature are informative enough that it can be aptly described using a one-line docstring.

#### Args
List each parameter by name. A description should follow the name, and be separated by a colon followed by either a space or newline. If the description is too long to fit on a single 80-character line, use a hanging indent of 2 or 4 spaces more than the parameter name (be consistent with the rest of the docstrings in the file). The description should include required type(s) if the code does not contain a corresponding type annotation. If a function accepts *foo (variable length argument lists) and/or **bar (arbitrary keyword arguments), they should be listed as *foo and **bar.

#### Returns: (or Yields: for generators)
Describe the semantics of the return value, including any type information that the type annotation does not provide. If the function only returns None, this section is not required. It may also be omitted if the docstring starts with “Return”, “Returns”, “Yield”, or “Yields” (e.g. """Returns row from Bigtable as a tuple of strings.""") and the opening sentence is sufficient to describe the return value. Do not imitate older ‘NumPy style’ (example), which frequently documented a tuple return value as if it were multiple return values with individual names (never mentioning the tuple). Instead, describe such a return value as: “Returns: A tuple (mat_a, mat_b), where mat_a is …, and …”. The auxiliary names in the docstring need not necessarily correspond to any internal names used in the function body (as those are not part of the API). If the function uses yield (is a generator), the Yields: section should document the object returned by next(), instead of the generator object itself that the call evaluates to.

#### Raises
We don't document raises in Arena

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
