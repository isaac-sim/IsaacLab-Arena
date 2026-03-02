# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isaac-Lab Arena is a scalable robotics simulation environment creation and evaluation framework built on top of NVIDIA Isaac Lab. It provides a composable system for defining robotic tasks, assembling scenes from modular assets, solving spatial object relationships, and evaluating learning policies.

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

## Common Commands

### Running Tests
Tests require Isaac Sim and run via pytest. All simulation tests use `run_simulation_app_function()` to handle Isaac Sim's process lifecycle:

```bash
# Run all tests
pytest isaaclab_arena/tests/

# Run a single test file
pytest isaaclab_arena/tests/test_asset_registry.py

# Run a specific test function
pytest isaaclab_arena/tests/test_asset_registry.py::test_default_assets_registered

# Run tests that require cameras
pytest isaaclab_arena/tests/ -m with_cameras
```

### Linting and Formatting
Pre-commit hooks enforce: black (line length 120), flake8, isort, pyupgrade (py310+), codespell, and Apache-2.0 license headers.

```bash
# Install pre-commit hooks
pre-commit install

# Run all pre-commit checks
pre-commit run --all-files

# Run individually
black --line-length 120 --unstable <file>
flake8 <file>
isort --profile black <file>
```

### Contributing
All commits must be signed off per DCO requirements:
```bash
git commit -s -m "Your commit message"
```

## Architecture

### Core Composition Model

An environment is assembled by composing four components into an `IsaacLabArenaEnvironment` descriptor, which `ArenaEnvBuilder` then translates into Isaac Lab's `ManagerBasedRLEnvCfg`:

```
IsaacLabArenaEnvironment(name, scene, embodiment, task, [teleop_device], [orchestrator])
    ↓ ArenaEnvBuilder.make_rl() / make_mimic()
IsaacLabArenaManagerBasedRLEnvCfg
    ↓
IsaacLabArenaManagerBasedRLEnv (wraps isaaclab.envs.ManagerBasedRLEnv)
```

### Key Packages

- **`isaaclab_arena/`** — Main package
  - **`environments/`** — `IsaacLabArenaEnvironment` (descriptor), `ArenaEnvBuilder` (config composer), `IsaacLabArenaManagerBasedRLEnv` (runtime)
  - **`scene/`** — `Scene` assembles assets (objects, backgrounds, HDR, ground planes) and exports to USD
  - **`embodiments/`** — Robot definitions (Franka, GR1T2, G1, DROID, AgiBot, Galbot); each implements `EmbodimentBase`
  - **`tasks/`** — Task implementations (pick-and-place, open/close door, sorting, assembly, etc.); each implements `TaskBase`
  - **`assets/`** — `AssetRegistry`/`DeviceRegistry` singletons for dynamic asset lookup; `Object`, `Background`, `HDRImage`, `ObjectSet`
  - **`relations/`** — Spatial constraint solver: `ObjectPlacer` + `RelationSolver` optimize positions using loss primitives (NextTo, On, Above, etc.)
  - **`affordances/`** — Object capability descriptors: `Openable`, `Turnable`, `Pressable`, `Placeable`
  - **`policy/`** — Policy interface (`PolicyBase`) and implementations: `ZeroActionPolicy`, `RslRlActionPolicy`, `ReplayActionPolicy`, `ActionChunking`
  - **`evaluation/`** — `PolicyRunner`, `EvalRunner`, `JobManager` for running and recording evaluations
  - **`metrics/`** — Pluggable metrics (`MetricBase`): success rate, object moved, revolute joint moved rate
  - **`remote_policy/`** — Client/server policy execution over network (`PolicyServer`, `PolicyClient`)
  - **`terms/`** — Custom Isaac Lab manager terms (events, articulations, transforms)
  - **`cli/`** — Argument parser (`get_isaaclab_arena_cli_parser`)
  - **`utils/`** — Pose math, USD helpers, joint utilities, bounding boxes, RNG management

- **`isaaclab_arena_environments/`** — Pre-built environment definitions
- **`isaaclab_arena_examples/`** — Example scripts and notebooks
- **`isaaclab_arena_gr00t/`** — GR00T humanoid robot utilities
- **`isaaclab_arena_g1/`** — G1 robot-specific code
- **`submodules/IsaacLab/`** — Core Isaac Lab framework
- **`submodules/Isaac-GR00T/`** — GR00T implementation

### Test Infrastructure

Tests use `run_simulation_app_function()` (`isaaclab_arena/tests/utils/subprocess.py`) to wrap simulation tests. This creates a persistent `SimulationApp` (reused across tests in a session) and handles Isaac Sim's process-termination quirk—Isaac Sim would otherwise kill the pytest process with exit code 0 even on test failures. The `conftest.py` tracks test failures and forces exit code 1 when needed.

Test functions follow the pattern:
```python
def _test_foo(simulation_app):  # inner function runs inside SimulationApp
    from isaaclab_arena.X import Y   # deferred imports after sim init
    ...
    return True  # indicates pass

def test_foo():  # pytest-visible outer function
    result = run_simulation_app_function(_test_foo)
    assert result
```

### Configuration System

Configs are `@configclass` dataclasses (Isaac Lab pattern). `ArenaEnvBuilder` dynamically merges configs from scene, embodiment, task, and optional components using `combine_configclass_instances()`. The `env_cfg_callback` on `IsaacLabArenaEnvironment` allows post-merge customization.

### Relation Solver

Object placement in a scene can be specified declaratively using spatial relations (e.g., `NextTo(table)`, `On(shelf)`). `ObjectPlacer` collects all objects with relations, identifies the anchor object (`IsAnchor`), and runs `RelationSolver` which minimizes a sum of loss primitives (defined in `loss_primitives.py`) to find valid non-overlapping poses.
