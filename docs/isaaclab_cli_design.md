# Arena integration with the Isaac Lab CLI

This proposal reflects Arena `e5e7aae2af` and its pinned Isaac Lab commit `53d23c8581`.

## Motivation

Arena workflows currently expose repository script paths and require explicit registration
callbacks when launched through Lab. External environments add more import paths and
registration steps.

Use `uv run isaaclab` as the common launcher, with Lab owning command dispatch and Arena
owning environment construction and evaluation. `uv` selects the installed environment;
Python entry points provide discovery.

## Goals

- Launch Arena evaluations without referring to repository scripts.
- Run Arena environments through Lab's existing training and demonstration workflows without
  specifying `--external_callback`.
- Discover environments installed from external packages.
- Preserve existing Experiment YAML, Hydra overrides, direct scripts, and explicit callbacks.
- Keep command discovery lightweight and defer simulation-dependent environment construction.

## No-Goals

- Replace Lab's training, teleoperation, or simulation implementations.
- Redesign Arena's environment builder or Experiment configuration.
- Unify every workflow's argument parser.
- Expose every Arena script as a new subcommand in this patch.

## High-level APIs

### User-facing commands

```bash
uv run isaaclab arena evaluate --experiment-config experiment.yaml
uv run isaaclab train --rl_library rsl_rl --task <environment>
uv run isaaclab play --rl_library rsl_rl --task <environment>
uv run isaaclab teleop record --task <environment>
uv run isaaclab teleop replay --task <environment> --dataset_file demos.hdf5
```

Replace `<environment>` with the registered environment name and supply any required
workflow-specific arguments, such as a playback checkpoint.

### Extension points

Three entry-point groups separate responsibilities:

| Entry-point group | Published by | Responsibility |
| --- | --- | --- |
| `isaaclab.cli` | Arena or another extension | Add a top-level command such as `arena`. |
| `isaaclab.tasks` | Arena | Import lightweight Gym task registrations. This Lab mechanism already existed. |
| `isaaclab_arena.environments` | External environment packages | Register named Arena environment factories and their typed configurations. |

Arena publishes:

```toml
[project.entry-points."isaaclab.cli"]
arena = "isaaclab_arena.cli.entrypoint:main"

[project.entry-points."isaaclab.tasks"]
arena = "isaaclab_arena.environments.isaaclab_task_registration"
```

An external package publishes:

```toml
[project.entry-points."isaaclab_arena.environments"]
my_environment = "my_package.environments:MyEnvironment"
```

The external entry-point name must match the factory's `name`, and the package must be
installed in the same Python environment as Arena.

### Execution

- **Arena evaluation:** Lab loads the selected command and forwards its arguments to Arena's
  `main(argv)`, which delegates to the existing Experiment Runner.
- **Lab workflows:** Arena registers Gym placeholders containing `task_registration_callback`.
  Lab invokes the selected task's callback before loading its full configuration. Arena
  constructs and registers the environment, consumes its arguments, and returns the arguments
  remaining for Lab/Hydra. An explicit `--external_callback` takes precedence.

Lab also applies its existing runtime setup to external commands. Command-name collisions are
rejected, and top-level help discovers command names without importing their implementations.

## What are missing

1. **A released Lab dependency contract.** The implementation requires the pinned Lab change.
   Document or check the required capability/version; installing Arena into an older Lab
   environment will not enable these hooks automatically.
2. **Broader workflow coverage.** Automatic registration currently covers RSL-RL train/play and
   demo record/replay. `teleop run`, zero/random agents, other RL backends, and export paths do
   not call the new registration helper. Either extend coverage or clearly document the
   supported subset.
3. **Consistent configuration forwarding.** The Lab interop path still uses the legacy argparse
   adapter. It supports registered variation overrides but does not forward builder overrides
   or recorded-variation replay configuration completely.
4. **Installation and end-to-end validation.** Add fresh-install tests using real package
   metadata, an external environment package, and execution outside the repository directory.
   Verify a short evaluation, training run, and recording/replay cycle. The 17 focused tests
   passed after rebasing, but they do not establish these full workflows.
5. **A cleaner callback contract as follow-up.** Registration currently temporarily replaces
   `sys.argv` and calls a zero-argument callback. Prefer an explicit `callback(argv)` interface,
   retaining an adapter for existing callbacks. This would make argument ownership clearer and
   testing simpler.
