# Hydra environment configuration example

This example proves one small path end to end:

```text
hydra_example_suite.yaml
  -> EnvironmentRegistry name lookup
     -> registered environment provider + concrete Cfg type
  -> Hydra environment config-group composition
  -> two independently composed ArenaExperimentCfg values
  -> two eval_runner Jobs
     (generic evaluation fields and variations; no environment CLI tokens)
  -> eval_runner.evaluate_jobs()
     -> PickAndPlaceMapleTableEnvironmentCfg from each experiment
     -> registered PickAndPlaceMapleTableEnvironment.build(cfg)
     -> IsaacLabArenaEnvironment -> ArenaEnvBuilder
     -> existing policy, rollout, cleanup, metrics, and report flow
```

The first YAML experiment ports
`isaaclab_arena_environments/eval_jobs_configs/droid_pnp_variations_config.json` into the typed
configuration shape. Its environment, zero-action policy, rollout, camera, HDR, rebuild, and three
variation settings are preserved. The second experiment is the matching no-variations control and
keeps the example exercising sequential experiments in one simulation app.

`PickAndPlaceMapleTableEnvironmentCfg` is pure structured data. The registered
`PickAndPlaceMapleTableEnvironment` provider advertises that Cfg type and owns `build(cfg)`. The
returned `IsaacLabArenaEnvironment` is the assembled environment consumed by `ArenaEnvBuilder`.

The existing argparse frontend remains compatible: `get_env()` translates its legacy Namespace
into the same Cfg and delegates to `build()`. Once that frontend is migrated, `get_env()`,
`add_cli_args()`, and the Namespace translation can disappear without changing the Cfg or builder.

The YAML's top-level `experiments` list is only a composition envelope. Each experiment identifies
its environment by the registry name under `environment.name`; it contains no Hydra `defaults`
list. The frontend projects registered providers that expose a Cfg type into Hydra's `environment`
config group and constructs the composition defaults in memory. Hydra then replaces the experiment
schema's required `environment` field with the selected concrete node before applying that
experiment's YAML values.

The reusable `ArenaExperimentCfg`, `PolicyCfg`, `RolloutCfg`, and `EnvironmentBuilderCfg` types live
in core evaluation code. `ArenaExperimentCfg` depends only on the core `ArenaEnvironmentCfg`, not on
Maple-table. Adding another environment requires a registered provider with a concrete Cfg type;
the experiment composer does not need another environment-specific branch. Simulator-dependent
construction remains inside `build()`, so every experiment is composed and validated before the
simulator starts.

The frontend then converts each typed experiment's generic evaluation fields into an eval-runner
`Job` and injects an environment loader that resolves the matching provider through
`EnvironmentRegistry` and calls `build(cfg)`. The generated Jobs deliberately have no
`arena_env_args`, so this path does not introduce a dataclass-to-CLI round trip. Core evaluation
code does not import the example package; the existing JSON/argparse frontend continues to use
`load_env()`.

`run.py` separates Hydra composition, existing-API adapters, and dispatcher integration explicitly.
Hydra composes core `ArenaExperimentCfg` values. The named adapters convert experiments to the
`Job` and `argparse.Namespace` shapes still expected by `eval_runner` and `ArenaEnvBuilder`. The
dispatcher separately owns process-wide camera aggregation, provider lookup, environment
construction, and sequential execution. Those responsibilities remain when the consumers
eventually accept the core Cfg types directly; only their compatibility translations can disappear.

The legacy JSON's `enable_cameras` value remains part of each typed environment configuration.
Before this example dispatcher launches one shared `SimulationApp`, it enables process-wide camera
support when any experiment requires it. A future dispatcher can instead group compatible
experiments or send their Jobs to separate workers without changing `ArenaExperimentCfg`.
Environment CLI names such as `embodiment` and `hdr` become typed fields such as
`embodiment_asset_name` and `high_dynamic_range_image_name`. The dotted keys under `variations`
remain experiment data and are passed into the existing `ArenaEnvBuilder` Hydra-variation channel
when the eval-runner `Job` is created.

Run the co-located configuration inside the Arena development container:

```bash
/isaac-sim/python.sh -m isaaclab_arena_examples.hydra_configuration.run \
  isaaclab_arena_examples/hydra_configuration/hydra_example_suite.yaml
```

The YAML path is a required positional argument; `run.py` has no implicit suite configuration.

Like the existing eval runner, this writes episode results and an HTML report beneath
`/eval/output`.

Pass Isaac Lab's visualizer flag to open the Kit window:

```bash
/isaac-sim/python.sh -m isaaclab_arena_examples.hydra_configuration.run \
  isaaclab_arena_examples/hydra_configuration/hydra_example_suite.yaml \
  --viz kit
```

Hydra overrides are applied independently to every configured experiment:

```bash
/isaac-sim/python.sh -m isaaclab_arena_examples.hydra_configuration.run \
  isaaclab_arena_examples/hydra_configuration/hydra_example_suite.yaml \
  environment.light_intensity=750 rollout.num_steps=10
```

The current `run.py` dispatcher evaluates both experiments sequentially inside one simulation app.
Sharing that app is an execution choice, not part of the experiment configuration model. The
example reuses the eval runner's variation, policy, rollout, cleanup, metrics, and HTML reporting
paths; the source rebuild count is preserved. Recording, chunking, migrating additional registered
environments to typed Cfg providers, and broader policy selection remain for later work.
