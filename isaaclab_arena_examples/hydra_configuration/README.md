# Hydra environment configuration example

This example proves one small path end to end:

```text
hydra_example_suite.yaml
  -> ArenaRunConfiguration
  -> eval_runner Job
     (policy and rollout only; no environment CLI tokens)
  -> eval_runner.evaluate_jobs()
     -> PickAndPlaceMapleTableEnvironmentConfiguration
        (new Hydra-native definition; intended surviving API)
     -> argparse.Namespace compatibility bridge
        (temporary MVP adapter)
     -> PickAndPlaceMapleTableEnvironment
        (legacy registered factory; temporary delegate)
     -> IsaacLabArenaEnvironment -> ArenaEnvBuilder
     -> existing policy, rollout, cleanup, metrics, and report flow
```

`PickAndPlaceMapleTableEnvironmentConfiguration` is a typed environment definition and factory,
not the running simulation environment. For this MVP, its `build()` method translates the typed
fields to legacy CLI argument names and delegates to `PickAndPlaceMapleTableEnvironment`. The
returned `IsaacLabArenaEnvironment` is the runtime object consumed by `ArenaEnvBuilder`.

The intended migration is to move the legacy `get_env()` body into `build()`. The legacy
`PickAndPlaceMapleTableEnvironment`, its `add_cli_args()` method, and the `argparse.Namespace`
bridge can then disappear; the Hydra-native configuration class remains.

The example frontend composes Hydra before launching Isaac Sim, converts the policy and rollout
settings into one existing eval-runner `Job`, and injects an environment loader that builds directly
from the typed environment configuration. The job deliberately has no `arena_env_args`, so this path
does not reintroduce a dataclass-to-CLI round trip. Core evaluation code does not import the example
package; the existing JSON/argparse frontend continues to use `load_env()`.

The environment configuration is a Hydra ConfigStore node. Selecting
`environment: pick_and_place_maple_table` makes the YAML's `environment` block type-checked
against that environment's dataclass. Heavy Isaac Sim imports remain inside `build()`, so the YAML
can be composed and validated without starting the simulator. The example runs headless by default.

Run the co-located suite inside the Arena development container:

```bash
/isaac-sim/python.sh -m isaaclab_arena_examples.hydra_configuration.run
```

Like the existing eval runner, this writes episode results and an HTML report beneath
`/eval/output`.

Pass Isaac Lab's visualizer flag to open the Kit window:

```bash
/isaac-sim/python.sh -m isaaclab_arena_examples.hydra_configuration.run --viz kit
```

Other tokens are composed as Hydra overrides:

```bash
/isaac-sim/python.sh -m isaaclab_arena_examples.hydra_configuration.run \
  environment.light_intensity=750 rollout.num_steps=10
```

This MVP configures one eval job and exposes its policy and step budget. It reuses the eval runner's
policy lifecycle, cleanup, metrics, and HTML reporting, while leaving multi-job suites, recording,
rebuilds, chunking, and dynamic variations on their existing interfaces for later migration.
