---
name: author-environment
description: Create or modify Isaac Lab-Arena environments, scenes, tasks, completion predicates, placement constraints, and environment variations by composing existing features. Use for environment authoring and feature discovery; use run-experiment to execute an existing evaluation.
---

# Author an Arena environment

Produce the smallest environment definition that expresses the requested behavior and follows the
current Arena examples. Work from the checked-out APIs; this alpha project's older tutorials may
use compatibility interfaces. Paths below are relative to the repository root unless linked.

## Discover before extending

Read the relevant rows of the [feature map](references/feature-map.md), then the linked source and
nearest maintained example. Use source search on the host; load Arena registries only in the
supported runtime after SimulationApp startup. Registry access can import USD transitively.

For each requirement, identify the existing component and the configuration or composition that
uses it. Keep this mapping brief in the work notes or final handoff; no extra design document is
required. If a requirement needs new behavior, state the missing capability before implementing
the smallest extension. Do not infer missing support from a single example or a filtered agent
catalogue; search the full registry implementation and feature directory.

## Choose the authoring surface

- Reuse an existing environment configuration when it already expresses the request.
- Use graph YAML for registered assets, spatial relations, and supported atomic/parallel/sequential
  task composition. Follow `isaaclab_arena_environments/robolab/tasks/` and `scenes/`; use
  `external_yaml` to share an existing scene. Read `environment_spec/arena_env_graph_types.py`
  and `arena_env_graph_task_conversion_utils.py` under `isaaclab_arena/` before adding task fields.
  YAML parsing alone does not prove fields are consumed. In particular, the current composite
  graph task does not expose `desired_subtask_success_state` or arbitrary predicate composition.
- Use Python when required semantics or integrations are not exposed in the graph schema. Follow
  `isaaclab_arena_environments/pick_and_place_maple_table_environment.py`: a dataclass derived from
  `ArenaEnvironmentCfg`, a directly parameterized `ArenaEnvironmentFactory[ConcreteCfg]`,
  `@register_environment`, and `build(cfg)` returning `IsaacLabArenaEnvironment`.
  Keep simulation imports inside `build` and type-only imports under `TYPE_CHECKING`.
  Give config fields usable defaults; pass `cfg.enable_cameras` to the embodiment. Follow the
  example's `_legacy_argparse_cfg_type` bridge when exposing the factory to current CLI runners;
  do not add another parser or extend deprecated `ExampleEnvironmentBase`.

Keep the environment focused on assembling `Scene`, embodiment, and task. Use task constructor
parameters for semantics and builder/Experiment configuration for execution settings. Do not edit
core APIs merely to make a one-off environment fit YAML. New registered modules in
`isaaclab_arena_environments/` are discovered automatically; do not add imports to `__init__.py`.

## Compose the requested behavior

**Assets and placement.** Verify registry names, constructor parameters, affordances, and prim
paths against the source/example. Give repeated asset instances distinct `instance_name` values.
Include task destinations and references in `Scene.assets`. Use `ObjectReference` for a prim
owned by another asset; do not spawn a second physics owner for it. Use `RigidObjectSet` only when
object identity should differ between environments, not as an episode-level object randomizer.

Declare `IsAnchor`, `On`, and other relations on assets and let `ArenaEnvBuilder` own solving,
collision checking, validation, and pooled reset placement. Do not add manual random-position
reset events or call `RelationSolver` from the factory. Do not set an explicit initial pose on the
same movable asset whose pose is relation-solved. `resolve_on_reset=True` draws layouts from the
pool; it does not promise a fresh solve or a unique layout on every reset. Use `placement_seed`
for repeatable layouts. If invalid fallback layouts would violate the request, use
`ObjectPlacerParams(allow_best_loss_fallbacks=False)` and retain required validation checks.
Check reachability and physical support through the existing placement validation workflow.

**Tasks and predicates.** Start with an existing task, such as `PickAndPlaceTask`, which already
supplies settling/lifting/placement predicates, contact sensors, drop failure, and progress criteria.
Compose a flat list with `CompositeTaskBase`; nested composites are unsupported. Set
`subtasks_are_sequential=True` only when order is required. Set the overall episode timeout
explicitly or use the composite's sum of subtask timeouts.

- Completed milestones latch until reset. Use `desired_subtask_success_state` when earlier
  subtask final conditions must still hold at success. For sequential tasks, `None` ignores a
  final condition but still requires that stage to finish before the next stage starts.
- `CompletionCriteria.predicate_sequence` orders milestones. Named `predicate_sequences` with
  `logical="all"`, `"any"`, or `"choose"` (`K`) combine independently completed sequences;
  they do not enforce simultaneous truth.
- For conditions that must hold together, combine existing instantaneous predicates into one
  Boolean tensor condition. Use `TrueForConsecutiveStepsCfg` for shared consecutive-step truth;
  do not write counters. Prefer task parameters such as `placement_consecutive_steps` when they
  already express the requirement. Preserve each existing task's sensors and failure conditions.
- Add a custom predicate only for a missing condition. It must return one Boolean per environment
  on the correct device; use Arena's scene access and frame conventions. Let `TaskSuccessTerm`
  advance `ProgressTracker` once per control step. Read recorded states/events and subtask
  completion for reporting; do not advance tracking again from metrics or maintain a second tracker.

**Variations.** Inspect the constructor inheritance chain, not just the concrete asset class:
`RootedObjectBase` in `isaaclab_arena/assets/object_base.py` already attaches `mass` and `disappear`
to rigid objects. Light and embodiment constructors also attach their supported variations.
Configure these existing variations; do not attach them again. `Asset.add_variation` rejects
duplicate names. In a ready runtime, `asset.get_variations()` lists attached variations and
`asset.get_variation(name)` retrieves one. Add a variation only when it is absent from the full
inheritance chain and required by the task. Use actual scene instance names in override keys.
Mass sampler bounds are absolute kilograms, not scale factors. Keep variations opt-in and cameras
optional unless requested. When defaults must remain off, disable them in every Run of a supplied
Experiment: naming a Run `variations_demo` does not make it optional; the runner executes all Runs.
Configure Experiment `runs.<name>.variations` with dotted keys as in
`experiment_configs/droid_pnp_variations_experiment.yaml` under `isaaclab_arena_environments/`.
Build-time variations need rebuilds to obtain more samples; reset-time variations sample per
resetting environment. Do not create duplicate reset events or another randomization system.

## Validate within the requested scope

Use `dev-container` to discover the clone's container and run Arena code as the host user. For
runtime preparation use `setup-arena`; for pytest follow `run-tests`. Run formatting/lint on the
host. If no runtime is ready, report that limit and complete static review instead of installing
dependencies or claiming runtime success. Environment authoring alone does not require policy
rollouts, model downloads, or policy servers.

Scale validation to the behavior changed:

- Check Python syntax, YAML, referenced symbols/constructor arguments, and skill links on the host
  without importing Arena. Review registration, defaults, frame conventions, and feature reuse.
- In a ready SimulationApp, load the graph or build the factory and inspect the compiled config.
  Use `ArenaEnvBuilder.make_registered()`, reset, and bounded direct environment steps to inspect
  placement, termination configuration, observations, and progress without a policy. Include
  multiple environments and a subset reset when validating per-environment state.
- For changed success logic, check early/out-of-order conditions, interruption of a temporal
  streak, removal of a previously placed object, and reset of progress. Reuse the focused tests
  for completion criteria, task success, and temporal composition; add behavioral coverage only
  for genuinely new logic. A zero-action step cannot demonstrate task solvability.
- To inspect variation paths without rollouts, use `experiment_runner.py --list_variations
  --experiment_config <file>` through `run-experiment`. Do not omit `--list_variations` for an
  inspection-only request. The interactive `environment_runner.py` requires one environment,
  Kit, and CPU PhysX; it is not a headless validation command.

Finish by naming the reference example, reused features, any custom behavior and why it is needed,
the definition/configuration paths, checks actually run, and remaining simulation validation.
