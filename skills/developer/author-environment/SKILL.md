---
name: author-environment
description: Create or modify Isaac Lab-Arena environments, scenes, tasks, completion predicates, placement constraints, and environment variations by composing existing features. Use for environment authoring and feature discovery; use run-experiment to execute an existing evaluation.
---

# Author an Arena environment

Produce the smallest environment definition that expresses the request and follows a maintained
Arena example. Prefer configuration, then composition, then a small extension for a demonstrated gap.
Prefer writing the environment as a graph YAML, and write a Python environment only when the YAML
cannot express the user's intent.

## Discover before extending

Use `rg --files` to find relevant concepts under `docs/pages/concepts/`, examples under
`isaaclab_arena_environments/` and `isaaclab_arena_examples/`, and implementations in
`isaaclab_arena/` or the relevant first-party extension. Inspect the checked-out source when docs
and examples differ; Arena's APIs are evolving.

Use any relevant Arena feature, including untagged classes and Python APIs outside registries.
`@agent_ready` filters the built-in environment generator's prompt catalogues; it does not restrict
coding agents. The references below are starting points, not an exhaustive feature inventory.
Inspect the [registries](../../../isaaclab_arena/assets/registries.py) for registered components.
Load registries only in the supported runtime after SimulationApp startup; imports can require USD.

Identify reusable components for each requirement before implementing. Check constructors and their
inheritance chains before adding capabilities. If reuse cannot express a requirement, name the
missing capability and add the smallest extension. Do not duplicate solvers, predicates, temporal
counters, reset loops, progress tracking, or variation systems that Arena already provides.

Prefer Arena's existing tools to build, inspect and check an environment, listed under
"Authoring and validation tools" below. Write a diagnostic script only when these cannot answer the
question.

## Assemble from maintained sources

Read only the references relevant to the requested behavior:

- **Authoring surface:** start from a matching graph YAML example:
  - **Table-top scene:** follow a RoboLab task, such as
    [banana in bowl](../../../isaaclab_arena_environments/robolab/tasks/banana_in_bowl.yaml) for a
    single pick and place, or
    [apple and yogurt in bowl](../../../isaaclab_arena_environments/robolab/tasks/apple_and_yogurt_in_bowl.yaml)
    for two subtasks. More are under `isaaclab_arena_environments/robolab/tasks/`.
  - **Scene in a larger background,** such as a kitchen: follow a KitchenBench environment, such as
    [mustard in bowl](../../../isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_peninsula_mustard_bowl.yaml)
    for a single pick and place, or
    [banana and orange in bowl](../../../isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_banana_and_orange_in_bowl.yaml)
    for two subtasks with a robot that moves along the counter. More are under
    `isaaclab_arena_environments/kitchen_bench/`.

  Check the fields you use against the
  [schema](../../../isaaclab_arena/environment_spec/arena_env_graph_types.py) and the
  [task conversion](../../../isaaclab_arena/environment_spec/arena_env_graph_task_conversion_utils.py).
  Only when the YAML cannot express a requirement, write Python following the
  [factory contract](../../../isaaclab_arena/environments/arena_environment_factory.py), and reuse
  [builder configuration](../../../isaaclab_arena/environments/arena_env_builder_cfg.py) for
  existing build and reset controls.
- **Assets:** follow the example's asset ownership and check identity, collision handling, and reset
  requirements. Choose `object_reference.prim_path` values with the scene inspector; see
  "Authoring and validation tools" below.
- **Layout:** read [placing objects and the robot](placement.md). For the full design, read
  [relations](../../../docs/pages/concepts/object_placement/relations.rst),
  [placement pools](../../../docs/pages/concepts/object_placement/pooled_placement.rst), and the
  [validation workflow](../../../docs/pages/concepts/object_placement/validation.rst).
  Follow the example's scene composition, and check physical support. When placement rejects many
  candidates or a fill fails, read the placement diagnostics before changing relations, collision
  settings or reachability.
- **Task semantics:** read [tasks in graph YAML](tasks.md). For the full design, read
  [task composition](../../../docs/pages/concepts/task/concept_composite_tasks_design.rst)
  and [predicates and progress](../../../docs/pages/concepts/task/concept_progress_tracking_design.rst)
  for ordered milestones, current final conditions, simultaneous truth, temporal holds, and reporting.
  Reuse existing tasks and preserve their sensors and failure conditions.
- **Variations:** read the [variation concepts](../../../docs/pages/concepts/variations/variations.rst)
  and inspect asset constructors, including [object base classes](../../../isaaclab_arena/assets/object_base.py),
  for existing attachments before adding one. Verify names, units, and sampling timing against the
  relevant implementation in `isaaclab_arena/variations/`.
- **Companion Experiments:** follow the [Experiment concepts](../../../docs/pages/concepts/concept_arena_experiments.rst)
  and [variation example](../../../isaaclab_arena_environments/experiment_configs/droid_pnp_variations_experiment.yaml).
  Check effective defaults across every supplied Run and constraints in the
  [Run configuration](../../../isaaclab_arena/evaluation/arena_run.py). Keep variations opt-in and
  cameras optional unless requested. Keep execution settings separate from task semantics.

## Validate and hand off

Use [dev-container](../dev-container/SKILL.md) for the clone's runtime and host-user execution,
[setup-arena](../../user/setup-arena/SKILL.md) for requested runtime preparation, and
[run-tests](../run-tests/SKILL.md) for pytest. Run lint and source inspection on the host.
If no runtime is ready, complete static review and report the limit; do not install dependencies
or claim runtime success. Authoring alone does not require policy rollouts or policy servers.

Check Python/YAML syntax, referenced constructors, registration, schema conversion, defaults, and
feature reuse without importing Arena on the host. In a ready runtime, build and reset the
environment, then run the placement pool validation. Read the candidate pass counts and the
success configuration in the build log, and the placement diagnostics if few layouts pass.
"Authoring and validation tools" below lists the tool for each step. The environment builds and
resets cleanly when both steps pass; stop there, without running zero-action episodes (unless
requested by the user). Use existing focused tests to check changed success logic, temporal
interruptions, and reset isolation across parallel environments. Parsing or passive stepping alone
does not establish task solvability.

When the user requests physics-based validation of an authored environment without a robot policy,
use the Object Control Runner.

For variation inspection, use the `--list_variations` path through
[run-experiment](../../user/run-experiment/SKILL.md). Check the
[Environment Runner](../../../isaaclab_arena/scripts/environment_runner.py) requirements before
using it for interactive inspection; do not assume it supports headless validation.

Finish with the reference example, reused features, any extension rationale, definition/configuration
paths, checks actually run, and remaining simulation validation. Keep this in the handoff; no extra
design document is required.

## Authoring and validation tools

- **Find prim paths in a background:**
  [`scene_inspector.py`](../../../isaaclab_arena_examples/agentic_environment_generation/scene_inspector.py)
  `prim-tree`. Lists a USD scene's authored physics hierarchy as text, or as JSON for one subtree,
  without starting the simulator. Use it to choose `object_reference.prim_path` values and support
  prims to anchor. Browse the text first, then request JSON only for the relevant subtree. It shows
  authored USD data, without Arena and Isaac Lab spawn-time overrides.
- **Validate environment code:**
  [`cli_runner.py`](../../../isaaclab_arena_examples/agentic_environment_generation/cli_runner.py)
  `--mode build`. Builds the environment, resets it once and takes a few steps. The log shows how
  many candidate layouts passed placement and the effective success configuration.
- **Validate the placement layouts:**
  [`run_placement_pool_validation.py`](../../../isaaclab_arena/scripts/run_placement_pool_validation.py).
  Applies every layout stored in the placement pool, as a reset would, and checks that each one
  settles under physics.
- **Find why layouts are rejected:**
  [placement diagnostics](../../../docs/pages/concepts/object_placement/placement_diagnostics.rst).
  Reports, per check (relations, `no_overlap`, `ik_reachable`), how many candidates passed and why
  the rest were rejected. Printed in the build log; an Experiment Run also writes
  `arena_run_placement_diagnostics.json`.
- **Check the task can be completed:**
  [Object Control Runner](../../../docs/pages/agentic_tools/object_control.rst)
  ([source](../../../isaaclab_arena/agentic_tools/object_control_runner.py)). Drives objects to
  pose and joint-position targets with bounded forces, without a robot policy, checks the task's
  success predicates, and diagnoses why an interaction or success condition fails. Its
  documentation covers recipe and live-command usage, commands and outputs.
- **List environment variations:** `--list_variations` through
  [run-experiment](../../user/run-experiment/SKILL.md). Lists the variations an environment
  declares, without running rollouts.
