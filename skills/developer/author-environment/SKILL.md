---
name: author-environment
description: Create or modify Isaac Lab-Arena environments, scenes, tasks, completion predicates, placement constraints, and environment variations by composing existing features. Use for environment authoring and feature discovery; use run-experiment to execute an existing evaluation.
---

# Author an Arena environment

Produce the smallest environment definition that expresses the request and follows a maintained
Arena example. Prefer configuration, then composition, then a small extension for a demonstrated gap.

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

## Assemble from maintained sources

Read only the references relevant to the requested behavior:

- **Authoring surface:** start from a matching existing environment. For Python, follow the
  [maintained Maple-table factory](../../../isaaclab_arena_environments/pick_and_place_maple_table_environment.py)
  and [factory contract](../../../isaaclab_arena/environments/arena_environment_factory.py).
  Reuse [builder configuration](../../../isaaclab_arena/environments/arena_env_builder_cfg.py) for
  existing build and reset controls; keep environment configuration focused on the scene and task.
  For graph YAML, inspect the [schema](../../../isaaclab_arena/environment_spec/arena_env_graph_types.py)
  and [task conversion](../../../isaaclab_arena/environment_spec/arena_env_graph_task_conversion_utils.py)
  alongside `isaaclab_arena_environments/robolab/` examples. Choose Python when the graph cannot
  express the required semantics; a field parsing successfully does not prove it is consumed.
- **Assets and layout:** follow the example's ownership and scene composition. Use the existing
  [relations](../../../docs/pages/concepts/object_placement/relations.rst),
  [placement pools](../../../docs/pages/concepts/object_placement/pooled_placement.rst), and
  [validation workflow](../../../docs/pages/concepts/object_placement/validation.rst).
  Check asset identity, collision handling, reachability, physical support, and reset requirements.
- **Task semantics:** read [task composition](../../../docs/pages/concepts/task/concept_composite_tasks_design.rst)
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
environment, then use bounded direct steps for inspection. Use existing focused tests to check
changed success logic, temporal interruptions, and reset isolation across parallel environments.
Parsing or passive stepping alone does not establish task solvability.

For variation inspection, use the `--list_variations` path through
[run-experiment](../../user/run-experiment/SKILL.md). Check the
[Environment Runner](../../../isaaclab_arena/scripts/environment_runner.py) requirements before
using it for interactive inspection; do not assume it supports headless validation.

Finish with the reference example, reused features, any extension rationale, definition/configuration
paths, checks actually run, and remaining simulation validation. Keep this in the handoff; no extra
design document is required.
