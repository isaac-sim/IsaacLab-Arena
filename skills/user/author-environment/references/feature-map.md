# Arena feature discovery

Paths in this map are relative to the repository root. Read only the rows relevant to the request.
The source directories and registries are the inventory; this map routes to them rather than
duplicating lists of every asset, parameter, or class.

| Need | Source of truth | Maintained example or concept |
| --- | --- | --- |
| Environment configuration, registration, builder | `isaaclab_arena/environments/arena_environment_factory.py`, `arena_env_builder.py`, `arena_env_builder_cfg.py`; `isaaclab_arena/assets/register.py` | `isaaclab_arena_environments/pick_and_place_maple_table_environment.py`; `docs/pages/concepts/environment/environment_definition.rst` |
| Declarative graph, scene reuse, node/parameter validation | `isaaclab_arena/environment_spec/arena_env_graph_spec.py`, `arena_env_graph_types.py`, `arena_env_graph_task_conversion_utils.py`, `arena_env_graph_yaml_loader.py` | `isaaclab_arena_environments/robolab/scenes/`, `robolab/tasks/`; `docs/pages/concepts/environment/environment_definition.rst` |
| Asset/embodiment/task/relation/device catalogues | `isaaclab_arena/assets/registries.py`, `register.py`, `object_library.py`, `background_library.py`; `isaaclab_arena/tasks/task_library.py`; `isaaclab_arena/embodiments/` | `docs/pages/concepts/scene/concept_assets_design.rst`; `isaaclab_arena/agentic_environment_generation/catalogues.py` (agent-filtered subset) |
| Scene ownership, references, rigid/deformable objects, cables | `isaaclab_arena/scene/scene.py`; `isaaclab_arena/assets/object_reference.py`, `object.py`, `deformable_object.py`, `cable.py`, `simready_object_library.py` | `isaaclab_arena_environments/franka_put_and_close_door_environment.py`, `droid_deformable_pick_and_place_environment.py` |
| Asset capabilities for doors, knobs, buttons, placement | `isaaclab_arena/affordances/` | `docs/pages/concepts/scene/concept_affordances_design.rst`; `isaaclab_arena_environments/gr1_open_microwave_environment.py`, `gr1_turn_stand_mixer_knob_environment.py`, `press_button_environment.py` |
| Heterogeneous objects sharing a task | `isaaclab_arena/assets/object_set.py` | `isaaclab_arena_environments/droid_table_multi_object_placement_environment.py`; `docs/pages/concepts/scene/concept_rigid_object_set.rst` |
| Relational object and robot placement, anchors, bounds, orientation | `isaaclab_arena/relations/relations.py`, `object_placer_params.py`, `relation_solver_params.py`; `isaaclab_arena/environments/relation_solver_interface.py` | `docs/pages/concepts/object_placement/relations.rst`, `solver.rst`; `isaaclab_arena_examples/relations/` |
| Collision modes, passive obstacles, reachability, validation | `isaaclab_arena/relations/collision_mode.py`, `passive_collision_objects.py`, `reachability_config.py`, `validation/` | `docs/pages/concepts/object_placement/collision_handling.rst`, `validation.rst`; `isaaclab_arena_environments/gr1_table_multi_object_no_collision_environment.py` |
| Layout pools, seeded resets, fixed/replayed placements | `isaaclab_arena/relations/pooled_object_placer.py`, `placement_layouts.py`, `placement_events.py` | `docs/pages/concepts/object_placement/pooled_placement.rst`; `isaaclab_arena_environments/experiment_configs/settled_placement_replay_experiment.yaml` |
| Physics settling, clutter, offline placement recording | `isaaclab_arena/offline_placement/`; `isaaclab_arena/scripts/record_placement_layouts.py`, `run_placement_pool_validation.py` | `docs/pages/concepts/offline_placement/recording.rst`, `clutter.rst` |
| Atomic manipulation, insertion, assembly, goal poses, sorting | `isaaclab_arena/tasks/`, `task_base.py`, `task_termination_cfg.py` | `isaaclab_arena_environments/tabletop_peginsert_environment.py`, `tabletop_gearmesh_environment.py`, `tabletop_place_upright_environment.py`, `sorting_environment.py` |
| Sequential/parallel tasks, final success conditions | `isaaclab_arena/tasks/composite_task_base.py` | `docs/pages/concepts/task/concept_composite_tasks_design.rst`; `isaaclab_arena_environments/franka_put_and_close_door_environment.py` |
| Predicates, ALL/ANY/CHOOSE, temporal requirements | `isaaclab_arena/tasks/predicates/`; `isaaclab_arena/progress_tracking/completion_criteria.py` | `docs/pages/concepts/task/concept_progress_tracking_design.rst`; `isaaclab_arena/tests/test_temporal_predicate_composition.py` |
| Subtask progress, events, metrics, episode/trajectory recording | `isaaclab_arena/progress_tracking/`, `recording/`, `metrics/`; `isaaclab_arena/tasks/composite_task_base.py` | `docs/pages/concepts/task/concept_progress_tracking_design.rst`, `concept_metrics_design.rst`; `isaaclab_arena/tests/test_task_success_from_progress.py` |
| Lighting, HDR, cameras, mass, disappearance, samplers | `isaaclab_arena/variations/`; inherited attachments in `isaaclab_arena/assets/object_base.py`, `object_library.py`; `isaaclab_arena/embodiments/embodiment_base.py` | `docs/pages/concepts/variations/variations.rst`; `isaaclab_arena_environments/experiment_configs/droid_pnp_variations_experiment.yaml`; `isaaclab_arena/tests/test_object_mass_variation.py` |
| Observations, actions, sensors, embodiment-specific control | `isaaclab_arena/embodiments/`, `tasks/observations/`; `isaaclab_arena/environments/arena_world.py`, `arena_world_scene_access.py` | `docs/pages/concepts/embodiment/`; `isaaclab_arena/tests/test_camera_observation.py`, `test_arena_world_scene_access.py` |
| Backend selection, asset physics, compiled-config overrides | `isaaclab_arena/utils/physics_backend.py`; `isaaclab_arena/assets/physics_config.py`; `isaaclab_arena/environment_spec/env_cfg_override.py` | `docs/pages/concepts/environment/physics_backend_selection.rst`, `physics_configuration.rst`, `env_cfg_override.rst` |
| Teleoperation, retargeting, Mimic, RL integration | `isaaclab_arena/assets/device_library.py`, `retargeter_library.py`; `isaaclab_arena/tasks/common/`; `isaaclab_arena_examples/policy/` | `docs/pages/concepts/embodiment/concept_teleop_devices_design.rst`; `docs/pages/example_workflows/sequential_static_manipulation/`, `reinforcement_learning/` |
| Experiments, policies, evaluation reports, sensitivity sweeps | `isaaclab_arena/evaluation/`, `policy/`; `isaaclab_arena_examples/sensitivity_analysis/` | `docs/pages/concepts/concept_arena_experiments.rst`, `concept_sensitivity_analysis.rst`; `skills/user/run-experiment/SKILL.md` (execution only when requested) |
| Natural-language graph generation, asset/prim inference, review UI | `isaaclab_arena/agentic_environment_generation/`; `isaaclab_arena_examples/agentic_environment_generation/` | `docs/pages/concepts/agentic_environment_generation/`; `docs/pages/example_workflows/agentic_env_gen/` |
| External environment integration | `isaaclab_arena/environments/arena_environment_factory.py`; `isaaclab_arena_environments/cli.py` | `docs/pages/arena_in_your_repo/external_environments.rst`; `isaaclab_arena_examples/external_environments/` (legacy CLI examples; check the current factory contract) |

Discover additions without waiting for this map to be updated:

```bash
rg --files isaaclab_arena isaaclab_arena_environments isaaclab_arena_examples docs/pages/concepts
rg -n '^class |^def |@register_|add_variation' isaaclab_arena/tasks isaaclab_arena/relations isaaclab_arena/variations
rg -n 'get_asset_by_name|instance_name|registry_name' isaaclab_arena_environments
rg -n 'CompletionCriteria|desired_subtask_success_state|TrueForConsecutiveStepsCfg' isaaclab_arena/tests
```

Registry methods such as `get_all_keys()` and `get_component_by_name()` provide a live inventory
after simulation startup in the supported runtime. Do not mistake the `@agent_ready` subset used
by built-in generation for the entire API. For a new feature, update its existing concept/example
and the relevant row here instead of copying its implementation into agent instructions.
