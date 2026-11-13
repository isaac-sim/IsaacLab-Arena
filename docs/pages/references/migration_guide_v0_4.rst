Migrating from v0.3 to v0.4
===========================

This guide lists the changes to Arena's public interfaces between ``v0.3.0`` and ``v0.4.0``.
It includes the changes released in ``v0.3.1``.
It covers the environment YAML format, the Python environment API, script entry points,
Experiment Definitions, and evaluation outputs.
For new features, see the :doc:`release notes <release_notes>`.

Each section lists breaking changes first, then changes in behavior that need no edits but can change results.


Environment YAML
----------------

Breaking changes
^^^^^^^^^^^^^^^^

**Placement validator settings moved to** ``placer_params``.
The ``placement_validators`` block was removed. Its fields, and other placer settings, now go under
``placer_params`` (v0.3.1).

.. code-block:: yaml

   # v0.3
   placement_validators:
     enabled_checks: [no_overlap, on_relation]
     required_checks: [no_overlap]

   # v0.4
   placer_params:
     enabled_checks: [no_overlap, on_relation]
     required_checks: [no_overlap]

**PositionLimitsBox takes limit pairs.**
Replace the single-bound parameters with ``[min, max]`` pairs. Use ``null`` for an open side (#1537).

.. code-block:: yaml

   # v0.3
   - kind: position_limits_box
     params: {x_min: 0.55, x_max: 0.7, z_min: 0.8}

   # v0.4
   - kind: position_limits_box
     params: {x_limits: [0.55, 0.7], z_limits: [0.8, null]}

**The** ``random_around_solution`` **relation was removed.**
It had no effect in environments built by ``ArenaEnvBuilder``. Bound the object's position with
``position_limits_box`` instead (#1494).

**Object** ``instance_name`` **was renamed to** ``name``.
This affects only YAML files that set ``instance_name`` explicitly. The node ``id`` is still the
default name (#1507).

.. code-block:: yaml

   # v0.3
   params: {instance_name: pick_up_object}

   # v0.4
   params: {name: pick_up_object}

**PickAndPlaceTask uses radians for the support cone.**
Replace ``support_cone_half_angle_deg`` with ``support_cone_half_angle_rad``. The default is unchanged
at 45 degrees (#1173).

.. code-block:: yaml

   # v0.3
   support_cone_half_angle_deg: 45.0

   # v0.4
   support_cone_half_angle_rad: 0.785

**Relations on a mesh-collision background fail early.**
A background with ``collision_mode: mesh`` that takes part in a relation now raises an error before
solving. Previously, placement failed after a console warning. Anchor an ``object_references`` entry
that points at the support surface instead (#1614).

**The Dexsuite lift task was removed.**
``DexsuiteLiftTask`` and the Kuka-Allegro embodiment have no replacement (#1666).

Behavior changes
^^^^^^^^^^^^^^^^

- **NextTo:** ``cross_position_ratio`` now defaults to ``None``, which places the object anywhere
  along the side. Set ``cross_position_ratio: 0.0`` to keep the previous centered placement. Gaps
  within ``tolerance_m`` of ``distance_m`` are now accepted, which gives more varied layouts (#1522, #1597).
- **Placement initializer:** ``initializer_type`` defaults to ``ON_TREE`` (#1369).
- **PickAndPlaceTask success:** the default ``velocity_threshold`` dropped from ``0.1`` to ``0.003``,
  so the at-rest check is stricter (#1178).
- **Object references:** a relative ``prim_path`` resolves under the parent asset's prim path, not
  under the background. This changes results only when ``parent_id`` is not the background (#1617).
- **Placement seeds:** consecutive ``placement_seed`` values now produce independent layouts, so a
  fixed seed gives a different layout than in v0.3 (#1523).
- **YAML booleans:** an unquoted ``kind: on`` is read by YAML as ``true``. It now fails with a
  message that asks you to quote it: ``kind: 'on'`` (#1603).

New options
^^^^^^^^^^^

These keys are new and optional:

- ``external_yaml`` accepts a list of files. A value may be defined in only one file (#1504).
- ``default_physics_backend`` and ``env_cfg_override`` set the physics backend and simulation
  settings (v0.3.1).
- The ``clutter_on`` relation and ``overlap`` on the ``on`` relation (v0.3.1), and the ``align_with``
  relation (#1522).
- ``initial_pose`` on assets (v0.3.1) and ``initial_joint_positions`` on articulation object
  references (#1617).


Python environments
-------------------

Environment builder
^^^^^^^^^^^^^^^^^^^

Rename ``ArenaEnvBuilder.compose_manager_cfg()`` to ``build_env_cfg()``. It returns the same
``(env_cfg, env_kwargs)`` pair (#1474).

.. code-block:: python

   # v0.3
   env_cfg, env_kwargs = arena_builder.compose_manager_cfg()

   # v0.4
   env_cfg, env_kwargs = arena_builder.build_env_cfg()

``variations_hydra.compose_variations_cfg_and_apply_overrides()`` is now
``build_variations_cfg_with_overrides()``.

Assets
^^^^^^

**Registry names.**
An asset class declares its registry key with ``registry_name`` instead of ``name``. This also applies
to embodiments and HDR images. ``name`` now means only the asset's name in the scene, so library objects
take ``name=`` instead of ``instance_name=`` (#1507).

.. code-block:: python

   # v0.3
   class MyBox(LibraryObject):
       name = "my_box"

   box = CrackerBox(instance_name="pick_up_object")

   # v0.4
   class MyBox(LibraryObject):
       registry_name = "my_box"

   box = CrackerBox(name="pick_up_object")

The registry lookups take ``registry_name=`` as their keyword: ``get_asset_by_name()``,
``get_hdr_by_name()`` and ``register_asset_factory()``.

**USD sources.**
``Object``, ``Background`` and library classes take either ``usd_path`` or ``usd_source``, not both.
After construction, ``usd_path`` is always a local path string. Lightwheel-backed assets declare
``usd_source = LightwheelUsdSource(...)``, which replaces ``LightwheelLazyPath`` (#1621).

``Background`` reordered its arguments, so positional calls must change:

.. code-block:: python

   # v0.3
   Background(name, usd_path, object_min_z)

   # v0.4
   Background(name, object_min_z, usd_path=usd_path)

**Object poses.**
``ObjectBase.get_object_pose()`` was removed. Read poses through ``ArenaWorld``, described below (#1187).

Tasks and success
^^^^^^^^^^^^^^^^^

**Success is declared through** ``TaskTerminationCfg``.
``TaskBase.get_termination_cfg()`` returns a ``TaskTerminationCfg``. Success is a list of
``CompletionCriteria``, which replace ``ProgressObjective`` and ``get_progress_objectives()``.
Failure terms go in ``failures``. Predicate groups become predicate sequences (#1255, #1316).

.. code-block:: python

   # v0.3
   def get_termination_cfg(self):
       return TerminationsCfg(
           success=TerminationTermCfg(func=ObjectOnDestinationTerm, params={...}),
           object_dropped=object_dropped,
       )

   def get_progress_objectives(self):
       return [ProgressObjective(name="pick_and_place", predicate_groups=[p1, p2])]

   # v0.4
   def get_termination_cfg(self) -> TaskTerminationCfg:
       return TaskTerminationCfg(
           timeout_s=self.episode_length_s,
           success=[CompletionCriteria(name="pick_and_place", predicate_sequence=[p1, p2])],
           failures={"object_dropped": object_dropped},
       )

To require a condition to hold for several steps, wrap the predicate in ``TrueForConsecutiveStepsCfg``
(#1305). ``ObjectOnDestinationTerm`` was removed; use the ``object_on_destination`` predicate in
``isaaclab_arena.tasks.predicates.spatial``.

**Sequential tasks.**
``SequentialTaskBase`` was removed. Use ``CompositeTaskBase`` with ``subtasks_are_sequential=True``.
Pass a flat list of subtasks; nested composite tasks are not supported (#1255).

Scene access at run time
^^^^^^^^^^^^^^^^^^^^^^^^

Predicates, rewards, terminations, observations and metrics read the scene through ``ArenaWorld``,
available as ``env.unwrapped.arena_world``. It replaces the helpers in
``tasks/predicates/predicate_utils.py`` and ``live_scene_geometry.py`` (#1173, #1187, #1195, #1197).

.. code-block:: python

   # v0.3
   t_W_O = get_root_pos_w(env, object_name)
   v_W = get_root_lin_vel_w(env, object_name)

   # v0.4
   t_W_O = env.unwrapped.arena_world.get_position_w(object_name)
   v_W = env.unwrapped.arena_world.get_root_linear_velocity_w(object_name)

Physics backend and callbacks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``IsaacLabArenaEnvironment`` takes ``default_physics_backend``, which applies when ``--presets`` is
not given. ``ArenaEnvBuilderCfg.presets`` is a ``PhysicsBackend`` instead of a string; strings are
still accepted.

``env_cfg_callback`` now runs after the physics backend is applied, so its physics settings are no longer
overwritten by ``--presets``. The callback must return the configuration and must not change the
backend type (#1262, #1266).

Embodiments
^^^^^^^^^^^

Embodiments expose their gripper through ``self.gripper``, for example ``PandaGripper`` or
``RobotiqGripper``. Custom embodiments must set it for gripper predicates such as
``gripper_not_grasping`` to work (#1293, #1370).

Custom variations and samplers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``VariationBase.configure_at_build_time()`` takes ``seed`` and ``variation_key``. A ``sampler`` is
required. Custom samplers must draw from the generator set through ``set_generator()``, so that the same
``--seed`` gives the same values (#1467).

Module moves
^^^^^^^^^^^^

No compatibility aliases were left at the old paths.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - v0.3
     - v0.4
   * - ``isaaclab_arena.utils.usd_helpers``
     - ``isaaclab_arena.utils.usd.helpers``
   * - ``isaaclab_arena.utils.usd_pose_helpers``
     - ``isaaclab_arena.utils.usd.pose``
   * - ``isaaclab_arena.utils.usd_prim_tree``
     - ``isaaclab_arena.utils.usd.prim_tree``
   * - ``isaaclab_arena.relations.relation_solver``
     - ``isaaclab_arena.relations.optimization.relation_solver``
   * - ``isaaclab_arena.relations.placement_events``
     - ``isaaclab_arena.relations.runtime.placement_events``
   * - ``isaaclab_arena.relations.placement_validators``
     - ``isaaclab_arena.relations.validation.pre_physics``
   * - ``isaaclab_arena.relations.placement_pool_validation``
     - ``isaaclab_arena.offline_placement.pool_validation``
   * - ``isaaclab_arena.visualization.episode_results_files``
     - ``isaaclab_arena.evaluation.episode_results_files``
   * - ``isaaclab_arena.visualization.report_data``
     - ``isaaclab_arena.evaluation.report_data``

Relation types and placers keep their paths, for example ``isaaclab_arena.relations.relations`` and
``isaaclab_arena.relations.object_placer``. The other modules in ``isaaclab_arena.relations`` moved
into the ``optimization``, ``geometry``, ``validation`` and ``runtime`` subpackages (#1534).
The USD helper moves are from #1218 and #1292, and the evaluation moves from #1604.

Removed
^^^^^^^

- ``RandomAroundSolution``: use ``PositionLimitsBox`` (#1494).
- The ``@agent_ready`` decorator: remove it from your classes. The environment generator selects its
  tasks and relations through the ``included_names`` argument of its catalogue builders (#1671).
- ``Scene.get_termination_cfg()``: declare terminations on the task.
- ``KukaAllegroEmbodiment``, ``DexsuiteLiftTask`` and the ``dexsuite_lift`` environment (#1666).


Script entry points
-------------------

Arena's evaluation scripts keep their paths and arguments: ``policy_runner.py``,
``experiment_runner.py`` and ``environment_runner.py``. Imitation learning still uses
Isaac Lab's scripts, with the same ``--external_callback`` argument as in v0.3. Two workflows change.

Running in Docker
^^^^^^^^^^^^^^^^^

The Arena image installs Isaac Lab into ``/opt/isaaclab-venv``. In an interactive container shell,
``python`` and ``pytest`` are aliased to Isaac Lab's launcher, so documented commands work unchanged.
In scripts and non-interactive ``docker exec`` commands, call ``isaaclab.sh -p`` explicitly.
``/isaac-sim/python.sh`` is no longer supported, because it bypasses Isaac Lab's environment (#1481).

.. code-block:: bash

   # v0.3
   docker exec <container> /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py ...

   # v0.4
   docker exec <container> isaaclab.sh -p isaaclab_arena/evaluation/policy_runner.py ...

Reinforcement learning training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train Arena tasks with Arena's ``train.py``, which wraps Isaac Lab's unified training entry point.
Select the RL library with ``--rl_library``. The other arguments are unchanged.

.. code-block:: bash

   # v0.3
   python submodules/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
     --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
     --task lift_object --rl_training_mode --num_envs 4096

   # v0.4
   python isaaclab_arena/scripts/train.py \
     --rl_library rsl_rl \
     --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
     --task lift_object --rl_training_mode --num_envs 4096

Runner behavior
^^^^^^^^^^^^^^^

- ``policy_runner.py --record_viewport_video`` no longer enables cameras. Pass ``--enable_cameras``
  when the environment needs them.
- ``--presets`` accepts only ``physx`` or ``newton``.
- ``environment_runner.py`` requires PhysX and fails early under ``--presets newton``. It now keeps
  Fabric enabled, which keeps CPU PhysX scenes stable (#1385, #1554).
- The policy runner and the Experiment Runner run exactly the requested number of episodes. Previously,
  parallel environments could finish more episodes than requested (#1371).
- Isaac Lab's recording and replay tools now apply Arena's variation overrides and success checks
  (#1397, #1435).


Experiment Definitions
----------------------

v0.3 Experiment Definitions still load. The changes are additions and one change to how shared settings
merge.

- **Experiment name:** every Experiment has a ``name``. It is taken from the top-level ``name:`` key, or
  from the file stem when the key is missing. Override it with ``name=<x>`` (#1585).
- **KPI exporters:** the optional top-level ``kpi_exporters`` list selects the KPI files written after
  all Runs complete. The default is ``[timing]`` (#1588).
- **Rollout limits:** ``rollout_limit.num_episodes_per_rebuild`` sets the episodes for each rebuild. Its
  length must equal ``num_rebuilds`` (#1544).
- **Shared rollout limits:** a ``rollout_limit`` declared on a Run now replaces the one in ``shared``
  instead of merging into it. A Run that sets ``num_steps`` no longer inherits a shared ``num_episodes``.
- **Recorded variation replay:** ``recorded_variation_samples_paths`` on a Run replays one recorded
  episode-results file per rebuild (#1394, #1429).
- **Builder options:** ``environment_builder`` accepts ``camera_resolution_mode``,
  ``record_trajectories`` and ``recorded_variation_samples_path`` (#1254, #1394, #1511).
- **Asset selection:** a rigid object's asset can vary per rebuild through the ``asset_selection``
  variation (#1458).

.. code-block:: yaml

   name: getting_started
   kpi_exporters: [success_rate, timing]
   runs:
     pick_and_place:
       # environment and policy omitted
       num_rebuilds: 3
       rollout_limit:
         num_episodes: 3
       variations:
         pick_up_object.asset_selection.enabled: true
         pick_up_object.asset_selection.registry_names: [mustard_bottle_hot3d_robolab, rubiks_cube_hot3d_robolab]

**Seeding:** variations are now seeded from ``--seed`` and the variation key. The same seed gives the
same variation values, but they differ from the values v0.3 produced (#1467).


Evaluation outputs
------------------

The Experiment output layout is unchanged: one timestamped directory, with one subdirectory per Run.
Metric names, such as ``success_rate``, are unchanged.

Combined result file
^^^^^^^^^^^^^^^^^^^^

``arena_experiment_result.json`` gains top-level ``git_stamp`` (``git_sha``, ``git_branch``) and
``experiment_name`` fields. Each Run gains ``start_timestamp_ns`` and ``end_timestamp_ns``
(#1540, #1585). Parsers that ignore unknown keys keep working.
``ArenaExperimentResult.from_experiment_output()`` reads these files, but requires the new fields, so
it does not read v0.3 results (#1586).

Episode results
^^^^^^^^^^^^^^^

The ``progress`` block in ``episode_results_rebuild<i>.jsonl`` renamed its keys to match
``CompletionCriteria`` (#1255):

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - v0.3
     - v0.4
   * - ``progress.objectives``
     - ``progress.criteria_by_name``
   * - ``completed_groups`` / ``total_groups``
     - ``completed_sequences`` / ``total_sequences``
   * - ``events[].objective`` / ``events[].group``
     - ``events[].criteria_name`` / ``events[].sequence_name``

New fields:

- ``progress.consecutive_step_summary``, for ``TrueForConsecutiveStepsCfg`` requirements (#1382).
- ``progress.tracked_events``, for tasks that define tracked events (#1457).
- ``placement``, the layout used for the episode (#1427).
- ``replay_source_record_index``, written when replaying recorded variations (#1394).

New files
^^^^^^^^^

- ``<run>/arena_run_timings.json`` and ``arena_experiment_timings.json`` record step timings (#1198, #1426).
- ``<run>/arena_run_placement_diagnostics.json`` records the placement checks that rejected candidates
  (#1560, #1602).
- ``kpis_timing.json``, ``kpis_success_rates.json`` and ``kpis_placement_success.json`` are written
  by the exporters listed in ``kpi_exporters`` (#1587, #1588, #1623).

Recordings and videos
^^^^^^^^^^^^^^^^^^^^^

- **HDF5 recordings:** the Experiment Runner writes ``<run>/dataset_<run>_rebuild<i>.hdf5`` in the Run
  directory. It no longer writes ``dataset_<run>.hdf5`` to ``/tmp/isaaclab/logs``, where Runs and
  concurrent jobs could overwrite each other (#1449).
- **Camera videos:** ``--record_camera_video`` records only RGB cameras. File names are unchanged.
- **Viewport videos:** the viewport video uses Gymnasium's default name, ``rl-video-step-0.mp4``,
  instead of ``viewport-rebuild<i>-env0-viewport-episode-0.mp4``.

Metric values
^^^^^^^^^^^^^

``success_rate`` no longer skips the first reset, and Runs finish exactly the requested number of
episodes. Success rates can therefore differ slightly from v0.3 for the same policy (#1371).

OSMO
^^^^

The OSMO output collector requires the timestamps and git fields above in ``experiment_runner_result.json``,
so outputs produced by v0.3 runners cannot be collected with v0.4. A Run that times out is recorded as
failed instead of failing the collector (#1426, #1540).
