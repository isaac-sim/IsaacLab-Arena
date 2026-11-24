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

Placement validator settings moved to ``placer_params``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``placement_validators`` block was removed. Its fields, and other placer settings, now go under
``placer_params`` (v0.3.1).

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item::

      .. code-block:: yaml

         # v0.3
         placement_validators:
           enabled_checks:
             - no_overlap
             - on_relation
           required_checks: [no_overlap]

   .. grid-item::

      .. code-block:: yaml

         # v0.4
         placer_params:
           enabled_checks:
             - no_overlap
             - on_relation
           required_checks: [no_overlap]

PositionLimitsBox takes limit pairs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace the single-bound parameters with ``[min, max]`` pairs. Use ``null`` for an open side (#1537).

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item::

      .. code-block:: yaml

         # v0.3
         - kind: position_limits_box
           params:
             x_min: 0.55
             x_max: 0.7
             z_min: 0.8

   .. grid-item::

      .. code-block:: yaml

         # v0.4
         - kind: position_limits_box
           params:
             x_limits: [0.55, 0.7]
             z_limits: [0.8, null]

Object ``instance_name`` was renamed to ``name``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This affects only YAML files that set ``instance_name`` explicitly. The node ``id`` is still the
default name (#1507).

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item::

      .. code-block:: yaml

         # v0.3
         params: {instance_name: pick_up_object}

   .. grid-item::

      .. code-block:: yaml

         # v0.4
         params: {name: pick_up_object}

PickAndPlaceTask uses radians for the support cone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace ``support_cone_half_angle_deg`` with ``support_cone_half_angle_rad``. The default is unchanged
at 45 degrees (v0.3.1, #1173).

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item::

      .. code-block:: yaml

         # v0.3
         support_cone_half_angle_deg: 45.0

   .. grid-item::

      .. code-block:: yaml

         # v0.4
         support_cone_half_angle_rad: 0.785

Other breaking changes
~~~~~~~~~~~~~~~~~~~~~~

- **random_around_solution:** the relation was removed. It had no effect in environments built by
  ``ArenaEnvBuilder``. Bound the object's position with ``position_limits_box`` instead (#1494).
- **Relations on a mesh-collision background:** a background with ``collision_mode: mesh`` that takes
  part in a relation now raises an error before solving. Previously, placement failed after a console
  warning. Anchor an ``object_references`` entry that points at the support surface instead (#1614).
- **Deformable asset names:** the deformable assets added in v0.3.1 have one variant per physics
  backend. Replace the asset names ``deformable_cube``, ``deformable_surface`` and
  ``deformable_teddy_bear`` with their ``_physx`` or ``_newton`` variant, for example
  ``deformable_cube_physx`` (#1441).

Behavior changes
^^^^^^^^^^^^^^^^

- **NextTo:** ``cross_position_ratio`` now defaults to ``None``, which places the object anywhere
  along the side. Set ``cross_position_ratio: 0.0`` to keep the previous centered placement. Gaps
  within ``tolerance_m`` of ``distance_m`` are now accepted, which gives more varied layouts (#1522, #1597).
- **Placement initializer:** ``initializer_type`` defaults to ``ON_TREE`` (v0.3.1, #1369).
- **PickAndPlaceTask success:** the default ``velocity_threshold`` dropped from ``0.1`` to ``0.003``,
  so the at-rest check is stricter (v0.3.1, #1178).
- **Object references:** a relative ``prim_path`` resolves under the parent asset's prim path, not
  under the background. This changes results only when ``parent_id`` is not the background (#1617).
- **Robot reachability:** cuRobo is now always installed, so the ``ik_reachable`` check runs by
  default for embodiments with a cuRobo config, such as DROID. It can reject layouts that v0.3
  accepted. To turn it off, list ``placer_params.enabled_checks`` without ``ik_reachable`` (#1550).
- **Placement seeds:** consecutive ``placement_seed`` values now produce independent layouts, so a
  fixed seed gives a different layout than in v0.3 (#1523).
- **YAML booleans:** an unquoted ``kind: on`` is read by YAML as ``true``. It now fails with a
  message that asks you to quote it: ``kind: 'on'`` (#1603).

New options
^^^^^^^^^^^

These keys are new and optional:

- ``external_yaml`` accepts a list of files as well as a single path. A value may be defined in only
  one file (#1504).
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
``build_variations_cfg_with_overrides()`` (#1474).

Assets
^^^^^^

Registry names
~~~~~~~~~~~~~~

An asset class declares its registry key with ``registry_name`` instead of ``name``. This also applies
to embodiments and HDR images. ``name`` now means only the asset's name in the scene, so library objects
take ``name=`` instead of ``instance_name=`` (#1507).

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item::

      .. code-block:: python

         # v0.3
         class MyBox(LibraryObject):
             name = "my_box"

         box = CrackerBox(
             instance_name="pick_up_object")

   .. grid-item::

      .. code-block:: python

         # v0.4
         class MyBox(LibraryObject):
             registry_name = "my_box"

         box = CrackerBox(
             name="pick_up_object")

The registry lookups ``get_asset_by_name()`` and ``get_hdr_by_name()`` take ``registry_name=`` as
their keyword, as does the ``register_asset_factory()`` decorator.

The deformable asset names also changed; see `Other breaking changes`_ under Environment YAML.

USD sources
~~~~~~~~~~~

``Object``, ``Background`` and library classes accept a ``usd_source`` as an alternative to
``usd_path``; pass only one. A ``usd_source`` is downloaded on first use and its local path is stored
in ``usd_path``. An explicit ``usd_path`` is kept as given. Lightwheel-backed assets declare
``usd_source = LightwheelUsdSource(...)``, which replaces ``LightwheelLazyPath`` (#1621).

``Background`` reordered its arguments, so positional calls must change:

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item::

      .. code-block:: python

         # v0.3
         Background(name, usd_path, object_min_z)

   .. grid-item::

      .. code-block:: python

         # v0.4
         Background(name, object_min_z,
                    usd_path=usd_path)

Object poses
~~~~~~~~~~~~

``ObjectBase.get_object_pose()`` was removed. Read poses through ``ArenaWorld``, described below (#1187).

Tasks and success
^^^^^^^^^^^^^^^^^

Success is declared through ``TaskTerminationCfg``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

A v0.3 ``predicate_groups`` list maps to ``predicate_sequence``. A dict of named groups maps to
``predicate_sequences`` (plural). ``CompletionCriteria`` takes exactly one of the two.

To require a condition to hold for several steps, wrap the predicate in ``TrueForConsecutiveStepsCfg``
(v0.3.1).

``ObjectOnDestinationTerm`` was removed; use the ``object_on_destination`` predicate in
``isaaclab_arena.tasks.predicates.spatial``. Its signature changed: ``destination_cfg`` is required,
every parameter except ``support_cone_half_angle_rad`` must be passed, and the support cone is in
radians. ``objects_on_destinations`` was removed. A v0.3 ``partial(object_on_destination, ...)`` that
relied on defaults raises ``TypeError``.

Sequential tasks
~~~~~~~~~~~~~~~~

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
not given. ``ArenaEnvBuilderCfg.presets`` is a ``PhysicsBackend`` instead of a string. Python code may
still pass a string; Experiment Definitions must use the enum names, as described below.

``env_cfg_callback`` now runs after the physics backend is applied, so its physics settings are no longer
overwritten by ``--presets``. The callback must return the configuration and must not change the
backend type (#1262, #1266).

Embodiments
^^^^^^^^^^^

- **Terminations:** ``EmbodimentBase.get_termination_cfg()`` and ``termination_cfg`` were removed. The
  builder reads terminations only from the task, so terminations set on an embodiment are dropped
  without an error. Move them to the task's ``TaskTerminationCfg.failures``.
- **Recorders:** ``get_recorder_term_cfg()`` takes ``record_trajectories``, which the builder passes by
  keyword. Overrides with the v0.3 signature raise ``TypeError``.
- **Mimic:** custom Mimic environments must subclass ``IsaacLabArenaManagerBasedRLMimicEnv``, not
  Isaac Lab's ``ManagerBasedRLMimicEnv``. Otherwise they have no ``arena_world``, and predicates that
  read the scene fail.
- **Gripper:** embodiments expose their gripper through ``self.gripper``, for example ``PandaGripper``
  or ``RobotiqGripper``. Set it so that tasks can get it through ``get_gripper()`` (#1293).

Custom variations and samplers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``VariationBase.configure_at_build_time()`` took no arguments in v0.3. It now takes ``seed``,
``variation_key`` and ``rebuild_index``, which the builder passes by keyword. A ``sampler`` is
required. Custom samplers must draw from the generator set through ``set_generator()``, so that the
same seed gives the same values (#1467).

Custom placement validators subclass ``PrePhysicsPlacementValidator`` and implement
``_validate_batch(batch, collision_objects, rejections)``. In v0.3 they subclassed
``PlacementValidator`` and implemented ``validate_batch()``. ``PlacementValidator`` is now in
``isaaclab_arena.relations.validation.base`` (#1534).

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
   * - ``isaaclab_arena.utils.usd_articulation``
     - ``isaaclab_arena.utils.usd.articulation``
   * - ``isaaclab_arena.relations.relation_solver``
     - ``isaaclab_arena.relations.optimization.relation_solver``
   * - ``isaaclab_arena.relations.placement_events``
     - ``isaaclab_arena.relations.runtime.placement_events``
   * - ``isaaclab_arena.relations.placement_validators``
     - ``isaaclab_arena.relations.validation.pre_physics``
   * - ``isaaclab_arena.relations.placement_validator_registry``
     - ``isaaclab_arena.relations.validation.registry``
   * - ``isaaclab_arena.relations.placement_validation``
     - ``isaaclab_arena.relations.validation.types``
   * - ``isaaclab_arena.relations.placement_pool_validation``
     - ``isaaclab_arena.offline_placement.pool_validation``
   * - ``isaaclab_arena.visualization.episode_results_files``
     - ``isaaclab_arena.evaluation.episode_results_files``
   * - ``isaaclab_arena.visualization.report_data``
     - ``isaaclab_arena.evaluation.report_data``

Modules you construct objects from keep their paths, for example ``relations.relations``,
``object_placer``, ``pooled_object_placer``, ``object_placer_params`` and ``placement_result``.
Solver, geometry, validation and run-time modules moved into the ``optimization``, ``geometry``,
``validation`` and ``runtime`` subpackages (#1534). The USD helper moves are from v0.3.1, and the
evaluation moves from #1604.

Removed
^^^^^^^

- ``RandomAroundSolution``: use ``PositionLimitsBox`` (#1494).
- The ``@agent_ready`` decorator: remove it from your classes. The environment generator selects its
  tasks and relations through the ``included_names`` argument of its catalogue builders (#1671).
- ``Scene.get_termination_cfg()``: declare terminations on the task (v0.3.1).
- ``KukaAllegroEmbodiment``, ``DexsuiteLiftTask`` and the ``dexsuite_lift`` environment (#1666).


Script entry points
-------------------

Arena's evaluation scripts keep their paths: ``policy_runner.py``, ``experiment_runner.py`` and
``environment_runner.py``. Imitation learning still uses Isaac Lab's scripts, with the same
``--external_callback`` argument as in v0.3.

Isaac Lab 3.0 removed ``--headless``. Drop it from every command, including data generation; runs are
headless unless a visualizer is selected with ``--viz`` (#1481).

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

The ``-c`` option of ``./docker/run_docker.sh``, v0.3.1's ``-t`` option, and the separate cuRobo image
were removed. cuRobo is now part of every Docker and native ``uv`` install (#1550).

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item::

      .. code-block:: bash

         # v0.3
         ./docker/run_docker.sh -c

   .. grid-item::

      .. code-block:: bash

         # v0.4
         ./docker/run_docker.sh

The host datasets, models and evaluation directories are mounted at ``/home/<user>/datasets``,
``/home/<user>/models`` and ``/home/<user>/eval`` instead of ``/datasets``, ``/models`` and
``/eval``. Update paths in commands and configs that run inside the container (#1612).

Reinforcement learning training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train Arena tasks with Arena's ``train.py``, which wraps Isaac Lab's unified training entry point.
Select the RL library with ``--rl_library``. The other arguments are unchanged (v0.3.1, #1166).

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

Most v0.3 Experiment Definitions still load. The exception is a lowercase ``presets`` value. The other
changes are additions and one change to how shared settings merge.

- **Physics presets:** ``environment_builder.presets`` is now validated against the ``PhysicsBackend``
  enum names. Replace ``presets: newton`` with ``presets: NEWTON``, and ``presets: physx`` with
  ``presets: PHYSX``.

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
- **Seeding:** variations are now seeded from the Run's ``environment_builder.seed`` (default ``42``)
  and the variation key; each rebuild adds its index to the seed. The Experiment Runner does not read
  ``--seed``. The same seed gives the same variation values, but they differ from the values v0.3
  produced (#1467).

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
``CompletionCriteria`` (v0.3.1):

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
  instead of ``viewport-rebuild<i>-env0-viewport-episode-0.mp4``. Every rebuild writes to the same
  file, so with ``num_rebuilds`` greater than 1 only the last rebuild's video is kept.

Metric values
^^^^^^^^^^^^^

``success_rate`` no longer skips the first reset, and Runs finish exactly the requested number of
episodes. Success rates can therefore differ slightly from v0.3 for the same policy (#1371).

OSMO
^^^^

The OSMO output collector reads ``git_sha``, ``git_branch``, ``start_timestamp_ns`` and
``end_timestamp_ns`` as top-level keys of each Run's ``experiment_runner_result.json``. It also
requires every completed Run to have ``arena_run_timings.json``. Outputs produced by v0.3 runners
therefore cannot be collected with v0.4. A Run that times out is recorded as failed instead of
failing the collector (#1426, #1540).
