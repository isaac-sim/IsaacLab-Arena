Graph YAML ``env_cfg_override``
=================================

Environment graph YAML (``ArenaEnvGraphSpec``) may include an ``env_cfg_override`` mapping.
``build_arena_env_from_graph_spec`` turns that mapping into an ``env_cfg_callback`` that calls
``apply_env_cfg_override`` after ``ArenaEnvBuilder`` assigns the default solver for the resolved
physics backend.

See :doc:`physics_backend_selection` for backend resolution, embodiment hooks, and
``replicate_physics`` behavior.

Physics configuration scopes and order
--------------------------------------

Choose configuration based on what owns the setting:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Scope
     - Configuration
     - Examples
   * - Physics backend
     - ``default_physics_backend`` or CLI ``--presets``
     - Select PhysX or Newton.
   * - Environment
     - ``env_cfg_override`` / ``env_cfg_callback``
     - Timestep, solver iterations, substeps, collision pipeline.
   * - Scene object
     - ``Object.spawn_cfg_addon``
     - Object-wide mass, collision properties, material, or selected-prim physics.
   * - Robot / end effector
     - Embodiment ``_configure_physics_backend()`` and ``spawn_cfg_addon``
     - Actuators, finger contacts, gripper colliders, and joint coupling.

The configuration and spawning stages run in this order:

1. Object construction prepares object spawn configs. Enabled build-time variations sample
   values and update configuration before the scene is composed.
2. The builder runs embodiment backend defaults, applies embodiment spawn addons, composes
   the scene and manager configs, and assigns the selected backend's default solver.
3. ``env_cfg_callback`` applies the environment's ``env_cfg_override`` to the composed config.
   These settings take precedence over earlier defaults; the selected backend stays the same.
4. Environment creation loads each USD using its ordinary spawn properties, then applies
   ``prim_physics`` overrides, then clones the configured asset and imports its physics.
   Every clone inherits the same spawn-time physics edits.

Concrete ``UsdPrimSpawnPhysicsCfg`` implementations can configure collision, material, mass,
joint, or backend-specific properties. Use schema APIs compatible with the selected backend.
Use actuator configuration for controlled joint gains because articulation initialization can
overwrite authored USD drives.

Runtime variations such as object mass update simulation state per reset. Build-time variations
can configure the spawn hook to apply a sampled physics value once USD prims exist. For an
already constructed object, update ``object_cfg.spawn``; changing only ``spawn_cfg_addon``
after construction does not rebuild that config.

See :doc:`../scene/concept_assets_design` for a primitive object example and
:doc:`../embodiment/index` for the robot configuration hook.

Minimal graph fragment
----------------------

Set the backend with ``default_physics_backend``. Tune solver fields under ``env_cfg_override``;
do not swap PhysX for Newton (or the reverse) through overrides.

.. code-block:: yaml

   default_physics_backend: newton
   env_cfg_override:
     sim:
       dt: 0.01
       physics:
         num_substeps: 4

Data-only overrides
-------------------

Scalar and nested dict fields merge into the composed ``ManagerBasedRLEnvCfg`` when they match
known Isaac Lab config fields. Examples:

.. code-block:: yaml

   env_cfg_override:
     decimation: 4
     sim:
       dt: 0.01
       physics:
         num_substeps: 4
         debug_mode: false

Hydra ``_target_`` nodes
------------------------

Use ``_target_`` when replacing a nested **configclass** field with a concrete Isaac Lab type
(for example a Newton solver or collision pipeline). The target must live under an approved
``isaaclab*`` package prefix and match the field annotation on the parent config.

.. code-block:: yaml

   env_cfg_override:
     sim:
       physics:
         solver_cfg:
           _target_: isaaclab_newton.physics.MJWarpSolverCfg
           solver: newton
           iterations: 100
         collision_cfg:
           _target_: isaaclab_newton.physics.NewtonCollisionPipelineCfg
           reduce_contacts: true

Nested ``_target_`` mappings anywhere in the tree are validated before any change is applied to
the live environment configuration.

Disallowed patterns
-------------------

The following are rejected at validation time:

- Swapping ``sim.physics`` to another backend via ``_target_`` — set
  ``default_physics_backend`` (or ``--presets``) instead.
- Overriding ``class_type`` (derived by Isaac Lab).
- OmegaConf interpolation (``${...}``) in override values.
- Hydra targets outside approved Isaac Lab packages (for example ``builtins.*``).
