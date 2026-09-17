Physics configuration scopes and order
======================================

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
   * - Composed environment configuration
     - ``env_cfg_override`` / ``env_cfg_callback``
     - Simulation settings (timestep, solver iterations, substeps, collision pipeline),
       scene settings, and manager configuration.
   * - Scene object
     - ``Object.spawn_cfg_addon``
     - Object-wide mass, collision properties, material, or selected-prim physics.
   * - Robot / end effector
     - Embodiment ``_configure_physics_backend()`` and ``spawn_cfg_addon``
     - Actuators, finger contacts, gripper colliders, and joint coupling.

``ArenaEnvBuilder`` supplies backend defaults, then calls ``env_cfg_callback`` with the
composed environment config. Graph YAML ``env_cfg_override`` is applied through that callback.
These mechanisms can update simulation, scene, and manager fields in ``env_cfg``; they are
not limited to settings on ``ArenaEnvBuilderCfg``.

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

See :doc:`../scene/concept_assets_design` for a primitive object example and
:doc:`../embodiment/index` for the robot configuration hook.

See :doc:`physics_backend_selection` for backend selection details and
:doc:`env_cfg_override` for YAML override syntax.

Differences from the variation system
------------------------------------

Spawn-time physics configuration applies settings before cloning and physics import.
The :doc:`variation system <../variations/variations>` controls sampling and when sampled
values are applied. Runtime variations such as object mass update simulation state per reset.
Build-time variations can configure the spawn hook to apply a sampled physics value once
USD prims exist.

For an already constructed object, update ``object_cfg.spawn``; changing only
``spawn_cfg_addon`` after construction does not rebuild that config.
