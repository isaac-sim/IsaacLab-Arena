Physics configuration
=====================

Motivation
----------

The same robot or object may need different physics parameters in different environments.
A gripper may need higher finger friction for lifting and lower friction for sliding an object
into place. An insertion task may need different collider offsets or solver settings from a
pick-and-place task. Switching physics backends may also require different actuator defaults.

Keep reusable defaults with the asset or embodiment, and expose task-dependent values through
configuration. Each environment can then tune its instances while reusing the same USD and
library assets, without changing the source USD or another environment's settings.

.. _physics-configuration-scopes:

Configuration scopes
--------------------

Choose where to configure a setting based on what it affects:

.. list-table::
   :header-rows: 1
   :widths: 15 20 35 30

   * - Scope
     - Applies to
     - Configuration
     - Examples
   * - Physics backend
     - The simulation
     - ``default_physics_backend`` or CLI ``--presets``;
       see :doc:`physics_backend_selection`.
     - Select PhysX or Newton.
   * - Environment configuration
     - The composed environment
     - :doc:`env_cfg_override` / ``env_cfg_callback``
     - Timestep, solver iterations, substeps, collision pipeline, scene, and managers.
   * - Asset physics
     - Objects, robots, and selected prims within them
     - Object or embodiment ``spawn_cfg_addon``; use nested ``prim_physics`` for a specific
       collider, rigid body, or joint. Embodiments may also use ``_configure_physics_backend()``
       for backend-dependent settings. See :doc:`../scene/concept_assets_design` and
       :doc:`../embodiment/index`.
     - Mass, materials, contacts, colliders, joints, and actuator settings.

Asset physics includes both whole-asset settings and per-prim overrides. Actuator settings
are applied when the robot initializes; spawn addons apply when its USD is loaded.

``ArenaEnvBuilder`` supplies backend defaults, then calls ``env_cfg_callback`` with the
composed environment config. Graph YAML ``env_cfg_override`` is applied through that callback.
These mechanisms can update simulation, scene, and manager fields in ``env_cfg``; they are
not limited to settings on ``ArenaEnvBuilderCfg``.

.. _physics-application-order:

Application order
-----------------

The settings in the :ref:`scope table <physics-configuration-scopes>` are applied in this order:

1. Object construction prepares spawn configs, including any ``prim_physics`` overrides.
   Enabled build-time variations sample values and update configuration before scene composition.
2. The builder runs embodiment backend defaults. ``get_scene_cfg()`` then applies its spawn
   addons, including ``prim_physics``. The builder composes the scene and manager configs and
   assigns the default solver.
3. ``env_cfg_callback`` applies the environment's ``env_cfg_override`` to the composed config.
   These settings take precedence over earlier defaults; the selected backend stays the same.
4. Environment creation loads each USD using its ordinary spawn properties, then applies
   ``prim_physics`` overrides, then clones the configured asset and imports its physics.
   Every clone inherits the same spawn-time physics edits.

.. _per-prim-spawn-physics:

Per-prim physics at spawn time
------------------------------

This is the selected-prim part of asset physics in the :ref:`scope table <physics-configuration-scopes>`.
It runs in step 4 of the :ref:`application order <physics-application-order>`. Steps 1 and 2
prepare the object and embodiment spawn configs, and step 3 can override those configs.
``prim_physics`` stays inside ``spawn_cfg_addon`` so ordinary and per-prim settings use the
same object, embodiment, and build-time variation API. Each per-prim value is a typed config.

Once the USD is loaded, the spawner checks the targets in ``prim_physics`` and calls each
config's ``apply()`` method before cloning and physics import.

Concrete ``UsdPrimSpawnPhysicsCfg`` implementations can configure collision, material, mass,
joint, or backend-specific properties. Use schema APIs compatible with the selected backend.
Use actuator configuration for controlled joint gains because articulation initialization can
overwrite authored USD drives.

See :doc:`../scene/concept_assets_design` for a primitive object example and
:doc:`../embodiment/index` for the robot configuration hook.

Differences from the variation system
-------------------------------------

Spawn-time physics configuration applies settings before cloning and physics import.
The :doc:`variation system <../variations/variations>` controls sampling and when sampled
values are applied. Runtime variations such as object mass update simulation state per reset.
Build-time variations can configure the spawn hook to apply a sampled physics value once
USD prims exist.

For an already constructed object, update ``object_cfg.spawn``; changing only
``spawn_cfg_addon`` after construction does not rebuild that config.
