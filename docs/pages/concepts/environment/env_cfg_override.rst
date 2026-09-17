Graph YAML ``env_cfg_override``
=================================

Environment graph YAML (``ArenaEnvGraphSpec``) may include an ``env_cfg_override`` mapping.
``build_arena_env_from_graph_spec`` turns that mapping into an ``env_cfg_callback`` that calls
``apply_env_cfg_override`` after ``ArenaEnvBuilder`` assigns the default solver for the resolved
physics backend.

See :doc:`physics_backend_selection` for backend resolution, embodiment hooks, and
``replicate_physics`` behavior.

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

There is no separate override schema or duplicated defaults table: available fields and their
defaults come from the concrete environment config produced by the builder and the selected
physics backend.

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

Use a plain mapping to patch a configclass instance that already exists. Use ``_target_`` when
selecting or replacing its concrete type, especially when the current field is ``None`` (for
example a Newton collision pipeline). The target must live under an approved ``isaaclab*``
package prefix and match the field annotation on the parent config.

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

Physics on selected asset parts
-------------------------------

Use ``PhysicsUsdFileCfg`` when an environment needs different physics for individual
colliders or joints inside the same USD asset. Its ``prim_physics`` mapping selects exact
paths relative to the spawned asset root. The ordinary USD spawn properties are applied
first, followed by these overrides, before the asset is cloned or imported by the physics
backend. Shared source USD files and other instances retain their defaults.

For example, an insertion environment can tune a finger's contacts and an existing
passive-jaw equality independently of an environment that uses the same robot for routing:

.. code-block:: yaml

   default_physics_backend: newton
   env_cfg_override:
     scene:
       right_robot:
         actuators:
           gripper:
             stiffness: 40000.0
             damping: 40.0
             effort_limit_sim: 160.0
         spawn:
           _target_: isaaclab_arena.assets.physics_config.PhysicsUsdFileCfg
           usd_path: /path/to/robot.usda
           prim_physics:
             finger/collision:
               _target_: isaaclab_arena.assets.physics_config.PrimPhysicsCfg
               collision_props:
                 - _target_: isaaclab_newton.sim.schemas.NewtonCollisionCfg
                   contact_gap: 0.0002
                 - _target_: isaaclab_newton.sim.schemas.MujocoCollisionCfg
                   condim: 4
                   solref: [0.004, 1.0]
                   solimp: [0.95, 0.999, 0.0005, 0.5, 2.0]
               physics_material:
                 _target_: isaaclab_newton.sim.schemas.NewtonMaterialPropertiesCfg
                 static_friction: 8.0
                 dynamic_friction: 8.0
                 torsional_friction: 0.002
               filtered_pairs: [housing/collision]
             passive/joint:
               _target_: isaaclab_arena.assets.physics_config.PrimPhysicsCfg
               mujoco_equality:
                 _target_: isaaclab_arena.assets.physics_config.MujocoEqualityPropertiesCfg
                 solref: [0.004, 1.0]

The paths above are illustrative; use the exact collider and joint paths in your asset.
Selecting a spawn ``_target_`` replaces the spawn config: provide its ``usd_path`` and retain
any needed scale, variants, or other spawn options explicitly. This spawner uses the ordinary
USD loading path; assets with custom spawn functions should integrate ``apply_prim_physics``
into their own spawner before cloning instead of replacing their spawn function.

``PrimPhysicsCfg`` supports:

- ``collision_props``: Isaac Lab collision fragments on a geometry prim. This can enable a
  collider on an authored visual mesh without adding procedural geometry.
- ``physics_material``: a new material bound to the selected collider, without editing a
  material shared with other parts or asset instances.
- ``joint_drive_props``: Isaac Lab joint-drive fragments on a revolute or prismatic joint.
  Prefer the embodiment's actuator configuration for controlled joint gains; it may overwrite
  authored drive values during articulation initialization.
- ``mujoco_equality``: ``solref`` and ``solimp`` on an existing ``MjcEqualityJointAPI``,
  ``MjcEqualityConnectAPI``, or ``MjcEqualityWeldAPI`` constraint. This does not create or convert
  mimic constraints, or change their leader joint or coupling coefficients. It is specific to
  Newton's MuJoCo solver.
- ``filtered_pairs``: additional asset-relative rigid-body or collider exclusions. Existing
  exclusions are preserved, and internal relationships are remapped when cloning.

All target paths are checked before per-prim overrides are authored. Missing paths, absolute
paths, paths outside the asset, incompatible prim types, and instance proxies are rejected.
Use ``make_uninstanceable: true`` on the spawn config when the selected prims are inside an
instanceable USD subtree and the additional stage memory is acceptable. Use ``.`` to select
the asset root. Paths do not support regular expressions or wildcard matching.

Only ``PhysicsUsdFileCfg``, ``PrimPhysicsCfg``, and ``MujocoEqualityPropertiesCfg`` are added
to the approved Arena target classes; arbitrary Arena classes remain disallowed.

MJWarp-VBD coupling (Newton deformable pick-and-place)
------------------------------------------------------

Newton **deformable** assets simulate with the VBD solver.
The **robot**, table, and other rigids stay on MJWarp. Replace ``sim.physics.solver_cfg`` with a
**coupled** proxy solver instead of a plain ``MJWarpSolverCfg``. Use Newton deformable spawn
properties on the pick object, and tune ``bodies`` / ``proxies`` regexes to your prim paths
(robot links, plate, ``soft_cube``, and so on).

.. code-block:: yaml

   default_physics_backend: newton
   env_cfg_override:
     sim:
       dt: 0.008333333
       use_newton_actuators: true
       physics:
         num_substeps: 2
         solver_cfg:
           _target_: isaaclab_contrib.coupling.coupler_cfg.CouplerProxyCfg
           iterations: 1
           entries:
             - name: rigid
               solver_cfg:
                 _target_: isaaclab_newton.physics.MJWarpSolverCfg
                 integrator: implicitfast
                 njmax: 300
                 nconmax: 200
                 cone: elliptic
                 ls_iterations: 20
                 use_mujoco_contacts: false
                 ccd_iterations: 100
               bodies:
                 - r"/World/envs/env_.*/Robot"
                 - r"/World/envs/env_.*/maple_table_robolab"
                 - r"/World/envs/env_.*/plate"
             - name: deformable
               solver_cfg:
                 _target_: isaaclab_newton.physics.VBDSolverCfg
                 iterations: 10
               bodies:
                 - r"/World/envs/env_.*/soft_cube"
               include_static_shapes: true
           proxies:
             - source: rigid
               destination: deformable
               bodies:
                 - r"/World/envs/env_.*/Robot"
                 - r"/World/envs/env_.*/maple_table_robolab"
                 - r"/World/envs/env_.*/plate"
               mode: lagged
               mass_scale: 1.0
               collide_interval: 2

Disallowed patterns
-------------------

The following are rejected at validation time:

- Swapping ``sim.physics`` to another backend via ``_target_`` — set
  ``default_physics_backend`` (or ``--presets``) instead.
- Overriding ``class_type`` (derived by Isaac Lab).
- OmegaConf interpolation (``${...}``) in override values.
- Hydra targets outside approved Isaac Lab packages (for example ``builtins.*``).
