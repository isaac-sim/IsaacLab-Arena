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
