Graph YAML ``env_cfg_override``
=================================

Environment graph YAML (``ArenaEnvGraphSpec``) may include an ``env_cfg_override`` mapping.
``build_arena_env_from_graph_spec`` turns that mapping into an ``env_cfg_callback`` that calls
``apply_env_cfg_override`` after ``ArenaEnvBuilder`` assigns the default solver for the resolved
physics backend.

See :doc:`physics_configuration` for configuration scopes and application order, and
:doc:`physics_backend_selection` for backend resolution and ``replicate_physics`` behavior.

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
