Environment configuration overrides
===================================

Environment graph YAML (``ArenaEnvGraphSpec``) may include an ``env_cfg_override`` mapping.
``build_arena_env_from_graph_spec`` turns that mapping into an ``env_cfg_callback`` that calls
``apply_env_cfg_override`` after ``ArenaEnvBuilder`` assigns the default solver for the resolved
physics backend.

See :doc:`physics_configuration` for configuration scopes and application order, and
:doc:`physics_backend_selection` for backend resolution and ``replicate_physics`` behavior.

Per-Run Experiment overrides
----------------------------

An Experiment Run can supply ``env_cfg_override`` for either a graph-YAML environment or a
registered Python environment. Place the mapping alongside ``environment`` and ``policy``:

.. code-block:: yaml

   runs:
     short_episodes:
       environment:
         type: isaaclab_arena_environments/robolab/tasks/clamp_in_right_bin.yaml
       policy:
         type: zero_action
       rollout_limit:
         num_episodes: 1
       env_cfg_override:
         episode_length_s: 1.0

The same mapping can be supplied on the Experiment CLI:

.. code-block:: bash

   '+runs.short_episodes.env_cfg_override={episode_length_s:1.0}'

The Hydra ``+`` prefix allows adding fields that are not already declared in the mapping.

Run overrides use the same validation as graph overrides and run through ``env_cfg_callback``
after the environment's existing callback, including any overrides in its graph YAML.
Fields not specified by the Run retain their existing values.
The effective Experiment YAML records the Run overrides for reproducibility.
Omit the mapping to preserve the environment's configuration. The override changes the generated
Isaac Lab configuration; it does not rewrite the task definition or source files.

``episode_length_s`` changes the duration of an enabled timeout; it does not enable a disabled
timeout termination. The clamp task already enables it and defaults to 60 seconds.

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
