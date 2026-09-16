Physics backend selection
=========================

Arena environments can target **PhysX** or **Newton**. Backend-specific assets (deformables, cables) and embodiment spawn hooks must match the simulation backend. Arena resolves the backend once during compilation and applies it before embodiment configuration is finalized.

Precedence
----------

When ``ArenaEnvBuilder`` composes the Isaac Lab configuration:

1. **CLI** — ``--presets`` / ``ArenaEnvBuilderCfg.presets`` when set.
2. **Environment default** — ``IsaacLabArenaEnvironment.physics_backend`` (defaults to PhysX).

There is no scene-level inference: authors declare the intended default on the environment object.

Authoring
---------

**PhysX (default)** — omit ``physics_backend`` on ``IsaacLabArenaEnvironment`` or leave the default. Example: ``droid_deformable_pick_and_place`` uses PhysX deformable assets.

**Newton-required environments** — pass ``physics_backend=PhysicsBackend.NEWTON`` in the constructor, for example dexsuite lift and gear-insertion Newton factories. Users can still override with ``--presets physx`` only when the scene and task support it.

**Runners** — ``environment_runner.py`` and ``policy_runner.py`` build through ``get_arena_builder_from_cli``, which backfills ``args_cli.presets`` from ``arena_env.physics_backend`` when the flag is omitted so builder and environment defaults stay aligned.

Callbacks and YAML overrides
----------------------------

``env_cfg_callback`` runs **after** the resolved backend is written to ``env_cfg.sim.physics``. Use it to tune solver parameters, collision settings, or scene flags under that backend.

Callbacks must **not** replace PhysX with Newton (or the reverse). The builder asserts the backend type is unchanged after the callback. Select the backend via ``--presets`` or ``physics_backend`` on the environment instead.

Embodiment backend hooks
------------------------

After resolution, ``ArenaEnvBuilder`` calls ``embodiment.configure_physics_backend(resolved)`` **before** scene and action configs are combined. That entry point:

- Runs at most once per embodiment instance (reconfiguration raises an assertion).
- Delegates to ``EmbodimentBase._configure_physics_backend``, which subclasses override for backend-specific spawn, actuators, actions, and observations.

Most embodiments use the default no-op hook. **DROID** is the shipped example: when the resolved backend is Newton, ``DroidEmbodimentBase`` switches robot spawn to ``spawn_newton_droid``, applies Newton material and gripper actuation, and control-mode subclasses add further Newton tuning (for example differential-IK controller settings). PhysX builds keep the default DROID spawn and gripper wiring.

Author new backend-specific robot behavior by overriding ``_configure_physics_backend`` on your embodiment class, not by swapping ``sim.physics`` in a callback.

Asset validation
----------------

Rigid and articulation assets do not inspect ``sim.physics`` by default. Backend-sensitive scene assets validate when the environment is registered:

``ArenaEnvBuilder.build_registered()`` calls ``scene.validate_simulation_cfg(env_cfg.sim)``, which forwards to every asset in the scene.

**Deformables** — each ``DeformableObject`` infers its preset from ``spawner_cfg.deformable_props``:

- ``PhysxDeformableBodyPropertiesCfg`` → PhysX-only.
- ``NewtonDeformableBodyPropertiesCfg`` → Newton-only.

At validation time the asset compares that preset to the resolved backend (treating ``sim.physics is None`` as PhysX, matching Isaac Lab). A mismatch fails with an assertion naming the object and the selected backend. Library deformables must therefore use the property type that matches the environment default (or the ``--presets`` override).

**Cables** — ``Cable`` assets require ``sim.physics`` to be a ``NewtonCfg``; PhysX or unset physics fails validation.

Rigid bodies and articulations are not checked automatically; pairing them with the wrong backend is still an authoring error, but only deformables and cables get an explicit Arena guard today.

Callback and registration checks
----------------------------------

Two layers keep callbacks compatible with the resolved backend:

1. **Builder guard** — ``backend_type_from_sim_cfg`` records the backend immediately after ``env_cfg.sim.physics`` is materialized. After ``env_cfg_callback`` returns, ``assert_same_physics_backend`` requires the same PhysX vs Newton type. Replacing ``PhysxCfg`` with ``NewtonCfg`` (or vice versa) fails even if solver fields look similar.

2. **Callback self-checks** — some shipped callbacks assert stricter requirements. For example, ``assembly_env_cfg_callback`` requires PhysX before it applies assembly-specific ``PhysxCfg`` tuning, so ``--presets newton`` on a PhysX-only assembly environment fails inside the callback.

Callbacks may still **replace** the active ``PhysxCfg`` or ``NewtonCfg`` instance with another config of the **same** type (solver iterations, contact offsets, and so on). They must not change the backend class.

Pipeline
--------

1. Resolve backend (CLI or environment default).
2. ``embodiment.configure_physics_backend(resolved)`` (subclass hooks).
3. Compose scene, actions, and observations from the configured embodiment.
4. Materialize ``env_cfg.sim.physics`` from ``ArenaPhysicsCfg`` (Newton also sets ``replicate_physics=True``).
5. Optional ``env_cfg_callback`` (tuning only; builder backend guard).
6. ``scene.validate_simulation_cfg`` when the gym environment is registered.

See also :doc:`../environment/index` for how scene, embodiment, and task compose into an environment.
