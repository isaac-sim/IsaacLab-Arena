Physics backend selection
=========================

Arena environments can target **PhysX** or **Newton** for physics simulation.
Each ``IsaacLabArenaEnvironment`` can specify a default physics backend, which can be overridden at runtime with the ``--presets`` command-line argument.

Embodiments are intended to be compatible with both backends, with backend-specific configuration hooks invoked based on the runtime-resolved backend.
Most USD-backed rigid and articulation assets are loadable with both backends, although the current registered object and background libraries are primarily verified for PhysX.
Deformable objects and cables are only supported for a single backend.

Backend selection precedence
----------------------------

When ``ArenaEnvBuilder`` composes the Isaac Lab configuration in ``compose_manager_cfg``, the selected backend is determined as follows:

1. **CLI** — ``--presets`` / ``ArenaEnvBuilderCfg.presets`` takes precedence when set.
2. **Environment default** — ``IsaacLabArenaEnvironment.default_physics_backend`` (defaults to PhysX) is used when no CLI ``--presets`` is provided.

Authoring and CLI inputs:

.. code-block:: python

   # Environment factory (Python)
   return IsaacLabArenaEnvironment(
       name="dexsuite_lift",
       scene=scene,
       embodiment=embodiment,
       task=task,
       default_physics_backend=PhysicsBackend.NEWTON,
   )

   # Runner override (optional)
   builder = ArenaEnvBuilder(
       arena_env,
       ArenaEnvBuilderCfg(presets=PhysicsBackend.PHYSX),  # wins over env default
   )

Overriding backend default solver settings
------------------------------------------

``ArenaPhysicsCfg`` provides the default solver and simulation settings for each backend, which are assigned to ``env_cfg.sim.physics`` (``physx`` or ``newton``) when the backend is resolved.

``env_cfg_callback`` runs after the default solver settings assignment. Use it to tune solver settings under the already-selected backend—for example, solver iterations, contact settings, and collision pipelines.

After the callback returns, the builder asserts the solver type is still consistent with the resolved backend.

.. code-block:: python

   # Assign default solver settings to env_cfg.sim.physics
   arena_physics = ArenaPhysicsCfg()
   if resolved_physics_backend is PhysicsBackend.PHYSX:
       env_cfg.sim.physics = arena_physics.physx
   elif resolved_physics_backend is PhysicsBackend.NEWTON:
       env_cfg.sim.physics = arena_physics.newton
       env_cfg.scene.replicate_physics = True

   # Run the environment-specific callback to tune the solver settings
   if self.arena_env.env_cfg_callback is not None:
       env_cfg = self.arena_env.env_cfg_callback(env_cfg)
       if resolved_physics_backend is PhysicsBackend.PHYSX:
           assert isinstance(env_cfg.sim.physics, PhysxCfg)
       elif resolved_physics_backend is PhysicsBackend.NEWTON:
           assert isinstance(env_cfg.sim.physics, NewtonCfg)

Scene replication (``replicate_physics``)
-----------------------------------------

``ArenaEnvBuilder`` seeds its internal ``InteractiveSceneCfg`` with ``replicate_physics=False``.
That default favors per-environment physics when scenes differ across clones.

When the **resolved** backend is Newton, ``compose_manager_cfg`` sets ``replicate_physics = True``
before ``env_cfg_callback`` runs. ``replicate_physics = False`` is not supported for Newton by
Isaac Lab and could lead to errors when running with ``num_envs > 1``.

``env_cfg_callback`` can modify ``replicate_physics`` to False to disable replication for a specific environment.

Embodiment backend hooks
------------------------

Immediately after resolution, ``compose_manager_cfg`` calls ``embodiment.configure_physics_backend(resolved)`` **before** scene and action configs are combined. That entry point:

- Runs at most once per embodiment instance (reconfiguration raises an assertion).
- Delegates to ``EmbodimentBase._configure_physics_backend``, which subclasses override for backend-specific spawn, actuators, actions, and observations.

Hooks key off the **resolved** backend (CLI or environment default), not on post-callback edits to ``sim.physics``.
Most embodiments use the default no-op hook.
**DROID** is the shipped example: when the resolved backend is Newton, ``DroidEmbodimentBase`` switches robot spawn to ``spawn_newton_droid``, applies Newton material and gripper actuation, and control-mode subclasses add further Newton tuning (for example differential-IK controller settings).
PhysX builds keep the default DROID spawn and gripper wiring.

Author new backend-specific robot behavior by overriding ``_configure_physics_backend`` on your embodiment class.

Next step in ``compose_manager_cfg`` (embodiment hooks, then merge):

.. code-block:: python

   embodiment = self.arena_env.embodiment or NoEmbodiment()
   embodiment.configure_physics_backend(resolved_physics_backend)

   scene_cfg = combine_configclass_instances(
       "SceneCfg",
       self.interactive_scene_cfg,
       self.arena_env.scene.get_scene_cfg(),
       embodiment.get_scene_cfg(),  # spawn/actuators already backend-specific
       task.get_scene_cfg(),
   )
   # ... observation, action, event, and other manager cfgs follow ...

Asset validation
----------------

Rigid and articulation assets do not inspect ``sim.physics`` by default.
Backend-sensitive scene assets validate when the environment is registered.

``build_registered()`` finishes ``compose_manager_cfg`` (including ``env_cfg_callback``), validates the scene against ``env_cfg.sim``, registers with Gym, then ``parse_env_cfg`` applies only ``sim.device``, ``scene.num_envs``, and ``sim.use_fabric``.

**Deformables** — each ``DeformableObject`` infers its preset from ``spawner_cfg.deformable_props``:

- ``PhysxDeformableBodyPropertiesCfg`` → PhysX-only.
- ``NewtonDeformableBodyPropertiesCfg`` → Newton-only.

At validation time the asset compares that preset to the backend implied by ``sim.physics`` (treating ``sim.physics is None`` as PhysX, matching Isaac Lab).
A mismatch fails with an assertion naming the object and the selected backend.
Library deformables must therefore use the property type that matches the environment default (or the ``--presets`` override).

**Cables** — ``Cable`` assets require ``sim.physics`` to be a ``NewtonCfg``; PhysX or unset physics fails validation.

Rigid bodies and articulations are not checked automatically; pairing them with the wrong backend is still an authoring error, but only deformables and cables get an explicit Arena guard today.

``build_registered`` (after ``compose_manager_cfg`` returns):

.. code-block:: python

   env_cfg, env_kwargs = self.compose_manager_cfg()
   self.arena_env.scene.validate_simulation_cfg(env_cfg.sim)

Per-asset checks, such as ``DeformableObject.validate_simulation_cfg``, verify the asset is compatible with the selected backend.

Pipeline
--------

1. Resolve backend: ``cfg.presets`` if set, else ``arena_env.default_physics_backend``.
2. ``embodiment.configure_physics_backend(resolved)`` (subclass hooks).
3. Compose scene, actions, and observations from the configured embodiment.
4. Assign ``env_cfg.sim.physics`` from ``ArenaPhysicsCfg().physx`` or ``.newton`` (Newton also sets ``replicate_physics=True``).
5. Optional ``env_cfg_callback`` (tuning only; builder ``isinstance`` guard).
6. ``scene.validate_simulation_cfg`` when the Gym environment is registered.
7. ``parse_env_cfg`` for device, parallel env count, and fabric (does not change physics).

See also :doc:`index` for how scene, embodiment, and task compose into an environment.
