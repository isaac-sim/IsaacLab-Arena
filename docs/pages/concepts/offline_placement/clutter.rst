Offline Clutter Settling
========================

``ClutterOn`` defines release poses. ``settle_clutter`` uses the relation solver
to generate those poses, advances physics, and returns accepted layouts.
This API runs after ``SimulationApp`` starts and the environment is constructed.

.. code-block:: python

   from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
   from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
   from isaaclab_arena.offline_placement.clutter_settling import settle_clutter

   # arena_env is an environment definition containing ClutterOn relations.
   env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(solve_relations=False)).make_registered()
   env.reset()
   layouts = settle_clutter(
       env, arena_env.get_placement_assets(), placer_params=arena_env.placer_params
   )
   poses = layouts[0].poses
   checks = layouts[0].validation
   env.close()

The result contains one layout per parallel environment, keyed by runtime scene
name. Positions are environment-local metres and rotations are xyzw quaternions.
The caller's initial scene and robot targets are restored after generation.
Pass ``arena_env.placer_params`` to retain the environment's solver settings;
omitting it uses fresh ``ObjectPlacerParams`` defaults.

Acceptance checks
-----------------

Releases must pass the configured solver checks, including ``no_overlap`` and
``clutter_on_relation``. The latter permits a positive height above the support
while requiring the release footprint to fit.

After physics, the default validators require consecutive quiet pose samples,
containment on the full support, and no excessive drift of passive bodies or
robot links. All enabled validators must pass. Failed environments are retried;
exhausting the attempt limit fails the call instead of returning unvalidated poses.

``build_post_physics_validators`` in ``offline_placement.clutter_validators``
builds the configured checks. Pass its result through ``validators=``. The builder
prints effective settings and disabled checks. Each result retains post-physics
configurations and outcomes; pre-physics evidence contains release verdicts only.
Only one rest validator may be enabled, so each reported threshold matches the
motion history used for acceptance.

For example, change the rest threshold while retaining the other default checks:

.. code-block:: python

   from isaaclab_arena.offline_placement.clutter_validators import (
       build_post_physics_validators,
       default_post_physics_validators,
   )

   configurations = default_post_physics_validators()
   configurations["rest"]["move_thresh_m"] = 0.001
   validators = build_post_physics_validators(configurations)
   layouts = settle_clutter(
       env,
       arena_env.get_placement_assets(),
       placer_params=arena_env.placer_params,
       validators=validators,
   )

Use this call in place of the default ``settle_clutter`` call above, before closing
the environment. Set a check's ``enabled`` field to ``False`` to skip it; the report
retains the skip reason. At least one post-physics check must remain enabled.

Scope
-----

Supports must be fixed anchors. Clutter members must be dynamic rigid bodies with
gravity enabled. Other placement must already be resolved to fixed anchors.
For each ``RigidObjectSet``, call ``assign_variants(num_envs, variant_seed=seed)``
before constructing the environment, using the same ``num_envs`` as the builder.
Settling rejects missing or differently sized assignments before solving or
writing scene state. It cannot safely assign new variants to an already spawned
scene.

Robot joints are not recorded, and reachability of the settled pile is not
certified. Post-physics acceptance checks rest, support containment and passive
drift; it does not rerun the release collision or relation validators.
Clutter with roll or pitch requires BBOX collision mode because the
solver's mesh checks currently transform geometry by yaw only.

The offline package depends on the solver and the shared validator base.
The solver and runtime replay do not import the offline package.
