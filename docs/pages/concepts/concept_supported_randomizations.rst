Supported Randomizations
========================

Arena has three randomization paths:

* **Arena variations** are named, configurable randomizations attached to assets. They can be
  discovered with ``--list_variations``, enabled with Hydra overrides, and their sampled values
  are automatically written to the per-episode ``variations`` record. **Recording is supported.**
* **Arena-native initial-state and scene randomization** uses Arena placement, pose, object-set,
  task, and embodiment APIs. **Recording is not currently supported.**
* **Isaac Lab event-based randomization** uses custom event terms merged from the scene,
  embodiment, and task configurations. **Recording is not supported by default**; a custom
  episode recorder term is required.

In the tables, **build time** means the value is selected while
``ArenaEnvBuilder.compose_manager_cfg()`` is compiling the environment and remains shared by all
parallel environments and episodes in that build. **Runtime** means the value can change on reset
without rebuilding the Arena environment. Object-set assignment happens before scene spawning and
is fixed for the lifetime of the built environment.


Arena variations
----------------

The current ``isaaclab_arena/variations/`` package contains exactly eight concrete variations:
HDR image, light intensity, light color, light color temperature, light direction, camera
extrinsics, camera intrinsics, and rigid-object mass.

Run the following command to list the exact variation hosts and override paths available in a
particular environment:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --list_variations \
     pick_and_place_maple_table

.. list-table::
   :header-rows: 1
   :widths: 45 20 35

   * - Randomization type and description
     - Build time or runtime
     - Code snippet to enable it
   * - **HDR environment image** (``HDRImageVariation``). Selects a registered HDR map for a dome
       light. An empty ``hdr_names`` list samples from every registered HDR.
     - Build time; use multiple environment rebuilds to obtain multiple images.
     - | ``light.hdr_image.enabled=true``
       | ``"light.hdr_image.hdr_names=[home_office_robolab,garage_robolab]"``
   * - **Light intensity** (``LightIntensityVariation``). Samples one light intensity; the default
       uniform range is ``[100, 2000]``.
     - Build time.
     - | ``light.intensity.enabled=true``
       | ``light.intensity.sampler_cfg.low=[500]``
       | ``light.intensity.sampler_cfg.high=[1500]``
   * - **Light RGB color** (``LightColorVariation``). Samples RGB channels independently; each
       channel defaults to ``[0, 1]``.
     - Build time.
     - | ``light.color.enabled=true``
       | ``"light.color.sampler_cfg.low=[0.5,0.5,0.5]"``
       | ``"light.color.sampler_cfg.high=[1,1,1]"``
   * - **Light color temperature** (``LightColorTemperatureVariation``). Samples a white-point
       temperature; the default range is ``[1000, 10000]`` K.
     - Build time.
     - | ``light.color_temperature.enabled=true``
       | ``light.color_temperature.sampler_cfg.low=[3000]``
       | ``light.color_temperature.sampler_cfg.high=[6500]``
   * - **Directional-light direction** (``LightDirectionVariation``). Samples azimuth in
       ``[-pi, pi]`` and elevation from overhead to 80 degrees by default. It can also dim a
       registered dome light so directional shadows remain visible.
     - Build time.
     - | ``directional_light.direction.enabled=true``
       | ``directional_light.direction.dome_intensity_when_active=500``
   * - **Camera extrinsics** (``CameraExtrinsicsVariation``). Adds an XYZ translation offset in
       the camera ROS optical frame. The default range is ±5 mm on each axis. It preserves a wrist
       camera's parent-relative mounting behavior.
     - Runtime; sampled independently for each resetting environment.
     - | ``droid_abs_joint_pos.camera_extrinsics_wrist_camera.enabled=true``
       | ``"droid_abs_joint_pos.camera_extrinsics_wrist_camera.sampler_cfg.low=[-0.01,-0.01,-0.01]"``
       | ``"droid_abs_joint_pos.camera_extrinsics_wrist_camera.sampler_cfg.high=[0.01,0.01,0.01]"``
   * - **Camera intrinsics** (``CameraIntrinsicsVariation``). Samples fractional focal-length
       perturbations ``(d_fx, d_fy)``; defaults are ±10%. It forces the camera rig to use untiled
       cameras during build so each environment can have distinct intrinsics.
     - Runtime samples, with a build-time camera setup change.
     - | ``droid_abs_joint_pos.camera_intrinsics_wrist_camera.enabled=true``
       | ``"droid_abs_joint_pos.camera_intrinsics_wrist_camera.sampler_cfg.low=[-0.05,-0.05]"``
       | ``"droid_abs_joint_pos.camera_intrinsics_wrist_camera.sampler_cfg.high=[0.05,0.05]"``
   * - **Rigid-object mass** (``ObjectMassVariation``). Samples an absolute mass in kilograms and
       optionally scales inertia by the sampled/default mass ratio. It supports one selected rigid
       body per Arena object. The default range is ``[0.05, 2.0]`` kg.
     - Runtime; sampled independently on reset.
     - | ``cracker_box.mass.enabled=true``
       | ``cracker_box.mass.sampler_cfg.low=[0.1]``
       | ``cracker_box.mass.sampler_cfg.high=[0.5]``
       | ``cracker_box.mass.recompute_inertia=true``

The host prefix in a Hydra variation path is the asset's registered name. For example,
``light.intensity`` is valid only when the selected environment contains an asset named ``light``.
Use ``--list_variations`` instead of assuming a host or camera name.

.. toctree::
   :maxdepth: 1

   variations/variations


Arena-native initial-state and scene randomization
--------------------------------------------------

These mechanisms are supported by Arena composition, but do not use the named variation system.
Their sampled values are therefore not recorded automatically.

.. list-table::
   :header-rows: 1
   :widths: 45 20 35

   * - Randomization type and description
     - Build time or runtime
     - Code snippet to enable it
   * - **Rigid object-set member** (``RigidObjectSet``). Gives each parallel environment one rigid
       object selected from a candidate set. ``random_choice=False`` cycles through the declared
       order; ``True`` samples independently. The assignment remains fixed across resets.
     - Build/assignment time, before spawning. A different member in an existing environment
       requires rebuilding; no rebuild is needed merely to evaluate heterogeneous members
       concurrently.
     - .. code-block:: python

          fruit = RigidObjectSet(name="fruit", objects=members, random_choice=True)
          fruit.add_relation(On(table))

          # Use for reproducible assignment.
          cfg = ArenaEnvBuilderCfg(num_envs=8, placement_seed=42)
   * - **Object root pose from a range** (``PoseRange``). Samples position and roll/pitch/yaw
       uniformly within the configured bounds on every reset. Supported by rooted rigid and
       articulated Arena objects; embodiments deliberately do not accept ``PoseRange``.
     - Runtime; sampled per resetting environment. The range is configured before build.
     - .. code-block:: python

          obj.set_initial_pose(
              PoseRange(
                  position_xyz_min=(0.3, -0.1, 0.1),
                  position_xyz_max=(0.5, 0.1, 0.1),
                  rpy_min=(0, 0, -0.2),
                  rpy_max=(0, 0, 0.2),
              )
          )
   * - **Externally generated per-environment object or embodiment root poses**
       (``PosePerEnv``). Supplies one pose per parallel environment and restores that pose on
       reset. This supports both objects and embodiments, and can represent random samples
       generated by user code before building.
     - Configured at build time; fixed per environment at runtime.
     - ``asset.set_initial_pose(PosePerEnv(poses=[pose_for_env_0, pose_for_env_1]))``
   * - **Relation-solved object and embodiment placement, resampled on reset.** Arena pre-solves a
       pool of layouts satisfying ``On``, ``NextTo``, ``NotNextTo``, ``AtPosition``,
       ``PositionLimitsBox``, and ``PositionLimitsCylindrical``; ``FaceTo`` controls heading.
       ``IsAnchor`` fixes references, and ``random_yaw_init=True`` adds uniform yaw initialization.
       The same path supports co-placement of an embodiment when relations are attached to it.
     - Runtime layout selection on reset; the layout pool and object geometry are prepared at
       build time.
     - .. code-block:: python

          table.add_relation(IsAnchor())
          obj.add_relation(On(table))
          obj.add_relation(NextTo(other))
          placer = ObjectPlacerParams(
              resolve_on_reset=True,
              random_yaw_init=True,
              placement_seed=42,
          )
          arena_env = IsaacLabArenaEnvironment(..., placer_params=placer)
   * - **Relation-solved fixed per-environment placement.** Uses the same relation solver but
       assigns one solved ``PosePerEnv`` layout and reuses it on every reset.
     - Build time; fixed during runtime.
     - .. code-block:: python

          placer = ObjectPlacerParams(resolve_on_reset=False, placement_seed=42)
          arena_env = IsaacLabArenaEnvironment(..., placer_params=placer)

``RandomAroundSolution`` is available to direct, single-environment ``ObjectPlacer`` users, where
it converts a solved pose into a ``PoseRange``. It is not currently applied by the default
``ArenaEnvBuilder`` relation-placement path, which owns pose application itself. Use an explicit
``PoseRange`` when reset-time jitter is required in a normally built Arena environment.

The built-in episode recorder records ``ArenaEnvBuilderCfg.seed`` for every episode. It does not
record the separate ``placement_seed``. Even when the relevant seed is available, it is not a
substitute for recording the actual object-set member, pose, relation layout, or joint-state draw.


Isaac Lab event-based randomization examples
--------------------------------------------

Users can also author custom Isaac Lab event-based randomization. Arena merges these event terms
from ``scene.events_cfg``, embodiment events, and task events into the manager-based environment.
The table below lists typical examples; it is not exhaustive.

.. list-table::
   :header-rows: 1
   :widths: 45 20 35

   * - Randomization type and description
     - Build time or runtime
     - Code snippet to enable it
   * - **Franka and DROID initial joint state.** Their configured Isaac Lab reset event adds a
       Gaussian offset to the default joint state; both default to ``mean=0`` and ``std=0.02``.
       Arena's ``set_initial_joint_pose(...)`` or ``set_joint_initial_pos(...)`` changes the
       default around which the event samples.
     - Runtime Gaussian draw on reset; the default joint pose is configured at build time.
     - .. code-block:: python

          embodiment.event_config.randomize_franka_joint_state.params.update(
              {"mean": 0.0, "std": 0.05}
          )
          embodiment.set_joint_initial_pos({"panda_joint1": 0.1})
   * - **Physics material: static/dynamic friction and restitution.**
       ``randomize_rigid_body_material`` supports rigid objects and articulation bodies. PhysX and
       OVPhysX use separate static/dynamic friction; Newton uses one friction coefficient and
       ignores ``dynamic_friction_range`` and ``num_buckets``.
     - Runtime reset or startup. PhysX recommends startup because material buckets are assigned
       through CPU tensors; Newton supports runtime writes.
     - .. code-block:: python

          EventTermCfg(
              func=mdp.randomize_rigid_body_material,
              mode="reset",
              params={
                  "asset_cfg": SceneEntityCfg("object"),
                  "static_friction_range": (0.4, 1.0),
                  "dynamic_friction_range": (0.4, 1.0),
                  "restitution_range": (0.0, 0.1),
                  "num_buckets": 64,
              },
          )
   * - **Rigid-body center of mass.** ``randomize_rigid_body_com`` adds sampled XYZ offsets to each
       selected body's default COM. Newton supports the write, but the current implementation
       warns that runtime COM changes may not fully recompute the mass matrix and can be unstable.
     - Runtime reset or startup; startup is safer on Newton.
     - .. code-block:: python

          EventTermCfg(
              func=mdp.randomize_rigid_body_com,
              mode="reset",
              params={
                  "asset_cfg": SceneEntityCfg("object"),
                  "com_range": {
                      "x": (-0.01, 0.01),
                      "y": (-0.01, 0.01),
                      "z": (-0.01, 0.01),
                  },
              },
          )
   * - **Collider rest/contact offsets.** ``randomize_rigid_body_collider_offsets`` varies PhysX
       rest/contact offsets; on Newton these map to shape margin and shape gap.
     - Runtime-capable, although initialization is recommended.
     - .. code-block:: python

          EventTermCfg(
              func=mdp.randomize_rigid_body_collider_offsets,
              mode="reset",
              params={
                  "asset_cfg": SceneEntityCfg("object"),
                  "rest_offset_distribution_params": (0.0, 0.002),
                  "contact_offset_distribution_params": (0.002, 0.01),
              },
          )
   * - **Actuator gains.** ``randomize_actuator_gains`` varies articulation stiffness and damping
       with uniform, log-uniform, or Gaussian draws and ``add``, ``scale``, or absolute operations.
     - Runtime-capable; startup is recommended for implicit actuators using CPU-backed writes.
     - .. code-block:: python

          EventTermCfg(
              func=mdp.randomize_actuator_gains,
              mode="reset",
              params={
                  "asset_cfg": SceneEntityCfg("robot"),
                  "stiffness_distribution_params": (0.8, 1.2),
                  "damping_distribution_params": (0.8, 1.2),
                  "operation": "scale",
              },
          )
   * - **Joint and fixed-tendon properties.** ``randomize_joint_parameters`` varies friction,
       armature, and lower/upper position limits. ``randomize_fixed_tendon_parameters`` varies
       tendon stiffness, damping, limits, rest length, and offset where supported; some tendon
       properties differ between PhysX and Newton.
     - Runtime-capable; startup is recommended for CPU-backed property writes.
     - .. code-block:: python

          EventTermCfg(
              func=mdp.randomize_joint_parameters,
              mode="reset",
              params={
                  "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                  "friction_distribution_params": (0.8, 1.2),
                  "armature_distribution_params": (0.8, 1.2),
                  "operation": "scale",
              },
          )
   * - **Physics-scene gravity.** ``randomize_physics_scene_gravity`` varies gravity by adding,
       scaling, or setting a sampled vector. PhysX and OVPhysX use one scene-wide vector; Newton
       supports a different vector per environment.
     - Runtime reset or interval.
     - .. code-block:: python

          EventTermCfg(
              func=mdp.randomize_physics_scene_gravity,
              mode="reset",
              params={
                  "gravity_distribution_params": (
                      [-0.1, -0.1, -10.0],
                      [0.1, 0.1, -9.6],
                  ),
                  "operation": "abs",
              },
          )
   * - **External wrench or velocity push.** ``apply_external_force_torque`` samples persistent
       forces/torques; ``push_by_setting_velocity`` samples a root-velocity change.
     - Runtime reset or interval.
     - .. code-block:: python

          EventTermCfg(
              func=mdp.apply_external_force_torque,
              mode="interval",
              interval_range_s=(1.0, 3.0),
              params={
                  "asset_cfg": SceneEntityCfg("robot"),
                  "force_range": (-10.0, 10.0),
                  "torque_range": (-1.0, 1.0),
              },
          )
   * - **General root, joint, and deformable nodal state.** ``reset_root_state_uniform`` samples
       rigid/articulation root pose and velocity; ``reset_root_state_with_random_orientation``
       samples orientation uniformly on SO(3); ``reset_joints_by_offset``,
       ``reset_joints_by_scale``, and ``reset_joints_within_limits_range`` sample articulation
       state; ``reset_nodal_state_uniform`` samples deformable position and velocity.
     - Runtime; sampled on reset.
     - .. code-block:: python

          EventTermCfg(
              func=mdp.reset_root_state_uniform,
              mode="reset",
              params={
                  "asset_cfg": SceneEntityCfg("object"),
                  "pose_range": {
                      "x": (-0.1, 0.1),
                      "yaw": (-0.2, 0.2),
                  },
                  "velocity_range": {},
              },
          )
   * - **Visual material, color, texture, and shape channels.** ``randomize_visual_material`` and
       ``randomize_visual_shape`` use renderer-backed runtime storage. The Replicator terms
       ``randomize_visual_texture_material`` and ``randomize_visual_color`` operate on USD visual
       prims and require scene replication to be disabled. Renderer/backend support differs by
       term.
     - Runtime reset or interval after renderer initialization. Some Replicator setup occurs when
       the event term is constructed.
     - .. code-block:: python

          EventTermCfg(
              func=mdp.randomize_visual_color,
              mode="reset",
              params={
                  "event_name": "object_color",
                  "asset_cfg": SceneEntityCfg("object"),
                  "colors": {
                      "r": (0.0, 1.0),
                      "g": (0.0, 1.0),
                      "b": (0.0, 1.0),
                  },
              },
          )
   * - **Rigid-object scale.** ``randomize_rigid_body_scale`` samples uniform or per-axis scale by
       editing USD. Articulations are not supported. Because physics parses scale before
       simulation starts, this must use the pre-play ``usd`` event mode.
     - Build/startup only. New scale values after simulation starts require recreating the
       environment.
     - .. code-block:: python

          EventTermCfg(
              func=mdp.randomize_rigid_body_scale,
              mode="usd",
              params={
                  "asset_cfg": SceneEntityCfg("object"),
                  "scale_range": (0.8, 1.2),
              },
          )
   * - **Other custom Isaac Lab event-term randomization.** Additional compatible
       ``EventTermCfg`` terms can use reset, interval, startup, prestartup, or USD timing as
       supported by that term and backend.
     - Depends on the event mode and implementation.
     - Define ``MyEventsCfg`` containing the event terms, then set
       ``scene.events_cfg = MyEventsCfg()`` before building.


Randomization that requires rebuilding
--------------------------------------

The following common changes alter scene composition and therefore cannot be applied to an already
built Arena environment:

* **Clutter object count.** Adding or removing clutter changes scene entities, manager
  configuration, collision geometry, and potentially sensors. The environment must be rebuilt.
  Poses of an existing fixed set of clutter objects can still be changed at runtime using
  ``PoseRange``, relation placement, or reset event terms.
* **Embodiment swap.** Changing the robot changes its articulation, actions, observations,
  cameras, sensors, controllers, and event terms. Select the new embodiment before building a new
  environment.

Other composition changes—such as replacing an ordinary object's USD outside a predeclared
``RigidObjectSet``, changing the number or type of sensors, or switching physics backends—likewise
require rebuilding.
