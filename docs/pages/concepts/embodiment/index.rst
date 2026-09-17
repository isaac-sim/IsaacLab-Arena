Embodiment
==========

An embodiment is the robot: its physical description, control interface, sensors, and cameras.
Because the embodiment is independent of the scene and task, you can swap the robot
without touching anything else. The same pick-and-place task works with a Franka or a G1.

.. code-block:: python

   embodiment = asset_registry.get_asset_by_name("franka_ik")(enable_cameras=True)

   environment = IsaacLabArenaEnvironment(
       name="kitchen_pick_and_place",
       embodiment=embodiment,
       scene=scene,
       task=task,
   )

Walkthrough
-----------

We load the embodiment from the registry, passing any options to its constructor:

.. code-block:: python

   embodiment = asset_registry.get_asset_by_name("franka_ik")(enable_cameras=True)
   embodiment.set_initial_pose(Pose(position_xyz=(0.5, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

The initial pose places the robot in world frame — relative to the scene origin.
This is usually set to position the robot in front of the workspace.

Available embodiments include the Franka Panda, Unitree G1, GR1T2, DROID, and others.
Each has one or more control variants registered separately.
For example, ``franka_ik`` uses differential IK control,
while ``franka_joint_pos`` uses direct joint position control.

**Cameras**

Passing ``enable_cameras=True`` adds the robot's onboard cameras to the observation space.
This is required for any policy that takes image observations, such as GR00T.

Robot and end-effector physics
-----------------------------

The embodiment owns robot physics, including end-effector contact materials, gripper
colliders, self-collision exclusions, joint coupling, and actuator configuration.

Configure the physics backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement backend-specific robot defaults in ``_configure_physics_backend(self, backend)``.
Use this hook to select compatible robot configuration and actuator settings for the resolved
backend. Task-dependent end-effector values can be exposed as embodiment configuration and
used by this hook.

.. list-table:: Backend configuration
   :header-rows: 1
   :widths: 28 36 36

   * - Change
     - Configure it in
     - How it is applied
   * - Backend-specific robot defaults
     - ``_configure_physics_backend(backend)``
     - Updates the embodiment's configuration before scene composition.
   * - Controlled joint stiffness, damping, and effort limits
     - The robot's ``ArticulationCfg.actuators`` in the backend hook
     - Articulation initialization creates the actuators and applies their settings.

Apply spawn config addons
~~~~~~~~~~~~~~~~~~~~~~~~~

``spawn_cfg_addon`` defines how the embodiment's USD is loaded and which physics properties
are authored during spawning. Its outer keys name entries in the embodiment's scene config:
``robot`` for a single robot, or ``left_robot`` / ``right_robot`` for a bimanual embodiment.
The base class copies this mapping per instance.

``_apply_spawn_cfg_addons()`` applies the mapping to those entries' spawn configs after the
backend hook finishes. It preserves unspecified USD paths, scales, variants, and other spawn
options. Every named entry must exist and have a spawn config; all replacements validate before
they are published.

.. list-table:: Spawn addons
   :header-rows: 1
   :widths: 28 36 36

   * - Change
     - Configure it in
     - How it is applied
   * - USD loading options and robot-wide collision/material settings
     - ``spawn_cfg_addon["robot"]``
     - The USD spawner uses these options when loading the robot.
   * - Selected finger contacts, colliders, joint coupling, or collision exclusions
     - ``spawn_cfg_addon["robot"]["prim_physics"]`` with a concrete ``UsdPrimSpawnPhysicsCfg``
     - The spawn hook edits selected prims after USD loading, before cloning and import.

For example, define or import the ``ColliderFrictionCfg`` implementation shown in
:doc:`../scene/concept_assets_design` and declare these defaults on your embodiment class:

.. code-block:: python

   spawn_cfg_addon = {
       "robot": {
           "prim_physics": {
               "finger/collision": ColliderFrictionCfg(friction=0.8),
           },
       },
   }

Use the exact collider path in the robot USD; ``finger/collision`` is illustrative.
When addon values depend on the backend, set them in ``_configure_physics_backend()``;
``_apply_spawn_cfg_addons()`` then applies the resulting mapping automatically.

Call order
~~~~~~~~~~

Before collecting the embodiment's scene configuration, the environment builder calls
``configure_physics_backend(backend)``. This public wrapper:

1. Calls ``_configure_physics_backend(backend)`` to configure robot defaults.
2. Calls ``_apply_spawn_cfg_addons()`` to update the robot's spawn configs.
3. Records the configured backend so repeating the same call does not apply settings again.

Both steps prepare configuration. USD loading and per-prim physics edits happen later during
spawning. See :doc:`../environment/env_cfg_override` for the complete application order.

More details
------------

The rest of this section covers further details of the embodiment component.

.. toctree::
   :maxdepth: 1

   concept_teleop_devices_design
