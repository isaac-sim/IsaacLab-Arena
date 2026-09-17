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
Implement backend-specific settings in ``_configure_physics_backend(self, backend)``.
The environment builder calls the public ``configure_physics_backend()`` wrapper before
collecting the embodiment's scene configuration.

For per-collider or per-joint settings, use ``with_prim_physics()`` from that hook:

.. code-block:: python

   from isaaclab_arena.assets.physics_spawner import with_prim_physics

   robot_cfg = self.scene_config.robot
   robot_cfg.spawn = with_prim_physics(robot_cfg.spawn, overrides)

The ``overrides`` mapping contains asset-relative prim paths and ``PrimPhysicsCfg`` values.
The helper returns an independent spawn config with the mapping and physics spawn wrapper;
the robot's USD path, scale, variants, and other spawn options are preserved.
An internal config subclass retains the settings when Isaac Lab copies the config.
Use the actuator configuration
for controlled joint gains. Task-dependent end-effector values should be exposed as embodiment
configuration and consumed by the same hook.

The hook prepares configuration; the spawner applies it after loading the robot USD and before
cloning and physics model import. An embodiment with a custom spawn function should integrate
``apply_prim_physics`` into that function before cloning.

Scene objects such as cubes, boxes, and fixtures use ``Object.spawn_cfg_addon`` for their own
contact properties. See :doc:`../scene/concept_assets_design` for the supported per-prim fields.

More details
------------

The rest of this section covers further details of the embodiment component.

.. toctree::
   :maxdepth: 1

   concept_teleop_devices_design
