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
------------------------------

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

Applying changes
~~~~~~~~~~~~~~~~

``spawn_cfg_addon`` defines how the embodiment's USD is loaded and which physics properties
are authored during spawning. Its outer keys name entries in the embodiment's scene config:
``robot`` for a single robot, or ``left_robot`` / ``right_robot`` for a bimanual embodiment.
The base class copies this mapping per instance.

``get_scene_cfg()`` applies the mapping to those entries' spawn configs whenever the scene
is collected. The builder configures backend defaults first. Unspecified USD paths, scales,
variants, and other spawn options are preserved. Every named entry must exist and have a spawn config; all replacements validate before
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

For example, reuse the existing Franka and expose contact friction as a constructor parameter:

.. code-block:: python

   from isaaclab.sim import RigidBodyMaterialCfg

   from isaaclab_arena.assets.register import register_asset
   from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment


   @register_asset
   class ContactFranka(FrankaIKEmbodiment):
       name = "contact_franka"

       def __init__(self, contact_friction: float = 0.8, **kwargs):
           super().__init__(**kwargs)
           self.spawn_cfg_addon["robot"] = {
               "physics_material": RigidBodyMaterialCfg(
                   static_friction=contact_friction,
                   dynamic_friction=contact_friction,
               ),
           }

To target individual colliders instead, define or import ``ColliderFrictionCfg`` from the
example in :doc:`../scene/concept_assets_design` and use ``prim_physics``:

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
``get_scene_cfg()`` then applies the resulting mapping automatically.

For per-prim settings in YAML, use the same constructor-parameter pattern to construct
``ColliderFrictionCfg`` from the supplied friction value.

Save the ``ContactFranka`` definition in ``my_project/robots.py``. Both examples set contact
friction to ``1.2`` for this environment:

.. tab-set::

   .. tab-item:: env.yaml

      Import ``my_project.robots`` in your runner before loading the environment graph so its
      registration runs. Add this embodiment block to your environment YAML:

      .. code-block:: yaml

         embodiment:
           id: robot
           registry_name: contact_franka
           params:
             contact_friction: 1.2

   .. tab-item:: env.py

      Pass the robot instance alongside your scene and task:

      .. code-block:: python

         from my_project.robots import ContactFranka

         from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

         environment = IsaacLabArenaEnvironment(
             name="franka_contact_task",
             embodiment=ContactFranka(contact_friction=1.2),
             scene=scene,
             task=task,
         )

See :doc:`../environment/physics_configuration` for configuration scopes and application order.

Call order
~~~~~~~~~~

The environment builder prepares robot physics in two steps:

1. ``configure_physics_backend(backend)`` calls the backend hook once to set robot defaults.
2. ``get_scene_cfg()`` applies spawn addons and returns the scene configuration. Direct callers
   also get the addons, and later calls pick up changes to the mapping.

USD loading and per-prim edits happen later during spawning. Custom USD spawners, including
DROID's Newton spawner, keep their setup: their single-prim body runs first, then the per-prim
edits, then cloning. These spawners must use Isaac Lab's ``@clone`` decorator, with no other
decorators or cloning inside the function body.

See :doc:`../environment/physics_configuration` for the complete application order.

More details
------------

The rest of this section covers further details of the embodiment component.

.. toctree::
   :maxdepth: 1

   concept_teleop_devices_design
