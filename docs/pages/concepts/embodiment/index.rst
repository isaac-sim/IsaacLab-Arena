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

More details
------------

The rest of this section covers further details of the embodiment component.

.. toctree::
   :maxdepth: 1

   concept_teleop_devices_design

Robot instance keys
-------------------

An instance key identifies one robot when several robots share an environment.
Franka embodiments accept ``instance_key="left"`` in their constructor.
The robot's asset name and primary scene key become ``left``.
The registered embodiment type remains available through ``embodiment_type``.
Omitting the key preserves the existing configuration.
Keys must be lowercase ASCII identifiers distinct from every original scene field.
For Franka, this excludes the robot, frame sensor, and camera field names.
This restriction keeps the capitalized robot prim names unique.

The configuration getters copy their output before applying these naming rules:

* The primary articulation uses the instance key. Other scene fields gain its prefix.
* Robot prim paths move from ``{ENV_REGEX_NS}/Robot`` to ``{ENV_REGEX_NS}/Left``.
* Observation groups and action, event, reward, curriculum, command, termination,
  and recorder terms gain the instance prefix.
* Entity references and frame-transformer sensor references follow the renamed scene fields.
* Logical target-frame names gain the prefix, keeping recorded poses distinct for each robot.
* Camera variation bindings follow the renamed cameras. Their catalogue names stay unchanged.
* Camera terms gain the prefix inside the shared ``camera_obs`` observation group.

Names within an ordinary observation group stay unchanged. A parameterless last-action
observation becomes a concatenation of that robot's raw action terms.
This allows a robot policy to receive the observation names it was trained against.

Every keyed term must pass entity parameters explicitly. Validation rejects omitted
entity defaults and literal scene or action-term lookups in inspectable Python functions.
The diagnostic names the configuration term and its unresolved reference.
Validation cannot inspect arbitrary helper calls or dynamically generated names.
Embodiment authors must avoid those references and validate the embodiment in simulation.
The shipped G1 controller contains a literal action lookup and does not accept a key.

Teleoperation and demonstration generation retain their single-robot assumptions.
