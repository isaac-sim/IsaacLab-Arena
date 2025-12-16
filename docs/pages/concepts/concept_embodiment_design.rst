Embodiment Design
==================

Embodiments define robot-specific configurations and behaviors. They provide a modular way to integrate different robots into environments, encapsulating kinematics, control actions, observations, and camera systems.

Core Architecture
-----------------

Embodiments use the ``EmbodimentBase`` abstract class that extends the asset system:

.. code-block:: python

   class EmbodimentBase(Asset):
       name: str | None = None
       tags: list[str] = ["embodiment"]
       mimic_arm_mode: MimicArmMode | None = None

       def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None, mimic_arm_mode: MimicArmMode | None = None):
           self.scene_config: Any | None = None
           self.action_config: Any | None = None
           self.observation_config: Any | None = None
           # ... other configs

       @abstractmethod
       def get_scene_cfg(self) -> Any:
           """Robot physical configuration and actuators."""

       @abstractmethod
       def get_action_cfg(self) -> Any:
           """Control interface definition."""

Embodiments in Detail
---------------------

**Configuration Components**
   Embodiments contribute multiple configuration aspects to environments:

   - **Scene Configuration**: Robot physical properties, actuators, initial pose, mounting stands
   - **Action Configuration**: Control interface (joint positions, end-effector poses, WBC)
   - **Observation Configuration**: Sensor data, joint states, end-effector poses, camera feeds
   - **Event Configuration**: Robot initialization, resets, randomization
   - **Camera Configuration**: Onboard cameras with positions and sensor properties (optional)
   - **XR Configuration**: XR device locations for teleop integration (optional)
   - **Mimic Configuration**: Mimic environment support for demonstration (optional)


**Available Embodiments**
   Robot assets with different capabilities and control modes:

   - **Franka Panda**: 7-DOF manipulator with differential IK control
   - **Unitree G1**: Humanoid with WBC, dual-arm manipulation, locomotion
   - **GR1T2**: Humanoid optimized for manipulation with dual end-effector control
   - **Control Variants**: Joint space vs. inverse kinematics control modes

**Camera Integration**
   Optional camera systems that add observation terms when enabled, supporting both manipulation and perception tasks with head-mounted or external cameras.


**Mimic Arm Mode**
    Embodiments expose an ``mimic_arm_mode`` attribute that declares which arms are
    movable for demonstration playback. This attribute uses the
    ``MimicArmMode`` enum from ``isaaclab_arena.embodiments.common.mimic_arm_mode``:

    - **SINGLE_ARM** – the robot has only one arm.
    - **DUAL_ARM** – bimanual robot, task is performed with both arms in the demonstration.
    - **LEFT** – bimanual robot, task is performed with the left arm and right arm is idle.
    - **RIGHT** – bimanual robot, task is performed with the right arm and left arm is idle.

    Tasks and mimic environment builders consult this property to request the correct
    subtask configurations, ensuring that mimic environments match the
    capabilities of the selected embodiment.

    .. code-block:: python

        class PickAndPlaceTask(TaskBase):
            """ Task class """

            def get_mimic_env_cfg(self, arm_mode: MimicArmMode):
                # gets the mimic environment configuration based on embodiment's mimic arm mode.
                return PickPlaceMimicEnvCfg(
                    arm_mode=arm_mode,
                    pick_up_object_name=self.pick_up_object.name,
                    destination_location_name=self.destination_location.name,
                )

        @configclass
        class PickPlaceMimicEnvCfg(MimicEnvCfg):
            """Mimic environment configuration class"""

            arm_mode: MimicArmMode = MimicArmMode.SINGLE_ARM

            def __post_init__(self):
                super().__post_init__()

                # defines the subtask configurations based on the mimic arm mode.
                if self.arm_mode == "single_arm":
                    # single arm subtask configuration
                elif self.arm_mode in ["left", "right"]:
                    # bimanual robot with one arm idle subtask configuration
                elif self.arm_mode == "dual_arm":
                    # dual arm subtask configuration
                else:
                    # raise error for unsupported mimic arm mode

        class ArenaEnvBuilder:
            """ ArenaEnvBuilder class """

            def compose_manager_cfg(self):
                # composes the mimic environment configuration based on embodiment's mimic arm mode.
                task_mimic_env_cfg = self.arena_env.task.get_mimic_env_cfg(
                    arm_mode=self.arena_env.embodiment.mimic_arm_mode
                )
                return task_mimic_env_cfg


Environment Integration
-----------------------

.. code-block:: python

   # Embodiment creation with camera support
   embodiment = asset_registry.get_asset_by_name("franka")(
       enable_cameras=True
   )

   # Set robot initial pose
   embodiment.set_initial_pose(
       Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
   )

   # Environment composition
   environment = IsaacLabArenaEnvironment(
       name="manipulation_task",
       embodiment=embodiment,  # Robot configuration
       scene=scene,
       task=task,
       teleop_device=teleop_device
   )

Usage Examples
--------------

**Manipulation Robot**

.. code-block:: python

   franka = asset_registry.get_asset_by_name("franka")(enable_cameras=True)
   task = PickAndPlaceTask(pick_object, destination, background)

**Humanoid Control Modes**

.. code-block:: python

   # Joint space control
   g1_joint = asset_registry.get_asset_by_name("g1_wbc_joint")()

   # Inverse kinematics control
   g1_ik = asset_registry.get_asset_by_name("g1_wbc_pink")()

**Teleop Integration**

.. code-block:: python

   embodiment = asset_registry.get_asset_by_name("gr1_pink")()
   teleop_device = device_registry.get_device_by_name("avp_handtracking")()

   environment = IsaacLabArenaEnvironment(
       embodiment=embodiment,
       teleop_device=teleop_device,
       scene=scene,
       task=task
   )
