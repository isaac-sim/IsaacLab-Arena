Embodiment Design
==================

Embodiments in Isaac Arena define robot-specific configurations and behaviors, providing a modular way to integrate different robots into environments. Each embodiment encapsulates all robot-specific aspects including kinematics, control actions, observations, and specialized behaviors.

Core Architecture
-----------------

The embodiment system is built around the ``EmbodimentBase`` abstract class, which defines a standard interface for all robots:

.. code-block:: python

   class EmbodimentBase(Asset):
       name: str | None = None
       tags: list[str] = ["embodiment"]

       def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
           # Robot-specific configurations
           self.scene_config: Any | None = None
           self.camera_config: Any | None = None
           self.action_config: Any | None = None
           self.observation_config: Any | None = None
           self.event_config: Any | None = None
           self.mimic_env: Any | None = None
           self.xr: Any | None = None

Configuration Components
------------------------

Each embodiment provides several configuration components:

**Scene Configuration**
   Defines the robot's physical properties, joint actuators, initial pose, and any robot-specific scene elements (e.g., mounting stands).

**Action Configuration**
   Specifies the robot's control interface - joint positions, end-effector poses, or specialized controllers like whole-body control (WBC).

**Observation Configuration**
   Defines what sensor data and state information the robot provides, including joint states, end-effector poses, and camera feeds.

**Event Configuration**
   Handles robot initialization, resets, and any robot-specific events during simulation.

**Camera Configuration** *(Optional)*
   Defines onboard cameras with their positions, orientations, and sensor properties. This is only used when ``enable_cameras`` is set to ``True``. When it is used an observation term is added to the observations config for each camera.

**XR Configuration** *(Optional)*
   Provides the location of the XR camera device.

**Mimic Configuration** *(Optional)*
   Provides the configuration for the mimic environment. This is only used when ``mimic_env`` is set to ``True``.

Available Embodiments
---------------------

**Franka Panda** (``franka``)
   - 7-DOF manipulator with parallel gripper
   - Differential inverse kinematics control
   - End-effector pose observations
   - Suitable for manipulation tasks

**Unitree G1** (``g1_wbc_joint``, ``g1_wbc_pink``)
   - Humanoid robot with arms and dexterous hands
   - Whole-body control (WBC) with walking capabilities
   - Dual-arm manipulation with navigation commands
   - Head-mounted camera support
   - Available in joint control and IK control variants

**Fourier GR1T2** (``gr1_joint``, ``gr1_pink``)
   - Humanoid robot optimized for manipulation
   - Joint position and inverse kinematics control modes
   - Dual end-effector control with hand articulation
   - Head-mounted cameras for perception

Mimic Configurations
---------------------

Each embodiment provides a specialized ``MimicEnv`` class for enabling mimic components of IsaacLab to be seamlessly integrated:


Usage Examples
--------------

**Basic Embodiment Usage**:

.. code-block:: bash

   python isaac_arena/examples/policy_runner.py --policy_type zero_action kitchen_pick_and_place --embodiment franka

**Humanoid Manipulation**:

.. code-block:: bash

   python isaac_arena/examples/policy_runner.py --policy_type zero_action kitchen_pick_and_place --embodiment g1_wbc_joint

**Different Control Modes**:

.. code-block:: bash

   # Joint control
   python isaac_arena/examples/policy_runner.py --embodiment gr1_joint

   # IK control
   python isaac_arena/examples/policy_runner.py --embodiment gr1_pink

Creating New Embodiments
-------------------------

To add a new robot embodiment:

1. **Create embodiment class** inheriting from ``EmbodimentBase``
2. **Define robot configuration** with actuators, joints, and physical properties
3. **Implement action space** specifying control interface
4. **Define scene configuration** for the robot
5. **Define camera configuration** for the robot
6. **Define observations** for sensors and state information
6. **Define event configuration** for the robot
5. **Create mimic environment** for using mimic support
6. **Register embodiment** using the ``@register_asset`` decorator

The modular embodiment system enables rapid integration of new robots while maintaining consistency across different robot platforms and control paradigms.
