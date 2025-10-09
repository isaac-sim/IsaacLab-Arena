Teleop Devices Design
======================

Teleop devices in Isaac Arena provide a unified interface for different input devices used in teleoperation and demonstration collection. They enable seamless switching between keyboard, spacemouse, and hand tracking devices through a common abstraction layer that integrates with embodiments and environments.

Core Architecture
-----------------

The teleop device system is built around the ``TeleopDeviceBase`` abstract class with automatic registration:

.. code-block:: python

   class TeleopDeviceBase(ABC):
       name: str | None = None

       def __init__(self, sim_device: str | None = None):
           self.sim_device = sim_device

       @abstractmethod
       def get_teleop_device_cfg(self, embodiment: object | None = None):
           """Return Isaac Lab DevicesCfg for the specific device."""
           raise NotImplementedError

   @register_device
   class KeyboardTeleopDevice(TeleopDeviceBase):
       name = "keyboard"

       def get_teleop_device_cfg(self, embodiment=None):
           return DevicesCfg(devices={"keyboard": Se3KeyboardCfg(...)})

Devices are automatically discovered through decorator-based registration and provide Isaac Lab-compatible configurations for seamless integration.

Teleop Devices in Detail
-------------------------

**Available Devices**
   Three primary input modalities for different use cases:

   - **Keyboard**: WASD-style SE3 manipulation with configurable sensitivity parameters
   - **SpaceMouse**: 6DOF precise spatial control for manipulation tasks
   - **Hand Tracking**: OpenXR-based hand tracking with GR1T2 retargeting for humanoid control

**Configuration System**
   Each device generates Isaac Lab ``DevicesCfg`` objects:

   - **Device-Specific Parameters**: Sensitivity settings, joint mappings, visualization options
   - **Embodiment Integration**: Optional embodiment parameter for robot-specific customization
   - **Automatic Configuration**: Seamless integration with Isaac Lab device factory system
   - **Callback Support**: Automatic binding of device inputs to robot actions

**Registration and Discovery**
   Decorator-based system for automatic device management:

   - **@register_device**: Automatic registration during module import
   - **Device Registry**: Central discovery mechanism for available devices
   - **CLI Integration**: Command-line device selection and fallback handling
   - **Runtime Creation**: Dynamic device instantiation based on environment requirements

Environment Integration
-----------------------

.. code-block:: python

   # Device selection during environment creation
   teleop_device = None
   if args_cli.teleop_device is not None:
       teleop_device = device_registry.get_device_by_name(args_cli.teleop_device)()

   # Environment composition with teleop support
   environment = IsaacArenaEnvironment(
       name="manipulation_task",
       embodiment=embodiment,
       scene=scene,
       task=task,
       teleop_device=teleop_device  # Optional human control interface
   )

   # Automatic device configuration and integration
   env = env_builder.make_registered()  # Handles device setup internally

Usage Examples
--------------

**Keyboard Teleoperation**

.. code-block:: bash

   # Basic keyboard control
   python isaac_arena/scripts/teleop.py --teleop_device keyboard kitchen_pick_and_place

**SpaceMouse Control**

.. code-block:: bash

   # Precise manipulation with SpaceMouse
   python isaac_arena/scripts/teleop.py --teleop_device spacemouse kitchen_pick_and_place --sensitivity 2.0

**Hand Tracking**

.. code-block:: bash

   # VR hand tracking for humanoid control
   python isaac_arena/scripts/teleop.py --teleop_device avp_handtracking gr1_open_microwave

**Environment Integration**

.. code-block:: python

   # Programmatic teleop device usage
   keyboard_device = device_registry.get_device_by_name("keyboard")()

   environment = IsaacArenaEnvironment(
       embodiment=franka_embodiment,
       scene=kitchen_scene,
       task=pick_and_place_task,
       teleop_device=keyboard_device
   )

The teleop device system provides consistent human input interfaces across different robot embodiments and tasks, enabling demonstration collection, manual control, and interactive debugging through a unified abstraction layer.
