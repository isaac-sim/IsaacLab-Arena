Teleop Devices Design
======================

Teleop devices defined in Arena are a thin wrapper around the Isaac Lab teleop devices.
We define this wrapper to allow for easy registration and discovery of teleop devices.

Core Architecture
-----------------

Teleop devices use the ``TeleopDeviceBase`` abstract class with automatic registration:

.. code-block:: python

   class TeleopDeviceBase(ABC):
       name: str | None = None

       @abstractmethod
       def get_device_cfg(self, pipeline_builder=None, embodiment=None):
           """Return an Isaac Lab device config for the specific device."""

   @register_device
   class KeyboardCfg(TeleopDeviceBase):
       name = "keyboard"

       def get_device_cfg(self, pipeline_builder=None, embodiment=None):
           return Se3KeyboardCfg(pos_sensitivity=0.05, rot_sensitivity=0.05)

Devices are automatically discovered through decorator-based registration and provide Isaac Lab-compatible configurations.

For XR teleoperation, the ``OpenXRCfg`` device produces an ``IsaacTeleopCfg`` that
references a **pipeline builder** -- a callable that constructs an ``isaacteleop``
retargeting pipeline graph. This pipeline converts XR tracking data (hand poses,
controller inputs) into robot action tensors.

.. code-block:: python

   @register_device
   class OpenXRCfg(TeleopDeviceBase):
       name = "openxr"

       def get_device_cfg(self, pipeline_builder=None, embodiment=None):
           return IsaacTeleopCfg(
               pipeline_builder=pipeline_builder,
               xr_cfg=embodiment.get_xr_cfg(),
           )

Teleop Devices in Detail
-------------------------

**Available Devices**
   Three primary input modalities for different use cases:

   - **Keyboard**: WASD-style SE3 manipulation with configurable sensitivity parameters
   - **SpaceMouse**: 6DOF precise spatial control for manipulation tasks
   - **XR Hand Tracking**: Isaac Teleop pipeline-based hand tracking for humanoid control,
     using ``isaacteleop`` retargeters (Se3AbsRetargeter, DexHandRetargeter, etc.) to map
     XR hand poses to robot joint commands

**Registration and Discovery**
   Decorator-based system for automatic device management:

   - **@register_device**: Automatic registration during module import
   - **Device Registry**: Central discovery mechanism for available devices
   - **@register_retargeter**: Associates a pipeline builder with a (device, embodiment) pair

Environment Integration
-----------------------

.. code-block:: python

   # Device selection during environment creation
   teleop_device = device_registry.get_device_by_name(args_cli.teleop_device)()

   # Environment composition with teleop support
   environment = IsaacLabArenaEnvironment(
       name="manipulation_task",
       embodiment=embodiment,
       scene=scene,
       task=task,
       teleop_device=teleop_device  # Optional human control interface
   )

   # Automatic device configuration and integration
   env = env_builder.make_registered()  # Handles device setup internally

For XR devices, the environment builder sets ``isaac_teleop`` on the env config
(an ``IsaacTeleopCfg``). For keyboard/spacemouse devices, standard Isaac Lab
device configs are used.

Usage Examples
--------------

**Keyboard Teleoperation**

.. code-block:: bash

   # Basic keyboard control
   python isaaclab_arena/scripts/imitation_learning/teleop.py --teleop_device keyboard kitchen_pick_and_place

**SpaceMouse Control**

.. code-block:: bash

   # Precise manipulation with SpaceMouse
   python isaaclab_arena/scripts/imitation_learning/teleop.py --teleop_device spacemouse kitchen_pick_and_place --sensitivity 2.0

**Hand Tracking**

.. code-block:: bash

   # XR hand tracking for humanoid control (requires CloudXR runtime via Isaac Teleop)
   python isaaclab_arena/scripts/imitation_learning/teleop.py --teleop_device avp_handtracking gr1_open_microwave
