Teleop Devices
==============

A teleop device is the human input interface for teleoperation — it translates
operator input (keyboard, SpaceMouse, or XR hand tracking) into robot commands
during demonstration collection.

Arena's teleop devices are a thin wrapper around Isaac Lab's built-in devices,
added so they can be registered by name and discovered from the device registry,
the same way assets and embodiments are.

.. code-block:: python

   from isaaclab_arena.assets.register import register_device

   @register_device
   class KeyboardCfg(TeleopDeviceBase):
       name = "keyboard"

       def get_device_cfg(self, pipeline_builder=None, embodiment=None) -> Se3KeyboardCfg:
           return Se3KeyboardCfg(pos_sensitivity=0.05, rot_sensitivity=0.05)

The ``@register_device`` decorator makes the device available by name. The wrapper
implements a single method, ``get_device_cfg()``, that returns the underlying
Isaac Lab device configuration.

To add teleoperation to an environment, pick a device by name and pass it in:

.. code-block:: python

   teleop_device = device_registry.get_device_by_name("keyboard")()

   environment = IsaacLabArenaEnvironment(
       name="kitchen_pick_and_place",
       embodiment=embodiment,
       scene=scene,
       task=task,
       teleop_device=teleop_device,
   )

   env = ArenaEnvBuilder(environment, args_cli).make_registered()

``teleop_device`` is optional — omit it for policy evaluation and include it
when collecting demonstrations.

Available devices
-----------------

**keyboard**
   WASD-style SE3 control with configurable sensitivity. Good for quick setup
   and simple manipulation tasks.

**spacemouse**
   6-DOF spatial control. More precise than the keyboard; suitable for
   fine manipulation.

**openxr**
   XR hand tracking via the ``isaacteleop`` retargeting pipeline. Maps hand
   poses and controller input to robot joint commands. Requires a pipeline
   builder (see below).

   For a full walkthrough — collecting demonstrations on a static manipulation
   task using an Apple Vision Pro or Meta Quest — see
   :doc:`../../example_workflows/static_manipulation/step_2_teleoperation`.

XR and pipeline builders
------------------------

Keyboard and SpaceMouse emit SE3 deltas that Isaac Lab can apply to any robot
with an IK controller — no robot-specific setup needed. XR hand tracking is
different: the headset gives you raw hand and finger poses, and translating those
into joint commands depends on the specific robot. A GR1T2 and a Unitree G1 have
different joint structures, so the mapping must be defined per robot.

That mapping is called a **pipeline builder** — a callable that constructs an
``isaacteleop`` retargeting graph for a specific (device, embodiment) pair.
Arena registers one per pair, so when you pass ``openxr`` and an embodiment to
the environment, the right pipeline is looked up and wired in automatically.
Built-in embodiments that support XR already have a pipeline builder registered.
You only need to provide one if you are adding a new embodiment.
