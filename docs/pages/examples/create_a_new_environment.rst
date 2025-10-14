Creating a New Environment
==========================

Learn how to create custom environments using the kitchen pick and place example.

Overview
--------

Isaac Arena environments have five main components:

1. **Background**: The scene/world where the task takes place
2. **Objects**: Interactive objects that can be manipulated
3. **Embodiment**: The robot or agent performing the task
4. **Task**: The objective and success/failure conditions
5. **Teleop Device**: Input device for controlling the agent (optional)

Environment Structure
---------------------

All environments inherit from ``ExampleEnvironmentBase`` and must implement two methods:

.. code-block:: python

   from isaac_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase

   class MyEnvironment(ExampleEnvironmentBase):
       name: str = "my_environment"

       def get_env(self, args_cli: argparse.Namespace):
           # Create and return IsaacArenaEnvironment
           pass

       @staticmethod
       def add_cli_args(parser: argparse.ArgumentParser) -> None:
           # Add command line arguments
           pass

Step-by-Step Implementation
---------------------------

Let's create a kitchen pick and place environment step by step:

1. **Import Required Components**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from isaac_arena.assets.object_base import ObjectType
   from isaac_arena.assets.object_reference import ObjectReference
   from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
   from isaac_arena.scene.scene import Scene
   from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTask
   from isaac_arena.utils.pose import Pose

2. **Get Assets from Registry**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The asset registry provides access to all registered assets:

.. code-block:: python

   def get_env(self, args_cli: argparse.Namespace):
       # Background scene
       background = self.asset_registry.get_asset_by_name("kitchen")()

       # Object to manipulate
       pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()

       # Robot embodiment
       embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
           enable_cameras=args_cli.enable_cameras
       )

**Available Assets:**

- **Backgrounds**: ``kitchen``, ``galileo_lab``
- **Objects**: ``cracker_box``, ``mustard_bottle``, ``sugar_box``, ``tomato_soup_can``, ``microwave``, ``brown_box``, etc.
- **Embodiments**: ``franka``, ``gr1_pink``, ``gr1_joint``, ``g1_wbc_joint``, ``g1_wbc_pink``

3. **Configure Object Positions**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set initial poses for objects in the scene:

.. code-block:: python

   # Position the object to be picked up
   pick_up_object.set_initial_pose(
       Pose(
           position_xyz=(0.4, 0.0, 0.1),
           rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
       )
   )

4. **Define Target Location**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For pick and place tasks, create an object reference for the destination:

.. code-block:: python

   # Create reference to existing scene object as destination
   destination_location = ObjectReference(
       name="destination_location",
       prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
       parent_asset=background,
       object_type=ObjectType.RIGID,
   )

5. **Set Up Teleop Device**
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure teleoperation if specified:

.. code-block:: python

   if args_cli.teleop_device is not None:
       teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
   else:
       teleop_device = None

**Available Teleop Devices**: ``keyboard``, ``spacemouse``, ``avp_handtracking``

6. **Create Scene and Task**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compose the scene and define the task:

.. code-block:: python

   # Create scene with all assets
   scene = Scene(assets=[background, pick_up_object, destination_location])

   # Define pick and place task
   task = PickAndPlaceTask(pick_up_object, destination_location, background)

7. **Build Environment**
^^^^^^^^^^^^^^^^^^^^^^^^

Create the final environment:

.. code-block:: python

   isaac_arena_environment = IsaacArenaEnvironment(
       name=self.name,
       embodiment=embodiment,
       scene=scene,
       task=task,
       teleop_device=teleop_device,
   )
   return isaac_arena_environment

8. **Add CLI Arguments**
^^^^^^^^^^^^^^^^^^^^^^^^

Define command line arguments for customization:

.. code-block:: python

   @staticmethod
   def add_cli_args(parser: argparse.ArgumentParser) -> None:
       parser.add_argument("--object", type=str, default="cracker_box")
       parser.add_argument("--embodiment", type=str, default="franka")
       parser.add_argument("--teleop_device", type=str, default=None)

Available Tasks
---------------

Choose from these pre-built tasks:

- **PickAndPlaceTask**: Move object from one location to another
- **OpenDoorTask**: Open articulated objects (doors, drawers, microwave)
- **G1LocomanipPickAndPlaceTask**: Locomotion + manipulation for humanoids
- **DummyTask**: Empty task for free-form environments

Complete Example
----------------

.. code-block:: python

   import argparse
   from isaac_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase

   class MyKitchenEnvironment(ExampleEnvironmentBase):

       name: str = "my_kitchen_environment"

       def get_env(self, args_cli: argparse.Namespace):
           from isaac_arena.assets.object_base import ObjectType
           from isaac_arena.assets.object_reference import ObjectReference
           from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
           from isaac_arena.scene.scene import Scene
           from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTask
           from isaac_arena.utils.pose import Pose

           # Get assets
           background = self.asset_registry.get_asset_by_name("kitchen")()
           pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
           embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
               enable_cameras=args_cli.enable_cameras
           )

           # Configure teleop
           if args_cli.teleop_device is not None:
               teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
           else:
               teleop_device = None

           # Position object
           pick_up_object.set_initial_pose(
               Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
           )

           # Define destination
           destination_location = ObjectReference(
               name="destination_location",
               prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
               parent_asset=background,
               object_type=ObjectType.RIGID,
           )

           # Create scene and task
           scene = Scene(assets=[background, pick_up_object, destination_location])
           task = PickAndPlaceTask(pick_up_object, destination_location, background)

           # Build environment
           return IsaacArenaEnvironment(
               name=self.name,
               embodiment=embodiment,
               scene=scene,
               task=task,
               teleop_device=teleop_device,
           )

       @staticmethod
       def add_cli_args(parser: argparse.ArgumentParser) -> None:
           parser.add_argument("--object", type=str, default="cracker_box")
           parser.add_argument("--embodiment", type=str, default="franka")
           parser.add_argument("--teleop_device", type=str, default=None)

Usage Tips
----------

1. **Asset Discovery**: Use ``asset_registry.get_assets_by_tag("tag_name")`` to find assets by category
2. **Pose Coordinates**: Positions are in meters, rotations use quaternions (w,x,y,z)
3. **Object References**: Use ``ObjectReference`` to refer to existing scene geometry as interaction targets
4. **Camera Enable**: Set ``enable_cameras=True`` for embodiments when using vision-based policies
5. **Validation**: Always validate that required assets exist before using them
