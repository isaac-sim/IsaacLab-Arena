Environment Setup and Validation
--------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

On this page we briefly describe the environment used in this example workflow
and validate that we can load it in Isaac Lab.


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: The Galileo G1 Locomanip Pick and Place Environment
   :animate: fade-in

   .. code-block:: python

       class GalileoG1LocomanipPickAndPlaceEnvironment(ExampleEnvironmentBase):

           name: str = "galileo_g1_locomanip_pick_and_place"

           def get_env(self, args_cli: argparse.Namespace):
               from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
               from isaaclab_arena.scene.scene import Scene
               from isaaclab_arena.tasks.g1_locomanip_pick_and_place_task import G1LocomanipPickAndPlaceTask
               from isaaclab_arena.utils.pose import Pose

               background = self.asset_registry.get_asset_by_name("galileo_locomanip")()
               pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
               blue_sorting_bin = self.asset_registry.get_asset_by_name("blue_sorting_bin")()
               embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

               teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()

               pick_up_object.set_initial_pose(
                   Pose(
                       position_xyz=(0.5785, 0.18, 0.0707),
                       rotation_wxyz=(0.0, 0.0, 1.0, 0.0),
                   )
               )
               blue_sorting_bin.set_initial_pose(
                   Pose(
                       position_xyz=(-0.2450, -1.6272, -0.2641),
                       rotation_wxyz=(0.0, 0.0, 0.0, 1.0),
                   )
               )
               embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.18, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

               scene = Scene(assets=[background, pick_up_object, blue_sorting_bin])
               task = G1LocomanipPickAndPlaceTask(pick_up_object, blue_sorting_bin, background),

               isaaclab_arena_environment = IsaacLabArenaEnvironment(
                   name=self.name,
                   embodiment=embodiment,
                   scene=scene,
                   task=task,
                   teleop_device=teleop_device,
               )
               return isaaclab_arena_environment


Step-by-Step Breakdown
^^^^^^^^^^^^^^^^^^^^^^^

**1. Interact with the Asset and Device Registry**

.. code-block:: python

    background = self.asset_registry.get_asset_by_name("galileo_locomanip")()
    pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
    blue_sorting_bin = self.asset_registry.get_asset_by_name("blue_sorting_bin")()
    embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

    teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()

Here, we're selecting the specific pieces we need for our locomanipulation task: the Galileo arena as our background environment,
an object to pick up, a blue sorting bin as our goal location, and the G1 embodiment.
The ``AssetRegistry`` and ``DeviceRegistry`` have been initialized in the ``ExampleEnvironmentBase`` class.
See :doc:`../../concepts/concept_assets_design` for details on asset architecture.


**2. Position the Objects**

.. code-block:: python

   pick_up_object.set_initial_pose(
       Pose(
           position_xyz=(0.5785, 0.18, 0.0707),
           rotation_wxyz=(0.0, 0.0, 1.0, 0.0),
       )
   )
   blue_sorting_bin.set_initial_pose(
       Pose(
           position_xyz=(-0.2450, -1.6272, -0.2641),
           rotation_wxyz=(0.0, 0.0, 0.0, 1.0),
       )
   )

Before we create the scene, we need to place our objects in the right locations. These initial poses are
currently set manually to create an achievable task.


**3. Compose the Scene**

.. code-block:: python

    scene = Scene(assets=[background, pick_up_object, blue_sorting_bin])

Now we bring everything together into a IsaacLab-Arena scene.
See :doc:`../../concepts/concept_scene_design` for scene composition details.

**4. Create the Locomanip Pick and Place Task**

.. code-block:: python

    task = G1LocomanipPickAndPlaceTask(pick_up_object, blue_sorting_bin, background),

The ``G1LocomanipPickAndPlaceTask`` encapsulates the task's goal of the
environment: pick up the specified object and place it in the blue sorting bin.

The task knows about the key objects involved (what to pick, where to place it, and the environment context) and will
provide rewards, compute observations, and determine when the episode should end.
See :doc:`../../concepts/concept_tasks_design` for task creation details.

**5. Create the IsaacLab Arena Environment**

.. code-block:: python

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name=self.name,
        embodiment=embodiment,
        scene=scene,
        task=task,
        teleop_device=teleop_device,
    )

Finally, we assemble all the pieces into a complete, runnable environment. The ``IsaacLabArenaEnvironment`` is the
top-level container that connects your embodiment (the robot), the scene (the world) and the task (the objective).
See :doc:`../../concepts/concept_environment_design` for environment composition details.


Step 1: Download a test dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run a robot in the environment we need some recorded demonstration data that
can be fed to the robot to control its actions.
We download a pre-recorded dataset from Hugging Face:

.. code-block:: bash

   hf download \
       nvidia/Arena-G1-Loco-Manipulation-Task \
       arena_g1_loco_manipulation_dataset_generated_small.hdf5 \
       --repo-type dataset \
       --local-dir $DATASET_DIR


Step 2: Validate Environment with Demo Replay
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replay the downloaded dataset to verify the environment setup

.. code-block:: bash

   python isaaclab_arena/scripts/replay_demos.py \
     --device cpu \
     --enable_cameras \
     --dataset_file <INPUT_DATASET_PATH> \
     galileo_g1_locomanip_pick_and_place \
     --object brown_box \
     --embodiment g1_wbc_pink

You should see the G1 robot replaying the generated demonstrations, performing box pick and place task in the Galileo lab environment.

.. note::

   The downloaded dataset was generated using CPU device physics, therefore the replay uses ``--device cpu`` to ensure reproducibility.

.. todo:: (amillane, 2025-10-22): screenshot
