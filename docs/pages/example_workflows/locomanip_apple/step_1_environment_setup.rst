Environment Setup and Validation
--------------------------------


On this page we briefly describe the environment used in this example workflow
and validate that we can load it in Isaac Lab.

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^

The apple-to-plate workflow reuses the ``GalileoG1LocomanipPickAndPlaceEnvironment`` (also used by the
box-to-bin workflow) with a different pick-up object and destination. The environment exposes
``--object`` and ``--destination`` CLI arguments, so switching between the two variants is just a
matter of passing the right asset names — no separate environment class is needed.

.. dropdown:: The Galileo G1 Loco-Manipulation Pick-and-Place Environment
   :animate: fade-in

   .. code-block:: python

       import math

       class GalileoG1LocomanipPickAndPlaceEnvironment(ExampleEnvironmentBase):

           name: str = "galileo_g1_locomanip_pick_and_place"

           def get_env(self, args_cli: argparse.Namespace):
               from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
               from isaaclab_arena.scene.scene import Scene
               from isaaclab_arena.tasks.locomanip_pick_and_place_task import LocomanipPickAndPlaceTask
               from isaaclab_arena.utils.pose import Pose, PoseRange

               background = self.asset_registry.get_asset_by_name("galileo_locomanip")()
               pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
               destination = self.asset_registry.get_asset_by_name(args_cli.destination)()
               embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

               if args_cli.teleop_device is not None:
                   teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
               else:
                   teleop_device = None

               XY_RANGE_M = 0.025
               pick_up_object.set_initial_pose(
                   PoseRange(
                       position_xyz_min=(0.5785 - XY_RANGE_M, 0.18 - XY_RANGE_M, 0.0707),
                       position_xyz_max=(0.5785 + XY_RANGE_M, 0.18 + XY_RANGE_M, 0.0707),
                       rpy_min=(math.pi, 0.0, math.pi),
                       rpy_max=(math.pi, 0.0, math.pi),
                   )
               )
               destination.set_initial_pose(
                   Pose(
                       position_xyz=(-0.2450, -1.6272, -0.2641),
                       rotation_xyzw=(0.0, 0.0, 1.0, 0.0),
                   )
               )
               embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.18, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

               scene = Scene(assets=[background, pick_up_object, destination])
               task = LocomanipPickAndPlaceTask(pick_up_object, destination, background)

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
    destination = self.asset_registry.get_asset_by_name(args_cli.destination)()
    embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

    if args_cli.teleop_device is not None:
        teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
    else:
        teleop_device = None

The ``teleop_device`` is left as ``None`` when ``--teleop_device`` is not supplied so that scripts like
``replay_demos.py`` and ``policy_runner.py`` (which don't need a teleop device) can launch the same
environment without instantiating an XR runtime.

Here, we're selecting the specific pieces we need for our loco-manipulation task: the Galileo arena as
our background environment, an apple to pick up, a clay plate as our destination, and the G1 embodiment.
Each registered asset carries its own default scale — ``Apple01ObjaverseRobolab`` bakes in the
``(0.01, 0.01, 0.01)`` factor that accounts for Objaverse assets being authored at ~100× real-world
size, while the HOT3D plate is authored at real size (~30 cm diameter) and needs no rescaling.
See :doc:`../../concepts/scene/concept_assets_design` for details on asset architecture.


**2. Position the Objects**

.. code-block:: python

   XY_RANGE_M = 0.025
   pick_up_object.set_initial_pose(
       PoseRange(
           position_xyz_min=(0.5785 - XY_RANGE_M, 0.18 - XY_RANGE_M, 0.0707),
           position_xyz_max=(0.5785 + XY_RANGE_M, 0.18 + XY_RANGE_M, 0.0707),
           rpy_min=(math.pi, 0.0, math.pi),
           rpy_max=(math.pi, 0.0, math.pi),
       )
   )
   destination.set_initial_pose(
       Pose(
           position_xyz=(-0.2450, -1.6272, -0.2641),
           rotation_xyzw=(0.0, 0.0, 1.0, 0.0),
       )
   )

The apple is placed on the shelf at a randomised pose within a small XY range, and the plate is placed on the
table to the right of the shelf. The poses were carried over from the box/bin variant and validated visually
in the simulator; if you swap in a different pick-up object or destination, you may need to tune the
z-heights so the object sits flush on the shelf and the destination sits flat on the table.


**3. Compose the Scene**

.. code-block:: python

    scene = Scene(assets=[background, pick_up_object, destination])

Now we bring everything together into an IsaacLab-Arena scene.
See :doc:`../../concepts/scene/index` for scene composition details.


**4. Create the Loco-Manipulation Pick and Place Task**

.. code-block:: python

    task = LocomanipPickAndPlaceTask(pick_up_object, destination, background)

We reuse the same ``LocomanipPickAndPlaceTask`` as the box/bin workflow. The task takes the pick-up object and
destination as constructor arguments, so swapping the apple and plate in for the box and bin works without
any task-class changes. Internally, the task passes both ``pick_up_object.name`` and
``destination_location.name`` through to the Isaac Lab Mimic configuration so that data generation references
the correct object frame and resolves a unique ``datagen_config.name`` per ``(object, destination)`` pair.

See :doc:`../../concepts/task/index` for task creation details.


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
See :doc:`../../concepts/concept_overview` for environment composition details.


Step 1: Validate Environment with Automated Tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A dedicated pytest module covers the apple-to-plate environment. It asserts that (i) the task does not
terminate when the apple is at its initial pose, (ii) the success termination fires when the apple is
teleported onto the plate, (iii) the Isaac Lab Mimic config correctly references both the apple object
and plate destination across all subtask configurations and its datagen key, (iv) the legacy
``(brown_box, blue_sorting_bin)`` pair still resolves to the preserved ``"locomanip_pick_and_place_D0"``
datagen name, and (v) any non-legacy ``brown_box`` pair (e.g. ``brown_box`` + plate) is routed to a
distinct templated datagen name so parallel Mimic runs cannot overwrite the box-to-bin dataset.

.. code-block:: bash

   python -m pytest isaaclab_arena/tests/test_g1_locomanip_apple_to_plate.py -v

You should see all five tests pass. The simulation-based tests
(``test_initial_state_not_terminated`` and ``test_apple_on_plate_succeeds``) run headless with cameras
enabled and take around a minute each; the three Mimic config tests are lightweight and complete in a
few seconds. This is the fastest way to confirm the scene, task, and Mimic config are wired up
correctly without requiring teleoperation hardware.


Step 2: Validate Environment with Demo Replay (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   **First-run asset cache**: The Objaverse apple USD streams from an S3 staging bucket and
   PhysX re-cooks its collision mesh at the baked-in ``0.01x`` scale. On a fresh machine the
   collision cook may land a frame after the first physics step, which can cause the apple to
   tunnel through the shelf surface on the very first load. Subsequent runs read the cached
   USD from ``/tmp/Assets/`` and spawn correctly. If you see the apple fall through the shelf,
   just re-run the command once.

Once a pre-recorded dataset is published for this workflow on Hugging Face, you can also validate the
environment by replaying the dataset. The replay runs the environment with the recorded actions and no
teleoperation device is needed, so this is a handy way to visually confirm the environment looks right
without an XR headset:

.. code-block:: bash

   python isaaclab_arena/scripts/imitation_learning/replay_demos.py \
     --viz kit \
     --device cpu \
     --enable_cameras \
     --dataset_file ${DATASET_DIR}/arena_g1_locomanip_apple_dataset_generated.hdf5 \
     galileo_g1_locomanip_pick_and_place \
     --object apple_01_objaverse_robolab \
     --destination clay_plates_hot3d_robolab \
     --embodiment g1_wbc_pink

.. note::

   The downloaded dataset will be generated using CPU device physics, so the replay uses ``--device cpu``
   to ensure reproducibility.

You should see the G1 robot replaying the generated demonstrations, performing the apple pick and place
task in the Galileo lab environment.
