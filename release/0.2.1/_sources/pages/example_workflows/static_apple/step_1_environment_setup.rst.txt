Environment Setup and Validation
--------------------------------


On this page we briefly describe the environment used in this example workflow
and validate that we can load it in Isaac Lab.

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^

The static apple-to-plate workflow ships its own ``GalileoG1StaticPickAndPlaceEnvironment`` registered
under ``galileo_g1_static_pick_and_place``. It reuses the same ``galileo_locomanip`` background USD
and the same OpenXR retargeter as the loco-manipulation apple-to-plate environment, but defaults to
the ``g1_wbc_agile_pink`` embodiment (AGILE end-to-end velocity policy) instead of ``g1_wbc_pink``
(HOMIE stand+walk pair). The static task never walks, so AGILE's single-policy backend is a better
fit than HOMIE's stand/walk model split — both share the same 23-D action layout and OpenXR
retargeter pipeline, so only the lower-body ONNX backend changes. Both the apple and the destination
plate are placed on the *same* shelf so the robot never needs to drive its base anywhere — WBC just
holds the standing pose. The ``g1_wbc_pink`` embodiment is still accepted as an override for users
who specifically want HOMIE.

.. note::

   **Recording vs. evaluation embodiment.** Teleoperation in :doc:`step_2_teleoperation` uses the
   default ``g1_wbc_agile_pink`` (PinkIK + AGILE), while closed-loop policy evaluation in
   :doc:`step_4_evaluation` uses ``g1_wbc_agile_joint`` (direct joint control + AGILE). Both share
   the same AGILE lower-body backend; the ``_joint`` twin just bypasses PinkIK at inference because
   the policy is trained on the joint-space targets that PinkIK *produced* during recording. This
   matters for finetuning: see :doc:`step_3_policy_training` for the rationale.

.. dropdown:: The Galileo G1 Static Pick-and-Place Environment
   :animate: fade-in

   .. code-block:: python

       from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

       SHELF_SURFACE_Z = -0.030
       SHELF_AIRGAP = 0.005
       PICK_UP_OBJECT_SPAWN_XY = (0.5785, 0.18)
       DESTINATION_SPAWN_XY = (0.5785, -0.06)

       _USD_ORIGIN_ABOVE_BOTTOM_M = {
           "apple_01_objaverse_robolab": 0.019,
           "clay_plates_hot3d_robolab": 0.0,
       }

       TUNED_PICK_UP_OBJECT_NAME = "apple_01_objaverse_robolab"
       TUNED_DESTINATION_NAME = "clay_plates_hot3d_robolab"


       class GalileoG1StaticPickAndPlaceEnvironment(ExampleEnvironmentBase):

           name: str = "galileo_g1_static_pick_and_place"

           def get_env(self, args_cli):
               from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
               from isaaclab_arena.scene.scene import Scene
               from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
               from isaaclab_arena.utils.pose import Pose

               background = self.asset_registry.get_asset_by_name("galileo_locomanip")()
               pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
               destination = self.asset_registry.get_asset_by_name(args_cli.destination)()
               embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
                   enable_cameras=args_cli.enable_cameras
               )

               teleop_device = (
                   self.device_registry.get_device_by_name(args_cli.teleop_device)()
                   if args_cli.teleop_device is not None else None
               )

               embodiment.set_initial_pose(
                   Pose(position_xyz=(0.0, 0.18, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
               )
               px, py = PICK_UP_OBJECT_SPAWN_XY
               dx, dy = DESTINATION_SPAWN_XY
               pick_up_object.set_initial_pose(
                   Pose(position_xyz=(px, py, _shelf_spawn_z(args_cli.object)),
                        rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
               )
               destination.set_initial_pose(
                   Pose(position_xyz=(dx, dy, _shelf_spawn_z(args_cli.destination)),
                        rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
               )

               scene = Scene(assets=[background, pick_up_object, destination])
               return IsaacLabArenaEnvironment(
                   name=self.name,
                   embodiment=embodiment,
                   scene=scene,
                   task=PickAndPlaceTask(
                       pick_up_object=pick_up_object,
                       destination_location=destination,
                       background_scene=background,
                       force_threshold=0.5,
                       velocity_threshold=0.1,
                   ),
                   teleop_device=teleop_device,
               )


Step-by-Step Breakdown
^^^^^^^^^^^^^^^^^^^^^^^

**1. Interact with the Asset and Device Registry**

.. code-block:: python

    background = self.asset_registry.get_asset_by_name("galileo_locomanip")()
    pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
    destination = self.asset_registry.get_asset_by_name(args_cli.destination)()
    embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
        enable_cameras=args_cli.enable_cameras
    )

The static workflow shares the ``galileo_locomanip`` background with the loco-manipulation variant
on purpose: the lighting, shelf geometry and 23-D action layout are already tuned, so the only
thing that changes is *where* the destination sits and which lower-body WBC backend balances the
robot. The default ``g1_wbc_agile_pink`` embodiment swaps HOMIE's stand+walk pair for AGILE's
single end-to-end velocity policy, which is a better fit for a no-locomotion task; ``g1_wbc_pink`` is
accepted as an override for users who want HOMIE behaviour. ``teleop_device`` is left as ``None``
when ``--teleop_device`` is not supplied so that scripts like ``replay_demos.py`` and
``policy_runner.py`` can launch the same environment without instantiating an XR runtime.


**2. Position the Objects**

.. code-block:: python

   embodiment.set_initial_pose(
       Pose(position_xyz=(0.0, 0.18, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
   )
   pick_up_object.set_initial_pose(
       Pose(position_xyz=(0.5785, 0.18, _shelf_spawn_z(args_cli.object)),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
   )
   destination.set_initial_pose(
       Pose(position_xyz=(0.5785, -0.06, _shelf_spawn_z(args_cli.destination)),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
   )

The robot pose mirrors the loco-manipulation env exactly so the WBC controller stands the robot up
in the same shelf-relative spot. The apple is placed at a measured on-shelf X/Y, and the plate is
offset 24 cm in -Y so its 30 cm footprint clears the apple without collision. ``_shelf_spawn_z``
consults a per-asset USD-origin table so each asset's *bottom face* — not its USD origin — lands
flush on the shelf surface; assets not in the table fall back to ``SHELF_SURFACE_Z + SHELF_AIRGAP``
and emit a warning so you know to verify the spawn pose visually before recording demonstrations.


**3. Compose the Scene**

.. code-block:: python

    scene = Scene(assets=[background, pick_up_object, destination])

The static env uses a single shelf-anchored background plus the pick-up object and the destination;
no second table is needed because the destination plate sits on the same shelf as the apple.
See :doc:`../../concepts/scene/index` for scene composition details.


**4. Create the Pick and Place Task**

.. code-block:: python

    task = PickAndPlaceTask(
        pick_up_object=pick_up_object,
        destination_location=destination,
        background_scene=background,
        force_threshold=0.5,
        velocity_threshold=0.1,
    )

The static env uses ``PickAndPlaceTask`` directly. The task wires the apple,
plate, and background into the standard pick-and-place termination logic;
``force_threshold`` / ``velocity_threshold`` mirror the locomanip env so success
metrics are directly comparable.

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

Finally, we assemble all the pieces into a complete, runnable environment.
See :doc:`../../concepts/concept_overview` for environment composition details.


Validate Environment with Automated Tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A dedicated pytest module covers the static apple-to-plate environment. It asserts that
(i) the task does not terminate when the apple is at its initial pose, and (ii) the success
termination fires after the apple is teleported above the plate and allowed to settle.

.. code-block:: bash

   python -m pytest isaaclab_arena/tests/test_g1_static_pick_and_place.py -v

You should see both tests pass. They run headless with cameras enabled and take around a
minute each. This is the fastest way to confirm the scene, task, and termination logic are
wired up correctly without requiring teleoperation hardware.

.. note::

   **First-run asset cache**: The Objaverse apple USD streams from an S3 staging bucket and PhysX
   re-cooks its collision mesh at the baked-in ``0.01x`` scale. On a fresh machine the collision cook
   may land a frame after the first physics step, which can cause the apple to tunnel through the
   shelf surface on the very first load. Subsequent runs read the cached USD from ``/tmp/Assets/``
   and spawn correctly. If you see the apple fall through the shelf, just re-run the command once.
