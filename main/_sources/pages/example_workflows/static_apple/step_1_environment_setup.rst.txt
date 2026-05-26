Environment Setup and Validation
--------------------------------


On this page we briefly describe the environment used in this example workflow
and validate that we can load it in Isaac Lab.

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^

The static apple-to-plate workflow ships its own ``GalileoG1StaticPickAndPlaceEnvironment``,
exposed by the example-environment registry as ``galileo_g1_static_pick_and_place``. It reuses the
same ``galileo_locomanip`` shelf scene and the same OpenXR retargeter as the loco-manipulation
apple-to-plate environment, but defaults to the ``g1_wbc_agile_pink`` embodiment for recording. The
static task never walks, so AGILE's single-policy lower-body backend is a better fit than HOMIE's
stand/walk model split. Both the apple and destination plate are placed on the same shelf, so the
WBC only needs to hold the standing pose.

.. note::

   **Recording vs. evaluation embodiment.** Teleoperation in :doc:`step_2_teleoperation` uses the
   default ``g1_wbc_agile_pink`` (PinkIK + AGILE), while closed-loop policy evaluation in
   :doc:`step_4_evaluation` uses ``g1_wbc_agile_joint`` (direct joint control + AGILE). Both share
   the same AGILE lower-body backend; the ``_joint`` twin just bypasses PinkIK at inference because
   the policy is trained on the joint-space targets that PinkIK produced during recording. This
   matters for finetuning: see :doc:`step_3_policy_training` for the rationale.


Environment Composition
^^^^^^^^^^^^^^^^^^^^^^^

The full implementation lives in
``isaaclab_arena_environments/galileo_g1_static_pick_and_place_environment.py``. This page keeps the
breakdown at the composition level so the docs stay focused on how an Arena environment is assembled
instead of mirroring every placement constant or scene utility in the source file.

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Component
     - Role in this workflow
   * - Environment name
     - ``galileo_g1_static_pick_and_place``
   * - Background
     - ``galileo_locomanip`` provides the Galileo shelf scene reused from the
       :doc:`loco-manipulation workflow <../locomanipulation/index>`.
   * - Embodiment
     - ``g1_wbc_agile_pink`` is the recording default; ``g1_wbc_agile_joint`` is used for policy
       evaluation after training.
   * - Pick-up object
     - ``apple_01_objaverse_robolab`` by default, selected through ``--object``.
   * - Destination
     - ``clay_plates_hot3d_robolab`` by default, selected through ``--destination``.
   * - Scene utilities
     - The registered environment handles tuned object placement, object scale, shelf support,
       finger contact friction, and background cleanup internally.
   * - Task
     - ``PickAndPlaceTask`` checks whether the apple is moved onto the plate and provides the
       success metrics used later in evaluation.
   * - Teleop device
     - ``openxr`` is attached only when ``--teleop_device openxr`` is passed, so the same
       environment can also be used for replay and policy evaluation without XR.

At a high level, defining an Arena environment is just selecting assets, composing a scene, choosing
a task, and returning an ``IsaacLabArenaEnvironment``:

.. code-block:: python

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

   scene = Scene(assets=[background, pick_up_object, destination])
   task = PickAndPlaceTask(
       pick_up_object=pick_up_object,
       destination_location=destination,
       background_scene=background,
   )

   return IsaacLabArenaEnvironment(
       name=self.name,
       embodiment=embodiment,
       scene=scene,
       task=task,
       teleop_device=teleop_device,
   )

The production environment adds task-specific defaults around this core composition: same-shelf
poses for the apple and plate, a short episode timeout, an invisible shelf support surface, tuned
contact/friction settings, and cleanup of a few reused background props. Those details are handled
inside the registered environment so workflow users can run it through the CLI without editing the
scene setup code.

See :doc:`../../concepts/concept_overview`, :doc:`../../concepts/scene/index`, and
:doc:`../../concepts/task/index` for more details on Arena environment composition.


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
