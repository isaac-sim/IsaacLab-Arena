Environment Setup and Validation
--------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

On this page we briefly describe the environment used in this example workflow
and validate that we can load it in Isaac Lab.

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^

The environment factory implements ``build(cfg)``, using its typed
``ArenaEnvironmentCfg`` subclass to configure the scene, embodiment, and task.


.. dropdown:: The GR1 Sequential Pick & Place and Close Door Environment
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/gr1_put_and_close_door_environment.py
      :language: python
      :start-at: from __future__ import annotations


Step-by-Step Breakdown
^^^^^^^^^^^^^^^^^^^^^^^

**1. Interact with the Asset and Device Registry**

.. code-block:: python

    camera_offset = Pose(position_xyz=(0.12515, 0.0, 0.06776), rotation_xyzw=(0.11204, -0.17712, -0.79108, 0.57469))
    embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
    embodiment.camera_config.robot_pov_cam.offset = CameraCfg.OffsetCfg(pos=camera_offset.position_xyz, rot=camera_offset.rotation_xyzw, convention="opengl")
    kitchen_background = self.asset_registry.get_asset_by_name(cfg.kitchen_background)()
    kitchen_prim_path = f"{{ENV_REGEX_NS}}/{kitchen_background.name}"
    kitchen_counter_top = ObjectReference(
        name="kitchen_counter_top",
        prim_path=f"{kitchen_prim_path}/counter_right_main_group/top_geometry",
        parent_asset=kitchen_background,
    )
    kitchen_counter_top.add_relation(IsAnchor())

    light = self.asset_registry.get_asset_by_name("light")()

    if cfg.teleop_device is not None:
        teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
    else:
        teleop_device = None

Here, we're selecting the components needed for our sequential static manipulation task:
The GR1 embodiment, the kitchen environment as our background,
and a light to illuminate the scene.
The ``AssetRegistry`` and ``DeviceRegistry`` have been initialized in the ``ArenaEnvironmentFactory`` class.
See :doc:`../../concepts/scene/concept_assets_design` for details on asset architecture.


**2. Position the Embodiment and Objects**

.. code-block:: python

    # Set initial poses
    embodiment.set_initial_pose(
        Pose(
            position_xyz=(3.943, -1.0, 0.995),
            rotation_xyzw=(0.0, 0.0, 0.7071068, 0.7071068),
        )
    )

    # ...

    if cfg.object_set is not None and len(cfg.object_set) > 0:
        objects = [self.asset_registry.get_asset_by_name(obj)() for obj in cfg.object_set]
        pickup_object = RigidObjectSet(name="object_set", objects=objects)
    else:
        pickup_object = self.asset_registry.get_asset_by_name(cfg.object)()

    pickup_object.add_relation(On(kitchen_counter_top))
    pickup_object.add_relation(AtPosition(x=4.05, y=-0.58))
    # Consider changing to other values for different objects, below is for ranch dressing bottle.
    yaw_rad = math.radians(-111.55)
    pickup_object.add_relation(RotateAroundSolution(yaw_rad=yaw_rad))
    pickup_object.add_relation(
        RandomAroundSolution(x_half_m=RANDOMIZATION_HALF_RANGE_X_M, y_half_m=RANDOMIZATION_HALF_RANGE_Y_M)
    )

Before we create the scene, we need to place our embodiment and objects in the right locations.
The embodiment is placed in a fixed spot while the object is placed on top of the kitchen counter using
the relational object placement APIs. A nonempty ``cfg.object_set`` selects a ``RigidObjectSet``;
otherwise, ``cfg.object`` selects a single object. Both use the same placement relations and
randomization range to add variability to the task.


**3. Create the Sequential Pick & Place and Close Door Task**

.. code-block:: python

    # Create pick and place task
    pick_and_place_task = PickAndPlaceTask(
        pick_up_object=pickup_object,
        destination_object=refrigerator,
        destination_location=refrigerator_shelf,
        background_scene=kitchen_background,
    )

    # Create close door task
    close_door_task = CloseDoorTask(
        openable_object=refrigerator,
        closedness_threshold=0.10,
        reset_openness=0.5,
    )

    # Create sequential task
    sequential_task = PutAndCloseDoorTask(subtasks=[pick_and_place_task, close_door_task], episode_length_s=10.0)

The sequential task is composed of two atomic subtasks: the pick and place task and the close door task.
See :doc:`../../concepts/task/index` for task creation details.


**4. Compose the Scene**

.. code-block:: python

    scene = Scene(
        assets=[kitchen_background, kitchen_counter_top, pickup_object, light, refrigerator, refrigerator_shelf]
    )

Now we bring everything together into an IsaacLab-Arena scene.
See :doc:`../../concepts/scene/index` for scene composition details.


**5. Create the IsaacLab Arena Environment**

.. code-block:: python

   from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import set_control_rate_50hz

   isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name=self.name,
        embodiment=embodiment,
        scene=scene,
        task=sequential_task,
        teleop_device=teleop_device,
        env_cfg_callback=set_control_rate_50hz,
    )

Finally, we assemble all the pieces into a complete, runnable environment. The ``IsaacLabArenaEnvironment`` is the
top-level container that connects the embodiment (the robot), the scene (the world), and the task (the objective).
See :doc:`../../concepts/environment/index` for environment composition details.


Step 1: Download a Test Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run a robot in the environment we need some recorded demonstration data that
can be fed to the robot to control its actions.

   .. code-block:: bash

      _tmp="$DATASET_DIR/_hf_download" && \
      hf download \
         nvidia/Arena-GR1-Manipulation-PlaceItemCloseDoor-Task \
         ranch_bottle_into_fridge/ranch_bottle_into_fridge_annotated.hdf5 \
         --repo-type dataset \
         --revision arena_v0.2_lab_v3.0 \
         --local-dir "$_tmp" && \
      mkdir -p "$DATASET_DIR" && \
      mv "$_tmp/ranch_bottle_into_fridge/ranch_bottle_into_fridge_annotated.hdf5" "$DATASET_DIR/" && \
      rm -rf "$_tmp"


Step 2: Validate the Environment by Replaying the Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replay the downloaded dataset to verify the environment setup:

.. code-block:: bash

   python submodules/IsaacLab/scripts/tools/replay_demos.py \
     --viz kit \
     --device cpu \
     --enable_cameras \
     --dataset_file "${DATASET_DIR}/ranch_bottle_into_fridge_annotated.hdf5" \
     --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
     --task put_item_in_fridge_and_close_door \
     --object ranch_dressing_hope_robolab \
     --embodiment gr1_pink

You should see the GR1 robot replaying the demonstrations, performing the sequential
pick & place and close door task in the kitchen environment.

.. figure:: ../../../images/gr1_sequential_static_manipulation_env.gif
   :width: 100%
   :alt: GR1 picking up and placing an object in a refrigerator and closing the door
   :align: center

   IsaacLab Arena GR1 picking up and placing an object in a refrigerator and closing the door
