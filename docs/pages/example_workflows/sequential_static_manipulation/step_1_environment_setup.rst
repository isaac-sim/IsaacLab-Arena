Environment Setup and Validation
--------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

On this page we briefly describe the environment used in this example workflow
and validate that we can load it in Isaac Lab.

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^


.. dropdown:: The GR1 Sequential Pick & Place and Close Door Environment
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/gr1_put_and_close_door_environment.py
      :language: python
      :start-at: # Pickup-object position
      :end-at: return isaaclab_arena_environment

See the `complete source <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/gr1_put_and_close_door_environment.py>`_.


Step-by-Step Breakdown
^^^^^^^^^^^^^^^^^^^^^^^

**1. Interact with the Asset and Device Registry**

.. literalinclude:: ../../../../isaaclab_arena_environments/gr1_put_and_close_door_environment.py
   :language: python
   :start-at: camera_offset = Pose(
   :end-before: # Set initial poses
   :dedent: 8

Here, we're selecting the components needed for our sequential static manipulation task:
The GR1 embodiment, the kitchen environment as our background, the object to pick and place,
and a light to illuminate the scene.
The ``AssetRegistry`` and ``DeviceRegistry`` have been initialized in the ``ArenaEnvironmentFactory`` class.
See :doc:`../../concepts/scene/concept_assets_design` for details on asset architecture.


**2. Position the Embodiment and Objects**

.. literalinclude:: ../../../../isaaclab_arena_environments/gr1_put_and_close_door_environment.py
   :language: python
   :start-at: # Set initial poses
   :end-at: pickup_object.add_relation(RotateAroundSolution
   :dedent: 8

Before we create the scene, we need to place our embodiment and objects in the right locations.
The embodiment is placed in a fixed spot while the object is placed on top of the kitchen counter using
the relational object placement APIs. ``PositionLimitsBox`` keeps the object within a small region of the
counter that the GR1 can reach, and ``RotateAroundSolution`` sets its yaw.
See :doc:`../../concepts/object_placement/relations` for the available relations.


**3. Compose the Scene**

.. literalinclude:: ../../../../isaaclab_arena_environments/gr1_put_and_close_door_environment.py
   :language: python
   :start-at: scene = Scene(
   :end-before: # Create pick and place task
   :dedent: 8

Now we bring everything together into an IsaacLab-Arena scene.
See :doc:`../../concepts/scene/index` for scene composition details.


**4. Create the Sequential Pick & Place and Close Door Task**

.. literalinclude:: ../../../../isaaclab_arena_environments/gr1_put_and_close_door_environment.py
   :language: python
   :start-at: # Create pick and place task
   :end-at: sequential_task = PutAndCloseDoorTask(
   :dedent: 8

The sequential task is composed of two atomic subtasks: the pick and place task and the close door task.
See :doc:`../../concepts/task/index` for task creation details.


**5. Create the IsaacLab Arena Environment**

.. literalinclude:: ../../../../isaaclab_arena_environments/gr1_put_and_close_door_environment.py
   :language: python
   :start-at: isaaclab_arena_environment = IsaacLabArenaEnvironment(
   :end-before: return isaaclab_arena_environment
   :dedent: 8

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
