Environment Setup and Validation
--------------------------------


On this page we briefly describe the environment used in this example workflow
and validate that we can load it in Isaac Lab.

:runtime_setup_tabs:


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: The Galileo G1 Locomanip Pick and Place Environment
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/galileo_g1_locomanip_pick_and_place_environment.py
      :language: python
      :start-at: @dataclass
      :end-at: return isaaclab_arena_environment

See the `complete source <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/galileo_g1_locomanip_pick_and_place_environment.py>`_.


Step-by-Step Breakdown
^^^^^^^^^^^^^^^^^^^^^^^

**1. Interact with the Asset and Device Registry**

.. literalinclude:: ../../../../isaaclab_arena_environments/galileo_g1_locomanip_pick_and_place_environment.py
   :language: python
   :start-at: background = self.asset_registry.get_asset_by_name("galileo_locomanip")()
   :end-before: XY_RANGE_M = 0.025
   :dedent: 8

Here, we're selecting the specific pieces we need for our locomanipulation task: the Galileo arena as our background environment,
an object to pick up, a destination (a blue sorting bin by default) as our goal location, and the G1 embodiment.
The ``AssetRegistry`` and ``DeviceRegistry`` have been initialized in the ``ArenaEnvironmentFactory`` class.
See :doc:`../../concepts/scene/concept_assets_design` for details on asset architecture.


**2. Position the Objects**

.. literalinclude:: ../../../../isaaclab_arena_environments/galileo_g1_locomanip_pick_and_place_environment.py
   :language: python
   :start-at: XY_RANGE_M = 0.025
   :end-at: embodiment.set_initial_pose(
   :dedent: 8

Before we create the scene, we need to place our objects and the robot in the right locations. These initial poses are
currently set manually to create an achievable task.


**3. Compose the Scene**

.. literalinclude:: ../../../../isaaclab_arena_environments/galileo_g1_locomanip_pick_and_place_environment.py
   :language: python
   :start-at: scene = Scene(
   :end-at: scene = Scene(
   :dedent: 8

Now we bring everything together into a IsaacLab-Arena scene.
See :doc:`../../concepts/scene/index` for scene composition details.

**4. Create the Pick and Place Task**

.. literalinclude:: ../../../../isaaclab_arena_environments/galileo_g1_locomanip_pick_and_place_environment.py
   :language: python
   :start-at: def _build_g1_pick_and_place_mimic_cfg(arm_mode):
   :end-before: scene = Scene(
   :dedent: 8

.. literalinclude:: ../../../../isaaclab_arena_environments/galileo_g1_locomanip_pick_and_place_environment.py
   :language: python
   :start-at: task=PickAndPlaceTask(
   :end-at: ),
   :dedent: 12

The task is passed to the ``IsaacLabArenaEnvironment`` in step 5.
``PickAndPlaceTask`` encapsulates the task's goal of the environment: pick up the specified object
and place it in the destination. The locomanip-specific Mimic config (3-subtask arm sequences
plus a 4-phase nav body) is injected via ``mimic_env_cfg_factory`` rather than via a task subclass,
because the cfg shape diverges from the parent ``PickPlaceMimicEnvCfg`` defaults.

The task knows about the key objects involved (what to pick, where to place it, and the environment context) and will
provide rewards, compute observations, and determine when the episode should end.
See :doc:`../../concepts/task/index` for task creation details.

**5. Create the IsaacLab Arena Environment**

.. literalinclude:: ../../../../isaaclab_arena_environments/galileo_g1_locomanip_pick_and_place_environment.py
   :language: python
   :start-at: isaaclab_arena_environment = IsaacLabArenaEnvironment(
   :end-before: return isaaclab_arena_environment
   :dedent: 8

Finally, we assemble all the pieces into a complete, runnable environment. The ``IsaacLabArenaEnvironment`` is the
top-level container that connects your embodiment (the robot), the scene (the world) and the task (the objective).
It also passes the task description and ``env_cfg_callback``, which sets the 50 Hz control rate that the
whole-body controller needs.
See :doc:`../../concepts/environment/index` for environment composition details.


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
       --revision arena_v0.2_lab_v3.0 \
       --local-dir $DATASET_DIR


Step 2: Validate Environment with Demo Replay
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replay the downloaded dataset to verify the environment setup

.. code-block:: bash

   python submodules/IsaacLab/scripts/tools/replay_demos.py \
     --viz kit \
     --device cpu \
     --enable_cameras \
     --dataset_file ${DATASET_DIR}/arena_g1_loco_manipulation_dataset_generated_small.hdf5 \
     --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
     --task galileo_g1_locomanip_pick_and_place \
     --object brown_box \
     --embodiment g1_wbc_pink

.. note::

   The downloaded dataset was generated using CPU device physics, therefore the replay uses ``--device cpu`` to ensure reproducibility.

You should see the G1 robot replaying the generated demonstrations, performing box pick and place task in the Galileo lab environment.


.. figure:: ../../../images/g1_locomanip_pick_and_place_task_view.png
   :width: 100%
   :alt: G1 Locomanip Pick and Place Task View
   :align: center

   IsaacLab Arena G1 Locomanip Pick and Place Task View
