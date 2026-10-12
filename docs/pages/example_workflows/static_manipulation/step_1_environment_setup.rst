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


.. dropdown:: The GR1 Open Microwave Environment
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/gr1_open_microwave_environment.py
      :language: python
      :start-at: from __future__ import annotations


Step-by-Step Breakdown
^^^^^^^^^^^^^^^^^^^^^^^

**1. Interact with the Asset and Device Registry**

.. code-block:: python

   background = self.asset_registry.get_asset_by_name("kitchen")()
   microwave = self.asset_registry.get_asset_by_name("microwave")()
   assets = [background, microwave]

   embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
   if cfg.teleop_device is not None:
       teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
   else:
       teleop_device = None

Here, we're selecting the components needed for our static manipulation task: the kitchen environment as our background,
a microwave with an openable door, and the GR1 embodiment (our robot).
The ``AssetRegistry`` and ``DeviceRegistry`` have been initialized in the ``ArenaEnvironmentFactory`` class.
See :doc:`../../concepts/scene/concept_assets_design` for details on asset architecture.

**2. Position the Objects**

.. code-block:: python

   microwave_pose = Pose(
       position_xyz=(0.4, -0.00586, 0.22773),
       rotation_xyzw=(0, 0, -0.7071068, 0.7071068),
   )
   microwave.set_initial_pose(microwave_pose)

Before we create the scene, we need to place our objects in the right locations. These initial poses are
currently set manually to create an achievable task. In this case, we place the microwave on the packing table.


**3. Compose the Scene**

.. code-block:: python

    scene = Scene(assets=assets)

Now we bring everything together into an IsaacLab-Arena scene.
See :doc:`../../concepts/scene/index` for scene composition details.

**4. Create the Open Door Task**

.. code-block:: python

    task = OpenDoorTask(microwave, openness_threshold=0.8, reset_openness=0.2, episode_length_s=5.0)

The ``OpenDoorTask`` encapsulates the goal of this environment: open the microwave door.
See :doc:`../../concepts/task/index` for task creation details.

**5. Create the IsaacLab Arena Environment**

.. code-block:: python

   from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import set_control_rate_50hz

   isaaclab_arena_environment = IsaacLabArenaEnvironment(
       name=self.name,
       embodiment=embodiment,
       scene=scene,
       task=task,
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
We download a pre-recorded dataset from Hugging Face.

.. code-block:: bash

   hf download \
       nvidia/Arena-GR1-Manipulation-Task \
       arena_gr1_manipulation_dataset_generated.hdf5 \
       --repo-type dataset \
       --revision arena_v0.2_lab_v3.0 \
       --local-dir $DATASET_DIR


Step 2: Validate the Environment by Replaying the Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replay the downloaded dataset to verify the environment setup:

.. code-block:: bash

   python submodules/IsaacLab/scripts/tools/replay_demos.py \
     --viz kit \
     --device cpu \
     --enable_cameras \
     --dataset_file "${DATASET_DIR}/arena_gr1_manipulation_dataset_generated.hdf5" \
     --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
     --task gr1_open_microwave \
     --embodiment gr1_pink

You should see the GR1 robot replaying the demonstrations, performing the microwave door
opening task in the kitchen environment.

.. figure:: ../../../images/gr1_open_microwave_task_view.png
   :width: 100%
   :alt: GR1 opening the microwave door
   :align: center

   IsaacLab Arena GR1 opening the microwave door
