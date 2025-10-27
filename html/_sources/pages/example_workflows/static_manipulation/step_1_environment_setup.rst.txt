Environment Setup and Validation
--------------------------------

On this page we briefly describe the environment used in this example workflow
and validate that we can load it in Isaac Lab.

Environment Description
^^^^^^^^^^^^^^^^^^^^^^^

The environment used in this example workflow has the following components:

**Embodiment Configuration:**

.. code-block:: python

   from isaaclab_arena.embodiments.gr1t2 import GR1T2PinkEmbodiment

   embodiment = GR1T2PinkEmbodiment(enable_cameras=True)

**Key Features:**

- 36 DOF control (upper body: torso, arms, hands)
- Head-mounted RGB camera (512x512)
- PINK IK controller for upper body manipulation
- Lower body fixed in standing pose, gravity disabled

**Scene Configuration:**

- Kitchen environment
- Microwave with articulated door (revolute joint)
- Task: Open door to 80% (success threshold)

**Task Configuration:**

.. code-block:: python

   from isaaclab_arena.tasks import OpenDoorTask
   from isaaclab_arena.affordances import Openable
   # define microwave object reference, and background scene

   microwave = Openable(
       name="microwave",
       articulation_cfg=microwave_cfg,
       joint_name="door_joint"
   )

   task = OpenDoorTask(
       openable_object=microwave,
       openness_threshold=0.8,  # >80% open = success
       background_scene=background_scene
   )

.. todo:: (amillane, 2025-10-22): Rework this section. Let's just step through the
    environment description in its entirety.


Step 1: Download a Test Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run a robot in the environment we need some recorded demonstration data that
can be fed to the robot to control its actions.
We download a pre-recorded dataset from Hugging Face (replace ``<INPUT_DATASET_PATH>`` with the actual path):

.. code-block:: bash

   huggingface-cli download \
       nvidia/Arena-GR1-Manipulation-Task \
       arena_gr1_manipulation_dataset_generated.hdf5 \
       --local-dir <INPUT_DATASET_PATH>

.. todo:: (alexmillane, 2025-10-23): Check this command works

.. todo:: (alexmillane, 2025-10-23): Move to specified paths.


Step 2: Start Isaac Lab - Arena
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start the Arena Docker container:

   :docker_run_default:


Step 3: Validate the Environment by Replaying the Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replay the downloaded dataset to verify the environment setup:

.. code-block:: bash

   python isaaclab_arena/scripts/replay_demos.py \
     --enable_cameras \
     --dataset_file <INPUT_DATASET_PATH> \
     gr1_open_microwave \
     --embodiment gr1_pink

You should see the GR1 robot replaying the demonstrations, performing the microwave door
opening task in the kitchen environment.

.. todo:: (amillane, 2025-10-22): screenshot
