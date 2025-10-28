Environment Setup and Validation
--------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

On this page we briefly describe the environment used in this example workflow
and validate that we can load it in Isaac Lab.

Environment Description
^^^^^^^^^^^^^^^^^^^^^^^

The environment used in this example workflow has the following components:

**Embodiment Configuration:**

.. code-block:: python

   from isaaclab_arena.embodiments.g1 import G1WBCPinkEmbodiment

   embodiment = G1WBCPinkEmbodiment(enable_cameras=True)

**Key Features:**

- 43 DOF control (legs, torso, arms, hands)
- Head-mounted RGB camera (480x640, FOV 128°x80°)
- Whole Body Controller (WBC) for lower body locomotion
- PINK IK controller for upper body manipulation
- Action space: EEF poses + gripper states + navigation commands (x velocity, y velocity, yaw velocity, base height, torso pose)

**Scene Configuration:**

- Galileo lab environment
- Brown box (spawned on shelf)
- Blue bin (target location)

**Task Configuration:**

.. code-block:: python

   from isaaclab_arena.tasks import PickAndPlaceTask
   # define brown box, blue bin object references, and background scene

   task = PickAndPlaceTask(
       pick_object=brown_box,
       place_target=blue_bin,
       background_scene=background_scene
   )

.. todo:: (amillane, 2025-10-22): Rework this section. Let's just step through the
    environment description in its entirety.


Step 1: Start Isaac Lab - Arena
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start the Arena Docker container:

   :docker_run_default:


.. todo:: (alexmillane, 2025-10-23): Unify the docker start-up specification.

Step 2: Download a test dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run a robot in the environment we need some recorded demonstration data that
can be fed to the robot to control its actions.
We download a pre-recorded dataset from Hugging Face (replace ``<INPUT_DATASET_PATH>`` with the actual path):

.. code-block:: bash

   huggingface-cli download \
       nvidia/Arena-GR1-Manipulation-Task \
       arena_gr1_manipulation_dataset_generated_small.hdf5 \
       --repo-type dataset \
       --local-dir <INPUT_DATASET_PATH>


Step 3: Validate Environment with Demo Replay
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replay the downloaded dataset to verify the environment setup
(replace ``<INPUT_DATASET_PATH>`` with the actual path):

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
