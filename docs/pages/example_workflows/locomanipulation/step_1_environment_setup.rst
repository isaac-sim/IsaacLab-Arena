Environment Setup and Validation
--------------------------------

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


Step 2: Download a test dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TBD


Step 3: Validate Environment with Demo Replay
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replay the generated dataset to verify the environment setup:

.. code-block:: bash

   python isaaclab_arena/scripts/replay_demos.py \
     --enable_cameras \
     --dataset_file /datasets/Arena-G1-Loco-Manipulation-Task/arena_g1_loco_manipulation_dataset_generated.hdf5 \
     galileo_g1_locomanip_pick_and_place \
     --object brown_box \
     --embodiment g1_wbc_pink

You should see the G1 robot replaying the generated demonstrations, performing box pick and place task in the Galileo lab environment.

.. todo:: (amillane, 2025-10-22): screenshot
