Data Generation
---------------

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

This workflow covers annotating and generating the demonstration dataset using
`Isaac Lab Mimic <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html>`_.


Step 1: Download Human Demonstration Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For this tutorial we will use the pre-recorded human demonstrations.
Note that, in contrast, in the :doc:`static manipulation workflow <../static_manipulation/index>`,
we support recording your own demonstrations.

.. note::

   Recording your own demonstrations for loco-manipulation workflows will be supported in a future release.

Download the pre-recorded human demonstrations (replace ``<INPUT_DATASET_PATH>`` with the actual path):

.. code-block:: bash

   huggingface-cli download \
       nvidia/Arena-G1-Loco-Manipulation-Task \
       arena_g1_loco_manipulation_dataset_annotated.hdf5 \
       --repo-type dataset \
       --local-dir <INPUT_DATASET_PATH>


Step 2: Generate Dataset with Isaac Lab Mimic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use Isaac Lab Mimic to generate additional demonstrations from the human
demonstrations by applying object and trajectory transformations to introduce
data variations.

Start the Arena Docker container, if you haven't already:

   :docker_run_default:

Generate the dataset (replace ``<INPUT_DATASET_PATH>`` and ``<OUTPUT_DATASET_PATH>`` with the actual paths):

.. code-block:: bash

   # Generate 100 demonstrations
   python isaaclab_arena/scripts/generate_dataset.py \
     --headless \
     --enable_cameras \
     --mimic \
     --input_file <INPUT_DATASET_PATH> \
     --output_file <OUTPUT_DATASET_PATH> \
     --generation_num_trials 100 \
     --device cpu \
     galileo_g1_locomanip_pick_and_place \
     --object brown_box \
     --embodiment g1_wbc_pink

Data generation takes 1-4 hours depending on your CPU/GPU.
You can remove ``--headless`` to visualize during data generation.


Step 3: Validate Generated Dataset (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To visualize the data produced, you can replay the dataset using the following command:
(replace ``<OUTPUT_DATASET_PATH>`` with the actual path):

.. code-block:: bash

   python isaaclab_arena/scripts/replay_demos.py \
     --device cpu \
     --enable_cameras \
     --dataset_file <OUTPUT_DATASET_PATH> \
     galileo_g1_locomanip_pick_and_place \
     --object brown_box \
     --embodiment g1_wbc_pink

You should see the robot successfully perform the task.

.. note::

   The dataset was generated using CPU device physics, therefore the replay uses ``--device cpu`` to ensure reproducibility.


.. todo:: (amillane, 2025-10-22): add screenshot
