Data Generation
---------------

This workflow covers generating a new dataset using
`Isaac Lab Mimic <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html>`_.

Note that this tutorial assumes that you've completed the
:doc:`preceding step (Teleoperation Data Collection) <step_2_teleoperation>`.
If you do not want to do the preceding step of recording demonstrations, you can jump to
you can download the pre-generated dataset either in
:ref:`step_1_annotate_demonstrations` or :ref:`step_2_generate_augmented_dataset`
below.


**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_default_code_block:


.. _step_1_annotate_demonstrations:

Step 1: Annotate Demonstrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step describes how to annotate the demonstrations recorded in the preceding step
such that they can be used by Isaac Lab Mimic.
The process of annotation involves segmenting demonstrations into subtasks (reach, grasp, pull):

To skip this step, you can download the pre-annotated dataset from Hugging Face as described below.

.. dropdown:: Download Pre-annotated Dataset (skip annotation step)
   :animate: fade-in

   These commands can be used to download the pre-annotated dataset,
   such that the annotation step can be skipped.

   To download run:

   .. code-block:: bash

      hf download \
         nvidia/Arena-GR1-Manipulation-Task \
         arena_gr1_manipulation_dataset_annotated.hdf5 \
         --repo-type dataset \
         --local-dir $DATASET_DIR

To start the annotation process run the following command:

.. code-block:: bash

   python isaaclab_arena/scripts/annotate_demos.py \
     --device cpu \
     --input_file $DATASET_DIR/arena_gr1_manipulation_dataset_recorded.hdf5 \
     --output_file $DATASET_DIR/arena_gr1_manipulation_dataset_annotated.hdf5 \
     --enable_pinocchio \
     --mimic \
     gr1_open_microwave

Follow the instructions described on the CLI to mark subtask boundaries:

1. **Reach:** Robot reaches toward the microwave door
2. **Open door:** Robot opens the door



.. _step_2_generate_augmented_dataset:

Step 2: Generate Augmented Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Isaac Lab Mimic generates additional demonstrations from a small set of
annotated demonstrations by using rigid body transformations to introduce variations.

This step can be skipped by downloading the pre-generated dataset from Hugging Face as described below.

.. dropdown:: Download Pre-generated Dataset (skip data generation step)
   :animate: fade-in

   These commands can be used to download the pre-generated dataset,
   such that the data generation step can be skipped.

   .. code-block:: bash

      hf download \
         nvidia/Arena-GR1-Manipulation-Task \
         arena_gr1_manipulation_dataset_generated.hdf5 \
         --repo-type dataset \
         --local-dir $DATASET_DIR


Generate the dataset:

.. code-block:: bash

   python isaaclab_arena/scripts/generate_dataset.py \
     --device cpu \
     --generation_num_trials 50 \
     --num_envs 10 \
     --input_file $DATASET_DIR/arena_gr1_manipulation_dataset_annotated.hdf5 \
     --output_file $DATASET_DIR/arena_gr1_manipulation_dataset_generated.hdf5 \
     --enable_pinocchio \
     --enable_cameras \
     --headless \
     --mimic \
     gr1_open_microwave

Data generation takes 30-60 minutes depending on hardware.
If you want to visualize the data generation process, remove ``--headless``
to visualize data generation.


Step 3: Validate Generated Data (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to validation the generated dataset, you can replay the generated data
through the robot, in order to check (visually) if the robot is able to perform the task successfully.
To do so, run the following command:

.. code-block:: bash

   python isaaclab_arena/scripts/replay_demos.py \
     --device cpu \
     --enable_cameras \
     --dataset_file $DATASET_DIR/arena_gr1_manipulation_dataset_generated.hdf5 \
     gr1_open_microwave \
     --embodiment gr1_joint

You should see the robot successfully perform the task.

.. todo:: (amillane, 2025-10-22): add screenshot
