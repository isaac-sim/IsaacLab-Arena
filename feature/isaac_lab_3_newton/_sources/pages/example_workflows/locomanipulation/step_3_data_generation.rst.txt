Data Generation
---------------

This workflow covers annotating and generating the demonstration dataset using
`Isaac Lab Mimic <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html>`_.


**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_default:



Step 1: Annotate Demonstrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step describes how to annotate the demonstrations recorded in the preceding step
so they can be used by Isaac Lab Mimic. For more details on Mimic annotation, see the
`Isaac Lab Mimic documentation <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html#annotate-the-demonstrationsl>`_.

To skip this step, you can download the pre-annotated dataset from Hugging Face as described below.

.. dropdown:: Download Pre-annotated Dataset (skip annotation step)
   :animate: fade-in

   These commands can be used to download the pre-annotated dataset,
   such that the annotation step can be skipped.

   To download run:

   .. code-block:: bash

      hf download \
         nvidia/Arena-G1-Loco-Manipulation-Task \
         arena_g1_loco_manipulation_dataset_annotated.hdf5 \
         --repo-type dataset \
         --revision arena_v0.2_lab_v3.0 \
         --local-dir $DATASET_DIR

To start the annotation process, run the following command:

.. code-block:: bash

   python isaaclab_arena/scripts/imitation_learning/annotate_demos.py \
     --visualizer kit \
     --device cpu \
     --input_file $DATASET_DIR/arena_g1_locomanipulation_dataset_recorded.hdf5 \
     --output_file $DATASET_DIR/arena_g1_locomanipulation_dataset_annotated.hdf5 \
     --enable_pinocchio \
     --mimic \
     galileo_g1_locomanip_pick_and_place

Follow the instructions described on the CLI to complete the annotation.



Step 2: Generate Augmented Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Isaac Lab Mimic generates additional demonstrations from the annotated demonstrations
by applying object and trajectory transformations to introduce data variations.

This step can be skipped by downloading the pre-generated dataset from Hugging Face as described below.

.. dropdown:: Download Pre-generated Dataset (skip data generation step)
   :animate: fade-in

   These commands can be used to download the pre-generated dataset,
   such that the data generation step can be skipped.

   .. code-block:: bash

      hf download \
         nvidia/Arena-G1-Loco-Manipulation-Task \
         arena_g1_loco_manipulation_dataset_generated.hdf5 \
         --repo-type dataset \
         --revision arena_v0.2_lab_v3.0 \
         --local-dir $DATASET_DIR

Generate the dataset:

.. code-block:: bash

   # Generate 100 demonstrations
   python isaaclab_arena/scripts/imitation_learning/generate_dataset.py \
     --headless \
     --enable_cameras \
     --mimic \
     --input_file $DATASET_DIR/arena_g1_loco_manipulation_dataset_annotated.hdf5 \
     --output_file $DATASET_DIR/arena_g1_loco_manipulation_dataset_generated.hdf5 \
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

.. code-block:: bash

   python isaaclab_arena/scripts/imitation_learning/replay_demos.py \
     --visualizer kit \
     --device cpu \
     --enable_cameras \
     --dataset_file $DATASET_DIR/arena_g1_loco_manipulation_dataset_generated.hdf5 \
     galileo_g1_locomanip_pick_and_place \
     --object brown_box \
     --embodiment g1_wbc_pink

You should see the robot successfully perform the task.

.. figure:: ../../../images/g1_locomanip_pick_and_place_task_view.png
   :width: 100%
   :alt: G1 Locomanip Pick and Place Task View
   :align: center

   IsaacLab Arena G1 Locomanip Pick and Place Task View

.. note::

   The dataset was generated using CPU device physics, therefore the replay uses ``--device cpu`` to ensure reproducibility.
