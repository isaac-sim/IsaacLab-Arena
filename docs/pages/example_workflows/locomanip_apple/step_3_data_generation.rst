Data Generation
---------------

This workflow covers annotating and generating the demonstration dataset using
`Isaac Lab Mimic <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html>`_.


**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:


Step 1: Annotate Demonstrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step describes how to annotate the demonstrations recorded in the preceding step
so they can be used by Isaac Lab Mimic. For more details on Mimic annotation, see the
`Isaac Lab Mimic documentation <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html#annotate-the-demonstrations>`_.

To start the annotation process, run the following command:

.. code-block:: bash

   python isaaclab_arena/scripts/imitation_learning/annotate_demos.py \
     --viz kit \
     --device cpu \
     --input_file $DATASET_DIR/arena_g1_locomanip_apple_dataset_recorded.hdf5 \
     --output_file $DATASET_DIR/arena_g1_locomanip_apple_dataset_annotated.hdf5 \
     --mimic \
     galileo_g1_locomanip_pick_and_place \
     --object apple_01_objaverse_robolab \
     --destination clay_plates_hot3d_robolab

Follow the instructions described on the CLI to complete the annotation.



Step 2: Generate Augmented Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Isaac Lab Mimic generates additional demonstrations from the annotated demonstrations
by applying object and trajectory transformations to introduce data variations.

Generate the dataset:

.. code-block:: bash

   # Generate 100 demonstrations
   python isaaclab_arena/scripts/imitation_learning/generate_dataset.py \
     --headless \
     --enable_cameras \
     --mimic \
     --input_file $DATASET_DIR/arena_g1_locomanip_apple_dataset_annotated.hdf5 \
     --output_file $DATASET_DIR/arena_g1_locomanip_apple_dataset_generated.hdf5 \
     --generation_num_trials 100 \
     --device cpu \
     galileo_g1_locomanip_pick_and_place \
     --object apple_01_objaverse_robolab \
     --destination clay_plates_hot3d_robolab \
     --embodiment g1_wbc_pink

Data generation takes 1-4 hours depending on your CPU/GPU.
You can remove ``--headless`` and add ``--viz kit``
(before specifying the task name ``galileo_g1_locomanip_pick_and_place``) to visualize during data generation.

.. note::

   Mimic relies on the ``pick_up_object_name`` and ``destination_name`` fields of
   ``LocomanipPickAndPlaceMimicEnvCfg`` to know which object and destination frames to use in its
   subtask configuration, and to derive a unique ``datagen_config.name`` per
   ``(object, destination)`` pair. The loco-manip task plumbs ``pick_up_object.name`` and
   ``destination_location.name`` through to the Mimic config automatically, so the same
   ``generate_dataset.py`` command that works for the brown-box + blue-bin workflow works here — just
   with ``apple_01_objaverse_robolab`` and ``clay_plates_hot3d_robolab``. The apple-to-plate dataset
   is written under a distinct templated datagen key, so it does not overwrite the brown-box dataset
   that keeps the preserved ``"locomanip_pick_and_place_D0"`` name.


Step 3: Validate Generated Dataset (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To visualize the data produced, you can replay the dataset using the following command:

.. code-block:: bash

   python isaaclab_arena/scripts/imitation_learning/replay_demos.py \
     --viz kit \
     --device cpu \
     --enable_cameras \
     --dataset_file $DATASET_DIR/arena_g1_locomanip_apple_dataset_generated.hdf5 \
     galileo_g1_locomanip_pick_and_place \
     --object apple_01_objaverse_robolab \
     --destination clay_plates_hot3d_robolab \
     --embodiment g1_wbc_pink

You should see the robot successfully perform the task.

.. note::

   The dataset was generated using CPU device physics, therefore the replay uses ``--device cpu`` to ensure reproducibility.
