Data Generation
---------------

This workflow covers generating a new dataset using
`Isaac Lab Mimic <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html>`_.

Note that this tutorial assumes that you've completed the
:doc:`preceding step (Teleoperation Data Collection) <step_2_teleoperation>`.
If you do not want to do the preceding step of recording demonstrations, you can download
the pre-recorded datasets and jump to :ref:`sequential_step_1_annotate_demonstrations`.


**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_default:


.. _sequential_step_1_annotate_demonstrations:

Step 1: Annotate Demonstrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step describes how to manually annotate the demonstrations recorded in the preceding step
such that they can be used by Isaac Lab Mimic. For automatic annotation the user needs to define
subtasks in their task definition, we do not show how to do this in this tutorial.
The process of annotation involves segmenting demonstrations into various subtasks for the left/right arms.
As this is a sequential composite task, there are additional Mimic subtasks to denote the end of each intermediate atomic task
(in this case an annotation to denote the end of the pick & place task):
For more details on mimic annotation, please refer to the
`Isaac Lab Mimic documentation <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html#annotate-the-demonstrationsl>`_.

To skip this step, you can download the pre-annotated dataset as described below.

.. dropdown:: Download Pre-annotated Dataset (skip annotation step)
   :animate: fade-in

   These commands can be used to download the pre-annotated dataset,
   such that the annotation step can be skipped.

   To download run:

   .. code-block:: bash

      hf download \
         nvidia/Arena-GR1-Manipulation-PlaceItemCloseDoor-Task \
         ranch_bottle_into_fridge/ranch_bottle_into_fridge_annotated.hdf5 \
         --repo-type dataset \
         --local-dir $DATASET_DIR

To start the annotation process run the following command:

.. code-block:: bash

   python isaaclab_arena/scripts/imitation_learning/annotate_demos.py \
     --device cpu \
     --input_file $DATASET_DIR/ranch_bottle_into_fridge/ranch_bottle_into_fridge_recorded.hdf5 \
     --output_file $DATASET_DIR/ranch_bottle_into_fridge/ranch_bottle_into_fridge_annotated.hdf5 \
     --enable_pinocchio \
     --mimic \
     put_item_in_fridge_and_close_door \
     --object ranch_dressing_bottle \
     --embodiment gr1_pink

Follow the instructions described on the CLI to mark subtask boundaries:

Left Arm:
1. **Pick & place completed:** Robot has completed the pick and place task

Right Arm:
1. **Grasp object:** Robot has grasped the object
2. **Pick & place completed:** Robot has completed the pick and place task
3. **Move to door:** Robot has moved its hand to the refrigerator door


.. _sequential_step_2_generate_augmented_dataset:

Step 2: Generate Augmented Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Isaac Lab Mimic generates additional demonstrations from a small set of
annotated demonstrations by using rigid body transformations to introduce variations.

This step can be skipped by downloading the pre-generated dataset as described below.

.. dropdown:: Download Pre-generated Dataset (skip data generation step)
   :animate: fade-in

   .. code-block:: bash

      hf download \
         nvidia/Arena-GR1-Manipulation-PlaceItemCloseDoor-Task \
         ranch_bottle_into_fridge/ranch_bottle_into_fridge_generated_100.hdf5 \
         --repo-type dataset \
         --local-dir $DATASET_DIR

Generate the dataset:

.. code-block:: bash

   python isaaclab_arena/scripts/imitation_learning/generate_dataset.py  \
     --device cpu \
     --generation_num_trials 100 \
     --num_envs 10 \
     --input_file $DATASET_DIR/ranch_bottle_into_fridge/ranch_bottle_into_fridge_annotated.hdf5 \
     --output_file $DATASET_DIR/ranch_bottle_into_fridge/ranch_bottle_into_fridge_generated_100.hdf5 \
     --enable_pinocchio \
     --enable_cameras \
     --headless \
     --mimic \
     put_item_in_fridge_and_close_door \
     --object ranch_dressing_bottle \
     --embodiment gr1_pink

Data generation takes 30-60 minutes depending on hardware.
If you want to visualize the data generation process, remove ``--headless``
to visualize data generation.


Step 3: Validate Generated Data (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to validation the generated dataset, you can replay the generated data
through the robot, in order to check (visually) if the robot is able to perform the task successfully.
To do so, run the following command:

.. code-block:: bash

   python isaaclab_arena/scripts/imitation_learning/replay_demos.py \
     --device cpu \
     --enable_cameras \
     --dataset_file $DATASET_DIR/ranch_bottle_into_fridge/ranch_bottle_into_fridge_generated_100.hdf5 \
     put_item_in_fridge_and_close_door \
     --object ranch_dressing_bottle \
     --embodiment gr1_pink

You should see the robot perform the manipulation task. Note that the robot's arms shake due to the action noise
added during data generation.

.. figure:: ../../../images/gr1_sequential_static_manip_mimic_datagen.gif
   :width: 100%
   :alt: GR1 picking up and placing an object in a refrigerator and closing the door
   :align: center

   IsaacLab Arena GR1 picking up and placing an object in a refrigerator and closing the door (with action noise)

.. note::

   The dataset was generated using CPU device physics, therefore the replay uses ``--device cpu`` to ensure reproducibility.
