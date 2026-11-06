Data Generation
---------------

This workflow covers annotating and generating the demonstration dataset using
`Isaac Lab Mimic <https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/teleop_imitation.html>`_.


**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:



Step 1: Annotate Demonstrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step describes how to annotate the demonstrations recorded in the preceding step
so they can be used by Isaac Lab Mimic. For more details on Mimic annotation, see the
`Isaac Lab Mimic documentation <https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/teleop_imitation.html#subtask-annotation>`_.

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

Run from the repository root with ``DATASET_DIR`` set to the directory used in
:doc:`Teleoperation Data Collection <step_2_teleoperation>`. Keep the task,
object, embodiment and physics device consistent with the recording.

To start manual annotation, run:

.. code-block:: bash

   python submodules/IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
     --viz kit \
     --device cpu \
     --input_file $DATASET_DIR/arena_g1_loco_manipulation_dataset_recorded.hdf5 \
     --output_file $DATASET_DIR/arena_g1_loco_manipulation_dataset_annotated.hdf5 \
     --mimic \
     --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
     --task galileo_g1_locomanip_pick_and_place \
     env.actions.g1_action.use_p_control=false

The final override keeps the recorded navigation commands during annotation.
Step 2 below uses the navigation controller to generate new demonstrations.

**Mark the subtask boundaries**

Click inside the **Isaac Sim viewport** before using the keyboard. These keys
are read by the simulation window, not the terminal:

* ``N``: begin or resume playback.
* ``B``: pause playback.
* ``S``: mark the next subtask boundary at the current frame.
* ``Q``: skip the entire episode without exporting it.

Annotation has three passes: **right arm**, **left arm**, then **body**. The
**entire episode** replays on each pass; mark only the named arm or body.
The terminal identifies the pass as ``eef "right"``, ``eef "left"`` or ``eef "body"``.
Press ``N`` to begin each pass, then ``S`` once at each boundary below, in order.
You can pause with ``B``, mark with ``S``, then resume with ``N``.

.. list-table::
   :header-rows: 1
   :widths: 13 32 55

   * - Pass
     - Signal
     - When to press ``S``
   * - Right arm
     - ``idle_right``
     - When the right arm finishes waiting and starts reaching for the box.
   * - Right arm
     - ``grasp_and_idle_right``
     - When grasping and carrying finish, just before the right arm starts
       placing the box in the bin.
   * - Left arm
     - ``idle_left``
     - When the left arm finishes waiting and starts reaching for the box.
   * - Left arm
     - ``grasp_and_idle_left``
     - When grasping and carrying finish, just before the left arm starts
       placing the box in the bin.
   * - Body
     - ``navigate_to_table``
     - When the robot reaches the pickup position in front of the box.
   * - Body
     - ``navigate_turn_inplace``
     - When the robot finishes turning toward the bin after picking up the box.
   * - Body
     - ``navigate_to_bin``
     - When the robot stops walking at the bin, before it squats to place the box.

Mark **two boundaries per arm and three for the body**. Let each replay finish
after marking them. The final subtask ends with the episode and needs no mark.

Each press prints ``Marked a subtask signal at action index: <index>``. If you
mistime a mark, press ``S`` once more before the replay ends; the wrong count
makes the script replay the pass. To drop the episode instead, press ``Q``.

**Check the result**

For each exported episode, the output includes:

.. code-block:: text

   All 2 subtask signals for eef "right" were annotated.
   All 2 subtask signals for eef "left" were annotated.
   All 3 subtask signals for eef "body" were annotated.
   Exported the annotated episode.

After the last episode, the script prints a summary. For example, a run on five
recorded episodes in which one replay failed prints:

.. code-block:: text

   Exported 4 (out of 5) annotated episodes.
   Successful task completions: 4

Annotation succeeds when at least one episode is exported. Some episodes may be
skipped if PhysX non-determinism causes borderline successful trajectories to
fail the task during replay.

.. important::

   **Do not start Step 2 with zero exported episodes.** The script creates the
   output file when it starts, replacing any existing file at
   ``--output_file``, so the file exists even when every episode fails. Its
   existence or size does not mean it contains annotated demonstrations.
   Recover the failed annotations below or use the pre-annotated download
   above.

**Recover from an unsuccessful annotation**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Message
     - Meaning and next action
   * - ``Only N out of M subtask signals ... were annotated.``
     - Replay succeeded, but the number of marks was wrong. The script clears
       the marks for this pass and pauses the retry until you press ``N``.
       Focus the viewport, press ``N``, and mark every boundary for that arm
       or body again.
   * - ``The final task was not completed.``
     - The replay failed the task's success check. The entire episode is
       skipped without retrying, even if an earlier pass succeeded. Replaying
       the same recorded actions can produce different outcomes, so a few
       episodes can fail this way at random. If every episode fails, check
       that the command includes ``env.actions.g1_action.use_p_control=false``
       and matches the recording's task and ``--device``. To retry skipped
       episodes, rerun annotation, which annotates every episode again, or
       use the pre-annotated dataset. A rerun replaces ``--output_file``; to
       keep an earlier output, pass a different filename and use that file in
       Step 2.

``Skipped exporting the episode due to incomplete subtask annotations`` also
appears after a failed task replay or after ``Q``; read the preceding message
for the cause.
Skipped episodes remain in the original recorded file.



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
   python submodules/IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
     --enable_cameras \
     --mimic \
     --input_file $DATASET_DIR/arena_g1_loco_manipulation_dataset_annotated.hdf5 \
     --output_file $DATASET_DIR/arena_g1_loco_manipulation_dataset_generated.hdf5 \
     --generation_num_trials 100 \
     --device cpu \
     --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
     --task galileo_g1_locomanip_pick_and_place \
     --object brown_box \
     --embodiment g1_wbc_pink

Data generation takes 1-4 hours depending on your CPU/GPU.
Add ``--viz kit`` to visualize during data generation.


Step 3: Validate Generated Dataset (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To visualize the data produced, you can replay the dataset using the following command:

.. note::

   Non-determinism may be observed when replaying demonstrations with PhysX because
   physics in Isaac Lab is not deterministically reproducible across ``env.reset()``
   calls. A demonstration may therefore fail during replay even though it succeeded
   during generation; all generated demonstrations remain valid for training.

   This dataset was generated using CPU physics, so the replay command uses
   ``--device cpu`` to reduce device-related differences between generation and replay.

.. code-block:: bash

   python submodules/IsaacLab/scripts/tools/replay_demos.py \
     --viz kit \
     --device cpu \
     --enable_cameras \
     --dataset_file $DATASET_DIR/arena_g1_loco_manipulation_dataset_generated.hdf5 \
     --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
     --task galileo_g1_locomanip_pick_and_place \
     --object brown_box \
     --embodiment g1_wbc_pink

You should see the robot successfully perform the task.

.. figure:: ../../../images/g1_locomanip_pick_and_place_task_view.png
   :width: 100%
   :alt: G1 Locomanip Pick and Place Task View
   :align: center

   IsaacLab Arena G1 Locomanip Pick and Place Task View
