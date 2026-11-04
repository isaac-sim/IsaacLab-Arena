Gear Insertion
==============

Use a keyboard or SpaceMouse to teleoperate and pick up a gear with the DROID robot and insert it onto the
peg board. This walkthrough covers previewing the environment, recording demonstrations,
replaying the saved dataset, and evaluating a vision-language model (VLM) agent as the policy.

Task Overview
-------------

**Task ID:** ``gear_insertion``

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Embodiment**
     - DROID with differential inverse kinematics control
   * - **Scene**
     - Maple table with a gear and peg board
   * - **Task**
     - Pick up the gear and seat it on the middle peg of the peg board
   * - **Physics**
     - Newton (selected automatically by the environment)
   * - **Teleoperation devices**
     - Keyboard or SpaceMouse


Prerequisites
-------------

Complete the :doc:`Arena installation <../../quickstart/installation>`.
Then open a **host terminal at the Arena repository root**.

.. _spacemouse_setup:

SpaceMouse Setup (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Skip this section if using keyboard input. **Stop the Arena container if it is running**,
then copy and paste the entire block below into a terminal on the Linux host:

.. code-block:: bash

   sudo groupadd -f plugdev
   sudo usermod -aG plugdev "$USER"
   sudo tee /etc/udev/rules.d/70-arena-spacemouse.rules <<'EOF'
   SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="256f", GROUP="plugdev", MODE="0660", TAG+="uaccess"
   SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="046d", ATTR{idProduct}=="c626|c628", GROUP="plugdev", MODE="0660", TAG+="uaccess"
   EOF
   sudo udevadm control --reload-rules

Unplug and reconnect the SpaceMouse, then log out and back in.

Open the Arena Runtime
^^^^^^^^^^^^^^^^^^^^^^

If using Docker, start the container from the repository root on the host:

.. code-block:: bash

   mkdir -p "$HOME/datasets"
   ./docker/run_docker.sh

Run the remaining commands from the repository root in the activated native
environment or inside the Docker container (``/workspaces/isaaclab_arena``).

Create a directory for your demonstrations. For native installations, replace
``/datasets/gear_insertion`` with ``$HOME/datasets/gear_insertion``:

.. code-block:: bash

   export DATASET_DIR=/datasets/gear_insertion
   mkdir -p "${DATASET_DIR}"

Keep using this terminal so ``DATASET_DIR`` remains set.


Step 1: Launch the Environment
------------------------------

Run a zero-action policy to preview the scene:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --num_steps 1000 \
     --num_envs 1 \
     --device cuda:0 \
     --viz kit \
     --enable_cameras \
     gear_insertion

You should see the DROID robot, the gear, and the peg assembly on the table. The simulation
steps with zero actions so you can inspect the scene. Newton is selected automatically.

The preview exits after 1,000 steps. Increase ``--num_steps`` for a longer preview, or press
``Ctrl+C`` in the terminal to stop it. Finish the preview before starting recording.

.. image:: ../../../images/gear_insertion_env.png
   :align: center
   :height: 400px


Step 2: Record Demonstrations
-----------------------------

Choose the tab for your input device.

.. tab-set::

   .. tab-item:: Keyboard
      :selected:

      .. code-block:: bash

         uv run isaaclab teleop record \
           --task gear_insertion \
           --device cuda:0 \
           --viz kit \
           --enable_cameras \
           --teleop_device keyboard \
           --num_demos 1 \
           --num_success_steps 1 \
           --no-reset_sim_buffer_each_episode \
           --dataset_file "${DATASET_DIR}/gear_insertion_teleop_demos.hdf5"

      **Controls**

      .. list-table::
         :widths: 60 40
         :header-rows: 1

         * - Action
           - Input
         * - Move along the X axis
           - ``W`` / ``S``
         * - Move along the Y axis
           - ``A`` / ``D``
         * - Move along the Z axis
           - ``Q`` / ``E``
         * - Rotate about the X axis
           - ``Z`` / ``X``
         * - Rotate about the Y axis
           - ``T`` / ``G``
         * - Rotate about the Z axis
           - ``C`` / ``V``
         * - Open or close the gripper
           - ``K``
         * - Reset the current attempt
           - ``R``

   .. tab-item:: SpaceMouse

      .. code-block:: bash

         uv run isaaclab teleop record \
           --task gear_insertion \
           --device cuda:0 \
           --viz kit \
           --enable_cameras \
           --teleop_device spacemouse \
           --num_demos 1 \
           --num_success_steps 1 \
           --no-reset_sim_buffer_each_episode \
           --dataset_file "${DATASET_DIR}/gear_insertion_teleop_demos.hdf5"

      **Controls**

      .. list-table::
         :widths: 60 40
         :header-rows: 1

         * - Action
           - Input
         * - Move the gripper
           - Move the cap laterally or vertically
         * - Change the gripper orientation
           - Rotate the cap
         * - Open or close the gripper
           - Press the left button
         * - Reset the current attempt
           - Press the right button

Use the selected teleop device to pick up the gear and insert it onto the **middle peg** of the peg board.

When the environment reports success, the script saves the demonstration and reports the
number of successful demonstrations recorded in the terminal. If more demonstrations are
needed, the environment resets for the next attempt. Resetting an unsuccessful attempt
discards that attempt.

The script exits after saving ``--num_demos`` successful demonstrations. Increase
``--num_demos`` to record more demonstrations, or press ``Ctrl+C`` in the terminal to stop earlier.

.. note::
  It may take a few seconds for the environment to reset after a successful demonstration as it saves the episode to the HDF5 dataset.

The recording options used here are:

* ``--num_demos 1``: collect one successful demonstration.
* ``--num_success_steps 1``: save as soon as the environment reports success. The task's own
  checks still require the gear to remain seated and settled before reporting success.
* ``--no-reset_sim_buffer_each_episode``: Newton requirement. Preserve simulation buffers between attempts.
  The environment still resets after each successful demonstration or manual reset.

.. image:: ../../../images/gear_insertion_env_teleop.gif
   :align: center
   :height: 400px


Step 3: Replay Demonstrations
-----------------------------

After recording finishes, replay the saved dataset:

.. code-block:: bash

   uv run isaaclab teleop replay \
     --task gear_insertion \
     --device cuda:0 \
     --viz kit \
     --enable_cameras \
     --num_envs 1 \
     --dataset_file "${DATASET_DIR}/gear_insertion_teleop_demos.hdf5"

You should see the recorded motions replayed. The script replays all
demonstrations in the dataset and exits when finished.

With the Kit window focused, press ``B`` to pause the replayed actions and ``N`` to resume.

.. image:: ../../../images/gear_insertion_env_replay.gif
   :align: center
   :height: 400px

.. note::

   Non-determinism may be observed when replaying demonstrations with Newton because
   physics in Isaac Lab is not deterministically reproducible across ``env.reset()``
   calls. A demonstration may therefore fail during replay even though it succeeded
   during recording. Demonstrations with more aggressive actions and borderline grasps
   are more susceptible to this issue. All recorded demonstrations in the HDF5 remain
   valid with respect to the task's success condition.


.. _gear-insertion-vlm-agent:

Step 4: Evaluate a VLM Agent Policy
-----------------------------------

A hosted VLM can also act as the policy, through ``VLMGoalPolicy``. Gear insertion needs the gear
within 3.5 mm of the peg, a precise test of how well the model locates parts from camera images.
See :ref:`vlm-agent-policies` for how the policy works, and set up an inference API key as in
:doc:`../../quickstart/running_a_real_policy/vlm_agent`.

Three experiment configurations run the policy on the ``droid_abs_ik`` embodiment with the full
1280x720 camera images. They differ only in what the model is given. Run them from the repository
root, so the prompt files they reference are found.

**Images only.** The model gets the camera images, the robot state, and the DROID default system
prompt, and must locate the gear and the pegs in the images:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_gear_insertion_vlm_goal_images_experiment.yaml

**Images with exact object poses.** ``object_poses`` adds the gear's exact pose and its seated
target pose to each request. A real robot does not observe these, so this run is a reference for
what the policy achieves once it knows where the parts are:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_gear_insertion_vlm_goal_object_poses_experiment.yaml

**Images with a task prompt.** ``system_prompt_files`` appends
``experiment_configs/prompts/droid_gear_insertion.txt`` to the default prompt. It describes the
gear and peg geometry and a grasp and insertion strategy, but no object positions:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_gear_insertion_vlm_goal_task_prompt_experiment.yaml

Add ``--viz kit`` to watch a rollout.

The same layout under each experiment: images only (left), exact object poses (middle), and the
task prompt (right).

.. image:: ../../../images/gear_insertion_vlm_agent.gif
   :align: center
   :width: 100%
