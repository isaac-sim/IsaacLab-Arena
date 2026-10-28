Deformable Pick and Place
=========================

Pick up a deformable cube with the DROID robot and place it on a plate. The same task can run
with either the Newton or PhysX deformable-body backend. This walkthrough covers previewing
the environment, evaluating an OpenPI policy, recording demonstrations, and replaying the
saved dataset.

Task Overview
-------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Task ID**
     - ``droid_deformable_pick_and_place``
   * - **Embodiment**
     - DROID; absolute joint-position control for OpenPI and differential inverse kinematics
       for teleoperation
   * - **Scene**
     - Maple table with a deformable cube and a plate
   * - **Task**
     - Pick up the cube, place it on the plate, and let it settle
   * - **Physics**
     - Newton by default; select the backend-specific deformable asset for PhysX

Four sampled initial placements are shown below. The placement relations are the same for
both physics backends.

.. image:: ../../../images/deformable_pick_and_place_sampled_placements.png
   :align: center
   :alt: Four sampled deformable pick-and-place initial placements

Prerequisites
-------------

Complete the :doc:`Arena installation <../../quickstart/installation>` and start your Arena
runtime. Run these commands from the repository root in the activated native environment or
inside the Base Docker container (``/workspaces/isaaclab_arena``).

Create a directory for your demonstrations:

.. code-block:: bash

   export DATASET_DIR=/datasets/deformable_pick_and_place
   mkdir -p "${DATASET_DIR}"

In Docker, use the ``/datasets`` host mount described in the installation guide so your data
persists outside the container. For a native installation, set ``DATASET_DIR`` to a writable
directory such as ``$HOME/datasets/deformable_pick_and_place``.

Step 1: Launch the Environment
------------------------------

Use a zero-action policy to inspect either backend:

.. tab-set::

   .. tab-item:: Newton
      :selected:

      .. code-block:: bash

         python isaaclab_arena/evaluation/policy_runner.py \
           --policy_type zero_action \
           --num_steps 1000 \
           --num_envs 1 \
           --viz kit \
           droid_deformable_pick_and_place \
           --pick_object deformable_cube_newton

   .. tab-item:: PhysX

      .. code-block:: bash

         python isaaclab_arena/evaluation/policy_runner.py \
           --policy_type zero_action \
           --num_steps 1000 \
           --num_envs 1 \
           --viz kit \
           droid_deformable_pick_and_place \
           --pick_object deformable_cube_physx

Step 2: Evaluate with OpenPI
----------------------------

Start the local OpenPI server in a separate terminal:

.. code-block:: bash

   ./isaaclab_arena_openpi/docker/run_openpi_server.sh

Then evaluate four parallel environments for 16 completed episodes:

.. tab-set::

   .. tab-item:: Newton
      :selected:

      .. code-block:: bash

         python isaaclab_arena/evaluation/policy_runner.py \
           --policy_type isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy \
           --num_episodes 16 \
           --num_envs 4 \
           --viz kit \
           --enable_cameras \
           droid_deformable_pick_and_place \
           --pick_object deformable_cube_newton

      .. note::
         A local 16-episode OpenPI ``pi05`` validation run succeeded in 9 episodes (56.25%).
         This small stochastic run validates the workflow.

   .. tab-item:: PhysX

      .. code-block:: bash

         python isaaclab_arena/evaluation/policy_runner.py \
           --policy_type isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy \
           --num_episodes 16 \
           --num_envs 4 \
           --viz kit \
           --enable_cameras \
           droid_deformable_pick_and_place \
           --pick_object deformable_cube_physx

      .. note::
         A local 16-episode OpenPI ``pi05`` validation run succeeded in 15 episodes (93.75%).
         This small stochastic run validates the workflow.

The following rollout shows one successful Newton episode from an external camera and the
robot wrist camera:

.. image:: ../../../images/deformable_pick_and_place_openpi.gif
   :align: center
   :alt: Successful Newton deformable pick-and-place rollout from external and wrist cameras

Step 3: Record Demonstrations
-----------------------------

The Isaac Lab demonstration recorder can control the task through Arena's interop callback.
The commands below use keyboard control; replace ``keyboard`` with ``spacemouse`` to use a
SpaceMouse. Press ``K`` to open or close the gripper and ``R`` to reset an attempt.

.. tab-set::

   .. tab-item:: Newton
      :selected:

      .. code-block:: bash

         python submodules/IsaacLab/scripts/tools/record_demos.py \
           --task droid_deformable_pick_and_place \
           --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
           --teleop_device keyboard \
           --num_demos 1 \
           --num_success_steps 1 \
           --no-reset_sim_buffer_each_episode \
           --dataset_file "${DATASET_DIR}/deformable_pick_and_place_newton.hdf5" \
           --viz kit \
           --pick_object deformable_cube_newton \
           --embodiment droid_differential_ik

   .. tab-item:: PhysX

      .. code-block:: bash

         python submodules/IsaacLab/scripts/tools/record_demos.py \
           --task droid_deformable_pick_and_place \
           --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
           --teleop_device keyboard \
           --num_demos 1 \
           --num_success_steps 1 \
           --dataset_file "${DATASET_DIR}/deformable_pick_and_place_physx.hdf5" \
           --viz kit \
           --pick_object deformable_cube_physx \
           --embodiment droid_differential_ik

``--no-reset_sim_buffer_each_episode`` preserves the Newton simulation buffers between
attempts. The environment still resets after success or manual reset.

Step 4: Replay Demonstrations
-----------------------------

After recording finishes, replay the saved dataset with the matching physics backend:

.. tab-set::

   .. tab-item:: Newton
      :selected:

      .. code-block:: bash

         python submodules/IsaacLab/scripts/tools/replay_demos.py \
           --task droid_deformable_pick_and_place \
           --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
           --viz kit \
           --num_envs 1 \
           --dataset_file "${DATASET_DIR}/deformable_pick_and_place_newton.hdf5" \
           --pick_object deformable_cube_newton \
           --embodiment droid_differential_ik

   .. tab-item:: PhysX

      .. code-block:: bash

         python submodules/IsaacLab/scripts/tools/replay_demos.py \
           --task droid_deformable_pick_and_place \
           --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
           --viz kit \
           --num_envs 1 \
           --dataset_file "${DATASET_DIR}/deformable_pick_and_place_physx.hdf5" \
           --pick_object deformable_cube_physx \
           --embodiment droid_differential_ik

The script replays every demonstration in the dataset and exits when finished. With the Kit
window focused, press ``B`` to pause the replayed actions and ``N`` to resume.
