Deformable Pick and Place
=========================

Pick up a deformable cube with the DROID robot and place it on a plate. The same task can run
with either the Newton or PhysX deformable-body backend. This walkthrough covers previewing
the environment, teleoperating it, and evaluating an OpenPI policy.

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
     - Newton by default, or PhysX with ``--presets physx``

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

Preview the Environment
-----------------------

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
           --presets newton \
           droid_deformable_pick_and_place

   .. tab-item:: PhysX

      .. code-block:: bash

         python isaaclab_arena/evaluation/policy_runner.py \
           --policy_type zero_action \
           --num_steps 1000 \
           --num_envs 1 \
           --viz kit \
           --presets physx \
           droid_deformable_pick_and_place

Teleoperate the Task
--------------------

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
           --dataset_file deformable_pick_and_place_newton.hdf5 \
           --viz kit \
           --presets newton \
           --embodiment droid_differential_ik

   .. tab-item:: PhysX

      .. code-block:: bash

         python submodules/IsaacLab/scripts/tools/record_demos.py \
           --task droid_deformable_pick_and_place \
           --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
           --teleop_device keyboard \
           --num_demos 1 \
           --num_success_steps 1 \
           --dataset_file deformable_pick_and_place_physx.hdf5 \
           --viz kit \
           --presets physx \
           --embodiment droid_differential_ik

``--no-reset_sim_buffer_each_episode`` preserves the Newton simulation buffers between
attempts. The environment still resets after success or manual reset.

Evaluate with OpenPI
--------------------

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
           --presets newton \
           droid_deformable_pick_and_place

   .. tab-item:: PhysX

      .. code-block:: bash

         python isaaclab_arena/evaluation/policy_runner.py \
           --policy_type isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy \
           --num_episodes 16 \
           --num_envs 4 \
           --viz kit \
           --enable_cameras \
           --presets physx \
           droid_deformable_pick_and_place

The following rollout shows one successful Newton episode from an external camera and the
robot wrist camera:

.. image:: ../../../images/deformable_pick_and_place_openpi.gif
   :align: center
   :alt: Successful Newton deformable pick-and-place rollout from external and wrist cameras

Local Validation Results
------------------------

The local OpenPI ``pi05`` validation runs used for this walkthrough produced:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Backend
     - Success
     - Failed episodes
   * - Newton
     - 9/16 (56.25%)
     - Six incomplete placements and one geometric-support rejection
   * - PhysX
     - 15/16 (93.75%)
     - One placement remained above the strict 0.001 m/s settling threshold at timeout

These small local runs validate the workflow; they are not policy benchmarks. OpenPI action
sampling is stochastic, and success rates vary between runs.
