Deformable Pick and Place
=========================

Pick up a deformable cube with the DROID robot and place it on a plate. The same task can run
with either the Newton or PhysX deformable-body backend. This walkthrough covers previewing
the environment, teleoperating it, recording and replaying demonstrations, and evaluating an OpenPI policy.

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
     - Newton by default; select the backend-specific deformable asset for PhysX. See Newton's
       `VBD solver tuning guide <https://newton-physics.github.io/newton/latest/concepts/simulation_tuning_vbd.html>`_
       for deformable simulation parameters.

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

The preview exits after 1,000 steps. Increase ``--num_steps`` for a longer preview, or press
``Ctrl+C`` in the terminal to stop it.

You should see the DROID robot with a deformable cube and a round plate on the maple table.
The placement relations are the same for both physics backends, with the initial poses sampled
as shown below.

.. image:: ../../../images/deformable_pick_and_place_sampled_placements.png
   :align: center
   :alt: Four sampled deformable pick-and-place initial placements

Teleoperate the Task
--------------------

The Isaac Lab teleoperation script can control the task through Arena's interop callback.
Choose the tab for your input device, then run the command for the simulation backend.

See :ref:`spacemouse_setup` for the required host permissions and Docker setup.

.. tab-set::

   .. tab-item:: Keyboard
      :selected:

      **Newton**

      .. code-block:: bash

         python submodules/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
           --task droid_deformable_pick_and_place \
           --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
           --teleop_device keyboard \
           --viz kit \
           --pick_object deformable_cube_newton \
           --embodiment droid_differential_ik

      **PhysX**

      .. code-block:: bash

         python submodules/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
           --task droid_deformable_pick_and_place \
           --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
           --teleop_device keyboard \
           --viz kit \
           --pick_object deformable_cube_physx \
           --embodiment droid_differential_ik

      .. include:: ../../_includes/keyboard_controls.rst.inc

   .. tab-item:: SpaceMouse

      **Newton**

      .. code-block:: bash

         python submodules/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
           --task droid_deformable_pick_and_place \
           --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
           --teleop_device spacemouse \
           --viz kit \
           --pick_object deformable_cube_newton \
           --embodiment droid_differential_ik

      **PhysX**

      .. code-block:: bash

         python submodules/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
           --task droid_deformable_pick_and_place \
           --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
           --teleop_device spacemouse \
           --viz kit \
           --pick_object deformable_cube_physx \
           --embodiment droid_differential_ik

      .. include:: ../../_includes/spacemouse_controls.rst.inc

.. note::

   Both cube presets load a packaged USD containing a pre-generated ``UsdGeom.TetMesh``.
   Their simulation topology is fixed across process launches: 69 nodes for Newton and
   21 for PhysX. These are separate assets preserving the presets' different meshing resolutions.
   Isaac Lab reuses these meshes instead of running fTetWild at scene creation.

Record and Replay
-----------------

Use the same cube asset, physics backend, embodiment, and environment settings for recording
and replay. The commands below select Newton. For PhysX, replace
``deformable_cube_newton`` with ``deformable_cube_physx`` in **both** commands.

Record one successful keyboard demonstration in a window:

.. code-block:: bash

   python submodules/IsaacLab/scripts/tools/record_demos.py \
     --task droid_deformable_pick_and_place \
     --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
     --pick_object deformable_cube_newton \
     --embodiment droid_differential_ik \
     --teleop_device keyboard \
     --viz kit \
     --no-reset_sim_buffer_each_episode \
     --num_demos 1 \
     --dataset_file datasets/deformable_pick_and_place.hdf5

Complete the pick-and-place task to export the demonstration. After recording exits, replay
in a fresh process without a window:

.. code-block:: bash

   python submodules/IsaacLab/scripts/tools/replay_demos.py \
     --task droid_deformable_pick_and_place \
     --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
     --pick_object deformable_cube_newton \
     --embodiment droid_differential_ik \
     --viz none \
     --experience isaaclab.python.kit \
     --validate_success_rate \
     --dataset_file datasets/deformable_pick_and_place.hdf5

The pinned Lab replay script creates a pause/resume keyboard even when running headlessly.
``--experience isaaclab.python.kit`` loads its input modules; ``--viz none`` keeps execution
headless. Use ``--viz kit`` instead to watch the replay. Fixed topology makes restoring the initial
nodal state possible; it does not guarantee numerically identical trajectories or task success.

.. warning::

   Re-record demonstrations made with the earlier procedural cubes. Their randomly generated
   simulation meshes may have different node counts, node ordering, or tetrahedron connectivity.
   A matching node count alone does not make an old dataset compatible. Arena rejects a nodal
   shape mismatch before resetting the environment, with an error naming the asset and the
   recorded and required shapes. Equal-sized topology differences cannot be detected from the
   recorded nodal state alone. Do not replace or regenerate a mesh between recording and replay,
   and do not replay a Newton recording using the PhysX cube or vice versa.

Bake a Custom Volume Deformable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For other volume deformables, Isaac Lab calls fTetWild if the source contains only a triangle
surface mesh. Its current Python wrapper exposes no random seed option, and seeding the
environment does not establish repeatable meshing. Generate the mesh once and keep the USD
alongside the recordings instead.

The baking tool accepts a registered Arena volume deformable. For example, to freeze the
Teddy Bear asset's simulation and visual geometry:

.. code-block:: bash

   python isaaclab_arena/scripts/bake_deformable_mesh.py \
     --asset deformable_teddy_bear_newton \
     --output /absolute/path/to/teddy_fixed.usda

The tool runs headlessly, saves one simulation ``UsdGeom.TetMesh`` and one visual
``UsdGeom.Mesh``, and refuses to overwrite an existing file. Baking a procedural asset twice
can produce different meshes; reuse the first output for all subsequent launches. The file
contains geometry in meters, with source transforms applied to its vertices. Do not apply the
original source scale again. Backend physics properties and materials are configured at spawn time.

In your custom registered ``DeformableObject`` definition, load the baked file with
``isaaclab.sim.UsdFileCfg(usd_path="/absolute/path/to/teddy_fixed.usda", ...)``.
Keep the same backend-specific ``deformable_props``, ``collision_props``, and
``physics_material`` settings. The ``DeformableCubeNewton`` and ``DeformableCubePhysx``
definitions in ``isaaclab_arena/assets/deformable_object_library.py`` show this configuration.
Register and select that asset before recording, then use the same definition for replay.
Package the generated USD with your custom asset, as the two cube presets do.

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
           droid_deformable_pick_and_place \
           --pick_object deformable_cube_newton

      A local 16-episode OpenPI ``pi05`` validation run succeeded in 9 episodes (56.25%).
      This small stochastic run validates the workflow.

      .. code-block:: text

         [Rank 0/1] Metrics: {'num_episodes': 16, 'success_rate': 0.5625}

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

      A local 16-episode OpenPI ``pi05`` validation run succeeded in 15 episodes (93.75%).
      This small stochastic run validates the workflow.

      .. code-block:: text

         [Rank 0/1] Metrics: {'num_episodes': 16, 'success_rate': 0.9375}

The following rollout shows one successful Newton episode from an external camera and the
robot wrist camera:

.. image:: ../../../images/deformable_pick_and_place_openpi.gif
   :align: center
   :alt: Successful Newton deformable pick-and-place rollout from external and wrist cameras
