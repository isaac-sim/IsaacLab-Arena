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

   The cube in this example uses ``MeshCuboidCfg``, which initially authors only a triangle
   surface mesh. A volume deformable also needs a tetrahedral simulation mesh, so Isaac Lab
   calls fTetWild during scene creation when it cannot find an existing ``UsdGeom.TetMesh``.
   This on-the-fly conversion is nondeterministic between process launches and may produce
   different node counts or connectivity. Demonstration datasets include the initial nodal
   state, which can be restored only into the same topology. Seeding the environment does
   not make this conversion repeatable: the current ``pytetwild`` wrapper exposes no random
   seed parameter.

   Before recording demonstrations, bake the desired cube **once** into a USD containing a
   ``UsdGeom.TetMesh``. Run the command for your backend from the repository root in your
   activated Arena runtime, before changing the object library:

   **Newton**

   .. code-block:: bash

      python isaaclab_arena/assets/preprocessing/bake_deformable_mesh.py \
        --asset deformable_cube_newton \
        --output /absolute/path/to/deformable_cube_newton.usda

   **PhysX**

   .. code-block:: bash

      python isaaclab_arena/assets/preprocessing/bake_deformable_mesh.py \
        --asset deformable_cube_physx \
        --output /absolute/path/to/deformable_cube_physx.usda

   Choose a persistent output location available to every recording and replay process.
   The script runs headlessly and refuses to overwrite an existing file. It saves the
   simulation TetMesh and visual Mesh without backend physics schemas or materials.
   The destination is created atomically only after the temporary USD exports successfully,
   so a failed or interrupted export does not leave an incomplete bake at that path.
   Generating another file can produce a different topology, so keep and reuse the first
   output. Generated USDs are local assets and are not included in this repository.

   Next, edit ``isaaclab_arena/assets/deformable_object_library.py`` to load that file.
   In ``DeformableCubeNewton``, change the start of ``spawner_cfg`` as follows:

   .. code-block:: diff

      -    spawner_cfg = sim_utils.MeshCuboidCfg(
      -        size=(0.15, 0.04, 0.04),
      -        edge_refinement=8.0,
      +    spawner_cfg = sim_utils.UsdFileCfg(
      +        usd_path="/absolute/path/to/deformable_cube_newton.usda",
               deformable_props=NewtonDeformableBodyPropertiesCfg(),

   For PhysX, make the corresponding change in ``DeformableCubePhysx``:

   .. code-block:: diff

      -    spawner_cfg = sim_utils.MeshCuboidCfg(
      -        size=(0.15, 0.04, 0.04),
      +    spawner_cfg = sim_utils.UsdFileCfg(
      +        usd_path="/absolute/path/to/deformable_cube_physx.usda",
               deformable_props=PhysxDeformableBodyPropertiesCfg(linear_damping=0.0),

   Leave the remaining backend-specific physics, collision, and material settings unchanged.
   Remove procedural meshing options such as ``size`` and ``edge_refinement``; they are not
   ``UsdFileCfg`` arguments. Source transforms are already applied to the baked vertices,
   so do not apply the original source scale again when baking other volume assets.

   Record new demonstrations after making this change, then use the same generated USD,
   library definition, and physics backend for every replay. Isaac Lab will reuse the
   TetMesh instead of calling fTetWild. Earlier procedural recordings must be recreated:
   even matching node counts do not guarantee matching node ordering or connectivity.
   Fixed topology permits restoring nodal state; it does not guarantee numerically
   identical trajectories or task success.

   The script also accepts the registered USD-backed Teddy Bear presets:
   ``deformable_teddy_bear_newton`` and ``deformable_teddy_bear_physx``. For example:

   .. code-block:: bash

      python isaaclab_arena/assets/preprocessing/bake_deformable_mesh.py \
        --asset deformable_teddy_bear_newton \
        --output /absolute/path/to/teddy_fixed.usda

   In ``DeformableTeddyBearNewton``, point the existing ``UsdFileCfg.usd_path`` at
   this output and remove ``scale=(0.01, 0.01, 0.01)`` because baking already applies
   that scale. Follow the same procedure for ``DeformableTeddyBearPhysx`` with its
   own output file. The tool accepts registry names rather than raw USD paths and
   requires one volume simulation TetMesh and one visual Mesh after spawning.
   Surface deformables such as the cloth presets are not supported. Only geometry
   is saved; source materials and textures are not copied.

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
