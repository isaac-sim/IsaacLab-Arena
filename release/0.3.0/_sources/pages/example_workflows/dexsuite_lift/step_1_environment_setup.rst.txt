Environment Setup and Validation
---------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^

The ``dexsuite_lift`` Arena environment wraps the Isaac Lab
``Isaac-Dexsuite-Kuka-Allegro-Lift-v0`` MDP for evaluation.
The physics backend defaults to PhysX and can be switched to Newton by passing
``--presets newton`` on the command line.

The environment is defined in
``isaaclab_arena_environments/dexsuite_lift_environment.py``:

.. dropdown:: The Dexsuite Lift Environment
   :animate: fade-in

   .. code-block:: python

      class DexsuiteLiftEnvironment(ExampleEnvironmentBase):

          name: str = "dexsuite_lift"

          def get_env(self, args_cli: argparse.Namespace):
              import math

              import isaaclab_tasks.manager_based.manipulation.dexsuite  # noqa: F401

              from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
              from isaaclab_arena.scene.scene import Scene
              from isaaclab_arena.tasks.lift_object_task import DexsuiteLiftTask
              from isaaclab_arena.utils.pose import Pose, PoseRange

              dexsuite_table = self.asset_registry.get_asset_by_name("procedural_table")()
              dexsuite_table.set_initial_pose(Pose(position_xyz=(-0.55, 0.0, 0.235)))

              manip_object = self.asset_registry.get_asset_by_name("procedural_cube")()
              manip_object.set_initial_pose(
                  PoseRange(
                      position_xyz_min=(-0.75, -0.1, 0.35),
                      position_xyz_max=(-0.35, 0.3, 0.75),
                      rpy_min=(-math.pi, -math.pi, -math.pi),
                      rpy_max=(math.pi, math.pi, math.pi),
                  )
              )

              ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
              light = self.asset_registry.get_asset_by_name("light")()

              embodiment = self.asset_registry.get_asset_by_name("kuka_allegro")()

              scene = Scene(assets=[dexsuite_table, manip_object, ground_plane, light])
              task = DexsuiteLiftTask(lift_object=manip_object, background_scene=dexsuite_table)

              dexsuite_rl_cfg_entry = (
                  "isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.agents."
                  "rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg"
              )

              return IsaacLabArenaEnvironment(
                  name=self.name,
                  embodiment=embodiment,
                  scene=scene,
                  task=task,
                  teleop_device=None,
                  rl_framework_entry_point="rsl_rl_cfg_entry_point",
                  rl_policy_cfg=dexsuite_rl_cfg_entry,
              )

.. note::

   The environment does not contain a physics-specific callback.
   Physics backend selection is handled globally by the ``--presets`` CLI flag
   (e.g. ``--presets newton``), which is applied by ``ArenaEnvBuilder``
   after all environment-specific configuration.


Step-by-Step Breakdown
^^^^^^^^^^^^^^^^^^^^^^^

**1. Embodiment: Kuka Allegro**

.. code-block:: python

   embodiment = self.asset_registry.get_asset_by_name("kuka_allegro")()

The ``KukaAllegroEmbodiment`` provides:

- **Scene**: Kuka LBR iiwa arm + Allegro Hand articulation, plus four fingertip contact
  sensors (``index_link_3``, ``middle_link_3``, ``ring_link_3``, ``thumb_link_3``).
- **Actions**: Relative joint position control for all 23 joints (``scale=0.1``).
- **Observations** (three groups, each with ``history_length=5``):

  - ``policy``: object quaternion, target pose command, last action.
  - ``proprio``: joint positions, joint velocities, hand-tip body states (palm + fingertips),
    fingertip contact forces.
  - ``perception``: object point cloud (64 points, flattened).

- **Events**: Joint-position randomization on reset (±0.5 rad offset from default).

**2. Scene and Task**

.. code-block:: python

   scene = Scene(assets=[dexsuite_table, manip_object, ground_plane, light])
   task = DexsuiteLiftTask(lift_object=manip_object, background_scene=dexsuite_table)

``DexsuiteLiftTask`` is evaluation-only: no rewards, no curriculum.
It provides the ``object_pose`` command (position-only, resampled every 2–3 s),
a success termination at 5 cm position tolerance, and a time-out termination.

**3. Physics Backend Selection**

The physics backend is selected via the common ``--presets`` CLI flag, handled
by ``ArenaEnvBuilder``:

- **Default (PhysX)**: no extra flag needed.
- **Newton**: pass ``--presets newton`` to ``policy_runner.py``.

When ``--presets newton`` is set, the builder automatically:

1. Applies the ``ArenaPhysicsCfg().newton`` configuration (MuJoCo-Warp solver
   with tuned parameters for dexterous manipulation).
2. Enables ``scene.replicate_physics = True`` (required by Newton).


Validation: Run Zero-Action Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Verify the environment loads correctly with a zero-action policy:

.. code-block:: bash

   # PhysX (default):
   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type zero_action \
     --num_steps 100 \
     dexsuite_lift

   # Newton:
   python isaaclab_arena/evaluation/policy_runner.py \
     --viz newton \
     --presets newton \
     --policy_type zero_action \
     --num_steps 100 \
     dexsuite_lift

You should see the Kuka Allegro hand in the scene with the cuboid on the table.

.. tip::

   ``--viz newton`` uses the MuJoCo viewer; ``--viz kit`` uses
   the Kit viewer. The visualizer setting is independent of the physics backend.
   For example, ``--viz kit --presets newton`` runs Newton physics with
   the Kit viewer.
