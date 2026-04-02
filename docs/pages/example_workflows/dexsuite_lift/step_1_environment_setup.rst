Environment Setup and Validation
---------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^

The ``dexsuite_lift`` Arena environment wraps the Isaac Lab
``Isaac-Dexsuite-Kuka-Allegro-Lift-v0`` MDP for evaluation under **Newton** physics.
The environment is defined in
``isaaclab_arena_environments/dexsuite_lift_environment.py``:

.. dropdown:: The Dexsuite Lift Environment
   :animate: fade-in

   .. code-block:: python

      class DexsuiteLiftEnvironment(ExampleEnvironmentBase):

          name: str = "dexsuite_lift"

          def get_env(self, args_cli: argparse.Namespace):
              import isaaclab_tasks.manager_based.manipulation.dexsuite  # noqa: F401

              from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
              from isaaclab_arena.reinforcement_learning.frameworks import RLFramework
              from isaaclab_arena.scene.scene import Scene
              from isaaclab_arena.tasks.lift_object_task import DexsuiteLiftTask

              dexsuite_table = self.asset_registry.get_asset_by_name("dexsuite_manip_table")()
              manip_object = self.asset_registry.get_asset_by_name("dexsuite_lift_object")()
              ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
              light = self.asset_registry.get_asset_by_name("light")()

              embodiment = self.asset_registry.get_asset_by_name("kuka_allegro")()

              scene = Scene(assets=[dexsuite_table, manip_object, ground_plane, light])
              task = DexsuiteLiftTask(lift_object=manip_object, background_scene=dexsuite_table)

              dexsuite_rl_cfg_entry = (
                  "isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.agents."
                  "rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg"
              )

              def _apply_dexsuite_cfg(env_cfg):
                  from isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg import (
                      KukaAllegroPhysicsCfg,
                  )
                  from isaaclab_tasks.manager_based.manipulation.dexsuite.dexsuite_env_cfg import EventCfg

                  env_cfg.sim.physics = KukaAllegroPhysicsCfg().newton
                  env_cfg.sim.dt = 1 / 120
                  env_cfg.decimation = 2
                  env_cfg.episode_length_s = 6.0
                  env_cfg.is_finite_horizon = False
                  env_cfg.events = EventCfg()
                  if hasattr(env_cfg, "scene") and env_cfg.scene is not None:
                      env_cfg.scene.replicate_physics = True
                  return env_cfg

              return IsaacLabArenaEnvironment(
                  name=self.name,
                  embodiment=embodiment,
                  scene=scene,
                  task=task,
                  teleop_device=None,
                  rl_framework=RLFramework.RSL_RL,
                  rl_policy_cfg=dexsuite_rl_cfg_entry,
                  env_cfg_callback=_apply_dexsuite_cfg,
              )


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

**2. Scene and Task**

.. code-block:: python

   scene = Scene(assets=[dexsuite_table, manip_object, ground_plane, light])
   task = DexsuiteLiftTask(lift_object=manip_object, background_scene=dexsuite_table)

``DexsuiteLiftTask`` is evaluation-only: no rewards, no curriculum.
It provides the ``object_pose`` command (position-only, resampled every 2–3 s),
a success termination at 5 cm position tolerance, and a time-out termination.

**3. Newton Physics via Environment Callback**

.. code-block:: python

   env_cfg.sim.physics = KukaAllegroPhysicsCfg().newton
   env_cfg.events = EventCfg()

The ``_apply_dexsuite_cfg`` callback switches the physics backend to **Newton**,
sets simulation rates (120 Hz physics, decimation 2 = 60 Hz control), and installs
reset-mode events (gravity scheduling, object/robot randomization). PhysX-only startup
events are excluded.


Validation: Run Zero-Action Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Verify the environment loads correctly under Newton with a zero-action policy:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --num_steps 100 \
     --visualizer newton \
     dexsuite_lift

You should see the Kuka Allegro hand in the scene with the cuboid on the table.
The robot will remain stationary (zero actions), confirming that the Newton physics
backend and scene load without errors.
