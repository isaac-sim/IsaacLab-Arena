Environment Setup and Validation
---------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^

The ``dexsuite_lift`` Arena environment wraps the Isaac Lab
``Isaac-Lift-KukaAllegro`` MDP for evaluation.
The physics backend defaults to Newton. Pass ``--presets physx`` to override
that environment default.

The environment is defined in
``isaaclab_arena_environments/dexsuite_lift_environment.py``:

.. dropdown:: The Dexsuite Lift Environment
   :animate: fade-in

   .. code-block:: python

      class DexsuiteLiftEnvironment(ArenaEnvironmentFactory):

          name: str = "dexsuite_lift"

          def build(self, cfg):
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

              return IsaacLabArenaEnvironment(
                  name=self.name,
                  embodiments=[embodiment],
                  scene=scene,
                  task=task,
                  rl_framework_entry_point="rsl_rl_cfg_entry_point",
                  rl_policy_cfg=DEXSUITE_RSL_RL_CFG,
                  default_physics_backend=PhysicsBackend.NEWTON,
                  env_cfg_callback=_match_isaac_lab_lift_cfg,
              )

.. note::

   The environment declares Newton as its default backend. The common
   ``--presets`` CLI flag can override that default.


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

- **Events**: Arena resets the procedural cube from its configured
  ``PoseRange`` and resets the Kuka-Allegro embodiment to its default state.

**2. Scene and Task**

.. code-block:: python

   scene = Scene(assets=[dexsuite_table, manip_object, ground_plane, light])
   task = DexsuiteLiftTask(lift_object=manip_object, background_scene=dexsuite_table)

The scene uses Arena's procedural table and cube. The cube's ``PoseRange``
generates its pose-reset event; no Isaac Lab conditional reset bank is used.
``DexsuiteLiftTask`` defines the policy command and evaluation termination
settings. Evaluation omits rewards and curriculum. The position-only
``object_pose`` target is regenerated every 4–6 seconds, episodes last 12
seconds, and success requires the object position to be within 5 cm of the
commanded target.

**3. Physics Backend Selection**

The physics backend is selected by ``ArenaEnvBuilder``:

- **Default (Newton)**: no extra flag needed.
- **PhysX override**: pass ``--presets physx`` to ``policy_runner.py``.

When Newton is resolved, the environment callback:

1. Applies Isaac Lab's ``PhysicsCfg.newton_mjwarp`` solver configuration.
2. Uses a 1/120-second simulation step and decimation of 4 (30 Hz control).
3. Enables ``scene.replicate_physics = True`` (required by Newton).


Validation: Run Zero-Action Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Verify the environment loads correctly with a zero-action policy:

.. code-block:: bash

   # PhysX override:
   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --presets physx \
     --policy_type zero_action \
     --num_steps 100 \
     dexsuite_lift

   # Newton (environment default):
   PYOPENGL_PLATFORM=glx python isaaclab_arena/evaluation/policy_runner.py \
     --viz newton_gl \
     --policy_type zero_action \
     --num_steps 100 \
     dexsuite_lift

You should see the Kuka Allegro hand with Arena's procedural cuboid.

.. tip::

   ``--viz newton_gl`` uses the MuJoCo viewer; ``--viz kit`` uses
   the Kit viewer. The visualizer setting is independent of the physics backend.
   For example, ``--viz kit --presets newton`` runs Newton physics with
   the Kit viewer.

   On Linux, set ``PYOPENGL_PLATFORM=glx`` before starting Python with the
   interactive Newton viewer. This avoids a PyOpenGL context initialization
   failure.
