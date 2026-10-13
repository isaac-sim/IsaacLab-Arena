Open a laptop and move a cup
===========================

The ``laptop_cup`` environment recreates a tabletop sequence: open the laptop,
then lift the cup from its left and place it upright on its right. It uses a
Franka arm with relative end-effector pose control (``franka_ik``).

The scene is an approximate reconstruction from a reference video. Its dimensions,
materials, camera, and lighting are chosen to resemble the scene; they are not
camera-calibrated measurements. The laptop is 34 cm wide and 24 cm deep, the cup
is 9.5 cm tall, and the tabletop is 74 cm above the floor.

Inspect the physical scene
--------------------------

Run these commands as the host user inside the running Arena container, from
``/workspaces/isaaclab_arena``. The interactive runner requires the Kit GUI and
CPU PhysX:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/environment_runner.py \
     laptop_cup --device cpu --viz kit --num_envs 1

Hold Shift and left-drag to interact with the laptop lid or cup. The laptop has
a fixed base and a passive revolute hinge with a 0–120 degree range. The cup is
a dynamic rigid body with a hollow interior and handle collision geometry.
The wooden table is a kinematic support surface. Reset closes the lid and
returns the cup to the left side.

Assets and textures are generated locally on first use and cached under
Arena's asset cache. No reference video or generated binary assets are stored
in the repository.

Run the reference demonstration
-------------------------------

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config isaaclab_arena_environments/experiment_configs/laptop_cup_experiment.yaml \
     --viz kit

The Experiment runs two checks, each with one environment and one episode:

* ``stationary_franka`` applies zero actions for two seconds. It should time
  out without completing either subtask.
* ``reference_scene_replay`` uses ``laptop_cup_scripted_replay`` to write the
  laptop hinge and cup states directly. It opens the lid to 105 degrees, pauses,
  lifts the cup 15 cm, and carries it around the front of the laptop. At about
  11 seconds it releases the cup and stops writing its state, allowing gravity
  and contact to settle it on the table. The Franka remains stationary.

This replay demonstrates the scene sequence and task reporting. It is not a
learned policy, a grasp demonstration, or evidence that the Franka can complete
the manipulation. A robot controller or recorded robot-action demonstration
must be supplied separately to evaluate physical manipulation.

Use ``--viz none`` for an unattended run. Cameras are disabled by default.
To record the fixed table view and the Franka wrist view, enable the environment's
cameras and request camera recording:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config isaaclab_arena_environments/experiment_configs/laptop_cup_experiment.yaml \
     --viz none --record_camera_video \
     shared.environment.enable_cameras=true

The runner enables rendering for camera recording automatically and writes
videos for ``table_cam`` and ``wrist_cam``. It creates a timestamped directory
beneath ``outputs/`` containing ``arena_experiment_result.json``, ``index.html``,
and per-Run episode results. The reference Run declares
``execution_kind: object_state_replay``. Its report shows **Scene checks passed**
and **Robot execution not evaluated**, preserves videos and progress events, and
excludes its episodes from robot-policy success and progress comparisons.
The ``stationary_franka`` Run remains a robot-action evaluation.

Success conditions
------------------

``LaptopCupTask`` composes Arena's ``OpenDoorTask`` and a specialized
``PickAndPlaceTask`` in order. With the default configuration, completion
requires:

#. Move the lid from its closed reset state and open it past 90 degrees.
#. After opening, lift the cup at least 6 cm above its recorded resting height.
#. Place the cup within 5.5 cm horizontally of its destination, supported by the
   table, within 10 degrees of upright, and below the linear and angular speed
   limits for ten consecutive control steps.

The laptop must still be open when the cup placement completes. Dropping the cup
below the table's configured failure height ends the episode unsuccessfully.
Ordered milestones, contact sensing, settling, and per-environment reset state
use Arena's existing task and progress components.

Coordinates and configuration
------------------------------

The laptop base defines the destination frame: +X points toward its hinge, +Y
points to the seated user's left, and +Z points upward. The default cup target
is ``(-0.02, -0.28, 0.0)`` meters in that frame. Its horizontal location determines
task placement; physical contact determines support height.

The environment's ``cup_destination_offset_xyz`` and the replay policy's
``goal_offset_xyz`` must match when changing the destination. For example, append
both overrides to the Experiment command to move the target farther right:

.. code-block:: bash

   'shared.environment.cup_destination_offset_xyz=[-0.02,-0.32,0.0]' \
   'runs.reference_scene_replay.policy.goal_offset_xyz=[-0.02,-0.32,0.0]'

Keep the destination clear of the laptop and on the table. The replay assumes
the supplied level laptop, upright cup, and Franka IK action configuration.
Changing the layout or task thresholds may require changing its path, opening
angle, lift height, or timing.

The implementation is in ``isaaclab_arena_environments/laptop_cup_environment.py``,
``laptop_cup_task.py``, ``laptop_cup_replay_policy.py``, and
``assets/laptop_cup/generate_assets.py``. Execution settings are in
``experiment_configs/laptop_cup_experiment.yaml``.
