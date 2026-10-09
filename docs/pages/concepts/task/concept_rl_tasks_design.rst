RL Tasks
========

RL tasks extend their imitation learning counterparts with the components Isaac Lab's
reinforcement-learning training scripts require: a command manager that samples
a new goal each episode, reward terms, and goal-conditioned observations.

The following schematic illustrates the pattern: an RL task subclasses the
corresponding IL task and implements the RL-specific parts:

.. code-block:: python

   class LiftObjectTaskRL(LiftObjectTask):

       def get_rewards_cfg(self):
          pass

       def get_commands_cfg(self):
          pass

This means the RL task inherits everything from the IL task (scene config,
termination conditions, metrics) and adds the RL-specific parts on top.
Note the the RL task is also likely to *modify* the IL task's configuration.
For example, adding privileged information to the observations.

Usage
-----

Set the object's initial pose before constructing the task, which uses it to
compute the target position ranges.

.. code-block:: python

   from isaaclab_arena.tasks.lift_object_task import LiftObjectTaskRL
   from isaaclab_arena.utils.pose import Pose

   lift_object = asset_registry.get_asset_by_name("cracker_box")()
   lift_object.set_initial_pose(Pose(position_xyz=(0.5, 0, 0.055), rotation_xyzw=(0, 0, 0, 1)))

   task = LiftObjectTaskRL(
       lift_object=lift_object,
       background_scene=table,
       embodiment=embodiment,
   )

``LiftObjectTaskRL`` adds a command manager that samples a random target
position each episode, reward terms for reaching the object, lifting it,
and tracking the goal, and goal-conditioned observations that tell the
policy where the target is.

See :doc:`../../example_workflows/reinforcement_learning/index` for a complete example
for how to use an RL task.

Training vs. evaluation mode
-----------------------------

RL tasks have an ``rl_training_mode`` flag (default ``True``).
During training, success does not terminate the episode — the robot
keeps acting until the time limit. This is standard practice to avoid
sparse termination signals early in training.
For evaluation, set ``rl_training_mode=False`` so episodes end on success:

.. code-block:: python

   # Training
   task = LiftObjectTaskRL(lift_object, table, embodiment, rl_training_mode=True)

   # Evaluation
   task = LiftObjectTaskRL(lift_object, table, embodiment, rl_training_mode=False)
