RL Tasks
========

RL tasks add the components Isaac Lab's reinforcement-learning training scripts
require: a command manager that samples a new goal each episode, reward terms,
and goal-conditioned observations.

An RL task is a ``TaskBase`` subclass that defines the usual task parts (scene config,
termination conditions, metrics) and also overrides the RL hooks that ``TaskBase``
leaves empty:

.. code-block:: python

   class LiftObjectTaskRL(TaskBase):

       def get_observation_cfg(self):
           pass

       def get_rewards_cfg(self):
           pass

       def get_commands_cfg(self):
           pass

Usage
-----

.. code-block:: python

   lift_object = asset_registry.get_asset_by_name("cracker_box")()

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
