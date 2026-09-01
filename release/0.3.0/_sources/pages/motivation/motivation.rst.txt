Isaac Lab Arena Overview
===============================

Generalist robot policies such as `GR00T <https://github.com/NVIDIA/Isaac-GR00T>`_
and `Pi_0 <https://github.com/Physical-Intelligence/openpi>`_ aim to solve many tasks across many scenes,
objects, embodiments, and deployment conditions.
Evaluating that claim requires more than a fixed benchmark suite. Lighting changes, clutter, object
substitutions, and robot morphology can all change policy behavior, and limited coverage can reward policies
tuned to benchmark-specific conditions rather than policies that generalize.
To address this, the evaluation set needed to test generalization grows combinatorially with tasks, scenes,
embodiments, objects, and environment variations, making manual environment authoring a central bottleneck.
Isaac Lab Arena provides a composable robotics simulation evaluation approach.

.. figure:: ../../images/task_duplications.png
   :width: 100%
   :alt: Task duplications in a task library
   :align: center

   Task duplications in a task library.
   When evaluating policies across different robot embodiments and objects, most of the environment setup and task logic remains the same, leading to significant code duplication.

Variational Approach to Robot Policy Evaluation
===============================================

.. figure:: ../../images/variation_axis.png
   :width: 100%
   :alt: Axis of variation for the pick and place task
   :align: center

   Axis of variation of a pick and place task.
   Each environment differs along two axes: the robot embodiment and the object to be manipulated.
   All other aspects of the environment and the task remain the same.


Task libraries often encode the same scenario repeatedly. A pick-and-place
benchmark, for example, may need to evaluate the same task across many target
objects, robot embodiments, object placements, or scene conditions. In these
cases, the core task definition does not change: the scene layout, observation
space, action space, rewards, and success criteria are mostly shared. Only one
or two dimensions vary.

Isaac Lab's manager-based environment API is convenient for expressing each
individual task, but representing a full family of related variations can lead
to many near-duplicate configurations. Isaac Lab Arena makes those differences
explicit as variation axes. It extends the manager-based interface so related
environments can be composed from reusable parts, while retaining the
modularity, performance, and accuracy of Isaac Lab.

Concretely, Isaac Lab Arena treats an environment as a composition of reusable
pieces instead of a standalone configuration for every variation. The shared
parts stay in one place, while the axes that should vary, such as the scene,
robot embodiment, or task, can be swapped independently. The environment is
assembled on the fly, which avoids maintaining duplicate task code for each new
combination.

.. figure:: ../../images/isaac_lab_arena_arch_overview.png
   :width: 100%
   :alt: Isaac Lab Arena Architecture Overview
   :align: center

Isaac Lab Arena decomposes each environment into three independent pieces:

* **Scene**: The physical layout and objects in the environment.
* **Embodiment**: The robot, its observations, actions, and sensors.
* **Task**: The objective, rewards, success criteria, and task-specific logic.

The ``ArenaEnvBuilder`` composes these pieces into a ``ManagerBasedRLEnvCfg``
that can be run in Isaac Lab.
