Why is Isaac Lab Arena needed?
===============================

Do we need a framework for building and running task libraries?


The Problem
===========

With the advent of *generalist* robot policies, such as `GR00T <https://github.com/NVIDIA/Isaac-GR00T>`_
and `Pi_0 <https://github.com/Physical-Intelligence/openpi>`_,
there is a growing need to evaluate these policies in a variety of tasks/environments.

Traditional approaches to building task libraries suffer from significant limitations.
Each new environment variation—whether testing a different robot embodiment or swapping objects—requires tedious manual creation of a new task configuration.
This leads to redundant, unscalable tasks where most of the environment setup, scene configuration, and task logic is duplicated across variations.
As the number of robot types and objects grows, maintaining and extending such task libraries becomes increasingly impractical.


.. figure:: ../../images/task_duplications.png
   :width: 100%
   :alt: Task duplications in a task library
   :align: center

   Task duplications in a task library.
   When evaluating policies across different robot embodiments and objects, most of the environment setup and task logic remains the same, leading to significant code duplication.



Can we simplify environment creation?
=====================================

.. figure:: ../../images/variation_axis.png
   :width: 100%
   :alt: Axis of variation for the pick and place task
   :align: center

   Axis of variation of a pick and place task.
   Each environment differs along two axis, the robot embodiment and the object to be manipulated.
   All other aspects of the environment and the task remain the same.


Tasks in a task library are typically highly redundant.
For example, you may want to test how well a policy performs on a pick and place task,
on many different objects.
In this example, each environment differs in the object-to-be-manipulated,
but all other aspects remain the same.
For example, the scene layout, the robot, the observations, actions, rewards, etc are all
conserved across the environments.
Isaac Lab's manager-based environment API is convenient for expressing one such task,
but does not naturally support expressing this type of variation.

Isaac Lab Arena extends the manager-based interface to provide
a convenient way of expressing task variation, while benefiting from
the modularity, performance, and accuracy of Isaac Lab.


Isaac Lab Arena
===============

Isaac Lab Arena is a framework that simplifies the creation and maintenance of such task/environment libraries.
To simplify the expression of task/environment variation in Isaac Lab Arena,
we *compose* the environment on-the-fly from independent sub-pieces.
Because the sub-pieces are independent, they can be reused and independently varied.
Furthermore, because the environment is built on the fly, we never need to write and maintain
duplicate code.

.. figure:: ../../images/isaac_lab_arena_arch_overview.png
   :width: 100%
   :alt: Isaac Lab Arena Architecture Overview
   :align: center

Isaac Lab Arena decomposes the environment into three independent sub-pieces:

* **Scene**: The physical environment layout. The scene is a collection of objects.
* **Embodiment**: The robot embodiment, its observations, actions, sensors etc.
* **Task**: A definition of what is to be accomplished in the environment.

The ``ArenaEnvBuilder`` composes the environment from these sub-pieces,
into a ``ManagerBasedRLEnvCfg`` which can be run in Isaac Lab.
