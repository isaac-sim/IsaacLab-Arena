Concept Overview
================

This section provides an overview of the core concepts in Isaac Lab Arena.

.. figure:: ../../images/isaaclab_arena_core_framework.png
   :width: 100%
   :alt: Isaac Lab Arena Workflow
   :align: center

   IsaacLab Arena Workflow Overview

In the core of the workflow, we have three main components, **Scene**, **Embodiment**, and **Task**. We compile most of the managers of the
manager-based RL environment from these three components. We strongly incline towards keeping these components as independent as possible. This
allows us to reuse the components in different environments and tasks, thus making the framework more modular and easier to extend.

**Embodiment**

Embodiments define robot-specific configurations and behaviors. They provide a modular way to integrate different robots into environments,
encapsulating kinematics, control actions, observations, terminations and camera systems. See :doc:`./concept_embodiment_design` for more details.

**Task**

Tasks define objectives, success criteria, and behavior logic for environments. They provide configurations for termination conditions, event handling,
metrics collection, and mimic components. See :doc:`./concept_tasks_design` for more details.

**Scene**

Scenes manage collections of assets that define the physical environment for simulation. They provide a unified interface for composing backgrounds,
objects, and interactive elements. See :doc:`./concept_scene_design` for more details.

When combining these three components we create the observation, action, event, termination, metrics, mimic components of the manager-based RL environment.
For more details on how to combine these components, see :doc:`./concept_environment_compilation`.

Other components of interest are the **Affordances** and the **Metrics**.

**Affordances**

Affordances define what interactions objects can perform - opening doors, pressing buttons, manipulating objects.
They provide standardized interfaces that integrate with tasks and embodiments. See :doc:`./concept_affordances_design` for more details.

**Metrics**

Metrics define the performance evaluation metrics for the environment.
Some metrics are independent of the task and the embodiment, such as the success rate metric,
while others are task-specific, such as open door rate metric. See :doc:`./concept_metrics_design` for more details.

These components together with teleoperation devices form the manager-based RL environment.
See :doc:`./concept_environment_design` for more details on how these components are easily combined to create our environments.
