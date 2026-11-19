Multimodal VLM as a Policy
==========================

Recent work [1]_ has shown that multimodal vision-language models (VLMs), such as Astra, can act as robot policies.
Arena ``v0.4`` adds support for VLM policies, allowing users to configure different models and
benchmark their action strategies using Arena's evaluation tools.

Try it with the :doc:`VLM policy quickstart <../quickstart/running_a_real_policy/vlm_agent>`.

.. video:: ../../images/whats_new/vlm_policy_astra_rollout.mp4
   :width: 60%
   :align: center
   :caption: GPT-6 Astra controlling a robot to stack colored blocks, seen from the wrist camera.
   :alt: Real-robot rollout of GPT-6 Astra stacking colored blocks, shown from the wrist camera
   :autoplay:
   :loop:
   :muted:
   :playsinline:
   :nocontrols:

How Arena connects a VLM to the robot
-------------------------------------

To use a VLM as a policy, Arena connects the model output to a robot controller.
Each request includes the task description and configured observations, such as camera images
and robot state, along with robot geometry.
The model then returns an action to be executed on the robot.
The Astranomy study [1]_ compares several approaches to using VLMs as robot policies.
Arena supports the following configuration options:

.. table::
   :class: arena-value-table
   :widths: 25 75

   +----------------+--------------------------------------------------------------------------------------------+
   | Design choice  | Arena                                                                                      |
   +================+============================================================================================+
   | Policy form    | **Direct Policy**: observation-to-action API calls, with no persistent conversation.       |
   +----------------+--------------------------------------------------------------------------------------------+
   | Actions        | **Absolute end-effector poses**, plus a command to open or close the gripper.              |
   +----------------+--------------------------------------------------------------------------------------------+
   | Action horizon | **Goal** (``VLMGoalPolicy``): one end-effector goal per call, tracked until it is reached. |
   +                +--------------------------------------------------------------------------------------------+
   |                | **Chunk** (``VLMChunkPolicy``): a fixed number of poses per call, set by ``chunk_size``.   |
   +                +--------------------------------------------------------------------------------------------+
   |                | **Step** (``VLMChunkPolicy`` with ``chunk_size: 1``): one pose per call.                   |
   +----------------+--------------------------------------------------------------------------------------------+
   | Observations   | Camera images, robot state, or both.                                                       |
   +----------------+--------------------------------------------------------------------------------------------+
   | History        | Current observation only. The goal policy also reports the outcome of the previous goal.   |
   +----------------+--------------------------------------------------------------------------------------------+

The action strategies determine how often the model receives updated observations and issues new actions.
The video below compares the ``goal``, ``chunk``, and ``step`` action strategies on a DROID
pick-and-place task, with three trials on each of three layouts.

.. video:: ../../images/whats_new/arena_v04_astra_pnp_eval_mosaic_3trials.mp4
   :width: 100%
   :alt: Grid of DROID pick-and-place rollouts with the goal, chunk, and step VLM policies, three trials on each of three layouts
   :autoplay:
   :loop:
   :muted:
   :playsinline:
   :nocontrols:


References
----------

.. [1] J. X. Yang, A. Khetan, and S. Zhang. `Astranomy: An empirical study of GPT-6 Astra as a
   native robot policy <https://research.nvidia.com/labs/srl/projects/astranomy/>`__. CoRL 2026
   Late-Breaking Results.
