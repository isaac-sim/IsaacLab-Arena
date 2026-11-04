Multimodal VLM as a Policy
==========================

Recent work [1]_ has shown that multimodal vision-language models (VLMs), for example Astra,
can act as a robot policy by connecting the model output to a robot controller.
Arena users can now adopt and benchmark this new approach to robot control.
The Arena ``v0.4`` release implements VLM-as-a-policy, which can be configured to use various VLM models.

Try it! Run a :doc:`../quickstart/running_a_real_policy/vlm_agent` in Arena!

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
At each request, the VLM is fed model inputs including the task description,
the robot state, the robot geometry, and the current camera images.
The model then returns an action to be executed on the robot.
The Astranomy study [1]_ compares several ways to use VLMs as robot policies:
Arena implements several of these approaches.

.. table::
   :class: arena-value-table
   :widths: 25 75

   +----------------+--------------------------------------------------------------------------------------------+
   | Design choice  | Arena                                                                                      |
   +================+============================================================================================+
   | Policy form    | **Direct Policy**: observation-to-action API calls, with no persistent conversation.       |
   +----------------+--------------------------------------------------------------------------------------------+
   | Actions        | **Absolute end-effector poses**, plus a gripper open or close command.                     |
   +----------------+--------------------------------------------------------------------------------------------+
   | Action horizon | **Goal** (``VLMGoalPolicy``): one end-effector goal per call, tracked until it is reached. |
   +                +--------------------------------------------------------------------------------------------+
   |                | **Chunk** (``VLMChunkPolicy``): a fixed number of poses per call, set by ``chunk_size``.   |
   +                +--------------------------------------------------------------------------------------------+
   |                | **Step** (``VLMStepPolicy``): a chunk of one pose per call.                                |
   +----------------+--------------------------------------------------------------------------------------------+
   | Observations   | Camera images, robot state, or both.                                                       |
   +----------------+--------------------------------------------------------------------------------------------+
   | History        | Current observation only. The goal policy also reports the outcome of the previous goal.   |
   +----------------+--------------------------------------------------------------------------------------------+

The video below compares several of these options in Arena, varying between the
``goal``, ``chunk``, and ``step`` action strategies on a DROID pick-and-place task showing three trials.
Try this on your own in Arena in our :doc:`../quickstart/running_a_real_policy/vlm_agent` workflow.

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
