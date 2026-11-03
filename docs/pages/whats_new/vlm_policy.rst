Multimodal VLM as a Policy
==========================

Recent work [1]_ has shown that multimodal vision-language models (VLMs), for example Astra [2]_,
can act as a robot policy.
At each request, the VLM is fed model inputs including the task, the robot state, the robot geometry, and the
current camera images.
The model then returns an action to be executed on the robot.
NVIDIA has conducted an in-depth study of the VLMs as robot policies in [1]_.

Arena users can now adopt and benchmark this new approach to robot control.
The Arena ``v0.4`` release implements VLM-as-a-policy and
Arena environments can now be configured to use this approach with various VLM models.

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

The Astranomy study [1]_ compares several ways to use VLM models as robot policies:
Arena implements the several of these approaches.

.. table::
   :class: arena-value-table
   :widths: 25 75

   +----------------+--------------------------------------------------------------------------------------------+
   | Design choice  | Arena                                                                                      |
   +================+============================================================================================+
   | Policy form    | Direct Policy: observation-to-action API calls, with no persistent conversation.           |
   +----------------+--------------------------------------------------------------------------------------------+
   | Actions        | Absolute end-effector poses, plus a gripper open or close command.                         |
   +----------------+--------------------------------------------------------------------------------------------+
   | Action horizon | **Goal** (``VLMGoalPolicy``): one end-effector goal per call, tracked until it is reached. |
   +                +--------------------------------------------------------------------------------------------+
   |                | **Chunk** (``VLMChunkPolicy``): a fixed number of poses per call, set by ``chunk_size``.   |
   +                +--------------------------------------------------------------------------------------------+
   |                | **Step**: a chunk of one pose per call.                                                    |
   +----------------+--------------------------------------------------------------------------------------------+
   | Observations   | Camera images, robot state, or both.                                                       |
   +----------------+--------------------------------------------------------------------------------------------+
   | History        | Current observation only. The goal policy also reports the outcome of the previous goal.   |
   +----------------+--------------------------------------------------------------------------------------------+

The video below compares several of these options in Arena, varying the between
goal, chunk, and step action strategies on a DROID pick-and-place task showing three trials.

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
