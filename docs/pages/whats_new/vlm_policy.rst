Multimodal VLM as a Policy
==========================

A multimodal vision-language model (VLM), for example Astra, can act as an Arena policy without
any robot training. At each request, Arena sends the model the task, the robot state, and the
current camera images. The model returns a JSON command, and Arena checks it and executes it on
the robot. The model is called through an OpenAI-compatible inference endpoint, so you do not need
to run a model server.

What's included
---------------

- **Policies in** ``isaaclab_arena_vlm_agent``: goal, chunk, and step execution strategies, each
  trading off the number of VLM calls against how much of the motion the model controls.
- **DROID pick-and-place examples:** ready-to-run experiment configurations for the goal and
  chunk strategies.
- **Extensible adapters:** connect your own robot or execution strategy through observation and
  action adapters.

Try it
------

- :doc:`../quickstart/running_a_real_policy/vlm_agent`: run the VLM policies on the DROID
  pick-and-place task and connect your own robot.
- :doc:`../concepts/policy/index`: how policies plug into the evaluation pipeline.

For the full list of changes in each release, see :doc:`../references/release_notes`.
