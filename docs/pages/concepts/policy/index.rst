Policy
======

A policy in Arena is a standard interface between your model and the evaluation
pipeline. You implement one method — ``get_action(env, obs)`` — and the policy
plugs into both the single-job runner and the Experiment Runner without any
changes to either. In bare IsaacLab you would write an ad-hoc inference loop
for each model; Arena's ``PolicyBase`` gives a consistent contract that all
runners depend on.

.. code-block:: python

   policy = ZeroActionPolicy(config=ZeroActionPolicyCfg())
   obs, _ = env.reset()
   action = policy.get_action(env, obs)

Built-in policies
-----------------

Arena ships with four policies:

**ZeroActionPolicy** (``"zero_action"``)
   Returns a zero-filled action tensor. Useful for validating an environment
   without a trained model.

**ReplayActionPolicy** (``"replay"``)
   Replays actions from a recorded episode stored in an HDF5 file.

**RslRlActionPolicy** (``"rsl_rl"``)
   Runs inference with a trained RSL-RL checkpoint. Loads the checkpoint and
   its accompanying ``params/agent.yaml`` automatically.


Writing a custom policy
-----------------------

Define a typed ``PolicyCfg``, subclass ``PolicyBase`` with that config, set a
``name``, register it with its config, and implement ``get_action``:

.. code-block:: python

   from dataclasses import dataclass

   import gymnasium as gym
   import torch
   from gymnasium.spaces.dict import Dict as GymSpacesDict

   from isaaclab_arena.assets.register import register_policy
   from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


   @dataclass
   class MyPolicyCfg(PolicyCfg):
       device: str = "cuda:0"


   @register_policy
   class MyPolicy(PolicyBase[MyPolicyCfg]):
       name = "my_policy"

       def __init__(self, config: MyPolicyCfg):
           super().__init__(config)

       def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
           # Your model inference here
           return torch.zeros(env.action_space.shape, device=torch.device(env.unwrapped.device))

Construct the policy by passing its typed configuration directly:

.. code-block:: python

   policy_cfg = MyPolicyCfg(device="cuda:0")
   policy = MyPolicy(policy_cfg)

The typed registration lets the single-job runner generate CLI flags from
``MyPolicyCfg`` and lets the Experiment Runner convert the current
``Job.policy_config_dict`` representation into that same type. See
:doc:`Arena Experiments <../concept_arena_experiments>` for details.

Config fields named ``device`` or ``num_envs`` reuse the corresponding shared
runner flags, so their defaults must match the runner defaults.

.. note::

   ``policy_runner.py`` remains an argparse frontend, but policies do not
   implement argparse methods. The runner generates their flags from the
   registered config and reconstructs it before creating the policy.

.. _vlm-agent-policies:

VLM agent policies
------------------

``VLMGoalPolicy`` and ``VLMChunkPolicy`` in the ``isaaclab_arena_vlm_agent`` package use a hosted
vision-language model as the policy. At each request, the policy sends the task, the robot state,
and the camera images to an OpenAI-compatible endpoint, then validates the JSON command it
returns and executes it. :doc:`../../quickstart/running_a_real_policy/vlm_agent` shows how to set
up an endpoint and run them on DROID pick-and-place.

What is inside
~~~~~~~~~~~~~~

Each request runs through these stages:

.. code-block:: text

   observation ─► observation adapter ─► images + JSON state ─┐
                                                              ├─► inference backend ─► JSON response
   embodiment preset ─► system prompt + response schema ──────┘                             │
                                                                                            ▼
   robot action ◄─ motion generator (goal only) ◄─ action adapter: validate + decode ◄──────┘

**System prompt.** The embodiment preset writes the system prompt. For DROID, it states:

* the pose and gripper command layout and the coordinate frames;
* how to use the camera calibration;
* for the goal policy and step mode, that ``base_link`` sits 0.131 m behind the point between
  the finger pads;
* for the goal policy, what the previous-goal outcomes mean;
* for step mode, how to grasp and keep holding.

Set ``policy.system_prompt`` to replace it.

**Request.** Each user message contains:

* the task text, plus the previous goal and its outcome for the goal policy;
* the policy observations, such as end-effector pose, gripper position, and the last action;
* the pose of the commanded body in the robot root frame;
* for each camera, the image size, intrinsics, and optical-frame pose in the world and robot
  root frames;
* the RGB images as JPEG, resized so their longer side is ``image_max_width_or_height`` pixels
  (384 by default).

``observation_modality`` selects the content: ``image``, ``state``, or ``image_state`` (the
default).

**Response.** The action adapter defines a strict JSON schema:

* goal: ``{"values": [...]}``
* chunk: ``{"commands": [{"values": [...]}, ...]}``

The adapter rejects a response with the wrong length, non-finite values, or a non-unit
quaternion. It returns the error to the model and asks for a corrected response, up to
``max_attempts`` times. Separately, the inference backend retries failed HTTP requests and
unparsable JSON.

**Motion generator.** The goal policy uses ``PoseTargetMotionGenerator``. It holds the goal pose,
closes or opens the gripper, and reports the goal as:

* **reached:** within ``position_tolerance_m`` and ``rotation_tolerance_rad``, with the gripper
  at its target or stopped while closing on an object;
* **blocked:** the arm and fingers stopped moving short of the goal for several steps.

Add your own implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework separates four parts. Each one can be replaced without touching the others.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Part
     - Responsibility
   * - Observation adapter
     - Subclass ``VLMObservationAdapter`` and implement
       ``extract_vlm_input_from_observation(env, observation, env_id)``. It returns the images by
       camera name and the JSON-serializable values to send.
   * - Action adapter
     - Subclass ``ChunkActionAdapter`` or ``GoalActionAdapter``. Set ``response_model``, a
       Pydantic model that defines the response schema. Implement
       ``is_robot_action_space_supported(env)`` to check the robot's action terms, and
       ``decode_command(response, arena_world, env_id)`` to convert a response into native
       commands. Raise ``InvalidAgentResponseError`` to ask the model for a correction.
   * - Motion generator
     - For goal policies, subclass ``MotionGenerator`` and implement
       ``compute_command(goal, env, env_id)`` and ``goal_reached(goal, env, env_id)``.
       Optionally implement ``goal_blocked`` to report stalls.
   * - Embodiment preset
     - A class with ``make_chunk_action_adapter(chunk_size)``,
       ``make_goal_action_adapter(position_tolerance_m, rotation_tolerance_rad)``, and
       ``get_system_prompt(action_adapter)``. It ties the parts above to one robot and writes
       the prompt. ``DroidVLMAdapter`` in ``isaaclab_arena_vlm_agent/droid_adapter.py`` is the
       reference.

**Support a new robot.** If the robot's controller accepts absolute pose or joint targets without
scaling, reuse ``AbsoluteChunkActionAdapter`` and ``AbsoluteGoalActionAdapter``. They read the
command layout from the active action terms. Then write a preset that describes the layout and
the frames in its prompt. Add the preset name where ``isaaclab_arena_vlm_agent/vlm_agent_policy.py`` validates
``vlm_embodiment_adapter`` (``VLMPolicyCfg``) and resolves it (``_get_vlm_embodiment_adapter``).
Then select it in the YAML:

.. code-block:: yaml

   policy:
     type: isaaclab_arena_vlm_agent.vlm_agent_policy.VLMGoalPolicy
     vlm_embodiment_adapter: my_robot
     max_steps_per_goal: 30

**Add an execution strategy.** Subclass ``VLMPolicy`` with its own typed configuration and
register it with ``@register_policy``, as in `Writing a custom policy`_. Then
implement two methods:

* ``_get_command(env, observation, env_id)``: decide when to request a new prediction, and call
  ``self._infer(self._request_content(...), ...)`` to get a validated one.
* ``reset(env_ids)``: discard pending predictions.

``VLMPolicy`` already handles request assembly, image encoding, response validation, and the
correction loop.
