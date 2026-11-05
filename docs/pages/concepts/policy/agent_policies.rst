.. _vlm-agent-policies:

Agent policies
==============

An agent policy uses a hosted foundation model as the controller, with no robot training. At
each request, it shows the model the task, the robot state, and the camera images, and asks for
a structured command. Arena validates the command and runs it. Swapping models means changing
the endpoint, not retraining.

``VLMGoalPolicy`` and ``VLMChunkPolicy`` live in the ``isaaclab_arena_vlm_agent`` package and
call any OpenAI-compatible endpoint. :doc:`../../quickstart/running_a_real_policy/vlm_agent`
runs them on DROID pick-and-place.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Class
     - ``name``
     - The model returns
   * - ``VLMGoalPolicy``
     - ``vlm_goal``
     - One goal pose and gripper state. A motion generator drives to it, then the model sees the
       outcome (reached, blocked, or timed out) and picks the next goal.
   * - ``VLMChunkPolicy``
     - ``vlm_chunk``
     - A short sequence of commands, executed one per step.

Pipeline
--------

.. code-block:: text

   observation ─► observation adapter ─► images + JSON state ─┐
                                                              ├─► inference backend ─► JSON
   embodiment preset ─► system prompt + response schema ──────┘                          │
                                                                                         ▼
   robot action ◄─ motion generator (goal only) ◄─ action adapter: validate + decode ◄──┘

* **Embodiment preset** (``DroidVLMAdapter``): writes the system prompt (command layout, frames,
  camera calibration) and builds the action adapter and motion generator for one robot.
* **Observation adapter** (``VLMObservationAdapter``): picks the images and state to send.
  ``observation_modality`` selects ``image``, ``state``, or ``image_state``.
* **Inference backend** (``InferenceBackend``): calls the endpoint with a strict response schema
  and retries failed requests.
* **Action adapter** (``ChunkActionAdapter`` / ``GoalActionAdapter``): validates the JSON and
  decodes it into robot commands. Invalid responses go back to the model with the error, up to
  ``max_attempts`` times.
* **Motion generator** (``MotionGenerator``, goal policy only): tracks the goal pose and reports
  when it is reached or blocked.

Requests from parallel environments run concurrently, up to ``max_concurrent_requests``. Each
environment keeps its own conversation.

Extend
------

Each stage is a separate class, so you replace one without touching the others.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - To support
     - Implement
   * - A new robot
     - An embodiment preset with ``make_chunk_action_adapter``, ``make_goal_action_adapter``, and
       ``get_system_prompt``. The absolute action adapters work unchanged when every action term
       is absolute pose IK, joint position, or a binary gripper (see
       `command_indices.py <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_vlm_agent/command_indices.py>`__).
   * - Different inputs
     - ``VLMObservationAdapter.extract_vlm_input_from_observation(env, obs, env_id)``.
   * - A different command format
     - An action adapter with a Pydantic ``response_model`` and ``decode_command``. Raise
       ``InvalidAgentResponseError`` to ask the model for a correction.
   * - A different controller
     - ``MotionGenerator.compute_command`` and ``goal_reached``; optionally ``goal_blocked``.
   * - A new execution strategy
     - A ``VLMPolicy`` subclass with ``_needs_new_prediction``, ``_set_prediction``,
       ``_get_next_command``, and ``reset``. Request assembly, concurrency, validation, and the
       correction loop are inherited.

Add a new preset name to the ``vlm_embodiment_adapter`` check in ``VLMPolicyCfg`` and to
``_get_vlm_embodiment_adapter``, then select it in YAML:

.. code-block:: yaml

   policy:
     type: isaaclab_arena_vlm_agent.vlm_agent_policy.VLMGoalPolicy
     vlm_embodiment_adapter: my_robot
     max_steps_per_goal: 30

The same structure scales beyond VLMs: an agent that plans, calls tools, or keeps memory is a
new execution strategy over the same adapters.
