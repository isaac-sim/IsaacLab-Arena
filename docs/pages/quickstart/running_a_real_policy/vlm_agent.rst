VLM Agent
=========

A vision-language model (VLM) can act as a policy without any robot training. At each request,
Arena sends the model the task, the robot state, and the current camera images. The model returns
a JSON command, and Arena checks it and executes it on the robot. The model is called through an
OpenAI-compatible inference endpoint, so you do not need to run a model server.

The policies live in the ``isaaclab_arena_vlm_agent`` package. This guide runs them on the DROID
pick-and-place task, explains what each request contains, and shows how to connect your own robot
or execution strategy.

.. figure:: ../../../images/vlm_agent_droid_3x3_grid.gif
   :width: 100%
   :alt: 3x3 grid of successful VLM agent rollouts: goal, chunk, and step policies on three cube and bowl layouts
   :align: center

   One successful episode for each execution strategy (rows) and layout (columns, placement seeds
   3, 2, and 19). Each layout is named by the cube and bowl distance from the robot base. The
   model is sampled, so repeated runs of one layout can differ; see
   :ref:`vlm-agent-strategies`.

Set the inference backend
-------------------------

The policy reads its endpoint from ``policy.endpoint``. If that field is not set, it reads the
``ARENA_INFERENCE_ENDPOINT`` environment variable. Each endpoint reads its API key from its own
environment variable:

.. list-table::
   :header-rows: 1

   * - Endpoint
     - API key variable
     - Default model
     - Access
   * - ``internal``
     - ``NV_API_KEY``
     - ``openai/openai/gpt-6-astra``
     - NVIDIA network only
   * - ``public``
     - ``NVIDIA_API_KEY``
     - ``deepseek-ai/deepseek-v4.1-flash``
     - `build.nvidia.com <https://build.nvidia.com/settings/api-keys>`_
   * - ``openai``
     - ``OPENAI_API_KEY``
     - ``gpt-5.6-terra``
     - OpenAI API

Export the key on the host **before** you start a native ``uv`` runner or the Docker container.
Docker forwards the configured keys when it creates the container. The example configurations
use the ``internal`` endpoint:

.. code-block:: bash

   export NV_API_KEY="YOUR_INTERNAL_API_KEY"

The same keys serve agentic environment generation; see
:ref:`agentic-env-gen-prerequisites` for how to create them.

To use another endpoint or model, override the policy fields:

* ``policy.endpoint``: ``internal``, ``public``, or ``openai``
* ``policy.model``: a model on that endpoint
* ``policy.base_url``: any OpenAI-compatible URL; the API key still comes from the endpoint's
  variable

The model must support strict JSON-schema structured outputs
(``response_format={"type": "json_schema", ...}``). With image observations, it must also accept
images. When the policy starts, it sends one short ping request, so a wrong key, URL, or model
fails before the simulation runs.

The pick-and-place task
-----------------------

Both example configurations run one episode of the same task:

* **Scene:** ``pick_and_place_maple_table`` with the ``home_office_robolab`` HDR.
* **Objects:** a Rubik's cube (``rubiks_cube_hot3d_robolab``) to pick up, and a bowl
  (``bowl_ycb_robolab``) as the destination.
* **Success:** the episode succeeds when the cube rests in the bowl. It ends after 1050 steps
  (70 s at 15 Hz) otherwise.
* **Robot:** the ``droid_abs_ik`` embodiment, a Franka arm with a Robotiq 2F-85 gripper. Each
  action is an absolute pose for the gripper's ``base_link`` in the robot root frame, given as
  XYZ in meters followed by an XYZW quaternion, then a binary gripper command (0 opens,
  1 closes). Differential IK tracks the pose.
* **Cameras:** two external cameras and a wrist camera.
* **Task text:** the ``language_instruction`` text is sent to the model as the task.

Run the example
---------------

Prepare the Arena runtime from the repository root, using either a native ``uv`` environment or
the base Docker container (see :doc:`../installation`):

.. tab-set::

   .. tab-item:: Native uv
      :selected:

      :uv_run_source:

   .. tab-item:: Docker Container

      :docker_run_default:

Arena includes one configuration for each policy type:

.. dropdown:: Goal policy (``droid_pnp_vlm_goal_experiment.yaml``)
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/droid_pnp_vlm_goal_experiment.yaml
      :language: yaml

.. dropdown:: Chunk policy (``droid_pnp_vlm_chunk_experiment.yaml``)
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/droid_pnp_vlm_chunk_experiment.yaml
      :language: yaml

Run the goal policy:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_vlm_goal_experiment.yaml

The runner prints the endpoint and model it connects to, then steps the episode. Each request
takes several seconds, and the simulation waits while a request is in flight. Run the chunk
policy the same way with ``droid_pnp_vlm_chunk_experiment.yaml``.

Override settings for one Run on the command line. Each line below is one example:

.. code-block:: bash

   # Step policy: one command per request
   runs.droid_pnp_vlm_chunk.policy.chunk_size=1
   # Longer chunks: fewer requests per episode
   runs.droid_pnp_vlm_chunk.policy.chunk_size=15
   # Another cube and bowl layout
   runs.droid_pnp_vlm_chunk.environment_builder.placement_seed=2
   # Another endpoint and model
   runs.droid_pnp_vlm_goal.policy.endpoint=openai
   runs.droid_pnp_vlm_goal.policy.model=<model>

For example, to run the step policy:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_vlm_chunk_experiment.yaml \
     runs.droid_pnp_vlm_chunk.policy.chunk_size=1

Add ``--output_base_dir ./output --record_camera_video`` to save one video per camera, and add
``--serve_evaluation_report`` to browse them, as for the :doc:`OpenPI <openpi>` policy.

.. _vlm-agent-strategies:

Goal, chunk, and step policies
------------------------------

The two registered policies differ in what one response controls:

.. list-table::
   :header-rows: 1
   :widths: 18 34 48

   * - Policy
     - One response contains
     - Executed as
   * - ``VLMGoalPolicy``
     - One goal pose and gripper state
     - A motion generator moves the arm toward the goal each step. The policy requests a new
       goal when the current one is **reached**, **blocked** (the arm and gripper stopped short
       of it), or **timed out** after ``max_steps_per_goal`` steps. The next request reports
       that outcome.
   * - ``VLMChunkPolicy``
     - ``chunk_size`` commands
     - One command per step. The policy requests the next chunk when the current one is used
       up.
   * - Step (``VLMChunkPolicy`` with ``chunk_size: 1``)
     - One command
     - One request per simulation step.

On the layouts in the figure, the policies needed:

* **Goal:** 8–9 requests per episode.
* **Chunk with** ``chunk_size: 15``: 4–6 requests.
* **Step:** 65–70 requests, one per step.

A request costs about 2,100 prompt tokens, including three 384-pixel images. Its output is
roughly 500–1,300 tokens, most of it model reasoning. Step mode therefore costs several times
more per episode.

It is also the most sensitive to a single bad response. A command executes on the very next step,
so one stray "open" while the robot carries the cube drops it. On the 0.66 m layout, step mode
succeeded in one of three attempts. Goal and long-chunk modes request far less often and give
the model time to observe the result of each response.

What is inside
--------------

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
---------------------------

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
register it with ``@register_policy``, as described in :doc:`../../concepts/policy/index`. Then
implement two methods:

* ``_get_command(env, observation, env_id)``: decide when to request a new prediction, and call
  ``self._infer(self._request_content(...), ...)`` to get a validated one.
* ``reset(env_ids)``: discard pending predictions.

``VLMPolicy`` already handles request assembly, image encoding, response validation, and the
correction loop.
