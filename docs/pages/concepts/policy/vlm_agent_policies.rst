.. _vlm-agent-policies:

VLM agent policies
==================

``VLMGoalPolicy`` and ``VLMChunkPolicy`` in the ``isaaclab_arena_vlm_agent`` package use a hosted
vision-language model as the policy. At each request, the policy sends the task, the robot state,
and the camera images to an OpenAI-compatible endpoint, then validates the JSON command it
returns and executes it. :doc:`../../quickstart/running_a_real_policy/vlm_agent` shows how to set
up an endpoint and run them on DROID pick-and-place.

Inference and execution flow
----------------------------

Each request runs through these stages:

.. code-block:: text

   observation ─► observation adapter ─► images + JSON state ─┐
                                                              ├─► inference backend ─► JSON response
   embodiment preset ─► system prompt + response schema ──────┘                             │
                                                                                            ▼
   robot action ◄─ motion generator (goal only) ◄─ action adapter: validate + decode ◄──────┘

The paragraphs below follow the diagram. Each one links to the code that implements the stage
for DROID.

**System prompt, from the embodiment preset.** ``DroidVLMAdapter`` in
`droid_adapter.py <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_vlm_agent/droid_adapter.py>`__ writes the system prompt.
It also creates the action adapter and, for the goal policy, the motion generator. The DROID
prompt states:

* the pose and gripper command layout and the coordinate frames;
* how to use the camera calibration;
* for the goal policy and step mode, that ``base_link`` sits 0.131 m behind the point between
  the finger pads;
* for the goal policy, what the previous-goal outcomes mean;
* for step mode, how to grasp and keep holding.

Set ``policy.system_prompt`` to replace it. Set ``policy.system_prompt_files`` to append text files
to it, such as a task's object geometry and strategy.

**Request, from the observation adapter.** The adapters in
`observation_adapter.py <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_vlm_agent/observation_adapter.py>`__ pick the
images and state to send. ``VLMPolicy._request_content`` in
`vlm_agent_policy.py <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_vlm_agent/vlm_agent_policy.py>`__ encodes the images
and builds the user message. It contains:

* the task text, plus the previous goal and its outcome for the goal policy;
* the policy observations, such as end-effector pose, gripper position, and the last action;
* the pose of the commanded body in the robot root frame;
* for each camera, the image size, intrinsics, and optical-frame pose in the world and robot
  root frames;
* the RGB images as JPEG, resized so their longer side is ``image_max_width_or_height`` pixels
  (384 by default).

``observation_modality`` selects the content: ``image``, ``state``, or ``image_state`` (the
default).

**Inference backend.** ``InferenceBackend`` in
`backend.py <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena/inference/backend.py>`__ sends the system prompt, the user
message, and the response schema to the endpoint. It retries failed HTTP requests and responses
that are not valid JSON.

**Response, from the action adapter.** ``AbsoluteChunkActionAdapter`` and
``AbsoluteGoalActionAdapter`` in
`absolute_action_adapter.py <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_vlm_agent/absolute_action_adapter.py>`__
define a strict JSON schema:

* goal: ``{"values": [...]}``
* chunk: ``{"commands": [{"values": [...]}, ...]}``

The adapter rejects a response with the wrong number of values, non-finite values, a quaternion
that is not unit length, or a gripper value other than open or closed. It sends the error back to
the model and asks for a corrected response, up to ``max_attempts`` times.

**Execution, by the motion generator (goal policy only).** ``PoseTargetMotionGenerator`` in
`pose_target_motion_generator.py <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_vlm_agent/pose_target_motion_generator.py>`__
holds the goal pose, closes or opens the gripper, and reports the goal as:

* **reached:** within ``position_tolerance_m`` and ``rotation_tolerance_rad``, with the gripper
  at its target or stopped while closing on an object;
* **blocked:** the arm and fingers stopped moving short of the goal for several steps.

Object poses
------------

Set ``object_poses`` to a list of scene keys to give the model their exact poses. The policy then
wraps the selected observation adapter in ``ObjectPoseObservationAdapter``, which adds each
object's pose in the robot root frame to the request as ``object_poses``. A real robot does not
observe these poses, so use them as a reference: they show what the policy achieves once it knows
where the objects are.

Pixel measurement
-----------------

Vision-language models judge where things are in an image only roughly, to a few centimeters.
That is enough to pick up a cube, but not to fit a gear onto a peg within a few millimeters. Set
``pixel_measurement: true`` to let the model measure positions instead of estimating them. The
policy then offers the model two tools from ``PixelMeasurementToolSet``, in
`pixel_measurement_tools.py <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_vlm_agent/pixel_measurement_tools.py>`__,
through the endpoint's tool calling:

* ``zoom`` takes a camera, a center pixel, and a half size. It returns a magnified crop of the
  rendered image, with a finer grid labelled in the same full-image pixels;
* ``measure_point`` takes a camera and a pixel. It returns the 3D point in the robot root frame
  where that pixel's ray hits a surface, read from the camera's depth image and calibration.

Every image carries a yellow pixel grid, labelled in the pixels the tools take. The model may call
tools for up to five rounds per decision before it must return an action. The last 12
measurements are repeated in later requests, so a part measured while visible stays usable after
the gripper hides it.

Every camera must also render ``distance_to_image_plane``.

Gear insertion example
----------------------

The DROID gear insertion experiment runs the goal policy three ways, one Run each:

* ``images``: camera images and robot state only;
* ``object_poses``: also the exact gear and target poses;
* ``pixel_measurement``: the model measures the gear and pegs from camera depth.

The cameras render at 1280x720 with depth, and the policy sends the full images. Every Run appends the gear
and peg geometry and a grasp and insertion strategy from
``experiment_configs/prompts/droid_gear_insertion.txt``, and the last two Runs add a short note on
their extra inputs. Run it from the repository root, adding ``~runs.<name>`` to skip a Run:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py --viz none \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_gear_insertion_vlm_goal_experiment.yaml

Add your own implementation
---------------------------

Every stage except the inference backend is a part you can replace without touching the others.
The rows follow the order of the stages above.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Pipeline stage
     - Part
     - What to implement
   * - System prompt
     - Embodiment preset
     - A class with ``make_chunk_action_adapter(chunk_size)``,
       ``make_goal_action_adapter(position_tolerance_m, rotation_tolerance_rad)``, and
       ``get_system_prompt(action_adapter)``. It ties the other parts to one robot and writes
       the prompt. ``DroidVLMAdapter`` is the reference.
   * - Request
     - Observation adapter
     - Subclass ``VLMObservationAdapter`` and implement
       ``extract_vlm_input_from_observation(env, observation, env_id)``. It returns the images by
       camera name and the JSON-serializable values to send. Adapters that send images also implement
       ``read_camera_views``, which returns each image's ``CameraView``: its calibration and any
       rendered depth.
   * - Tools
     - Tool set
     - Subclass ``VLMToolSet`` and implement ``get_tool_definitions`` and
       ``call_tool(name, arguments, env_id)``. Optionally implement ``prepare_request`` to change
       the images and prompt values of each decision, and ``get_system_prompt_addition`` to explain
       the tools. Pass the tool set to the policy as ``tool_set``; ``PixelMeasurementToolSet`` is the
       reference.
   * - Response
     - Action adapter
     - Subclass ``ChunkActionAdapter`` or ``GoalActionAdapter``. Set ``response_model``, a
       Pydantic model that defines the response schema. Implement
       ``is_robot_action_space_supported(env)`` to check the robot's action terms, and
       ``decode_command(response, arena_world, env_id)`` to convert a response into native
       commands. Raise ``InvalidAgentResponseError`` to ask the model for a correction.
   * - Execution (goal policy only)
     - Motion generator
     - Subclass ``MotionGenerator`` and implement ``compute_command(goal, env, env_id)`` and
       ``goal_reached(goal, env, env_id)``. Optionally implement ``goal_blocked`` to report
       stalls.

**Support a new robot.** ``AbsoluteChunkActionAdapter`` and ``AbsoluteGoalActionAdapter`` read
the command layout from the robot's active action terms and pass the model's values to them
unchanged. You can reuse them if every action term is one of these:

* ``DifferentialInverseKinematicsAction`` with ``command_type="pose"``, absolute mode
  (``use_relative_mode=False``), ``scale`` 1, and no ``body_offset``. It takes seven values: XYZ
  and an XYZW quaternion.
* ``JointPositionAction`` with ``scale`` 1, ``offset`` 0, and ``use_default_offset=False``.
* ``BinaryJointPositionAction`` or ``BinaryJointPositionZeroToOneAction`` with one value.

No term may set ``clip``. If any term does not match, ``is_robot_action_space_supported``
returns ``False`` and the policy stops with an assertion error before its first request.
`command_indices.py <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_vlm_agent/command_indices.py>`__
implements these checks. For other controllers, write your own action adapter.

Next, write a preset that describes the layout and the frames in its prompt. In
``isaaclab_arena_vlm_agent/vlm_agent_policy.py``, add the preset name where ``VLMPolicyCfg``
validates ``vlm_embodiment_adapter`` and where ``_get_vlm_embodiment_adapter`` creates the
preset. Then select it in the YAML:

.. code-block:: yaml

   policy:
     type: isaaclab_arena_vlm_agent.vlm_agent_policy.VLMGoalPolicy
     vlm_embodiment_adapter: my_robot
     max_steps_per_goal: 30

**Add an execution strategy.** Subclass ``VLMPolicy`` with its own typed configuration and
register it with ``@register_policy``, as in :ref:`writing-a-custom-policy`. Then
implement two methods:

* ``_get_command(env, observation, env_id)``: decide when to request a new prediction, and call
  ``self._infer(self._request_content(...), ...)`` to get a validated one.
* ``reset(env_ids)``: discard pending predictions.

``VLMPolicy`` already handles request assembly, image encoding, response validation, and the
correction loop.
