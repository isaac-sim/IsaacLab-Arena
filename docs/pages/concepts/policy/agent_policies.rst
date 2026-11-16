.. _vlm-agent-policies:

Agent as Policy
===============

In agent as policy, a hosted foundation model is the controller, with no robot training. At
each request, the policy shows the model the task, the robot state, and the camera images, and
asks it for a structured command. Arena checks the command and runs it. To try a different
model, you change the endpoint instead of retraining.

Arena includes two such policies in the ``isaaclab_arena_vlm_agent`` package. Both call any
OpenAI-compatible endpoint. ``VLMGoalPolicy`` (``vlm_goal``) asks for one goal pose and gripper
state at a time. A motion generator drives the robot to that goal, and the next request tells
the model whether the goal was reached, blocked, or timed out. ``VLMChunkPolicy``
(``vlm_chunk``) asks for a short sequence of commands and runs one per step.
:doc:`../../quickstart/running_a_real_policy/vlm_agent` runs both on DROID pick-and-place.


How a request works
-------------------

Each request passes through a small pipeline. Every stage is its own class:

.. code-block:: text

   observation ─► observation adapter ─► images + JSON state ─┐
                                                              ├─► inference backend ─► JSON
   embodiment preset ─► system prompt + response schema ──────┘                          │
                                                                                         ▼
   robot action ◄─ motion generator (goal only) ◄─ action adapter: validate + decode ◄──┘

The **embodiment preset** (``DroidVLMAdapter``) ties the pipeline to one robot. It writes the
system prompt, which explains the command layout, the coordinate frames, and the camera
calibration, and it builds the action adapter and motion generator. The **observation adapter**
(``VLMObservationAdapter``) chooses the images and state to send. ``observation_modality``
selects ``image``, ``state``, or ``image_state``. The **inference backend**
(``InferenceBackend``) calls the endpoint with a strict response schema and retries failed
requests.

The **action adapter** (``ChunkActionAdapter`` or ``GoalActionAdapter``) checks the model's JSON
and converts it into robot commands. If the response is invalid, the adapter sends the error
back to the model and asks for a correction, up to ``max_attempts`` times. For the goal policy,
the **motion generator** (``MotionGenerator``) tracks the goal pose and reports when it is
reached or blocked.

When several environments need a response on the same step, their requests run concurrently,
up to ``max_concurrent_requests`` at a time. Each environment keeps its own conversation with
the model.


.. _vlm-grasp-point-observations:

Grasp and contact point observations
------------------------------------

Set ``policy.grasp_points`` to add named surface points to either VLM policy.
``GraspPointObservationAdapter`` reads each point's live object pose for the
requested environment and reports its position in every commanded robot's root
frame. The point can follow a rigid object or a moving articulation part exposed
as an object reference. Multiple named points can refer to the same object.

For example, a point near a container's upper rim can be configured as:

.. code-block:: yaml

   policy:
     type: isaaclab_arena_vlm_agent.vlm_agent_policy.VLMGoalPolicy
     vlm_embodiment_adapter: droid
     max_steps_per_goal: 90
     grasp_points:
       bin_rim:
         scene_key: white_bin
         aabb_fraction: [0.5, 0.0, 1.0]
         offset_O_m: [0.0, 0.012, -0.004]
         normal_O: [0.0, -1.0, 0.0]

Frame O is the object's local geometry frame. ``aabb_fraction`` selects a point
between its spawned local bounding-box minimum (0) and maximum (1) on each axis;
``offset_O_m`` then adds a metric offset in O. Omit ``aabb_fraction`` to specify
a point directly relative to the object origin. Offsets are in meters at the
spawned size; they are not automatically multiplied by the asset scale.

The observation is available under
``simulator_ground_truth_assistance.grasp_points.bin_rim.points_in_robot_roots``.
Each robot entry contains ``position_root_m`` and, when configured,
``outward_normal_root``. Normals are unit vectors. The adapter preserves other
observations such as object poses and joint positions.

For a hinged part, add ``revolute_motion`` to the point:

.. code-block:: yaml

   revolute_motion:
     scene_key: socket_set
     body_name: Case_base
     pivot_B_m: [0.0, 0.074, 0.03]
     axis_B: [-1.0, 0.0, 0.0]
     direction: -1

Frame B is the named articulation body's frame, or the scene object's frame when
``body_name`` is omitted. The pivot and axis define a right-handed rotation;
``direction`` selects its positive or negative direction. This adds the live unit
``motion_tangent_root`` at the contact point. The workbench toolcase experiment
uses this configuration to preserve its authored lid-closing geometry.

These are **privileged, authored contact candidates**. They do not predict grasps
or validate reachability, collision clearance, or grasp stability. The policy must
account for its gripper's contact offset when converting a surface point into a
commanded-body pose.


Extend the pipeline
-------------------

Because each stage is a separate class, you can replace one without touching the others.

To support a new robot, write an embodiment preset with ``make_chunk_action_adapter``,
``make_goal_action_adapter``, and ``get_system_prompt``. The included absolute action adapters
work unchanged when every action term is absolute pose IK, joint position, or a binary gripper
(see
`command_indices.py <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_vlm_agent/command_indices.py>`__).
Add the preset name to the ``vlm_embodiment_adapter`` check in ``VLMPolicyCfg`` and to
``_get_vlm_embodiment_adapter``, and then select it in YAML:

.. code-block:: yaml

   policy:
     type: isaaclab_arena_vlm_agent.vlm_agent_policy.VLMGoalPolicy
     endpoint: internal  # internal, public, or openai
     model: openai/openai/gpt-6-astra  # Must be served by the endpoint
     vlm_embodiment_adapter: my_robot
     max_steps_per_goal: 30

To send different inputs, subclass ``VLMObservationAdapter`` and implement
``extract_vlm_input_from_observation``. To change the command format, write an action adapter
with a Pydantic ``response_model`` and a ``decode_command`` method. Raise
``InvalidAgentResponseError`` there to ask the model for a correction. To use a different
controller, subclass ``MotionGenerator``.

To change how commands are executed over time, subclass ``VLMPolicy`` and implement
``_needs_new_prediction``, ``_set_prediction``, ``_get_next_command``, and ``reset``. The subclass
inherits request assembly, concurrent requests, validation, and the correction loop. The same
structure extends beyond VLMs: an agent that plans, calls tools, or keeps memory is a new
execution strategy over the same adapters.


Next steps
----------

Continue to :doc:`code_policies` for policies that run programs instead of commands.
