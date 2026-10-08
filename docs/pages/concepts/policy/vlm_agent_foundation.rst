VLM agent foundation
====================

Arena provides shared inference and command contracts for building VLM agents.
The environment-generation agent and future robot policies use the same backend
in ``isaaclab_arena.inference.backend``. The existing environment-generation
import path remains available for compatibility.

This foundation does not register a runnable VLM policy. DROID adapters, policy
orchestration, goal and chunk execution, and Experiment/OSMO integration are
separate follow-up changes.

Shared inference
----------------

Use a typed configuration for new clients. Credentials are read from the named
environment variable and are not stored in the configuration. Endpoint presets
(``internal``, ``public``, and ``openai``) supply defaults; an explicit URL,
model, credential variable, and capability settings support custom endpoints.

.. code-block:: python

   from isaaclab_arena.inference.backend import (
       InferenceBackend,
       InferenceBackendCfg,
       InferenceRequest,
   )

   backend = InferenceBackend(
       config=InferenceBackendCfg(
           base_url="https://your-endpoint.example/v1",
           model="your-vision-model",
           api_key_env_var="ARENA_VLM_API_KEY",
           supports_images=True,
       )
   )
   try:
       response = backend.infer(
           InferenceRequest(
               messages=messages,  # OpenAI-compatible text and image_url parts.
               response_schema=command_schema,
           )
       )
       command = action_adapter.decode_command(response.data, measured_state)
   finally:
       backend.close()

Typed construction does not contact the endpoint. The legacy keyword constructor
retains its health check. Existing ``run_json(StructuredOutputRequest(...))``
calls use the shared transport, with their original text parsing conveniences:
unescaped control characters, provider envelopes, and the reasoning-content
fallback. New ``infer()`` calls require a JSON object in the completion content.

The backend retries connection failures and HTTP 408, 409, 429, and 5xx responses.
SDK retries are disabled for inference requests. Both APIs now propagate permanent
HTTP errors immediately and raise ``InferenceResponseError`` for malformed,
refused, or unfinished output. These errors are not retried by the transport;
command repair belongs to the calling agent. This replaces the former
environment-generation behavior that retried every exception.

``request_timeout_s`` limits SDK request waits. ``retry_budget_s`` prevents
starting attempts or backoff beyond the remaining budget and caps each attempt's
timeout. It is not a hard cancellation deadline for an in-flight HTTP request.
Responses report model identity, elapsed wall time, attempt count, provider
request ID, and token usage when supplied by the endpoint.

Model capabilities must match the selected model. Enable ``supports_images``
only for vision models. Override ``max_tokens_parameter`` and
``supports_temperature`` where a model differs from its endpoint preset.
Select ``response_format="json_object"`` for endpoints without JSON-schema
support; the schema is then included in the prompt. The caller validates the
returned data in either mode. JSON-schema dialect support varies by endpoint.

Commands and adapters
---------------------

``isaaclab_arena_vlm_agent_policy.commands`` defines validated move-to,
set-gripper, wait, and pose-chunk commands. Adapters may define other command
subclasses for different control domains.

* A pose represents ``T_A_B``: controlled frame B mapped into reference frame A.
  Both frames are named explicitly. Translation is in meters and rotation is a
  unit quaternion in XYZW order.
* Gripper opening is normalized from zero (closed) to one (open). A missing
  gripper target preserves the active target. Adapters map these values to the
  embodiment's controller convention.
* Step counts refer to calls to ``env.step()``, not physics substeps. Chunk
  references declare their duration in those steps.
* Models reject invalid numeric values, non-unit quaternions, nonpositive step
  counts, empty chunks, and unknown fields. Adapters additionally enforce known
  frames, workspace constraints, maximum horizons, and motion limits.

``VLMObservationAdapter`` extracts per-environment decision inputs and lightweight
tracking state. Camera calibration must describe the actual encoded image,
including resizing and cropping, and match its capture time. Calibration alone
does not determine the depth of an image pixel.

``AgentActionAdapter`` binds to an environment, declares a response schema,
validates model output, and encodes one action row from a control reference.
``AgentCommandExecutor`` owns reference generation, tracking, completion, and
timeouts. A later policy will stack action rows and manage replanning. Interfaces
do not import Isaac Sim or initialize a simulation.

Context and feedback
--------------------

``AgentContext`` records accepted decisions separately from execution updates,
linked by a caller-provided decision ID. It retains a bounded number of events
per environment and copies inputs and query results to prevent accidental mutation.
``reset([env_id])`` clears only the selected environments; ``reset()`` clears all
events and notes. Policy code converts tensor reset IDs to Python integer lists.

Task-progress snapshots are optional: the caller must explicitly authorize
privileged evaluator feedback. Execution success means that a command completed,
not that the task succeeded. Arena's evaluator remains authoritative for task
success. Notes and prompt token/image budgets are owned by the calling policy;
the context's event bound is not a byte or token limit.
