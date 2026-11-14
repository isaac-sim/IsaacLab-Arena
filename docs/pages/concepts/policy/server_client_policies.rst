Server–Client Policies
======================

Robot foundation models such as vision-language-action models (VLAs) and world-model policies
bring their own dependencies, GPUs, and release schedules. Arena keeps them out of the
simulation process. The model runs as a **server**, and Arena runs a thin **client policy**
that implements ``PolicyBase``. The two talk over a websocket, so the server can run in another
container, on another GPU, or on another machine.

.. code-block:: text

   ┌──────────── Arena process ────────────┐            ┌──── model server ────┐
   │ env obs ─► EmbodimentAdapter          │  request   │                      │
   │            extract + pack_request ────┼──────────► │  infer(request)      │
   │                                       │            │                      │
   │ env.step ◄─ action ◄─ chunk replay ◄──┼─────────── │  {"actions": (H, D)} │
   └───────────────────────────────────────┘  response  └──────────────────────┘

Arena includes clients for Cosmos (``CosmosRemotePolicy``), π0 and π0.5 from OpenPI
(``Pi0RemotePolicy``), GR00T (``Gr00tRemoteClosedloopPolicy``), and DreamZero
(``DreamZeroRemotePolicy``). The :doc:`OpenPI <../../quickstart/running_a_real_policy/openpi>`
and :doc:`GR00T <../../quickstart/running_a_real_policy/gr00t>` quickstarts start a server and
run its client end to end.


How the client works
--------------------

The Cosmos and OpenPI clients share one implementation,
`remote_policy_base.py <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena/policy/remote_policy_base.py>`__,
which separates *what* is sent from *how* it is sent.

An ``EmbodimentAdapter`` decides what is sent for one robot and one server format. Its
``extract(obs, env_id)`` method pulls one environment's camera images and joint state out of
Arena's observation dictionary. ``pack_request(extracted, instruction)`` turns them into the
request the server expects. ``action_dim`` declares the width of each action the server returns.

``RemoteChunkReplayPolicy`` decides how it is sent, and every adapter shares it. For each
environment, it sends one request and receives an action chunk of shape ``(H, action_dim)``. It
replays the first ``open_loop_horizon`` actions, one per step, and then requests a new chunk.
When an environment resets, or the connection drops and reconnects, the client discards the
cached chunk so that it never replays stale actions. ``close()`` disconnects the client but
leaves the server running.

A Run with ``num_episodes`` stops assigning new episodes once the limit is reached, but the
simulation keeps stepping every environment until the last assigned episode ends. The client
requests no chunks for the environments whose episode is over and has no replacement; they apply
the action that holds the robot in place until the Run ends. With many environments and few
episodes per environment this removes a large share of the inference, since only environments
whose results count reach the server. The client prints the number of held env-steps when it
closes.

Requests and responses are flat dictionaries of NumPy arrays and strings, packed with
``msgpack_numpy``. For example, the Cosmos DROID adapter sends:

.. code-block:: python

   {
       "observation/image": uint8[H, W, 3],          # wrist view on top, two exterior views below
       "observation/joint_position": float32[7],
       "observation/gripper_position": float32[1],
       "prompt": "put the banana in the bowl",
   }
   # response: {"action": float32[H, 8]}

The client's configuration holds the connection settings: ``remote_host``, ``remote_port``,
``ping_interval``, ``ping_timeout``, and ``policy_device``. The RoboLab example below evaluates
π0.5 and Cosmos on the same two tasks:

.. dropdown:: Configuration file (``robolab_2_tasks_pi0_and_cosmos.yaml``)
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/robolab/experiment_configs/robolab_2_tasks_pi0_and_cosmos.yaml
      :language: yaml
      :lines: 6-

GR00T and DreamZero servers use their own protocols, so Arena keeps their native clients. Both
still implement ``PolicyBase``, and the runners treat them like any other policy. GR00T also
converts chunks into per-step actions through a configurable ``ActionScheduler`` (see
`action_scheduling/ <https://github.com/isaac-sim/IsaacLab-Arena/tree/main/isaaclab_arena/policy/action_scheduling>`__).


Add a model or a robot
----------------------

To run an existing model on a new robot, write an ``EmbodimentAdapter`` for that robot's
cameras, joint state, and action layout. Then add its name to the client's adapter lookup,
which ``cosmos_embodiment_adapter`` or ``openpi_embodiment_adapter`` selects. The transport
does not change.

To add a model whose server speaks the OpenPI protocol, subclass ``RemoteChunkReplayPolicy`` and
``PolicyBase[YourCfg]``. Set ``name`` and ``server_response_actions_key``, and pass your adapter
and horizon to ``super().__init__``. ``CosmosRemotePolicy`` is the reference: it differs from
the OpenPI client only in its response key and adapter.

.. dropdown:: ``cosmos_remote_policy.py``
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_cosmos/policy/cosmos_remote_policy.py
      :language: python
      :lines: 6-

If the server uses its own protocol, implement ``PolicyBase`` directly, as the GR00T client
does. Keep the client thin. Image preprocessing, chunk handling, and reconnects belong in the
client, and model code belongs in the server.

Because the model sits behind a socket, scaling up is a deployment choice, not a code change.
Point ``remote_host`` at a larger GPU or another node, and raise ``num_envs`` to run more
environments per client. On OSMO, each client policy is paired with a server task in
``osmo/tasks/``, so one workflow launches both. To serve a new model there, subclass
``PolicyServerTask`` and register it with ``@register_server_task``.


Next steps
----------

Continue to :doc:`agent_policies` to use a hosted foundation model as a policy without robot
training. See :doc:`Arena Experiments <../concept_arena_experiments>` to submit a server–client
evaluation to OSMO.
