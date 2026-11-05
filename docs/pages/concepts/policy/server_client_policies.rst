Server–client policies
======================

Large robot foundation models (VLAs and world-model policies) bring their own dependencies,
GPUs, and release cadence. Arena keeps them out of the simulation process: the model runs as a
**server**, and Arena runs a thin **client policy** that implements ``PolicyBase``. The two talk
over a websocket, so the server can sit in another container, on another GPU, or on another
machine.

.. code-block:: text

   ┌──────────── Arena process ────────────┐            ┌──── model server ────┐
   │ env obs ─► EmbodimentAdapter          │  request   │                      │
   │            extract + pack_request ────┼──────────► │  infer(request)      │
   │                                       │            │                      │
   │ env.step ◄─ action ◄─ chunk replay ◄──┼─────────── │  {"actions": (H, D)} │
   └───────────────────────────────────────┘  response  └──────────────────────┘

Supported models
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 33 45

   * - Model
     - Client policy
     - Transport
   * - Cosmos policy
     - ``CosmosRemotePolicy``
     - openpi websocket protocol
   * - π0 / π0.5 (OpenPI)
     - ``Pi0RemotePolicy``
     - openpi websocket protocol
   * - GR00T N1.x
     - ``Gr00tRemoteClosedloopPolicy``
     - GR00T's native ``PolicyClient``
   * - DreamZero
     - ``DreamZeroRemotePolicy``
     - Websocket with DreamZero's msgpack encoding

Quickstarts: :doc:`OpenPI <../../quickstart/running_a_real_policy/openpi>` and
:doc:`GR00T <../../quickstart/running_a_real_policy/gr00t>`. On OSMO, each client policy is
paired with a server task in ``osmo/tasks/``, so one workflow launches both.

The contract
------------

Cosmos and OpenPI share one client,
`remote_policy_base.py <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena/policy/remote_policy_base.py>`__.
It splits the work into two classes:

**EmbodimentAdapter**: what is sent. One adapter per robot and wire format.

* ``action_dim``: width of each action row the server returns.
* ``extract(obs, env_id)``: pull one environment's images and proprioception out of Arena's
  observation dict.
* ``pack_request(extracted, instruction)``: build the request dict the server expects.

**RemoteChunkReplayPolicy**: how it is sent. Shared by every adapter.

* Sends one request per environment and expects an action chunk of shape ``(H, action_dim)``
  under ``server_response_actions_key``.
* Replays the first ``open_loop_horizon`` rows, one per step, then asks again.
* Clears an environment's chunk on ``reset(env_ids)`` and every chunk after a reconnect, so a
  stale chunk is never replayed.
* Reconnects up to three times on a dropped connection. ``close()`` closes the client only; the
  server keeps running.

**Wire format.** Requests and responses are flat dicts of numpy arrays and strings, packed with
``msgpack_numpy``. For example, the Cosmos DROID adapter sends:

.. code-block:: python

   {
       "observation/image": uint8[H, W, 3],          # wrist on top, two exterior views below
       "observation/joint_position": float32[7],
       "observation/gripper_position": float32[1],
       "prompt": "put the banana in the bowl",
   }
   # response: {"action": float32[H, 8]}

The client config carries the connection: ``remote_host``, ``remote_port``, ``ping_interval``,
``ping_timeout``, and ``policy_device``. In an Experiment Definition:

.. code-block:: yaml

   policy:
     type: isaaclab_arena_cosmos.policy.cosmos_remote_policy.CosmosRemotePolicy
     cosmos_embodiment_adapter: droid
     remote_host: 127.0.0.1
     remote_port: 8000

GR00T and DreamZero keep their native clients because their servers use a different protocol.
They still implement ``PolicyBase``, so runners treat them the same way. GR00T also turns chunks
into per-step actions through a pluggable ``ActionScheduler``
(`action_scheduling/ <https://github.com/isaac-sim/IsaacLab-Arena/tree/main/isaaclab_arena/policy/action_scheduling>`__).

Extend
------

**New robot, existing model server.** Write an ``EmbodimentAdapter`` for the robot's cameras,
proprioception, and action layout, then add its key to the policy's adapter lookup
(``cosmos_embodiment_adapter`` or ``openpi_embodiment_adapter``). The transport stays unchanged.

**New model that speaks the openpi protocol.** Subclass ``RemoteChunkReplayPolicy`` and
``PolicyBase[YourCfg]``, set ``name`` and ``server_response_actions_key``, and pass your adapter
and horizon to ``super().__init__``. ``CosmosRemotePolicy`` is the reference: it changes only
the response key and the adapter.

**New model with its own protocol.** Implement ``PolicyBase`` directly, as GR00T does. Keep
the client thin: preprocessing, chunking, and reconnects in the client; model code in the
server.

**Scale out.** Because the model is behind a socket, scaling is a deployment choice, not a code
change: point ``remote_host`` at a bigger GPU or another node, and run many environments per
client with ``num_envs``. To run a new server on OSMO, subclass ``PolicyServerTask`` for the client
policy and register it with ``@register_server_task``.
