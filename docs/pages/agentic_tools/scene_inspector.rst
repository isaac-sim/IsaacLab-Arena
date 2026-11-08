Scene Inspector
===============

The Scene Inspector renders assets from an Arena environment graph, records a
zero-action simulation preview, and reports a USD prim hierarchy. Rendering and
preview commands use a persistent SimulationApp backend so repeated requests do
not pay the application startup cost.

Run the inspector inside the Arena container. The backend and every client must
see the same environment-spec, USD, and output paths.

Start the backend
-----------------

Start one backend in a dedicated terminal:

.. code-block:: bash

   isaaclab.sh -p -m isaaclab_arena.agentic_tools.scene_inspector_backend

The backend accepts one option:

``--socket PATH``
   Listen on this Unix socket instead of the selected default.

The backend prints its socket path when it is ready and processes requests
sequentially. Keep its terminal open while using ``status``, ``snapshot``, or
``preview``.

Select a socket
---------------

Both inspector programs select their socket in this order:

#. ``--socket PATH``
#. The ``ARENA_REVIEW_SIMAPP_SOCKET`` environment variable
#. A checkout-specific path under ``/tmp``

Put the client ``--socket`` option before the subcommand:

.. code-block:: bash

   INSPECTOR=isaaclab_arena.agentic_tools.scene_inspector
   isaaclab.sh -p -m "$INSPECTOR" \
     --socket /tmp/arena-inspector.sock status

The default path includes the user ID and checkout identity, allowing separate
checkouts to use independent backends. An existing socket is never removed
automatically. Remove a stale socket only after confirming that its backend has
exited.

Check backend status
--------------------

.. code-block:: bash

   isaaclab.sh -p -m isaaclab_arena.agentic_tools.scene_inspector status

``status`` has no command-specific options. It exits successfully and reports
``"status": "ready"`` when the selected backend responds. An unavailable
backend produces a JSON error and a nonzero exit status.

Render asset snapshots
----------------------

.. code-block:: bash

   isaaclab.sh -p -m isaaclab_arena.agentic_tools.scene_inspector snapshot \
     --env-spec scene.yaml \
     --output /tmp/scene-snapshots

``snapshot`` supports:

``--env-spec PATH``
   Required Arena environment graph YAML.

``--output DIRECTORY``
   Required destination directory. The inspector writes one PNG per rendered
   graph node and reports each node-to-file mapping as JSON.

``--background-panorama``
   Render the background as a 360-degree panorama. Other graph assets retain
   their normal snapshot views.

Command-line snapshots bypass the backend image cache. A failure for one asset
is returned as a warning when the remaining images can still be produced. The
JSON result also reports each asset's axis-aligned bounding-box dimensions in
meters.

Record a simulation preview
---------------------------

.. code-block:: bash

   isaaclab.sh -p -m isaaclab_arena.agentic_tools.scene_inspector preview \
     --env-spec scene.yaml \
     --output /tmp/scene-preview.mp4 \
     --num-envs 2 \
     --num-steps 120 \
     --env-spacing 30 \
     --show-placement-overlay

``preview`` supports:

``--env-spec PATH``
   Required Arena environment graph YAML.

``--output FILE``
   Required destination MP4 path.

``--num-envs INTEGER``
   Number of environments to construct. Must be positive; the default is ``1``.

``--num-steps INTEGER``
   Number of zero-action simulation steps to record. Must be positive; the
   default is ``10``.

``--env-spacing FLOAT``
   Distance in meters between environment origins. Must be positive and finite;
   the default is ``30.0``.

``--show-placement-overlay``
   Draw relation-solver bounds and mesh collision geometry in the recorded
   viewport. If the environment has no placement pool, the preview continues
   without the overlay and reports a warning in the backend log.

The preview command builds the environment, solves its spatial relations,
resets it, advances it with zero actions, and copies the completed viewport
video to ``--output``. A preview failure is an error rather than a partial
result.

Inspect a USD prim tree
-----------------------

``prim-tree`` reads USD data directly and does not require or start the
SimulationApp backend.

.. code-block:: bash

   isaaclab.sh -p -m isaaclab_arena.agentic_tools.scene_inspector prim-tree \
     --usd scene.usd \
     --output /tmp/prim-tree.txt

   isaaclab.sh -p -m isaaclab_arena.agentic_tools.scene_inspector prim-tree \
     --usd scene.usd \
     --prim Selected \
     --format json \
     --output /tmp/prim-tree.json

``prim-tree`` supports:

``--usd PATH_OR_URL``
   Required local USD path or supported remote USD URL.

``--prim PATH``
   Limit the report to an absolute prim path or a path relative to the USD
   default prim.

``--format {text,json}``
   Write a compact hierarchy as ``text`` (the default), or complete structured
   metadata as ``json``.

``--output FILE``
   Required destination report file. Missing parent directories are created.

Stop the backend
----------------

.. code-block:: bash

   isaaclab.sh -p -m isaaclab_arena.agentic_tools.scene_inspector shutdown

``shutdown`` has no command-specific options. It asks only the backend at the
selected socket to exit.

Output, concurrency, and cleanup
--------------------------------

Every client command writes one machine-readable JSON result to standard output.
Failures include ``status: failed``, an error code, and an error message. Each
request uses a separate connection, while the backend executes operations one at
a time. A health check can therefore time out while a long preview is running.
A timeout leaves the operation's outcome unknown; do not automatically retry it
or remove the socket.

Completed client commands copy requested artifacts to caller-owned output
paths. A crashed or timed-out client can leave ``preview-*`` job directories
under ``~/.cache/isaaclab_arena/agentic_env_gen_sim_preview``. Snapshot cache
indexes and immutable images live in the
``agentic_env_gen_thumbnails`` sibling directory. Remove these directories only
while all inspector backends and GUI clients are stopped.

The Agentic Environment Generation GUI can attach to the same backend by passing
the selected path through its ``--socket`` option. Closing an attached GUI does
not stop the shared backend.
