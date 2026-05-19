OpenPi
======

The `openpi <https://github.com/Physical-Intelligence/openpi>`_ project (Physical
Intelligence) publishes Pi0 / Pi05 checkpoints fine-tuned on DROID. Arena
ships a thin WebSocket client (``Pi0RemotePolicy``) that talks to openpi's
``serve_policy.py`` running in a separate process / container.

The setup uses two terminals: the **openpi server** (terminal 1, hosts the model)
and the **arena policy runner** (terminal 2, runs the simulation and sends
observations / receives actions over WebSocket).

Terminal 1 — openpi server
---------------------------

**Build and run**

Arena ships a wrapper script that builds a self-contained Docker image (cloning
upstream openpi at a pinned commit on first run) and starts the inference server:

.. code-block:: bash

   ./isaaclab_arena_openpi/docker/run_openpi_server.sh

The first invocation builds ``isaaclab_arena_openpi-server:latest`` (~3 min,
~19 GB) and then downloads the ~11 GB checkpoint into the container on startup;
subsequent invocations reuse the cached image. Pass ``-r`` to force a rebuild,
``-v pi0`` to serve the pi0 variant instead of pi05, or ``-s <path>`` to build
from a local openpi checkout.

When you see:

.. code-block:: text

   INFO:websockets.server:server listening on 0.0.0.0:8000

the server is ready. Leave the terminal running.

The wrapper passes ``--policy.config`` (architecture + data transforms) and
``--policy.dir`` (params + normalization stats) for the selected variant; see
the supported-variants table below for the exact values.

Terminal 2 — arena policy runner
---------------------------------

**Run pi05 closed-loop**

Open a second terminal and point the arena policy runner at the server:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy \
     --num_episodes 3 \
     --enable_cameras --num_envs 1 \
     --language_instruction "Pick up the Rubik's cube and place it in the bowl." \
     pick_and_place_maple_table \
       --embodiment droid_abs_joint_pos \
       --pick_up_object rubiks_cube_hot3d_robolab \
       --destination_location bowl_ycb_robolab \
       --hdr home_office_robolab

Defaults: ``--openpi_embodiment_adapter droid``, ``--policy_variant pi05``,
``--remote_host localhost``, ``--remote_port 8000``. Pass ``--remote_host`` if the
server is on a different machine.

The server terminal will start logging connection and inference events as the arena
Kit window shows the droid arm reacting to pi0's commanded joint positions.

.. figure:: ../../../../images/openpi_droid_get_started.png
   :width: 100%
   :alt: Arena Kit viewport showing the DROID arm above the maple table with the Rubik's cube and bowl
   :align: center

   Arena Kit viewport during a pi05 rollout: the DROID arm above the maple table with the
   Rubik's cube and destination bowl, with the home_office_robolab HDR.

Supported variants
------------------

The ``Pi0DroidAdapter`` (selected via ``--openpi_embodiment_adapter droid``) supports
three openpi checkpoint variants on DROID:

.. list-table::
   :header-rows: 1

   * - ``--policy_variant``
     - ``--policy.config``
     - ``--policy.dir``
     - Pair with arena ``--embodiment``
     - ``open_loop_horizon``
   * - ``pi05`` (default)
     - ``pi05_droid_jointpos_polaris``
     - ``gs://openpi-assets-simeval/pi05_droid_jointpos``
     - ``droid_abs_joint_pos``
     - 15
   * - ``pi0``
     - ``pi0_droid_jointpos_polaris``
     - ``gs://openpi-assets-simeval/pi0_droid_jointpos``
     - ``droid_abs_joint_pos``
     - 10

To add a new embodiment, subclass ``Pi0EmbodimentAdapter`` (in
``isaaclab_arena_openpi/policy/pi0_remote_policy.py``), then add a branch in
``_resolve_openpi_embodiment_adapter`` and an entry to the
``--openpi_embodiment_adapter`` argparse ``choices`` list.
