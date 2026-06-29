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

**Sequential batch evaluation across object variations**

To measure success rates across several variations of the environment in a single command:

.. code-block:: bash

   python isaaclab_arena/evaluation/eval_runner.py \
     --viz kit \
     --eval_jobs_config isaaclab_arena_environments/eval_jobs_configs/droid_pnp_srl_openpi_jobs_config.json

This runs nine jobs sequentially — each varying the object, background, and destination — and reports a per-job success rate.
Each evaluation is run without restarting Isaac Sim to save on the startup time.

.. figure:: ../../../../images/openpi_droid_3x3_grid.gif
   :width: 100%
   :alt: 3x3 grid of pi05 DROID runs across different objects, backgrounds, and destinations
   :align: center

   9 closed-loop evaluation runs of pi05 on the DROID embodiment — each cell varies the
   pick-up object, background HDR, and destination.

At the end of the run you will see a job summary table followed by a metrics report:

.. code-block:: text

   +----------------------------------------------+-----------+----------------------------------------------------------------+----------+-----------+--------------+
   |                   Job Name                   |   Status  |                          Policy Type                           | Num Envs | Num Steps | Num Episodes |
   +----------------------------------------------+-----------+----------------------------------------------------------------+----------+-----------+--------------+
   |      droid_pnp_srl_openpi_billiard_hall      | completed | isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy |    1     |    None   |      3       |
   | droid_pnp_srl_openpi_rubiks_cube_home_office | completed | isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy |    1     |    None   |      3       |
   |    droid_pnp_srl_openpi_alphabet_soup_can    | completed | isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy |    1     |    None   |      3       |
   |         droid_pnp_srl_openpi_orange          | completed | isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy |    1     |    None   |      3       |
   |          droid_pnp_srl_openpi_lemon          | completed | isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy |    1     |    None   |      3       |
   |    droid_pnp_srl_openpi_tomato_sauce_can     | completed | isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy |    1     |    None   |      3       |
   |     droid_pnp_srl_openpi_mustard_bottle      | completed | isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy |    1     |    None   |      3       |
   |        droid_pnp_srl_openpi_sugar_box        | completed | isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy |    1     |    None   |      3       |
   |           droid_pnp_srl_openpi_mug           | completed | isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy |    1     |    None   |      3       |
   +----------------------------------------------+-----------+----------------------------------------------------------------+----------+-----------+--------------+

   ======================================================================
   METRICS SUMMARY
   ======================================================================

   droid_pnp_srl_openpi_alphabet_soup_can:
     num_episodes                            3
     object_moved_rate                  0.6667
     success_rate                       1.0000

   droid_pnp_srl_openpi_billiard_hall:
     num_episodes                            3
     object_moved_rate                  1.0000
     success_rate                       1.0000

   droid_pnp_srl_openpi_rubiks_cube_home_office:
     num_episodes                            3
     object_moved_rate                  1.0000
     success_rate                       1.0000

   droid_pnp_srl_openpi_sugar_box:
     num_episodes                            3
     object_moved_rate                  1.0000
     success_rate                       0.0000

   ...


pi05 succeeds on most of these variations zero-shot — eight of the nine jobs hit a 1.0
success rate over three episodes, with ``sugar_box`` as the lone outright failure
despite the object being moved in every episode. Performance is strong but not
uniform, consistent with the broader picture that VLA models are improving but
not yet fully robust under zero-shot distribution shift. See
`[robolab] <https://gitlab-master.nvidia.com/xuningy/robolab/-/blob/main/docs/analysis.md>`_
for a cross-model comparison.

Viewing rollouts as an HTML report
----------------------------------

Both ``policy_runner.py`` and ``eval_runner.py`` can collect the rollouts into a browsable
HTML evaluation report. For visualization add ``--record_camera_video`` to record one mp4 per camera, per
episode; the runner writes an ``index.html`` which is then served over HTTP.

.. code-block:: bash

   python isaaclab_arena/evaluation/eval_runner.py \
     --viz kit \
     --eval_jobs_config isaaclab_arena_environments/eval_jobs_configs/droid_pnp_srl_openpi_jobs_config.json \
     --video_base_dir ./output \
     --record_camera_video --serve_evaluation_report

You can also (re)build and serve a report later by pointing the standalone tool at the output
root — it picks the most recent run:

.. code-block:: bash

   python isaaclab_arena/visualization/report.py --video_dir ./output

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
