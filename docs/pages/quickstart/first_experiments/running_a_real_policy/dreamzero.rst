DreamZero
=========

DreamZero is a World Action Model with a checkpoint fine-tuned on DROID
(`GEAR-Dreams/DreamZero-DROID`). Arena ships a thin WebSocket client
(``DreamZeroRemotePolicy``) that talks to a DreamZero inference server running in a
separate process / container.

The setup uses two terminals: the **DreamZero server** (terminal 1, hosts the model)
and the **arena policy runner** (terminal 2, runs the simulation and sends
observations / receives actions over WebSocket + MessagePack).

Terminal 1 — DreamZero server
------------------------------

**Build and push the server image**

Arena ships everything needed to build the DreamZero inference server image and run
it as an OSMO job. Log in to the NGC registry once:

.. code-block:: bash

   docker login nvcr.io -u '$oauthtoken' -p <YOUR_NGC_API_KEY>

Then build and push (requires a HuggingFace token to bake the checkpoint into the image):

.. code-block:: bash

   ./isaaclab_arena_dreamzero/docker/push_to_ngc.sh <YOUR_HF_TOKEN> -p
   # Optional overrides:
   #   -t <tag>  Image tag (default: latest)
   #   -n <name> Override image name (default: dreamzero_inference_server)
   #   -R        Build without cache

This produces ``nvcr.io/nvidian/dreamzero_inference_server:<tag>`` with the
``GEAR-Dreams/DreamZero-DROID`` checkpoint baked in at
``/workspace/dreamzero/checkpoints/DreamZero-DROID``.

**Submit the OSMO job**

.. code-block:: bash

   osmo workflow submit isaaclab_arena_dreamzero/docker/dreamzero_inference_server.yaml \
       --set port=5000

The job starts the WebSocket inference server on the requested port using 2 H100
GPUs. No HuggingFace download at runtime — the checkpoint is already in the image.
Once the job is running, find its IP from the OSMO job logs; you will pass it to the
policy as ``--dreamzero_host`` below.

Terminal 2 — arena policy runner
----------------------------------

**Run DreamZero closed-loop**

Open a second terminal and point the arena policy runner at the server. All global
and policy-specific flags must appear **before** the environment name (subcommand);
flags like ``--embodiment`` that are specific to the environment go after it.

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type isaaclab_arena_dreamzero.policy.dreamzero_remote_policy.DreamZeroRemotePolicy \
     --dreamzero_host <OSMO_JOB_IP> \
     --dreamzero_port 5000 \
     --enable_cameras \
     --num_episodes 3 \
     --language_instruction "Pick up the Rubik's cube and place it in the bowl." \
     pick_and_place_maple_table \
       --embodiment droid_abs_joint_pos \
       --pick_up_object rubiks_cube_hot3d_robolab \
       --destination_location bowl_ycb_robolab \
       --hdr home_office_robolab

Defaults: ``--dreamzero_host localhost``, ``--dreamzero_port 5000``,
``--dreamzero_embodiment droid`` (the only embodiment the checkpoint currently
supports). Run headless by swapping ``--viz kit`` for ``--headless``.

Run inside the container:

.. code-block:: bash

   docker exec "$ARENA_CONTAINER" su $(id -un) -c \
       "cd /workspaces/isaaclab_arena && <command above>"

**Sequential batch evaluation across object variations**

To measure success rates across several variations of the environment in a single command,
use the dotted import path as ``policy_type`` and pass config fields directly in
``policy_config_dict``:

.. code-block:: json

   {
     "name": "dreamzero_pick_and_place",
     "arena_env_args": {
       "enable_cameras": true,
       "environment": "pick_and_place_maple_table",
       "embodiment": "droid_abs_joint_pos"
     },
     "num_episodes": 5,
     "language_instruction": "Pick up the cube and place it in the bowl.",
     "policy_type": "isaaclab_arena_dreamzero.policy.dreamzero_remote_policy.DreamZeroRemotePolicy",
     "policy_config_dict": {
       "remote_host": "localhost",
       "remote_port": 5000
     }
   }

Pass this file to:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/evaluation/eval_runner.py \
       --eval_jobs_config <path/to/config.json>

This runs the configured jobs sequentially — each varying the object, background, and
destination — and reports a per-job success rate. Each evaluation is run without
restarting Isaac Sim to save on the startup time.

Viewing rollouts as an HTML report
------------------------------------

Both ``policy_runner.py`` and ``eval_runner.py`` can collect the rollouts into a browsable
HTML evaluation report. For visualization add ``--record_camera_video`` to record one mp4 per
camera, per episode; the runner writes an ``index.html`` which is then served over HTTP.

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/evaluation/eval_runner.py \
       --eval_jobs_config <path/to/config.json> \
       --output_base_dir ./output \
       --record_camera_video --serve_evaluation_report

You can also (re)build and serve a report later by pointing the standalone tool at the output
root — it picks the most recent run:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/visualization/report.py --video_dir ./output

Configuration reference
--------------------------

All options have defaults matching the DreamZero wire protocol. Only override what
differs from your setup.

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Description
   * - ``--dreamzero_host``
     - ``localhost``
     - Hostname of the DreamZero inference server
   * - ``--dreamzero_port``
     - ``5000``
     - Port the server listens on
   * - ``--dreamzero_open_loop_horizon``
     - ``24``
     - Action steps replayed per server inference call
   * - ``--dreamzero_num_arm_joints``
     - ``7``
     - Arm DOF count; remainder of ``robot_joint_pos`` is treated as gripper
   * - ``--dreamzero_embodiment``
     - ``droid``
     - Embodiment the checkpoint was trained on; only ``droid`` is currently supported
   * - ``--dreamzero_cam_exterior_left``
     - ``external_camera_rgb``
     - Arena camera key → ``observation/exterior_image_0_left``
   * - ``--dreamzero_cam2_source``
     - ``black``
     - Source for ``observation/exterior_image_1_left``: ``black``, ``duplicate``, ``right``, or ``head``
   * - ``--dreamzero_cam_exterior_right``
     - ``external_camera_2_rgb``
     - Camera used when ``cam2_source=right``
   * - ``--dreamzero_cam_head``
     - ``head_camera``
     - Camera used when ``cam2_source=head``
   * - ``--dreamzero_cam_wrist``
     - ``wrist_camera_rgb``
     - Arena camera key → ``observation/wrist_image_left``
   * - ``--policy_device``
     - ``cuda``
     - Torch device for the returned action tensor

The environment must expose these keys in its observation dict:

- ``observation["camera_obs"][cam_exterior_left]`` — uint8 RGB tensor ``(num_envs, H, W, 3)``
- ``observation["camera_obs"][cam_wrist]`` — uint8 RGB tensor ``(num_envs, H, W, 3)``
- ``observation["policy"]["robot_joint_pos"]`` — float tensor ``(num_envs, num_arm_joints + 1)``

Images are resized to ``180 x 320`` with letterbox padding before being sent to the server.
