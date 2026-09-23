Astra VLM agent policy
======================

The Astra example connects Arena directly to NVIDIA's OpenAI-compatible hosted
inference endpoint. It sends the current wrist and external camera images,
camera calibration, and measured DROID state to ``openai/openai/gpt-6-astra``.
The model returns structured end-effector commands that Arena validates and
executes through differential inverse kinematics.

Prerequisites
-------------

Prepare an Arena runtime by following :doc:`../../quickstart/installation`.
Access to the internal inference endpoint and an API key in ``NV_API_KEY`` are
required. Export the key in the shell that launches Arena:

.. code-block:: bash

   export NV_API_KEY="YOUR_API_KEY"

The Docker launcher forwards ``NV_API_KEY`` into the container. Never add the
key to an experiment YAML file or commit it to the repository.

Launch the pick-and-place evaluation
------------------------------------

The goal-command configuration runs one 70-second episode in which a DROID arm
must pick up a banana and place it in a bowl. It uses three calibrated camera
views, retains up to 16 previous textual decisions, and records action traces.

.. dropdown:: Configuration file (``droid_pnp_agent_commanded_goal_experiment.yaml``)
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/droid_pnp_agent_commanded_goal_experiment.yaml
      :language: yaml

From the repository root, start the evaluation:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_agent_commanded_goal_experiment.yaml \
     --viz none \
     --record_camera_video \
     --output_base_dir outputs/astra_vlm_pick_and_place

Use ``/isaac-sim/python.sh`` instead of ``python`` when invoking the command
non-interactively through ``docker exec``.

During startup, the policy prints the selected model, action adapter, and trace
path. Each model decision can take substantially longer than a simulator step;
physics pauses while inference is in progress. The run ends on task success or
after the configured 70 simulated seconds.

Inspect the results
-------------------

The Experiment Runner writes an HTML report, per-camera MP4 recordings, episode
results, and timing data below the timestamped directory in
``outputs/astra_vlm_pick_and_place``. Measured state, model responses, accepted
commands, and executed actions are stored separately in
``outputs/agent_commanded_goal_traces``.

The recorded episode is a single rollout, not a success-rate estimate. Run more
episodes or environment variations before comparing policy quality.

Use action chunks instead
-------------------------

To request 15 absolute end-effector poses per model decision, run the paired
action-chunk configuration:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_agent_commanded_chunk_experiment.yaml \
     --viz none \
     --record_camera_video \
     --output_base_dir outputs/astra_vlm_pick_and_place_chunk

The chunk example uses the same model, cameras, task, and 70-second limit, but
does not retain textual decision history by default.
