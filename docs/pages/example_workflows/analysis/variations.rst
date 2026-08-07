Run an Evaluation
=================

An Arena evaluation runs the same policy and task repeatedly and records the outcome of each
episode. In this workflow, Arena also samples a controlled change to the environment: the
wrist-camera position.

Collect data with a controlled variation sweep
----------------------------------------------

To get started, this example runs a small 10-episode sweep with an OpenPI policy on the Rubik's-cube
pick-and-place task. Only the wrist-camera position varies; the object, destination, background,
and lighting remain fixed. The camera offset is drawn independently along three axes between
-30 mm and 30 mm. Larger sweeps are needed for reliable sensitivity conclusions, but 10 episodes
are enough to walk through the workflow.

.. figure:: ../../../images/droid_camera_sensitivity_run_high_quality_cropped.gif
   :width: 100%
   :alt: Accelerated Kit viewport showing DROID OpenPI Rubik's-cube pick-and-place rollouts
   :align: center

   The DROID arm performs repeated Rubik's-cube pick-and-place attempts while Arena resets the task
   and samples a new wrist-camera position for each episode.

The experiment is defined by this configuration:

.. dropdown:: Configuration file (``droid_pnp_camera_sensitivity_openpi_experiment.yaml``)
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/droid_pnp_camera_sensitivity_openpi_experiment.yaml
      :language: yaml

In the Base Docker container, set the output directory used by this workflow:

.. code-block:: bash

   export CAMERA_SENSITIVITY_OUTPUT_DIR="/eval/camera_sensitivity_workflow"
   mkdir -p "${CAMERA_SENSITIVITY_OUTPUT_DIR}"

Arena requires this directory to be empty. With the OpenPI server running, run the evaluation from
the repository root:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --record_camera_video \
     --serve_evaluation_report \
     --evaluation_report_port 8001 \
     --experiment_output_directory "${CAMERA_SENSITIVITY_OUTPUT_DIR}" \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_camera_sensitivity_openpi_experiment.yaml

At each episode reset, Arena samples a new wrist-camera position. OpenPI then uses observations from
that shifted camera to control the arm through the pick-and-place attempt.

.. figure:: ../../../images/droid_wrist_camera_position_variations.gif
   :width: 100%
   :alt: Wrist-camera views sampled from different camera positions
   :align: center

   Wrist-camera views from different sampled camera positions.

Review the evaluation report
----------------------------

After the final episode, the runner prints the aggregate metrics and the address of the evaluation
report. The run shown on this page produced:

.. code-block:: text

   ======================================================================
   METRICS SUMMARY
   ======================================================================

   droid_pnp_camera_sensitivity_openpi:
     num_episodes                           10
     object_moved_rate                  0.9000
     success_rate                       1.0000
   ======================================================================

   Wrote evaluation report with 1 task(s), 1 run(s) and 10 episode(s) to: /eval/camera_sensitivity_workflow/index.html (+2 linked page(s))
   Serving evaluation report at http://localhost:8001/index.html (Ctrl+C to stop).

Your metrics may differ. On the host, open
`http://localhost:8001/index.html <http://localhost:8001/index.html>`__ in a browser. The landing
page summarizes the runs in the experiment and their episode success rates. Select the task to
open its run overview:

.. figure:: ../../../images/droid_camera_sensitivity_evaluation_report_overview.png
   :width: 100%
   :alt: Evaluation report showing success rate, task progress, and episode outcomes
   :align: center

   The run overview summarizes its success rate, progress through the task objectives, and the
   outcome of each episode.

Select **Open droid_pnp_camera_sensitivity_openpi to watch the videos** to inspect the individual
episodes. The recording below shows the three DROID camera views for the first episode:

.. video:: ../../../images/droid_camera_sensitivity_evaluation_report_episode.mp4
   :width: 100%
   :alt: Evaluation report episode with external-camera and wrist-camera videos
   :autoplay:
   :loop:
   :muted:
   :playsinline:
   :align: center
   :caption: Each episode includes its outcome, task progress, recorded camera views, and sampled variation values.

Press :kbd:`Ctrl+C` in the Base Docker container when you finish reviewing the report. This stops
the report server; the HTML pages, videos, and episode results remain in the output directory.

Look beyond the success rate
----------------------------

The evaluation report shows how often the policy succeeded and lets you inspect what happened in
each episode. It does not show how the outcome relates to the sampled wrist-camera positions.

Continue to :doc:`Sensitivity Analysis <../sensitivity_analysis/sensitivity_analysis>` to estimate
and inspect the posterior distribution over camera offsets conditioned on success.
