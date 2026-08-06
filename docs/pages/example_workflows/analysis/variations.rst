Variations
==========

Sensitivity analysis starts with simulated data from a controlled variation sweep. An Arena
experiment runs the same policy and task repeatedly, samples the selected environment conditions,
and records both the sampled values and the outcome of each episode.

Collect data with a controlled variation sweep
----------------------------------------------

To get started, this example runs a small 10-episode sweep with an OpenPI policy on the Rubik's-cube
pick-and-place task. Only the wrist-camera position varies; the object, destination, background,
and lighting remain fixed. The camera offset is drawn independently along three axes between
-30 mm and 30 mm. Larger sweeps are needed for reliable sensitivity conclusions, but 10 episodes
are enough to walk through the workflow.

.. figure:: ../../../images/droid_wrist_camera_position_variations.gif
   :width: 100%
   :alt: Wrist-camera views sampled from different camera positions
   :align: center

   Wrist-camera views from different sampled camera positions.

The experiment is defined by this configuration:

.. dropdown:: Configuration file (``droid_pnp_camera_sensitivity_openpi_experiment.yaml``)
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/droid_pnp_camera_sensitivity_openpi_experiment.yaml
      :language: yaml

With the OpenPI server running, run the evaluation from the repository root in the Base Docker
container:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_camera_sensitivity_openpi_experiment.yaml

At each episode reset, Arena samples a new wrist-camera position. OpenPI then uses observations from
that shifted camera to control the arm through the pick-and-place attempt.

.. figure:: ../../../images/droid_camera_sensitivity_run_high_quality_cropped.gif
   :width: 100%
   :alt: Accelerated Kit viewport showing DROID OpenPI Rubik's-cube pick-and-place rollouts
   :align: center

   The DROID arm performs repeated Rubik's-cube pick-and-place attempts while Arena resets the task
   and samples a new wrist-camera position for each episode.

The runner writes the episode results to:

.. code-block:: text

   outputs/<timestamp>/droid_pnp_camera_sensitivity_openpi/episode_results_rebuild0.jsonl

The JSONL file contains one object per completed episode with its sampled variation values and
outcome.

Continue to :doc:`Sensitivity Analysis <../sensitivity_analysis/index>` to generate and interpret
a report from this file.
