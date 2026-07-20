Sensitivity Analysis
====================

A policy's overall success rate tells you how often it completed a task, but not under which
conditions it worked or failed. During a variation sweep, Arena records every tested condition
alongside the episode result. Sensitivity analysis uses those records to estimate which
combinations of conditions are associated with success or failure.

.. figure:: ../../../images/sensitivity_report_200_trails.png
   :width: 100%
   :align: center

   An example sensitivity report which shows the sensitivity of the Pi0.5 policy to displacements of the
   wrist-camera. We will build this figure later in this section.


How Arena generates sensitivity reports
---------------------------------------

For an experiment run with Arena, the input to the sensitivity analysis pipeline
is the episode-results file discussed in :doc:`variation_system`.

Arena considers all selected conditions together and estimates the distribution for the
outcome you choose, such as success or failure, over the selected conditions.
For example, a user may select to analyze the distribution of success over variations
in the wrist-camera position offsets.

Note that you may select to analyze the distribution of a single condition over multiple conditions.
Considering the conditions together preserves patterns where two conditions matter in combination,
and reduces the risk of crediting one condition for a pattern that is actually linked to another.

Arena chooses an estimator automatically from the recorded conditions:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Recorded conditions
     - Estimator
     - Example
   * - All conditions are numeric
     - NPE
     - Light intensity and camera-position offsets
   * - Mixed numeric conditions and named choices
     - MNPE
     - Light intensity together with background or material choices

NPE stands for *neural posterior estimation*. MNPE is its mixed-data counterpart, used when
the conditions include both numbers and named choices. Both produce a joint posterior: an
estimated distribution of the recorded conditions and the selected outcome.
You do not need to choose or configure the estimator yourself.


Run the camera-sensitivity example
----------------------------------

The preceding documentation page, :doc:`variation_system`, demonstrated our variation system.
In particular, we varied, among other factors, the wrist camera displacement.
In that example we used a zero-action policy that kept the robot still.
This example continues with the same Rubik's-cube pick-and-place task,
but uses an OpenPI policy so that each episode has a meaningful success or failure outcome.
Only the wrist-camera position varies; the object, destination, background, and lighting remain fixed.

The camera offset is drawn independently along three axes between -10 mm and 10 mm. Because
all three recorded conditions are numeric, Arena will use NPE for this analysis.

.. dropdown:: Configuration file (``droid_pnp_camera_sensitivity_openpi_experiment.yaml``)
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/droid_pnp_camera_sensitivity_openpi_experiment.yaml
      :language: yaml

Start the OpenPI server in one terminal:

.. code-block:: bash

   ./isaaclab_arena_openpi/docker/run_openpi_server.sh

Leave the server running. For installation, model variants, and server options, see
:doc:`../../quickstart/first_experiments/running_a_real_policy/openpi`.

In the Base Docker container, run the evaluation from the repository root:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --output_base_dir /eval/camera_sensitivity \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_camera_sensitivity_openpi_experiment.yaml

This places the results in the default output path ``/eval/camera_sensitivity/<timestamp>/droid_pnp_camera_sensitivity_openpi``.
In particular, Arena stores one row per completed episode detailing the sampled variation_system
and the per-episode outcome.


Generate the report
^^^^^^^^^^^^^^^^^^^

We now generate a sensitivity report from the results we just collected.

Point the report command at that episode-results file.
The flag ``--factors droid_abs_joint_pos.camera_extrinsics_wrist_camera`` selects the wrist-camera
variation and keeps all three components of its recorded offset:

.. code-block:: bash

   python -m isaaclab_arena.analysis.sensitivity.generate_report \
     --outcome success \
     --factors droid_abs_joint_pos.camera_extrinsics_wrist_camera \
     --output /eval/camera_sensitivity_report.png \
     --episode_results /eval/camera_sensitivity/<timestamp>/droid_pnp_camera_sensitivity_openpi/episode_results_rebuild0.jsonl

This places the report as a ``.png`` file at the requested output path ``/eval/camera_sensitivity_report.png``.

.. figure:: ../../../images/sensitivity_report_5_trails.png
   :width: 100%
   :alt: Sensitivity report for three wrist-camera translation components
   :align: center

   Example sensitivity report from the five-episode experiment. Horizontal axes show offsets in metres. The
   blue curve is the estimated distribution for successful episodes, its shading marks the 5%
   to 95% range, and the grey dashed line is the uniform sampling distribution. A blue curve
   close to the dashed line suggests no clear relationship; concentration in one region
   suggests a stronger association with success.

Note that your report will look different due to randomness in the experiments.


Read the report
^^^^^^^^^^^^^^^

This section explains how to read the report.

.. note::

   For this section we use data from an experiment generated from more episodes
   than the command we ran above (here we use 200 episodes).
   In general, you need a large number of episodes to generate consistent results.
   The data file we use is included in the repository under
   ``isaaclab_arena_examples/sensitivity_analysis/example_results/episode_results_camera_displacement.jsonl``

Generate the report using:

.. code-block:: bash

   python -m isaaclab_arena.analysis.sensitivity.generate_report \
     --outcome success \
     --factors droid_abs_joint_pos.camera_extrinsics_wrist_camera \
     --output /eval/camera_sensitivity_report.png \
     --episode_results isaaclab_arena_examples/sensitivity_analysis/example_results/episode_results_camera_displacement.jsonl

.. figure:: ../../../images/sensitivity_report_200_trails.png
   :width: 100%
   :align: center

   The sensitivity report from a 200-episode experiment included in the Isaac Lab - Arena repo.


Each panel shows one axis/direction of the wrist-camera offset for the selected outcome. The report
title shows the outcome used for the analysis; here, ``success=1`` means that the report focuses
on successful episodes. In the camera's optical frame:

* ``[0]`` is horizontal displacement: negative moves left and positive moves right;
* ``[1]`` is vertical displacement: negative moves up and positive moves down; and
* ``[2]`` is depth displacement: negative moves backward and positive moves forward.

Each of the plots shows the probability distribution over the varied quantity (i.e.
the particular axis of the wrist-camera offset) for the selected outcome.
A flat, i.e. uniform, distribution indicates that the outcome is insensitive to the varied quantity.
In this case such a graph would suggest that the policy is insensitive to changes in the
wrist-camera offset in that direction.
A peaked distribution suggests that the outcome is sensitive to the varied quantity.

Our generated report shows that:

* **Horizontal (x) displacement:** Pi0.5 appears to be sensitive to horizontal displacement of the wrist-camera.
  The proportion of successful episodes drops as horizontal distance from the nominal position increases.
  At the extremum of our experiment (5cm displacement), there are significantly fewer successful episodes.
* **Vertical displacement (y):** Pi0.5 appears relatively **more** sensitive to vertical displacement of the
  camera. This is indicated by the posterior distribution being more peaked than the uniform reference, or
  the posteriors from x and z. The success of the policy drops rapidly as the camera moves away from
  the nominal position in y.

Insights you can take from the report
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A sensitivity report can support several practical decisions:

* **Find a robust operating range.** A broad, flat distribution associated with success,
  suggests that changes of the varied quantity in that region do not affect the policy's success rate.
* **Identify the most sensitive direction.** Compare the horizontal, vertical, and depth
  panels. A strong concentration or a clear difference between the success and failure reports
  points to a direction that deserves closer attention.
* **Improve the training-data distribution.** Because the real-world is highly varied,
  factors of high sensitivity are candidates for additional training examples, to improve
  the policy's robustness, and therefore performance on the real-world.
* **Compare policies.** One is able to compare the sensitivity of different policies to the same variation.
  In general, policies that are less sensitive are more desirable.


Running on OSMO
^^^^^^^^^^^^^^^

Running experiments for several environments and many episodes can be time-consuming.
We use OSMO to orchestrate running experiments quickly on multi-node clusters.

.. note::

  OSMO docs coming soon...
