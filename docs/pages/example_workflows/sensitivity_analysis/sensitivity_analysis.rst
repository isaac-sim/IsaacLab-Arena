Sensitivity Analysis
====================

A policy's overall success rate tells you how often it completed a task, but not under which
conditions it worked or failed. During a variation sweep, Arena records every tested condition
alongside the episode result. Sensitivity analysis uses those records to estimate which
combinations of conditions are associated with success or failure.

Imagine varying several properties of the lighting, such as intensity, direction, and
background illumination. Looking at each property separately can hide combinations that
matter. For example, low intensity may work with frontal lighting but fail when the object is
backlit. Arena analyzes the recorded conditions together and produces a report showing where
successful or failed episodes are concentrated.

How Arena analyzes the results
------------------------------

Arena considers all selected conditions together and estimates their distribution for the
outcome you choose, such as success or failure. Considering the conditions together preserves
patterns where two conditions matter in combination and reduces the risk of crediting one
condition for a pattern that is actually linked to another.

Arena chooses the estimator automatically from the recorded conditions:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Recorded conditions
     - Estimator
     - Example
   * - All conditions are numeric
     - NPE
     - Light intensity and camera-position offsets
   * - Numeric conditions and named choices
     - MNPE
     - Light intensity together with background or material choices

NPE stands for *neural posterior estimation*. MNPE is its mixed-data counterpart, used when
the conditions include both numbers and named choices. Both produce a joint posterior: an
estimated distribution of the recorded conditions after focusing on the selected outcome.
You do not need to choose or configure the estimator yourself.

The report presents this joint result as one panel per condition. The deeper statistical
details and the complete input rules are covered in
:doc:`../../concepts/policy/concept_sensitivity_analysis`.

Run the camera-sensitivity example
----------------------------------

The :doc:`variation_system` example showed the wrist camera moving while a zero-action policy
kept the robot still. This example continues with the same Rubik's-cube pick-and-place task,
but uses an OpenPI policy so that each episode has a meaningful success or failure outcome.
Only the wrist-camera position varies; the object, destination, and lighting remain fixed.

The camera offset is drawn independently along three axes between -10 mm and 10 mm. Because
all three recorded conditions are numeric, Arena will use NPE for this analysis.

.. dropdown:: Configuration file (``droid_pnp_camera_sensitivity_openpi_config.json``)
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/eval_jobs_configs/droid_pnp_camera_sensitivity_openpi_config.json
      :language: json

Start the OpenPI server in one terminal:

.. code-block:: bash

   ./isaaclab_arena_openpi/docker/run_openpi_server.sh

Leave the server running. For installation, model variants, and server options, see
:doc:`../../quickstart/first_experiments/running_a_real_policy/openpi`.

In the Base Docker container, run the evaluation from the repository root:

.. code-block:: bash

   python isaaclab_arena/evaluation/eval_runner.py \
     --output_base_dir /eval/camera_sensitivity \
     --eval_jobs_config isaaclab_arena_environments/eval_jobs_configs/droid_pnp_camera_sensitivity_openpi_config.json

The standard container maps ``/eval`` to ``$HOME/eval`` on the host, so the results remain
available after the container stops.

Arena stores one row per completed episode in:

.. code-block:: text

   /eval/camera_sensitivity/<timestamp>/droid_pnp_camera_sensitivity_openpi/episode_results_rebuild0.jsonl

Generate the report
^^^^^^^^^^^^^^^^^^^

Point the report command at that episode-results file. ``--factors`` selects the wrist-camera
variation and keeps all three components of its recorded offset:

.. code-block:: bash

   python -m isaaclab_arena.analysis.sensitivity.generate_report \
     --episode_results /eval/camera_sensitivity/<timestamp>/droid_pnp_camera_sensitivity_openpi/episode_results_rebuild0.jsonl \
     --outcome success \
     --factors droid_abs_joint_pos.camera_extrinsics_wrist_camera \
     --output /eval/camera_sensitivity_report.png

The report generation runs on the CPU and does not start Isaac Sim.

Read the report
^^^^^^^^^^^^^^^

Each panel shows one direction of the wrist-camera offset for the selected outcome. The report
title shows the outcome used for the analysis; here, ``success=1`` means that the report focuses
on successful episodes. In the camera's optical frame:

* ``[0]`` is horizontal displacement: negative moves left and positive moves right;
* ``[1]`` is vertical displacement: negative moves up and positive moves down; and
* ``[2]`` is depth displacement: negative moves backward and positive moves forward.

The three directions are fitted together, not analyzed independently. Each panel shows one
*marginal* of that joint result: its curve summarizes over the sampled values of the other two
directions. This keeps the individual direction readable, although interactions between
directions are not visible in these one-dimensional panels.

.. todo::

   Replace this dry-run figure with the final plot from an evaluation that contains enough
   successful and failed episodes to support an interpretation.

.. figure:: ../../../images/droid_camera_sensitivity_dry_run.png
   :width: 100%
   :alt: Dry-run sensitivity report for three wrist-camera translation components
   :align: center

   Temporary report from the five-episode dry run. Horizontal axes show offsets in metres. The
   blue curve is the estimated distribution for successful episodes, its shading marks the 5%
   to 95% range, and the grey dashed line is the uniform sampling distribution. A blue curve
   close to the dashed line suggests no clear relationship; concentration in one region
   suggests a stronger association with success.

As an illustration of how to read the shapes, imagine that the same pattern came from a larger
evaluation containing both successes and failures. The takeaways could then be written as:

* **Horizontal displacement:** Pi0.5 appears more successful when the wrist camera moves to the
  right of its nominal position.
* **Vertical displacement:** Pi0.5 appears relatively robust to vertical movement because the
  posterior remains close to the uniform reference.
* **Depth displacement:** Pi0.5 appears more successful when the wrist camera moves about 3 mm
  forward from its nominal position.

Insights you can take from the report
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A sensitivity report can support several practical decisions:

* **Find a robust operating range.** A broad region associated with success, but not with
  failure, is usually more useful than one sharp peak. It suggests that small changes within
  that region may be less important to the policy.
* **Identify the most sensitive direction.** Compare the horizontal, vertical, and depth
  panels. A strong concentration or a clear difference between the success and failure reports
  points to a direction that deserves closer attention.
* **Choose a candidate deployment setting.** The success and failure reports can suggest a camera
  position or other operating point to test. Prefer the middle of a broad successful region over
  a narrow edge, then confirm the choice with a focused evaluation.
* **Improve the training-data distribution.** Conditions associated with failure are candidates
  for additional training examples. Include both successful and failure-prone regions so that
  the training data covers the intended operating range rather than only its easiest settings.
* **Compare policies.** Run the same variation sweep for each policy and compare their overall
  success rates together with their success and failure reports. A policy that succeeds across
  more of the tested range is less sensitive to that variation.

A posterior peak is not automatically the best deployment value. It shows where the selected
outcome was concentrated. Interpret it alongside the uniform reference, the failure report,
and the overall success rate before drawing an operational conclusion.

Compare success and failure
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default report focuses on success. To inspect conditions associated with failure, set the
observation to ``0``:

.. code-block:: bash

   python -m isaaclab_arena.analysis.sensitivity.generate_report \
     --episode_results /eval/camera_sensitivity/<timestamp>/droid_pnp_camera_sensitivity_openpi/episode_results_rebuild0.jsonl \
     --outcome success \
     --observation 0 \
     --factors droid_abs_joint_pos.camera_extrinsics_wrist_camera \
     --output /eval/camera_sensitivity_failures.png

Comparing the success and failure reports can make a weak region easier to recognize. Use
``--factors`` when you want to limit the report to a few recorded variations.
