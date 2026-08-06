Sensitivity Analysis
====================

A single success rate tells you how often a policy completed a task. It does not tell you
which conditions made the task harder or where the policy is most likely to fail. Sensitivity
analysis connects the sampled conditions recorded for each episode to its outcome.

For background information, see the :doc:`Variations concept page
<../../concepts/variations/variations>` and the :doc:`Sensitivity Analysis concept page
<../../concepts/concept_sensitivity_analysis>`.

Generate the report
-------------------

First, complete the :doc:`Variations workflow <../analysis/variations>` to collect episode results.
We now generate a sensitivity report from those results.

Point the report command at that episode-results file.
The flag ``--factors droid_abs_joint_pos.camera_extrinsics_wrist_camera`` selects the wrist-camera
variation and keeps all three components of its recorded offset:

.. code-block:: bash

   python -m isaaclab_arena.analysis.sensitivity.generate_report \
     --outcome success \
     --factors droid_abs_joint_pos.camera_extrinsics_wrist_camera \
     --output /eval/camera_sensitivity_report.png \
     --episode_results outputs/<timestamp>/droid_pnp_camera_sensitivity_openpi/episode_results_rebuild0.jsonl

This places the report as a ``.png`` file at the requested output path ``/eval/camera_sensitivity_report.png``.

.. figure:: ../../../images/sensitivity_report_5_trails.png
   :width: 100%
   :alt: Sensitivity report for three wrist-camera translation components
   :align: center

   Illustrative report layout from a separate five-episode experiment that sampled offsets
   between -10 mm and 10 mm. Horizontal axes show offsets in metres. The blue curve is the
   estimated distribution for successful episodes, its shading marks the 5% to 95% range, and
   the grey dashed line is the uniform sampling distribution. A blue curve close to the dashed
   line suggests no clear relationship; concentration in one region suggests a stronger
   association with success.

Your 10-episode report will look different because it uses a wider sampling range and newly
generated episode outcomes.


Read the report
---------------

This section explains how to read the report.

.. note::

   For this section, we use bundled results from a separate 200-episode banana
   pick-and-place experiment. It samples the wrist-camera position between -50 mm and 50 mm
   along each axis.
   In general, you need a large number of episodes to generate consistent results.
   The data file we use is included in the repository under
   ``isaaclab_arena_examples/sensitivity_analysis/example_results/episode_results_camera_displacement.jsonl``

The following configuration uses the same task and camera-offset range to collect 1,000 new
episodes:

.. dropdown:: 1,000-episode configuration (``droid_banana_camera_sensitivity_openpi_experiment.yaml``)
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/droid_banana_camera_sensitivity_openpi_experiment.yaml
      :language: yaml

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
---------------

Running experiments for several environments and many episodes can be time-consuming.
We use OSMO to orchestrate running experiments quickly on multi-node clusters.

.. note::

  OSMO docs coming soon...
