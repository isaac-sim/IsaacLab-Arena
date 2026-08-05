Plan a Sensitivity Experiment
=============================

A sensitivity analysis compares a policy's results with the conditions in which each episode ran.
Before generating a report, choose the conditions you want to test, set a range for each one, and
plan how to collect enough samples. Arena represents these conditions as variations and records
each sampled value with the episode result.


Choose the conditions to test
-----------------------------

Start with the question you want the report to answer. For example, to measure sensitivity to
wrist-camera placement, vary the camera position and keep unrelated conditions fixed. This makes
the report easier to interpret.

Configure the conditions and their sampling ranges through the experiment run's ``variations``
mapping. See :doc:`the Variations concept page <../../concepts/variations/variations>` for the
configuration syntax and available variations. To see what these changes look like before running
a policy, try the visual, zero-action :doc:`variation example
<../../quickstart/first_experiments/exploring_variations>`.


Plan how samples are collected
------------------------------

Some variations are sampled before the environment is created, while others are sampled whenever
an environment resets. Arena calls these *build-time* and *run-time* variations. This distinction
determines how you collect different values during the experiment.

.. list-table::
   :header-rows: 1
   :widths: 18 25 37 20

   * - Type
     - When it changes
     - Scope of one draw
     - How to collect more draws
   * - Build-time
     - Before the environment is built
     - Every parallel environment and episode in that build
     - Rebuild the environment
   * - Run-time
     - When an environment resets
     - One episode in one parallel environment
     - Run more episodes

For a build-time condition, increasing the number of episodes within one build does not produce a
new draw. The experiment must rebuild the environment. A run-time condition can produce a new draw
on every reset without rebuilding the scene. See :ref:`build-time-run-time-variations` for the full
explanation.


Use the episode records
-----------------------

At the end of each episode, Arena writes the result together with the exact draw from every enabled
variation. The report can use any recorded variation as a factor. A record from a camera-sensitivity
experiment can look like this:

.. code-block:: json

   {
     "success": true,
     "variations": {
       "droid_abs_joint_pos.camera_extrinsics_wrist_camera": [0.001, -0.002, 0.0]
     }
   }

These paired conditions and outcomes are the input to the sensitivity report. The next page runs
a trained policy and shows how to generate and read the :doc:`sensitivity report
<sensitivity_analysis>`.
