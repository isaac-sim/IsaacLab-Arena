Collecting Variation Data
=========================

Sensitivity analysis connects the conditions sampled in each episode to that episode's result.
If you have not used Arena's variation system before, start with the visual, zero-action
:doc:`variation example <../../quickstart/first_experiments/exploring_variations>`.

For a useful sensitivity report, a sweep must sample each condition at the right point and
record every draw alongside the outcome. Configure the enabled factors and their sampling ranges
through the experiment run's ``variations`` mapping. See :doc:`the Variations concept page
<../../concepts/variations/variations>` for the configuration syntax and available variations.


Build-time and run-time variations
----------------------------------

Some variations are sampled before the environment is created, while others are sampled whenever
an environment resets. Arena calls these *build-time* and *run-time* variations.

.. list-table::
   :header-rows: 1
   :widths: 20 25 35 20

   * - Type
     - When it changes
     - Where the drawn value applies
     - Examples
   * - Build-time
     - Before the environment is built
     - Every parallel environment and episode in that build
     - Background and lighting changes
   * - Run-time
     - When an environment resets
     - One episode in one parallel environment
     - Camera extrinsics and intrinsics

To collect several values of a build-time variation, rebuild the environment several times. A
run-time variation can produce a new value on each reset without rebuilding the scene. See
:ref:`build-time-run-time-variations` for the full explanation.


What Arena records
------------------

At the end of an episode, Arena writes the exact draw from every enabled variation alongside the
episode result. A record can look like this:

.. code-block:: json

   {
     "success": true,
     "variations": {
       "light.hdr_image": "home_office_robolab",
       "light.intensity": [1250.0],
       "droid_rel_joint_pos.camera_extrinsics_wrist_camera": [0.001, -0.002, 0.0]
     }
   }

The zero-action example makes variation draws easy to see, but it cannot produce meaningful
success or failure outcomes. The next page runs a trained policy, records its outcomes, and
generates a :doc:`sensitivity report <sensitivity_analysis>` from those records.
