Sensitivity Analysis
====================

A single success rate tells you how often a policy completed a task. It does not tell you
*why* the policy succeeded, which conditions made the task harder, or where the policy is
most likely to fail.

This workflow answers those questions in two parts:

* :doc:`variation_system` explains when variation values are sampled and what Arena records for
  each episode.
* :doc:`sensitivity_analysis` connects the exact conditions in each episode to the result of
  that episode. This reveals which conditions are most closely associated with success or
  failure.

The two parts are designed to work together. Arena draws a value for each enabled variation,
runs the policy, and records both the drawn values and the episode result. The sensitivity
report then looks for useful patterns across all recorded episodes.

Where to start
--------------

If variations are new to you, first run the visual, zero-action
:doc:`variation example <../../quickstart/first_experiments/exploring_variations>`. Then learn how
to :doc:`collect variation data <variation_system>` and :doc:`generate and read the report
<sensitivity_analysis>`.

.. toctree::
   :maxdepth: 1

   variation_system
   sensitivity_analysis
