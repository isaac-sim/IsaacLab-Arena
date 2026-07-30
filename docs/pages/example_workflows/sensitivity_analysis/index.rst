Sensitivity Analysis
====================

A single success rate tells you how often a policy completed a task. It does not tell you
*why* the policy succeeded, which conditions made the task harder, or where the policy is
most likely to fail.

This workflow answers those questions in two parts:

* :doc:`variation_system` creates controlled changes in the environment. For example, it can
  change the lighting, choose a different background, or move a wrist camera slightly away
  from its expected position.
* :doc:`sensitivity_analysis` connects the exact conditions in each episode to the result of
  that episode. This reveals which conditions are most closely associated with success or
  failure.

The two parts are designed to work together. Arena draws a value for each enabled variation,
runs the policy, and records both the drawn values and the episode result. The sensitivity
report then looks for useful patterns across all recorded episodes.

Where to start
--------------

Start with the :doc:`variation_system` page to learn what a variation is and when it changes.
Then continue to :doc:`sensitivity_analysis` to learn how to plan a useful evaluation and read
the resulting report.

.. toctree::
   :maxdepth: 1

   variation_system
   sensitivity_analysis
