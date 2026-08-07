Sensitivity Analysis
====================

A single success rate tells you how often a policy completed a task. It does not tell you
*why* the policy succeeded, which conditions made the task harder, or where the policy is
most likely to fail.

This workflow varies the position of a robot's wrist camera while running an OpenPI policy.
Arena records each sampled camera offset with the episode result, then uses those records to
show which offsets are most closely associated with success or failure.

Follow the :doc:`camera-sensitivity workflow <sensitivity_analysis>` to run the experiment,
generate a report, and read the results.

For background information, see the :doc:`Variations concept page
<../../concepts/variations/variations>` and the :doc:`Sensitivity Analysis concept page
<../../concepts/concept_sensitivity_analysis>`.

.. TODO(cvolk): Rework the Sensitivity Analysis workflow to continue from the preceding
   Variations workflow and move the remaining general explanations to the concept pages.

.. toctree::
   :maxdepth: 1

   sensitivity_analysis
