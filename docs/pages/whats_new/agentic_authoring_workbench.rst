Workbench Environment
=====================

.. _whats-new-workbench-authoring:

.. role:: gain
   :class: arena-value-gain

.. container:: arena-value-eyebrow

   Isaac Lab - Arena · Agentic Authoring · Workbench Task

Faster authoring. Evaluation infrastructure included.
-----------------------------------------------------

Coding agents are now the predominant way that developers interact with code.
Agents are quickly becoming the predominant way to interact with 3D environments.
Arena ``v0.4`` includes improvements to agentic environment authoring.

Study
-----

On this page we describe a study that aims to show the efficacy of Arena's evaluation building system for agentic environment authoring.
As a baseline, we compared against Isaac Lab alone, without the Arena extension,
which lacks the evaluation-oriented building blocks that Arena provides.

**TL;DR:** With Arena, the agent:
 * completed the initial setup **5.25× faster**,
 * spent **65% less** on tokens,
 * wrote **no custom code**,
 * and used Arena's batched placement solver to get **physically stable layout variations** with no additional setup.

The agent's task
^^^^^^^^^^^^^^^^

The agent gets a single prompt describing a workbench cleanup evaluation environment:
five tools must be moved from a bin into an open drawer, and the drawer must then be closed.

.. dropdown:: The prompt given to the agent

   Set up an environment for a workbench cleanup task evaluation. Use assets in
   ``~/datasets/workbench/assets``. On the tabletop of the workbench, there is a bin containing a
   clutter of five tools, such as a wrench and screwdriver, an oscilloscope, and an open toolbox.
   The top-left drawer of the workbench is open.

   Task: put all the tools into the drawer, then close the drawer. Validate task success correctly.

The agent's process
^^^^^^^^^^^^^^^^^^^

.. video:: ../../images/whats_new/agentic_authoring_workbench_walkthrough.mp4
   :width: 100%
   :caption: A coding agent authors, validates, and runs a workbench cleanup evaluation with Arena.
   :alt: Walkthrough of a coding agent authoring a workbench cleanup evaluation in Isaac Lab-Arena
   :muted:
   :playsinline:

The agent's solution
^^^^^^^^^^^^^^^^^^^^

The agent wrote a 172-line environment YAML. The scene places five tools as clutter inside a bin
and varies the tabletop layout between resets. Object references expose the tabletop and nested
articulated drawer as placement and task targets without modifying the source workbench USD.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: ../../images/whats_new/workbench_cleanup_10_reset_layouts.gif
         :width: 100%
         :alt: Ten reset layouts of the Arena workbench cleanup environment

         Ten reset layouts of the agent-authored environment.

   .. grid-item::

      .. figure:: ../../images/whats_new/workbench_cleanup_spatial_relations.svg
         :width: 100%
         :alt: Graph of the spatial relations and object references in the workbench environment

         Placement relations and nested object references in the environment YAML.

The task has one placement milestone for each tool followed by drawer closure. Success checks use
the live drawer body pose and require the tools to remain contained, supported, and settled.

Analysis
^^^^^^^^

Let's compare the solution above to the baseline solution written without Arena's evaluation building blocks.

.. container:: arena-value-stats

   .. container:: arena-value-stat

      **5.25× faster setup**

      initial setup took 4 min instead of 21 min

   .. container:: arena-value-stat

      **65% lower token cost**

      authoring and validation cost $3.95 instead of $11.30

   .. container:: arena-value-stat

      **0 Python**

      authored for Arena (172 lines of YAML)

.. list-table::
   :header-rows: 1
   :widths: 22 22 22 34
   :class: arena-value-table

   * - Measurement
     - Without Arena
     - With Arena
     - Details
   * - Initial setup time
     - 21 min
     - 4 min

       :gain:`5.25× faster setup`
     - Time for the agent's initial environment setup.
   * - Model cost
     - $11.30
     - $3.95

       :gain:`65% lower token cost`
     - Token cost of authoring and validation with GPT-6 Astra.
   * - Environment code
     - 302 lines of custom Python code
     - 0 lines of Python + 172 lines of environment YAML

       :gain:`no new code to maintain`
     - Environment is a composition of Arena placement relations, object references, and tasks.
   * - Ten physically stable layout variations
     - 19.4 min + $11.1 + 301 lines of custom Python code
     - 0 additional setup
     - Arena's native batched placement solver handles the variations.

In summary, with Arena, the agent:
 * completed the initial setup **5.25× faster**,
 * spent **65% less** on tokens,
 * wrote **no custom code**,
 * and used Arena's batched placement solver to get **physically stable layout variations** with no additional setup.

Beyond initial setup
^^^^^^^^^^^^^^^^^^^^

Arena provides more than a shorter environment definition:

* **Collision-free batched placement.** The placement solver samples the full layout together,
  rejects object collisions, and supports batched collision and robot-reachability validation.
  The workbench environment gets ten physically stable layout variations without another custom
  placement script.
* **Composable environment changes.** Swapping the workbench takes a 16-line environment YAML;
  adding two tasks takes a 35-line environment YAML. Neither requires Python.
* **Evaluation at scale.** A 43-line Experiment YAML runs 100 episodes over two variations either
  locally or on an OSMO cluster. Experiment runs produce an HTML summary and episode recordings
  and work with Arena's built-in sensitivity analysis.

Going further
^^^^^^^^^^^^^

The environment YAML plugs straight into an Arena Experiment runner and can be used in all Arena
evaluation workflows. For example,

* **Parallel evaluation:** The above environment can be run across many parallel environments.
  See :doc:`../example_workflows/analysis/recorded_variations`.
* **Multi-node evaluation:** The same environment can be evaluated across a multi-node cluster with OSMO.
  See :doc:`../example_workflows/multi_node_evaluation/multi_node_evaluation`.
* **Variations:** Expose the policy to controlled perturbations through Arena's variation system.
  See :doc:`../example_workflows/analysis/variations`.
* **Metrics:** By default, Arena collects evaluation-oriented metrics which can be viewed after an experiment run.
  See :doc:`../example_workflows/sensitivity_analysis/sensitivity_analysis`.
