Workbench Environment
=====================

.. _whats-new-workbench-authoring:

.. role:: gain
   :class: arena-value-gain

.. container:: arena-value-eyebrow

   Arena vs. Isaac Lab · Authoring study · 2 Nov 2026

Faster authoring. Evaluation infrastructure included.
-----------------------------------------------------

We asked the same coding agent to build the same workbench cleanup evaluation twice: once as Arena
YAML, and once as a standalone Isaac Lab Python script. With Arena, initial setup was faster and
cheaper, required no Python, and connected the task to reusable placement, variation, validation,
and distributed evaluation workflows.

The problem
^^^^^^^^^^^

The agent gets a single prompt: arrange a varied workbench scene, define a cleanup task that ends
by closing an articulated drawer, and validate success correctly.

.. dropdown:: The prompt given to the agent

   Set up an environment for a workbench cleanup task evaluation. Use assets in
   ``~/datasets/workbench/assets``. On the tabletop of the workbench, there is a bin containing a
   clutter of five tools, such as a wrench and screwdriver, an oscilloscope, and an open toolbox.
   The top-left drawer of the workbench is open.

   Task: put all the tools into the drawer, then close the drawer. Validate task success correctly.

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
   :widths: 36 30 34
   :class: arena-value-table

   * - Decision factor
     - Isaac Lab (agent writes custom code)
     - Isaac Lab - Arena (agent writes YAML)
   * - Initial setup time
     - 21 min
     - 4 min

       :gain:`5.25× faster setup`
   * - Initial authoring code
     - 302 lines of Python
     - 0 lines of Python + 172 lines of YAML
   * - Authoring and validation token cost
     - $11.30
     - $3.95

       :gain:`65% lower token cost`
   * - Add four physically stable layout variations
     - 19.4 min + $11.1 + 301 lines of Python
     - 0 additional setup; Arena's native batched placement solver handles the variations

Beyond initial setup
^^^^^^^^^^^^^^^^^^^^

Arena provides more than a shorter environment definition:

* **Collision-free batched placement.** The placement solver samples the full layout together,
  rejects object collisions, and supports batched collision and robot-reachability validation.
  The workbench environment gets four physically stable layout variations without another custom
  placement script.
* **Composable environment changes.** Swapping the workbench takes a 16-line environment YAML;
  adding two tasks takes a 35-line environment YAML. Neither requires Python.
* **Evaluation at scale.** A 43-line Experiment YAML runs 100 episodes over two variations either
  locally or on an OSMO cluster. Experiment runs produce an HTML summary and episode recordings
  and work with Arena's built-in sensitivity analysis.

Going further
^^^^^^^^^^^^^

The environment's scene, task, and validation recipe are reusable independently. The same
registered assets and object references can support robot-policy evaluations or additional
workbench tasks without embedding environment-specific success logic in the runner.
