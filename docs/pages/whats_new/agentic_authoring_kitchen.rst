Kitchen Environment
===================

.. _whats-new-framework-value:

.. role:: gain
   :class: arena-value-gain

.. container:: arena-value-eyebrow

   Isaac Lab - Arena · Agentic Authoring · Kitchen Task

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
 * reached a working environment **4× faster**,
 * wrote **no custom code**,
 * used Arena's relational placement system to achieve **highly-varied layouts**,
 * and used Arena's battle-tested tasking system to describe **complex composite tasks**.

The agent's task
^^^^^^^^^^^^^^^^

The agent gets a single prompt describing a kitchen pick-and-place evaluation environment:
two pieces of fruit must be placed in two bowls on a cluttered kitchen countertop.

.. dropdown:: The prompt given to the agent

   Can you build a new evaluation environment in a kitchen? The evaluation will require an agent
   to perform a composite pick and place task. Write it at
   ``isaaclab_arena_environments/kitchen_bench/kitchen_bench_round2_peninsula_banana_and_orange_in_bowls.yaml``.
   Use the Replicator peninsula kitchen with the DROID robot on its stand, standing on the floor
   at the long side of the island and facing it. Set the stand to 0.8 m. Put two bowls on the
   countertop, one behind the other from the robot's view: a wooden bowl closer to the robot and a
   red YCB bowl farther away. Put a banana next to the wooden bowl and an orange next to the red
   bowl, both on the countertop. The layout should vary as much as possible between resets while
   keeping this arrangement: the robot should be able to stand anywhere along the island's long
   side, with the bowls and fruit moving along the counter with it, on either side of the things
   already on the counter. The task is to put the banana in the wooden bowl and the orange in the
   red bowl, and it only succeeds when both are in their bowls. Don't look at the remote repo or
   GitHub.

The agent's solution
^^^^^^^^^^^^^^^^^^^^

The agent wrote a 140-line environment YAML that combines Arena's placement relations with two
existing pick-and-place tasks. The robot follows the bowls along the counter, so each reset puts
the whole group somewhere new.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: ../../images/whats_new/peninsula_kitchen_10_reset_layouts_arena_agent_authored.gif
         :width: 100%
         :alt: Ten reset layouts of the Arena agent-authored peninsula kitchen environment

         Ten reset layouts of the agent-authored environment.

   .. grid-item::

      .. figure:: ../../images/whats_new/peninsula_kitchen_arena_agent_authored_spatial_relations.svg
         :width: 100%
         :alt: Graph of the spatial relations in the agent-authored environment YAML

         The spatial relations the agent wrote in the environment YAML.

Analysis
^^^^^^^^

Let's compare the solution above to the baseline solution written without Arena's evaluation building blocks.

.. container:: arena-value-stats

   .. container:: arena-value-stat

      **4× faster**

      to a working environment (34 min vs. 8 min)

   .. container:: arena-value-stat

      **0 Python**

      authored for Arena (140 lines of YAML)

   .. container:: arena-value-stat

      **0 debug code**

      vs. about 950 throwaway lines with Isaac Lab

.. list-table::
   :header-rows: 1
   :widths: 22 22 22 34
   :class: arena-value-table

   * - Measurement
     - Without Arena
     - With Arena
     - Details
   * - Authoring time
     - 34 min
     - 8 min

       :gain:`4× faster`
     - Time until agent reached an environment that passes task success checks.
   * - Tokens
     - 15.3M
     - 4.4M
     - Input and output tokens over all model calls (115 vs. 49).
   * - Model cost
     - $20.97
     - $6.83

       :gain:`3× cheaper`
     - GPT-6 Astra through Codex, at standard API rates.
   * - Environment code
     - 388 lines of custom Python code
     - 0 lines of Python + 140 lines of environment YAML

       :gain:`no new code to maintain`
     - Environment is a composition of existing Arena tasks and spatial relations.
   * - Debug code
     - About 950 lines of custom throwaway Python code
     - 0
     - Arena's validation commands replace one-off checking scripts.
   * - Task success
     - Hand-written environment-specific code
     - ``CompositeTask`` with two ``PickAndPlaceTask`` sub-tasks
     - Reuses Arena's existing task and its success condition.
   * - Asset placement
     - Robot fixed; objects have hand-coded collision-free uniform ranges.
     - Spatial relation graph solved to produce collision-free highly-varied placements.
     - Arena places objects relative to the counter and each other, avoids the kitchen's built-in
       objects using their actual (mesh) geometry, and checks IK reachability on every layout.

In summary, with Arena, the agent:
 * reached a working environment **4× faster**,
 * wrote **no custom code**,
 * used Arena's relational placement system to achieve **highly-varied layouts**,
 * and used Arena's battle-tested tasking system to describe **complex composite tasks**.

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
