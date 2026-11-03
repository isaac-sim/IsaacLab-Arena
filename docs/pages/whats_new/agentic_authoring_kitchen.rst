Kitchen Environment
===================

.. _whats-new-framework-value:

.. role:: gain
   :class: arena-value-gain

.. container:: arena-value-eyebrow

   Arena vs. Isaac Lab · Authoring study · 3 Nov 2026

Faster authoring. Evaluation infrastructure included.
-----------------------------------------------------

We asked the same coding agent to build the same kitchen evaluation environment twice: once as
Arena YAML, and once as a standalone Isaac Lab Python script. With Arena, the agent reached a
working environment faster, wrote no Python, and reused Arena's placement, validation, and task
checks instead of writing its own.

The problem
^^^^^^^^^^^

The agent gets a single prompt: build a kitchen pick-and-place evaluation with a DROID robot, two
bowls and two pieces of fruit, whose layout varies as much as possible between resets, and which
succeeds only when both fruits are in their bowls.

.. dropdown:: The prompt given to the agent (Arena version, without asset paths and reporting instructions)

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

   * - Decision factor
     - Isaac Lab (agent writes custom code)
     - Isaac Lab - Arena (agent writes YAML)
     - Evidence
   * - Authoring time
     - 34 min
     - 8 min

       :gain:`4× faster`
     - Time until every check passed. About 20 minutes of the Isaac Lab time went into a DROID
       physics instability that Arena's DROID configuration already avoids.
   * - Tokens
     - 15.3M (99% cached)
     - 4.4M (97% cached)
     - Input and output tokens over all model calls (115 vs. 49).
   * - Model cost
     - $20.97
     - $6.83

       :gain:`3× cheaper`
     - GPT-6 Astra through Codex, at standard API rates: $10 per million input tokens, $1 per
       million cached input tokens, $50 per million output tokens.
   * - Environment code
     - 388 lines of Python
     - 0 lines of Python + 140 lines of YAML
     - Fully declarative; the YAML selects existing Arena tasks and placement relations.
   * - Throwaway debug code
     - About 950 lines
     - 0
     - Arena's two validation commands replace one-off checking scripts.
   * - Task success check
     - Hand-written environment-specific code
     - 2 × ``PickAndPlaceTask``
     - Reuses Arena's existing task and its success condition.
   * - Asset placement
     - Robot fixed; bounding boxes around the clutter
     - Robot moves about 0.9 m along the island; the kitchen's mesh
     - Arena places objects relative to the counter and each other, avoids the kitchen's built-in
       objects using their actual geometry, and checks IK reachability on every layout.
   * - Parallel environments
     - One
     - Many
     - The Isaac Lab script hard-codes coordinates for this one kitchen and runs one environment.

Going further
^^^^^^^^^^^^^

The environment YAML plugs straight into an Arena Experiment, so the same file that runs locally
can be evaluated across a multi-node cluster with OSMO. See :doc:`../example_workflows/multi_node_evaluation/multi_node_evaluation`.
