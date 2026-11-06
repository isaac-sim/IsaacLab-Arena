What's New
==========

The Isaac Lab-Arena **v0.4 Developer Preview** is available now on ``main``,
with three new capabilities for authoring and running evaluations.

The tested v0.4 release, called the **Qualified Release**, is expected around 8 December.

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: Agentic Authoring
      :link: agentic_authoring
      :link-type: doc

      .. image:: ../../images/whats_new/peninsula_kitchen_10_reset_layouts_arena_agent_authored.gif
         :width: 100%
         :alt: Ten reset layouts of the Arena agent-authored peninsula kitchen environment

      Use coding agents to author evaluation environments with Arena's placement relations and
      composite tasks.

   .. grid-item-card:: Newton-Based Workflows
      :link: newton_evaluation
      :link-type: doc

      .. video:: ../../images/whats_new/arena_v04_newton_deformables_rendered.mp4
         :width: 100%
         :alt: DROID placing a deformable block on a plate, simulated with Newton
         :autoplay:
         :loop:
         :muted:
         :playsinline:
         :nocontrols:

      Evaluate policies on insertion, contact-rich assembly, and deformable-object tasks using
      the Newton physics backend and Arena's evaluation tools.

   .. grid-item-card:: Multimodal VLM as a Policy
      :link: vlm_policy
      :link-type: doc

      .. video:: ../../images/whats_new/vlm_policy_astra_rollout.mp4
         :width: 100%
         :alt: Real-robot rollout of GPT-6 Astra stacking colored blocks, shown from the wrist camera
         :autoplay:
         :loop:
         :muted:
         :playsinline:
         :nocontrols:

      Use a multimodal vision-language model such as Astra as a robot policy, and compare action strategies
      using Arena's evaluation tools.

.. toctree::
   :maxdepth: 1
   :hidden:

   agentic_authoring
   newton_evaluation
   vlm_policy
