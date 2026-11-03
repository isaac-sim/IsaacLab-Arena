Agentic Evaluation Authoring Improvements
=========================================

Agents are now fluent in Arena's placement and tasking system. We compared a coding agent
authoring the same evaluation environment with Arena and with Isaac Lab alone, in two settings.
Each card below links to the results for one environment.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Workbench
      :link: agentic_authoring_workbench
      :link-type: doc

      .. image:: ../../images/whats_new/workbench_cleanup_10_reset_layouts.gif
         :width: 100%
         :alt: Ten reset layouts of the Arena workbench cleanup environment

      An agent authors a workbench environment and extends it with new layouts and tasks.

   .. grid-item-card:: Kitchen
      :link: agentic_authoring_kitchen
      :link-type: doc

      .. image:: ../../images/whats_new/peninsula_kitchen_10_reset_layouts_arena_agent_authored.gif
         :width: 100%
         :alt: Ten reset layouts of the Arena agent-authored peninsula kitchen environment

      An agent authors a kitchen pick-and-place environment with varied layouts and success checks.

To try agentic authoring yourself, see :doc:`../concepts/agentic_environment_generation/index`.
For the full list of changes in each release, see :doc:`../references/release_notes`.

.. toctree::
   :maxdepth: 1
   :hidden:

   agentic_authoring_workbench
   agentic_authoring_kitchen
