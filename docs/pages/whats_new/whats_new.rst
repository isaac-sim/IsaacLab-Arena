What's New
==========

The upcoming release of Isaac Lab-Arena, currently scheduled for 8 December 2026, adds three
capabilities, which you can already try on the ``main`` branch. Each summary below links to a
details page.

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: Agentic Authoring
      :link: agentic_authoring
      :link-type: doc

      .. image:: ../../images/whats_new/agentic_authoring_env_gen_as_skills.gif
         :width: 100%
         :alt: An agent generating an Arena evaluation environment using skills

      Improvements to agentic benchmark authoring. Agents are now fluent in Arena's placement and
      tasking system. In an authoring study, authoring an evaluation in Arena cost 85% less and was
      10.5× faster than generating a custom environment, with zero lines of Python.

   .. grid-item-card:: Newton-based Workflows
      :link: newton_evaluation
      :link-type: doc

      .. image:: ../../images/whats_new/newton_usb_insertion.gif
         :width: 100%
         :alt: USB insertion task running on the Newton physics backend

      Connect the Newton physics engine to Arena's evaluation tools to move beyond pick-and-place
      tasks to insertion, contact-rich assembly, and deformable tasks. Example workflows use
      Newton-backed DROID robot control.

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

      Use a multimodal vision-language model (for example, Astra) as a policy. Astra achieves
      state-of-the-art performance.

For the full list of changes in each release, see :doc:`../references/release_notes`.

.. toctree::
   :maxdepth: 1
   :hidden:

   agentic_authoring
   newton_evaluation
   vlm_policy
