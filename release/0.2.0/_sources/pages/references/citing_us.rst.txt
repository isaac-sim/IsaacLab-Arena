Citing Isaac Lab-Arena
======================

We encourage the community to build and publish benchmarks on Isaac Lab-Arena. The recommended workflow is:

1. **Maintain your benchmark in your own repository.** Create a branch or package that integrates with
   Isaac Lab-Arena (e.g. an ``IsaacLab-Arena`` branch). See
   `RoboTwin <https://github.com/RoboTwin-Platform/RoboTwin/tree/IsaacLab-Arena>`_ for a reference
   example. For detailed setup instructions - including repository layout, Dockerfile setup, and how to
   register custom environments/robots/tasks - see the
   `Arena in Your Repository <https://isaac-sim.github.io/IsaacLab-Arena/main/pages/arena_in_your_repo/index.html>`_
   guide.
2. **Reference your benchmark and Isaac Lab-Arena in publications.** When publishing on ArXiv or
   elsewhere, cite both your benchmark (by name, with a link to your repository) and Isaac Lab-Arena
   as the underlying evaluation framework.
3. **List it here.** Open a PR to add your benchmark to the
   `Published Benchmarks <https://isaac-sim.github.io/IsaacLab-Arena/main/pages/references/published_benchmarks.html>`_
   list above. This README serves as the single source of truth for the Arena benchmark ecosystem so that
   the community can discover and reuse.

If you use Isaac Lab-Arena in your research, please cite:

.. code-block:: bibtex

   @misc{isaaclab-arena2025,
       title   = {Isaac Lab-Arena: Composable Environment Creation and Policy Evaluation for Robotics},
       author  = {{NVIDIA Isaac Lab-Arena Contributors}},
       year    = {2025},
       url     = {https://github.com/isaac-sim/IsaacLab-Arena}
   }
