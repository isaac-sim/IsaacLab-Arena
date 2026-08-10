Run Agentic Environment Generation
----------------------------------

**Docker Container**: Base (see :doc:`../../../quickstart/installation` for more details)

The Arena environment generation agent infers an ``ArenaEnvGraphSpec`` YAML from a user prompt.
The agent runs in two modes:

* **GUI runner** — a browser live editor. Generate from a prompt, then edit,
  visualize, and simulation-preview the spec in the same session.
* **CLI runner** — a one-shot, non-interactive pipeline. It writes the YAML and editing can be done manually in a text editor.
  Use it for scripted or batch generation.

Name the background prims the task depends on — the counter top and the floor — so the agent emits them as
``object_references`` instead of spawning new assets. Also describe where the robot stands, because in a
kitchen the robot is placed in the scene rather than mounted on the task surface.

.. note::

   We recommend the GUI runner for this workflow because it takes interactive editing to disambiguate the
   countertop and refine the robot placement.

.. tab-set::

   .. tab-item:: GUI runner (live editing)
      :selected:

      Start the live editor and open ``http://localhost:8501`` in a browser:

      .. code-block:: bash

         python isaaclab_arena_examples/agentic_environment_generation/gui_runner.py

      In the ``Generate from prompt`` panel, enter the prompt and click
      ``Generate spec``:

      .. code-block:: text

         There is a counter top in the lightwheel_robocasa_kitchen background.
         DROID picks up a mustard bottle on the counter top and places it in a bowl.

      The generated environment graph contains the kitchen, the mustard bottle, the bowl, and a reference to
      a counter surface, but no placement for the robot itself.

      .. figure:: ../../../../images/agentic_environment_generation/agentic_ui_kitchen_pnp_prompt_counter.png
         :width: 100%
         :alt: GUI runner view of the environment graph spec generated from the first prompt.
         :align: center

         The first prompt generates the object placement and the pick-and-place task.

      Replace the prompt with one that also describes the robot placement and regenerate:

      .. code-block:: text

         There is a center-right counter top and a floor in the
         lightwheel_robocasa_kitchen background. DROID picks up a mustard bottle on
         the counter top and places it in a bowl. DROID is next to the counter top
         and on the floor.

      .. figure:: ../../../../images/agentic_environment_generation/agentic_ui_kitchen_pnp_prompt_robot.png
         :width: 100%
         :alt: GUI runner view of the environment graph spec with DROID placement relations.
         :align: center

         The second prompt adds the floor reference and the DROID ``on`` and ``next_to`` relations.

   .. tab-item:: CLI runner (no editing)

      Run the runner in ``resolve`` mode:

      .. code-block:: bash

         python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
            --mode resolve \
            --prompt "There is a center-right counter top and a floor in the lightwheel_robocasa_kitchen background. DROID picks up a mustard bottle on the counter top and places it in a bowl. DROID is next to the counter top and on the floor."

      The runner prints the generated graph and writes ``<env_name>.yaml`` under
      ``isaaclab_arena_environments/agent_generated/``.
