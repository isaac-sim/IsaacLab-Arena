Run Agentic Environment Generation
----------------------------------

**Docker Container**: Base (see :doc:`../../../quickstart/installation` for more details)

The Arena environment generation agent infers an ``ArenaEnvGraphSpec`` YAML from a user prompt.
The agent runs in two modes:

* **GUI runner** — a browser live editor. Generate from a prompt, then edit,
  visualize, and simulation-preview the spec in the same session.
* **CLI runner** — a one-shot, non-interactive pipeline. It writes the YAML and editing can be done manually in a text editor.
  Use it for scripted or batch generation.

.. tab-set::

   .. tab-item:: GUI runner (live editing)
      :selected:

      Start the live editor and open ``http://localhost:8501`` in a browser:

      .. code-block:: bash

         python isaaclab_arena_examples/agentic_environment_generation/gui_runner.py

      In the ``Generate from prompt`` panel, enter the prompt and click
      ``Generate spec``:

      .. code-block:: text

         Droid picks up the banana from the maple table and places it on the plate.
         There are two bagels and one bowl on the table.

      The returned YAML is loaded into the editor and assets are rendered on the right side of the editor.
      You can see the thumbnails of each object in the scene and its spatial relationships with each other.
      You can also see the task description in the lower part of the editor.

      .. figure:: ../../../../images/tabletop_agentic_env_banana_bagel_plate_gui.png
         :width: 100%
         :alt: GUI runner view of the environment graph spec.
         :align: center

         GUI runner view of the environment graph spec: the YAML editor on the left, and the environment graph
         visualization and task description on the right.

   .. tab-item:: CLI runner (no editing)

      Run the runner in ``resolve`` mode:

      .. code-block:: bash

         python isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py \
            --mode resolve \
            --prompt "Droid picks up the banana from the maple table and places it on the plate. There are two bagels and one bowl on the table."

      The runner prints the generated graph and writes ``<env_name>.yaml`` under
      ``isaaclab_arena_environments/agent_generated/``.
