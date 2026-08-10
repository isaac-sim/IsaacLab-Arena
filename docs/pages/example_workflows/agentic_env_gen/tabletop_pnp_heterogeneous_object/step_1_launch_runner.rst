Run Agentic Environment Generation
----------------------------------

**Docker Container**: Base (see :doc:`../../../quickstart/installation` for more details)

The Arena environment generation agent infers an ``ArenaEnvGraphSpec`` YAML from a user prompt.
The agent runs in two modes:

* **GUI runner** — a browser live editor. Generate from a prompt, then edit,
  visualize, and simulation-preview the spec in the same session.
* **CLI runner** — a one-shot, non-interactive pipeline. It writes the YAML and editing can be done manually in a text editor.
  Use it for scripted or batch generation.

Describe the object to pick up as a category rather than as a single asset, and
say that it varies across environments. That is the cue for the agent to emit an
``object_sets`` entry instead of a fixed ``objects`` entry.

.. tab-set::

   .. tab-item:: GUI runner (live editing)
      :selected:

      Start the live editor and open ``http://localhost:8501`` in a browser:

      .. code-block:: bash

         python isaaclab_arena_examples/agentic_environment_generation/gui_runner.py

      In the ``Generate from prompt`` panel, enter the prompt and click ``Generate spec``:

      .. code-block:: text

         Droid picks up a fruit from the maple table and places it into the bowl on the table.
         Each environment should get a different fruit.

      The returned YAML is loaded into the editor and assets are rendered on the right side of the editor.
      An object set is drawn as a single node, with a thumbnail per member, so you can
      check the whole set at a glance.

      .. figure:: ../../../../images/agentic_environment_generation/tabletop_agentic_env_fruits_gui.png
         :width: 100%
         :alt: GUI runner view of the environment graph spec.
         :align: center

         GUI runner view of the environment graph spec: the YAML editor on the left, and the environment graph
         visualization and task description on the right.

   .. tab-item:: CLI runner (no editing)

      Run the runner in ``resolve`` mode:

      .. code-block:: bash

         python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
            --mode resolve \
            --prompt "Droid picks up a fruit from the maple table and places it into the bowl on the table. Each environment should get a different fruit."

      The runner prints the generated graph and writes ``<env_name>.yaml`` under
      ``isaaclab_arena_environments/agent_generated/``.
