Run Agentic Environment Generation
----------------------------------

Complete the shared :ref:`agentic-env-gen-prerequisites` before running this workflow.

The Arena environment generation agent infers an ``ArenaEnvGraphSpec`` YAML from a user prompt.
The agent runs in two modes:

* **GUI runner** — a browser live editor. Generate from a prompt, then edit,
  visualize, and simulation-preview the spec in the same session.
* **CLI runner** — a one-shot, non-interactive pipeline. It writes the YAML and editing can be done manually in a text editor.
  Use it for scripted or batch generation.

SimReady search can be enabled in both modes. With it on, the agent will search the SimReady service
for assets that match the prompt, if assets are not found in the Arena asset library.

.. tab-set::

   .. tab-item:: GUI runner (live editing)
      :selected:

      Start the live editor and open ``http://localhost:8501`` in a browser:

      .. code-block:: bash

         python isaaclab_arena_examples/agentic_environment_generation/gui_runner.py

      Open the **SimReady search** expander and tick **Enable SimReady search**.

      In the ``Generate from prompt`` panel, enter the prompt and click
      ``Generate spec``:

      .. code-block:: text

         Droid picks up the pepsi can and the bean can from the maple table and places
         them into the mini plastic basket. There is a hammer next to the pepsi can and
         a tuna can on the table, and the bean can sits next to the basket.

      The returned YAML is loaded into the editor and assets are rendered on the right side of the editor.
      The task panel shows one row per subtask, so a composite task is visible as two
      ``PickAndPlaceTask`` rows under the root task.

      .. figure:: ../../../../images/agentic_environment_generation/tabletop_agentic_env_cans_simready_gui.png
         :width: 100%
         :alt: GUI runner view of the environment graph spec.
         :align: center

         GUI runner view of the environment graph spec: the YAML editor on the left, and the environment graph
         visualization and task description on the right.

   .. tab-item:: CLI runner (no editing)

      Run the runner in ``resolve`` mode with SimReady search enabled:

      .. code-block:: bash

         python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
            --mode resolve \
            --enable_simready_search \
            --prompt "Droid picks up the pepsi can and the bean can from the maple table and places them into the mini plastic basket. There is a hammer next to the pepsi can and a tuna can on the table, and the bean can sits next to the basket."

      The runner prints the generated graph and writes ``<env_name>.yaml`` under
      ``isaaclab_arena_environments/agent_generated/``.

To learn more about SimReady, see the
`SimReady Overview <https://docs.omniverse.nvidia.com/simready/latest/overview.html>`_.
