Launch the Runner
-----------------

**Docker Container**: Base (see :doc:`../../../quickstart/installation` for more details)

The runner resolves the prompt into an ``ArenaEnvGraphSpec`` YAML. It comes in
two modes:

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

   .. tab-item:: CLI runner (no editing)

      Run the runner in ``resolve`` mode with SimReady search enabled:

      .. code-block:: bash

         python isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py \
            --mode resolve \
            --enable_simready_search \
            --prompt "Droid picks up the pepsi can and the bean can from the maple table and places them into the mini plastic basket. There is a hammer next to the pepsi can and a tuna can on the table, and the bean can sits next to the basket."

      The runner prints the resolved graph and writes ``<env_name>.yaml`` under
      ``isaaclab_arena_environments/agent_generated/``.

.. todo:: add link to concept page covering simready search
