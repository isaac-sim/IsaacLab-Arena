Agentic Environment Generation GUI
==================================

The agentic environment-generation GUI is a Streamlit live editor for creating,
reviewing, editing, saving, visualizing, and simulation-previewing
``ArenaEnvGraphSpec`` YAML files.

Run the GUI from inside the Isaac Lab-Arena development container:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/gui_runner.py

You can also open an existing environment graph spec:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/gui_runner.py \
      --env_graph_spec_yaml isaaclab_arena/tests/test_data/pick_and_place_maple_table_env_graph.yaml

By default, generated YAML files are written under
``isaaclab_arena_environments/agent_generated``. Use ``--out_dir`` to choose a
different output directory, or ``--port`` to run Streamlit on a different port.

.. figure:: ../../../images/agentic_env_gen_gui.gif
   :alt: Agentic environment generation GUI

   The GUI is intended as a human-in-the-loop review surface: generate a draft
   from a prompt, inspect the compiled graph, edit the YAML, and preview the
   result before using it in policy evaluation.

UI Panels
---------

The page is split into a left editing column and a right preview column.

Generate from prompt
   Enter a natural-language task and scene description, then click
   ``Generate``. The GUI calls the environment-generation agent and loads the
   returned ``ArenaEnvGraphSpec`` YAML into the editor. After a successful
   generation, the panel can show the agent reasoning from the last run.

YAML editor
   Edit the generated or loaded ``ArenaEnvGraphSpec`` directly. The editor
   validates the YAML as you work and shows either a valid-spec summary or the
   parse/validation error. The ``Save YAML`` button writes the spec to
   ``<env_name>.yaml`` in the configured output directory.

Visualization
   Shows an automatically refreshed dashboard for valid YAML. The dashboard
   includes graph nodes, node thumbnails when available, the graph layout, task
   rows, and initial-state information. If the YAML is invalid, the panel waits
   until the error is fixed before rendering.

Sim preview
   Runs the full Arena environment construction from YAML, relation
   solving, and zero-action rollout in a SimulationApp side process.
   Controls let you set the number of parallel environments, zero-action steps,
   and environment spacing. The result shows the viewport at the start and after
   the requested rollout steps.

Editing and Update Flow
-----------------------

The main update flow is:

#. Type a prompt and click ``Generate``.
#. The agent receives the prompt and returns an ``ArenaEnvGraphSpec``. The
   generated YAML is loaded into the editor and saved as ``<env_name>.yaml``.
#. The user can manually edit the YAML in the editor. Once the edited YAML
   passes validation, click ``Save YAML`` to write it to the output directory.
   Use ``Change output directory`` to choose a different output location. The
   filename is derived from ``env_name`` and can be changed by editing
   ``env_name`` in the YAML editor.
#. The graph visualization refreshes automatically when the valid YAML text
   changes.
#. Click ``Run relation solver preview`` to manually trigger the simulation
   preview. This action sends the current editor text to the SimApp preview
   service, builds the Arena environment, solves relations, runs the configured
   zero-action rollout, and displays two viewport captures.
