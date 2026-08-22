GUI Runner
==========

The agentic environment-generation GUI is a Streamlit live editor for creating,
reviewing, editing, saving, visualizing, and simulation-previewing
``ArenaEnvGraphSpec`` YAML files.

What comes back depends on the model behind the selected endpoint — see
:doc:`model_selection`. Pick the endpoint in the GUI generation panel
(only endpoints whose API key is set are listed), or set
``ARENA_INFERENCE_ENDPOINT`` as the default. Since the model is
non-deterministic, review and correct what it returns.

Run the GUI from inside the Isaac Lab-Arena development container:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/gui_runner.py

You can also open an existing environment graph spec:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/gui_runner.py \
      --env_spec isaaclab_arena/tests/test_data/pick_and_place_maple_table_env_graph.yaml

By default, generated YAML files are written under
``isaaclab_arena_environments/agent_generated``. Use ``--out_dir`` to choose a
different output directory, or ``--port`` to run Streamlit on a different port.

.. figure:: ../../../images/agentic_environment_generation/agentic_env_gen_gui.gif
   :alt: Running the full agentic environment-generation pipeline

   Run the full pipeline from a natural-language prompt to generated YAML,
   automatically updated asset snapshots and graph visualization, and a
   simulation preview.

UI Panels
---------

The page is split into a left editing column and a right preview column.

Generate from prompt
   Choose an inference endpoint (only endpoints whose API key is set are
   listed), enter a natural-language task and scene description, then click
   ``Generate spec``. The GUI calls the environment-generation agent and loads
   the returned ``ArenaEnvGraphSpec`` YAML into the editor. When validation
   fails, the invalid YAML is still loaded into the editor and the validation
   traces are shown alongside it.

   .. image:: ../../../images/agentic_env_gen_gui_panel_generate.png
      :alt: Generate-from-prompt panel with inference endpoint selection
      :width: 50%

YAML editor
   Edit the generated or loaded ``ArenaEnvGraphSpec`` directly. The editor
   validates the YAML as you work and shows either a valid-spec summary or the
   parse/validation error. The ``Save YAML`` button writes the spec to
   ``<env_name>.yaml`` in the configured output directory. A searchable
   background prim-tree panel helps identify prim paths while editing.

   .. image:: ../../../images/agentic_env_gen_gui_panel_edit.png
      :alt: ArenaEnvGraphSpec YAML editor panel
      :width: 50%

Visualization
   Shows an automatically refreshed dashboard for valid YAML. The dashboard
   includes graph nodes, node thumbnails when available, the graph layout, task
   rows, and initial-state information. Snapshot axis overlays use red for
   :math:`+X`, green for :math:`+Y`, and blue for :math:`+Z`. If the YAML is
   invalid, the panel waits until the error is fixed before rendering.

   When the spec contains an entry under ``object_references``, expand
   **Background prim tree** to search the background USD for the referenced
   prim. The searchable, collapsible tree helps verify or correct the
   reference's ``prim_path``.

   .. grid:: 2
      :gutter: 2

      .. grid-item::

         .. image:: ../../../images/agentic_env_gen_gui_panel_visualize.png
            :alt: Environment graph visualization panel
            :width: 100%

      .. grid-item::

         .. image:: ../../../images/agentic_env_gen_gui_panel_visualize_kitchen.png
            :alt: Kitchen environment graph visualization panel
            :width: 100%

Sim preview
   Runs the full Arena environment construction from YAML, relation solving,
   and zero-action rollout in a SimulationApp side process. Controls let you
   set the number of parallel environments, zero-action steps, and environment
   spacing.

   .. image:: ../../../images/agentic_env_gen_gui_panel_sim_preview.png
      :alt: Simulation preview controls and viewport recording
      :width: 50%

   .. note::

      The preview uses the task's default viewer configuration and records the
      Kit viewport camera. Increase **Zero-action steps** (``num_steps``) to
      extend the rollout, and move the viewport camera in the Kit window to
      change the recorded view.

Editing and Update Flow
-----------------------

The main update flow is:

#. Type a prompt and click ``Generate spec``.
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
   zero-action rollout, and displays the recorded viewport video.

.. figure:: ../../../images/agentic_environment_generation/agentic_env_gen_gui_edit.gif
   :alt: Editing and validating generated environment YAML

   Edit the YAML with live validation, save it to a file, and review the asset
   snapshots and graph visualization as they update automatically.
