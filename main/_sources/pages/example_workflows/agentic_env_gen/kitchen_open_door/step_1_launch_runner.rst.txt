Run Agentic Environment Generation
----------------------------------

**Docker Container**: Base (see :doc:`../../../quickstart/installation` for more details)

The Arena environment generation agent infers an ``ArenaEnvGraphSpec`` YAML from a user prompt.
The agent runs in two modes:

* **GUI runner** — a browser live editor. Generate from a prompt, then edit,
  visualize, and simulation-preview the spec in the same session.
* **CLI runner** — a one-shot, non-interactive pipeline. It writes the YAML and editing can be done manually in a text editor.
  Use it for scripted or batch generation.

Name the background prims the task depends on — the fridge and the floor — so the agent emits them as
``object_references`` instead of spawning new assets, and state the openness the door has to reach to complete the task.
That is, the task is to open the fridge door to the 0.2 openness threshold in the ``OpenDoorTask``.

.. tab-set::

   .. tab-item:: GUI runner (live editing)
      :selected:

      Start the live editor and open ``http://localhost:8501`` in a browser:

      .. code-block:: bash

         python isaaclab_arena_examples/agentic_environment_generation/gui_runner.py

      In the ``Generate from prompt`` panel, enter the prompt and click
      ``Generate spec``:

      .. code-block:: text

         There is a floor and a fridge in the lightwheel_robocasa_kitchen kitchen.
         DROID is on the floor, next to the fridge with 0.1 meter distance and facing
         it. DROID opens the fridge door to the 0.2 openness threshold.

      .. figure:: ../../../../images/agentic_ui_kitchen_open_door.png
         :width: 100%
         :alt: GUI runner view of the environment graph spec.
         :align: center

         The prompt generates the floor and fridge references, DROID placement
         relations, and the fridge-opening task.

   .. tab-item:: CLI runner (no editing)

      Run the runner in ``resolve`` mode:

      .. code-block:: bash

         python isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py \
            --mode resolve \
            --prompt "There is a floor and a fridge in the lightwheel_robocasa_kitchen kitchen. DROID is on the floor, next to the fridge with 0.1 meter distance and facing it. DROID opens the fridge door to the 0.2 openness threshold."

      The runner prints the resolved graph and writes ``<env_name>.yaml`` under
      ``isaaclab_arena_environments/agent_generated/``.
