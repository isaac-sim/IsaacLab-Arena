CLI Runner
==========

``cli_runner.py`` is the scripted entry point for inspecting
the generation interface, resolving a prompt, building an existing graph spec,
or running the complete pipeline. Run it inside the Isaac Lab-Arena development
container:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
      --mode full \
      --prompt "Franka picks up a cube from the maple table and places it into a bowl."

Modes and system stages
-----------------------

The ``--mode`` option selects which parts of the
:doc:`system overview <system_overview>` run:

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Mode
     - What it does
     - System stages
   * - ``schema``
     - Prints the Pydantic JSON schema for ``ArenaEnvGraphSpec`` and exits.
       It does not call the model or start Isaac Sim.
     - Inspects the **spec** contract.
   * - ``catalog``
     - Prints the asset, relation, and task catalogs exposed to the agent and
       exits. It does not call the model or start Isaac Sim.
     - Inspects the **environment generation agent** inputs.
   * - ``prim_tree``
     - Prints the background prim tree from ``--env_spec`` and exits. It does
       not call the model or start Isaac Sim.
     - Inspects background object-reference paths.
   * - ``resolve``
     - Sends the prompt and catalogs to the agent, validates the returned spec,
       prints its graph, and writes ``<env_name>.yaml``.
     - **Prompt** → **agent** → **spec**.
   * - ``build``
     - Loads ``--env_spec``, converts it to an Arena environment,
       builds it, and runs a zero-action policy.
     - **Spec** → **Arena environment** → **placement and validation** →
       **evaluation smoke test**.
   * - ``full``
     - Resolves a prompt, writes the generated YAML, builds it, and runs a
       zero-action policy in one process. This is the default.
     - **Prompt** through **evaluation**, without a manual-edit pause.

Use ``resolve`` followed by ``build`` when the generated YAML needs manual
review:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
      --mode resolve \
      --prompt "Droid places the mustard bottle in the grey bin."

   # Review or edit the generated YAML, then:
   python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
      --mode build \
      --env_spec isaaclab_arena_environments/agent_generated/<env_name>.yaml \
      --headless

Runner options
--------------

These options are defined by the environment generation runner:

.. list-table::
   :header-rows: 1
   :widths: 28 42 30

   * - Option
     - Purpose
     - System stage
   * - ``--mode {full,resolve,build,schema,catalog,prim_tree}``
     - Selects the phases described above. Default: ``full``.
     - Pipeline selection
   * - ``--prompt TEXT``
     - Natural-language environment and task description.
     - Prompt
   * - ``--model MODEL_ID``
     - Overrides the generation agent's default model.
     - Agent
   * - ``--inference_endpoint {internal,public,openai}``
     - Selects the configured inference endpoint. See
       :doc:`model_selection` for endpoint defaults and API keys.
     - Agent
   * - ``--temperature FLOAT``
     - Sets model sampling temperature. Default: ``0.2``.
     - Agent
   * - ``--out_dir PATH``
     - Chooses where generated graph-spec YAML files are written. Default:
       ``isaaclab_arena_environments/agent_generated``.
     - Spec
   * - ``--enable_simready_search``
     - Searches SimReady for prompt objects missing from Arena's asset catalog.
     - Agent
   * - ``--simready_source SOURCE``
     - Chooses the SimReady search backend. Default: ``isaac-sim-ga``.
     - Agent
   * - ``--simready_s3_url URL``
     - Overrides the S3 root for S3-based SimReady sources.
     - Agent
   * - ``--simready_service_url URL``
     - Overrides the hosted USD Search service URL.
     - Agent
   * - ``--simready_max_results_per_object N``
     - Limits retained SimReady hits per searched object. Default: ``1``.
     - Agent
   * - ``--num_steps N``
     - Sets the number of zero-action simulation steps. Default: ``20``.
     - Evaluation smoke test

Build and simulation options
----------------------------

The runner also inherits Arena and Isaac Lab launcher options. The options most
relevant to ``build`` and ``full`` are:

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Option
     - Purpose
     - System stage
   * - ``--env_spec PATH``
     - Input graph-spec YAML. Required by ``--mode build`` and ``prim_tree``.
     - Spec
   * - ``--headless``
     - Runs Isaac Sim without a viewport.
     - Build and evaluation
   * - ``--num_envs N``
     - Sets the number of parallel environments. Default: ``1``.
     - Arena environment
   * - ``--env_spacing FLOAT``
     - Sets spacing between parallel environments. Default: ``30.0``.
     - Arena environment
   * - ``--seed N``
     - Sets the simulation random seed. Default: ``42``.
     - Build and evaluation
   * - ``--no_solve_relations``
     - Disables spatial-relation solving.
     - Placement solving
   * - ``--placement_seed N``
     - Makes object-placement solving reproducible.
     - Placement solving
   * - ``--resolve_on_reset`` / ``--no-resolve_on_reset``
     - Enables or disables re-placement of pooled objects at reset.
     - Placement solving
   * - ``--disable_fabric``
     - Uses USD I/O instead of Fabric.
     - Arena environment
   * - ``--mimic``
     - Builds a mimic environment for demonstration workflows.
     - Environment compilation
   * - ``--presets {physx,newton}``
     - Selects a physics-backend preset.
     - Environment compilation

Run the script with ``--help`` for the complete launcher-specific option list,
including device and livestream settings supplied by Isaac Lab.
