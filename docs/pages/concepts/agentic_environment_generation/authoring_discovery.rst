Discover existing authoring features
====================================

Start with the existing registries and runners before writing a new factory,
task, or variation. Discovery describes the interfaces already available in
this checkout; it does not add task behavior or simulator capabilities.

Inspect the catalogue and graph schema
--------------------------------------

Run inside the Arena runtime, before starting simulation:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
      --mode catalog --format json > /tmp/arena_catalogue.json
   python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
      --mode schema > /tmp/arena_graph_schema.json

The versioned catalogue lists registered environments and their typed config
fields, assets, agent-ready tasks, and relations. Constructor types, defaults,
enum choices and inherited affordances come from existing declarations. Optional
``authoring_metadata`` on a component supplies units, constraints or reset
semantics; absent metadata means these details have not been declared.
Catalogue and schema inspection do not construct assets or start SimulationApp.
Use ``--external_environment_class_path`` to load an extension's declarations.
The default text catalogue remains available.

Choose the smallest extension:

* Configure a registered environment when it already implements the task. Its
  catalogue entry describes the configuration fields. Start from the
  :doc:`first Experiment example <../../quickstart/arena_experiment>`.
* Compose existing tasks and assets in graph YAML when the graph schema exposes
  the required wiring and options.
* Use a Python factory for options the graph cannot express, reusing existing
  tasks, predicates and variations inside it. For example, the Python composite
  task supports current final-state requirements that the graph does not expose.

Constructor metadata does not expand the graph's supported inputs. In particular,
existing graph validation treats string task parameters as graph-node references;
a constructor's literal string option may still require a Python factory.

Validate before building
------------------------

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
      --mode validate --format json --env_spec environment.yaml

This opt-in mode calls the existing graph schema and agent-ready catalogue
checks. It returns zero on success and one on invalid input. JSON reports contain
``valid``, ``validation_scope: schema_and_catalogue`` and ``issues`` with messages
and JSON Pointer paths (``/`` for catalogue-wide errors). Checks cover graph
structure/references, registry names and required/unsupported parameters.
Declared metadata is descriptive; this command does not add enforcement of
parameter units, bounds, capabilities or physical feasibility.

Build and simulation checks remain necessary for asset availability, geometry,
placement, reachability and task completion:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
      --mode build --env_spec environment.yaml --num_envs 1 --num_steps 3 --viz none

Use ``--viz none`` for headless execution. CLI preset spelling is ``--presets
physx``; typed Experiment YAML uses ``presets: PHYSX``. Omitting it keeps the default.

Reuse the Experiment Runner
---------------------------

An Experiment's ``environment.type`` accepts a registered environment name or a
graph YAML path. A graph needs no custom factory or runner wrapper:

.. code-block:: yaml

   shared:
     environment:
       type: /absolute/path/to/environment.yaml
       enable_cameras: false
     environment_builder:
       num_envs: 1
     policy:
       type: zero_action
     rollout_limit:
       num_steps: 3
   runs:
     baseline: {}

Run ``python isaaclab_arena/evaluation/experiment_runner.py --viz none
--experiment_config experiment.yaml``. Graph paths resolve from the process
working directory. Three steps verify construction and stepping; complete-episode
reporting needs ``num_episodes`` instead of ``num_steps`` and a suitable timeout.

For a short timeout, reuse a graph containing neither ``external_yaml`` nor
``env_cfg_override`` through a sibling ``smoke.yaml``:

.. code-block:: yaml

   external_yaml: environment.yaml
   env_cfg_override:
     episode_length_s: 1.5

Select it with ``shared.environment.type=/absolute/path/to/smoke.yaml`` on the
same Experiment command. Includes resolve relative to the including file, accept
disjoint top-level keys, and support one level. This changes the generated
environment's timeout while preserving the task declaration.

Discover attached variations
----------------------------

Add ``--list_variations --variations_format json --variations_output
/tmp/arena_variations.json`` to the existing policy or Experiment Runner command.
This starts SimulationApp and constructs the selected environment definition to
inspect its attached variations; it does not run a rollout or sample variations.
Parse the output file because console output also contains simulator logs.

Entries describe exact enable/override paths, field types, initial and effective
values, build/reset timing, and any declared semantics. Use those paths in a
Run's ``variations`` mapping. Mass bounds are absolute kilograms, not scale
factors; static discovery does not measure native USD masses.

See the :doc:`variation guide <../variations/variations>` for existing sampling
and configuration behavior. Discovery does not introduce independent variation
seeds, recorded-mass replay or new task restrictions.
