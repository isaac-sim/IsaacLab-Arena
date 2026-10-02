Discovering Arena's authoring interfaces
========================================

Discover before inventing
-------------------------

Before writing a new environment, inspect the existing registries and graph
schema. Run these commands from the repository root in the Arena runtime:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
      --mode schema > /tmp/arena_graph_schema.json
   python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
      --mode catalog --format json > /tmp/arena_catalog.json

Both modes run without SimulationApp, asset construction, asset generation,
or an inference request. ``catalog`` without ``--format json`` preserves the
human-readable vocabulary. The JSON catalogue includes assets, relations,
agent-ready tasks, and registered environment factories, including their typed
configuration parameters. The CLI loads the normal first-party environment
modules. For an extension, pass the existing
``--external_environment_class_path package.module:Factory`` option; discovery
imports its declarations without calling the factory. The Python catalogue
function describes the registries currently loaded by its caller.

Choose the smallest extension first:

* If a registered environment already implements the task, configure it in an
  Experiment Definition. The catalogue's ``environments`` entries describe its
  configuration fields. Start from the
  :doc:`first Experiment example <../../quickstart/arena_experiment>`.
* For a new combination of existing tasks and assets, use a graph YAML file.
  Graphs support flat sequential or parallel composition, an overall
  ``task.episode_length_s``, and ``task.desired_subtask_success_state``.
* Use a Python factory when the graph cannot express the required wiring or
  behavior. Reuse existing tasks and predicates inside it before adding new ones.

Use this checklist when authoring:

* Select exact registry names. Reuse an existing task or affordance before
  adding a new implementation. Registration alone does not make a task
  available to generation; the task must carry ``@agent_ready``.
* Read parameter types, required flags, defaults, enum values, units, and
  declared bounds. ``x-arena-reference`` means a graph node id, while ordinary
  string parameters retain literal values. Fixed tuples have ``prefixItems``
  and length constraints.
* Match each task argument's ``requires`` capabilities to the selected asset's
  ``provides`` capabilities. Existing affordance classes such as ``Openable``
  are discovered from inheritance. Explicit component metadata may add more.
* Read ``constraints`` and ``reset_semantics``. Textual constraints document
  obligations; runtime checks still establish geometry and physical validity.
* Validate the graph before building. Then inspect the selected environment's
  variation paths and task restrictions before enabling variations.

For a long task, compose existing tasks with ``CompositeTaskBase`` and use
``desired_subtask_success_state`` when earlier conditions must still hold at
completion. Reuse ``TrueForConsecutiveStepsCfg`` for stable completion and the
predicates in ``isaaclab_arena/tasks/predicates/`` for instantaneous conditions.
``PlaceInRegionTask`` provides full supported collision-shape containment with
settling and optional measured release. If several conditions share instruments,
certificates, or episode-owned mechanisms, use the optional ``TaskRuntimeCfg``
contract described in :doc:`the environment builder guide <../environment/env_builder>`.
That guide specifies update/reset ordering and its limitations. The
:doc:`variation guide <../variations/variations>` covers sampling, replay, and
the distinction between variation traces and placement layouts.

Run a graph through the existing Experiment Runner
--------------------------------------------------

An Experiment's ``environment.type`` accepts either a registered environment
name or a graph YAML path. A graph does not need a Python factory or a runner
wrapper. For example, save this as ``experiment.yaml`` after authoring your graph:

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
--experiment_config experiment.yaml``. This builds and steps the environment;
three steps do not establish task success or produce a complete episode. Use
``num_episodes`` instead of ``num_steps`` and a short task timeout to verify
episode reporting. Graph paths resolve from the process working directory, so
use an absolute path or run from the repository root consistently.

For a timeout smoke, reuse a graph that has neither ``env_cfg_override`` nor
``external_yaml`` with this sibling ``smoke.yaml`` instead of copying its task
and scene:

.. code-block:: yaml

   external_yaml: environment.yaml
   env_cfg_override:
     episode_length_s: 1.5

Select it in the same Experiment with the CLI override
``shared.environment.type=/absolute/path/to/smoke.yaml``. The include path is
relative to the including file; includes accept disjoint top-level keys and
one level only. This changes the generated environment's timeout; the task
declaration stays unchanged.

Inspect this Experiment's attached variations with ``--list_variations
--variations_format json --variations_output /tmp/arena_variations.json`` before
adding a Run's ``variations`` overrides. Mass sampler bounds are absolute
kilograms, not scale factors; derive them from the asset's configured nominal
mass. Static discovery does not measure USD mass or other physical properties.

Static validation
-----------------

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
      --mode validate --format json --env_spec /path/to/environment.yaml

Then build and step that graph with the current launcher's headless option:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
      --mode build --env_spec /path/to/environment.yaml \
      --num_envs 1 --num_steps 3 --viz none

Use ``--viz none`` for headless execution. If explicitly selecting a physics
preset, CLI spelling is ``--presets physx`` while typed Experiment YAML uses
``environment_builder: {presets: PHYSX}`` (the enum member name). Omitting the
preset keeps the default; inspect the runner's ``--help`` for CLI spellings.

The validation command returns zero on success and one on invalid input. Its
report has ``validation_scope: schema_and_declared_semantics`` and an ``issues`` array.
Every issue includes a code, JSON Pointer path, message, expected schema or
capabilities, and compatible choices where available. For example, a task
argument pointing at an object without ``Openable`` identifies
``/task/subtasks/0/params/openable_object`` and lists compatible node ids.

The Python equivalent is:

.. code-block:: python

   from isaaclab_arena.agentic_environment_generation.catalogues import build_catalogue_dict
   from isaaclab_arena.agentic_environment_generation.semantic_validation import validate_authoring_spec
   from isaaclab_arena.environment_spec.arena_env_graph_yaml_loader import load_env_graph_spec_dict

   catalogue = build_catalogue_dict()
   graph = load_env_graph_spec_dict("environment.yaml")
   report = validate_authoring_spec(graph)
   assert report["valid"], report["issues"]

A successful static check does not verify USD prim existence, collision
geometry, placement, reachability, reset stability, or policy success. Use the
normal ``--mode build`` flow for a simulation smoke check. Constructor
``**kwargs`` and unresolved annotations are explicitly reported rather than
treated as fully specified contracts. Omitted units or ranges mean unknown,
not unrestricted physical behavior.

Declaring component semantics
-----------------------------

Attach metadata to the registered class or factory itself. The registries
remain the source of component identity; there is no separate metadata registry.

.. code-block:: python

   from isaaclab_arena.agentic_environment_generation.authoring_metadata import (
       AuthoringMetadata,
       ParameterMetadata,
   )

   class MyTask:
       authoring_metadata = AuthoringMetadata(
           parameters={"clearance": ParameterMetadata(units="m", minimum=0.0)},
           requires={"target": ("Openable",)},
           constraints=("The target joint must be reachable by the embodiment.",),
           reset_semantics="Restores the target joint's configured reset angle.",
       )

       def __init__(self, target, clearance: float = 0.01):
           ...

The same declaration works on a callable registered with
``register_asset_factory``. Discovery inspects its signature and never invokes
the factory. Types and defaults come from the constructor; metadata supplies
only semantics that cannot be inferred reliably. Dataclass default factories
are named without executing them. The normal asset/task registration and
``@agent_ready`` rules still apply.

Variation paths and effective configuration
-------------------------------------------

Both runners accept ``--list_variations --variations_format json``. For a
machine-readable artifact, pass ``--variations_output`` instead of redirecting
stdout, which also contains simulator startup and shutdown messages:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
      --list_variations --variations_format json \
      --variations_output /tmp/arena_variations.json \
      franka_put_and_close_door

Unlike the static catalogue and schema commands above, attached variation
discovery starts SimulationApp and constructs the selected environment's assets;
it requires the normal simulation runtime and asset access. The output file
contains only the catalogue. ``--variations_output`` requires
``--list_variations``, creates missing parent directories, and replaces an
existing file. The console output remains available for inspection.

The Python interface uses the same attached variation objects and Hydra
configuration path resolution:

.. code-block:: python

   from isaaclab_arena.variations.variations_printing import get_variations_catalogue_as_dict

   variations = builder.get_all_variations()
   report = get_variations_catalogue_as_dict(
       variations,
       hydra_overrides=["part.mass.enabled=true", "part.mass.sampler_cfg.low=[0.2]"],
       restrictions=arena_env.task.get_variation_restrictions(),
   )

Use paths returned by the selected environment rather than assuming that the
example's ``part.mass`` exists. Each entry reports its exact enable path,
build-time or run-time application, effective configuration, and fields with
both the original values and post-override values. Unsupported task/variation
combinations include the task's restriction reason. This operation composes
configuration without sampling or mutating the variations; obtaining the
builder's attached objects uses the normal environment construction flow.

Variation classes can supply ``AuthoringMetadata.configuration`` keyed by
relative configuration paths, such as ``sampler_cfg.low``, to expose units and
documented bounds. The existing variation ``validate_cfg()`` and task
restriction hooks remain responsible for runtime enforcement.
