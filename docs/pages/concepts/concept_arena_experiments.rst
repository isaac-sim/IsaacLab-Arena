Arena Experiments
=================

An Arena Experiment groups one or more named Runs into one evaluation. Each Run selects an
existing Environment Definition and combines it with a policy and the settings that control the
rollout.

The **Experiment Definition** is the YAML file that describes those Runs. It is the main interface
for evaluation: run the same definition locally with the Experiment Runner, or submit it to OSMO
for managed execution.

.. note::

   A Run can reuse either form of :doc:`Environment Definition
   <environment/environment_definition>`:

   * a registered Python environment, selected by name in ``environment.type``; or
   * an existing ``ArenaEnvGraphSpec`` YAML file, selected by its path in ``environment.type``.

   Different Runs in the same Experiment can use different forms.

   The Experiment Definition describes the evaluation. The Environment Definition describes the
   scene, embodiment, and task. Each Run builds a fresh runtime environment from that definition.

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: Experiment Definition
      :class-card: sd-shadow-sm

      The typed YAML file. It holds shared values and one or more named Runs.

   .. grid-item-card:: Run
      :class-card: sd-shadow-sm

      One Environment Definition combined with one policy and rollout settings. A Run can contain
      many parallel simulated environments.

   .. grid-item-card:: Execution
      :class-card: sd-shadow-sm

      The Experiment Runner executes locally. OSMO schedules the same Runs on managed compute.


The Experiment Definition
-------------------------

The YAML has a required ``runs`` mapping. A ``shared`` mapping is optional. When present, its
values are reused by every Run. Each key below ``runs`` is a Run name:

.. code-block:: yaml

   shared:
     environment:
       type: pick_and_place_maple_table
       embodiment: droid_rel_joint_pos
       hdr: home_office_robolab
     policy:
       type: zero_action
     rollout_limit:
       num_steps: 50

   runs:
     baseline: {}

     billiard_hall:
       environment:
         hdr: billiard_hall_robolab

``baseline`` inherits all shared values. ``billiard_hall`` inherits them too, then replaces only
the background. The Run name comes from its key below ``runs``; there is no separate ``name``
field. Arena uses this name in command-line overrides, output directories, and reports.

Runs keep their YAML order and execute locally in that order.


Reuse values with ``shared``
----------------------------

``shared`` removes repetition. Arena applies these values to every Run, then places each Run's
own values on top. A Run only needs to declare what is different.

Values are applied in this order, from lowest to highest priority:

.. code-block:: text

   typed configuration defaults
                 ↓
   shared values, including shared.* CLI overrides
                 ↓
   values written in an individual Run
                 ↓
   runs.<name>.* CLI overrides

.. dropdown:: Configuration rules
   :animate: fade-in

   * The only top-level fields are ``shared`` and ``runs``.
   * ``runs`` must contain at least one Run.
   * A Run name starts with a letter or underscore. It can then contain letters, numbers,
     underscores, or hyphens.
   * An override can change a field on a Run declared in the YAML, but it cannot add another Run.
   * A ``shared.*`` override must point to a field already present below ``shared``. The Hydra
     operators ``+``, ``++``, and ``~`` are not supported for shared values.


.. _sequential-batch-experiment-runner:

One definition, two execution paths
-----------------------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Local — Experiment Runner
      :class-card: sd-shadow-sm

      Loads the YAML once and executes its Runs in order in one process and one
      SimulationApp. Every Run builds a fresh environment. The runner stops at the first failure
      unless ``--continue_on_error`` is set.

   .. grid-item-card:: Managed — OSMO
      :class-card: sd-shadow-sm

      Turns every Run into an independently scheduled group. Runs can execute at the same time
      when resources are available. OSMO then collects their outputs into one combined result.

.. tab-set::

   .. tab-item:: Run locally

      .. code-block:: bash

         python isaaclab_arena/evaluation/experiment_runner.py \
           --experiment_config path/to/experiment.yaml

   .. tab-item:: Preview an OSMO workflow

      .. code-block:: bash

         python -m osmo.submit_arena_experiment \
           --experiment_cfg path/to/experiment.yaml \
           --dry_run

For remote policies, configure the policy client inside its Run like any other policy. Start the
policy server separately for local execution. OSMO can co-schedule a server for supported policy
types.

Follow :doc:`First Arena Experiment <../quickstart/arena_experiment>` for a complete local
example.
For OSMO setup and submission options, see :doc:`Multi-node Evaluation
<../example_workflows/multi_node_evaluation/multi_node_evaluation>`.


Choosing a runner
-----------------

.. note::

   Use the **Experiment Runner** for policy evaluations defined in YAML. Use the **Environment
   Runner** (``isaaclab_arena/scripts/environment_runner.py``) to inspect and physically manipulate
   an environment without a policy. The **Policy Runner** is currently needed only for ``torchrun``
   multi-GPU execution or policy evaluation of an external environment loaded with
   ``--external_environment_class_path``.


Related concepts
----------------

* :doc:`Environment Design <environment/index>`
* :doc:`Policy Design <policy/index>`
* :doc:`Metrics Design <task/concept_metrics_design>`
* :doc:`Environment Variations <variations/index>`
