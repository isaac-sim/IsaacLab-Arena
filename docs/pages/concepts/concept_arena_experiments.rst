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
values are reused by every Run. An optional top-level ``name`` string names the Experiment; if it is not
given, the name is taken from the YAML file name (without extension). An optional top-level ``kpi_exporters``
list names the registered KPI exporters (see the `KPI exporter registry
<https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_benchmarking/benchmarks/registry.py>`_)
that write the Experiment's KPI records once every Run has completed. An empty list exports no KPIs. KPIs are
aggregated over all Runs of the Experiment. Each key below ``runs`` is a Run name:

.. code-block:: yaml

   name: my_experiment
   kpi_exporters: [success_rate, timing]
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

``rollout_limit`` sets when a Run stops. ``num_rebuilds`` is a Run field next to it that sets how
many times the Run builds a fresh environment. These settings give:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Settings
     - Result
   * - ``num_steps: 50``
     - Each rebuild runs 50 environment steps.
   * - ``num_episodes: 30``, ``num_rebuilds: 3``
     - 10 / 10 / 10 episodes. With 31, the first rebuild gets the extra one: 11 / 10 / 10.
   * - ``num_episodes_per_rebuild: [20, 5, 5]``, ``num_rebuilds: 3``
     - 20 / 5 / 5 episodes.
   * - ``num_episodes: 30``, ``num_episodes_per_rebuild: [20, 5, 5]``, ``num_rebuilds: 3``
     - 20 / 5 / 5 episodes. ``num_episodes`` matches the sum of the list.
   * - No limit
     - The policy's own length. Only a policy with a fixed action-sequence length supports this.

These combinations are rejected:

* ``num_steps`` with ``num_episodes`` or ``num_episodes_per_rebuild``.
* ``num_episodes`` smaller than ``num_rebuilds``.
* ``num_episodes_per_rebuild`` with a length other than ``num_rebuilds``.
* ``num_episodes`` that differs from the sum of ``num_episodes_per_rebuild``.
* ``num_episodes_per_rebuild`` with replay from recorded variation samples.

``num_rebuilds`` sits at the Run level, beside ``rollout_limit``:

.. code-block:: yaml

   num_rebuilds: 3
   rollout_limit:
     num_episodes_per_rebuild: [20, 5, 5]   # rebuild 0: 20, rebuild 1: 5, rebuild 2: 5

See :doc:`Record and Replay Experiments <experiment_record_replay>` for variation recording,
partial and complete replay, and episode scheduling across one or more rebuilds.

Runs keep their YAML order and execute locally in that order.


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

The CLI override root depends on the execution path:

.. tab-set::

   .. tab-item:: Run locally

      The Experiment Runner receives the Experiment Definition directly.

      * Use ``shared.<path>=<value>`` to change a shared value.
      * Use ``runs.<name>.<path>=<value>`` to change one Run.
      * Use ``kpi_exporters=[<name>,<name>,...]`` to change the KPI exporters.

      .. code-block:: bash

         python isaaclab_arena/evaluation/experiment_runner.py \
           --experiment_config path/to/experiment.yaml \
           shared.rollout_limit.num_steps=100

   .. tab-item:: Preview an OSMO workflow

      OSMO nests the Experiment below ``experiment_cfg``, so Experiment overrides need that prefix.

      * Use ``experiment_cfg.shared.<path>=<value>`` to change a shared value.
      * Use ``experiment_cfg.runs.<name>.<path>=<value>`` to change one Run.
      * Use ``experiment_cfg.kpi_exporters=[<name>,<name>,...]`` to change the KPI exporters.

      .. code-block:: bash

         python -m osmo.submit_arena_experiment \
           --experiment_cfg path/to/experiment.yaml \
           --dry_run \
           experiment_cfg.shared.rollout_limit.num_steps=100

Both commands set the same ``shared.rollout_limit.num_steps`` field. A value written inside an
individual Run still takes priority over the shared override.

.. _experiment-without-file:

Run an Experiment without a file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Without ``--experiment_config``, the Experiment Runner takes the overrides as the whole Experiment.
OSMO submission always needs a file. This command runs one Run named ``quick``:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     runs.quick.environment.type=kitchen_pick_and_place \
     runs.quick.environment.object=mustard_bottle \
     runs.quick.policy.type=zero_action \
     runs.quick.rollout_limit.num_steps=50

The Experiment is named ``<timestamp>_experiment``, for example
``2026-11-27_14-42-54_experiment``, unless you pass ``name=<name>``. Without a file and without
overrides, the runner runs the smoke test.

Override rules
~~~~~~~~~~~~~~

* ``path=value`` sets a value and ``~path`` removes one, for example ``~runs.<name>`` drops a Run.
  ``+path=value`` and ``++path=value`` mean the same as ``path=value``.
* These Hydra forms are rejected: sweeps such as ``path=1,2``, ``~path=value``,
  ``path@package=value``, and ``extend_list(...)``. Write a list as ``path=[1,2]``.
* With a file, an override can set ``name``, ``kpi_exporters``, and any path below ``shared`` or
  below a Run the file declares. It cannot add a Run, and a misspelled Run name fails with the list
  of Runs in the file.
* An override changes only the key it names, before ``shared`` is merged into each Run. So
  ``runs.<name>.policy.type=<type>`` keeps the policy fields that Run writes, and the Run no longer
  inherits ``shared.policy`` when the types differ (see :ref:`experiment-shared-values`). The same
  holds for ``environment``. Likewise, ``runs.<name>.rollout_limit.num_steps=5`` replaces a limit
  the Run inherits from ``shared`` but merges into a limit the Run writes.
* A mapping value replaces the whole mapping. Use it to drop the fields written for an old type or
  limit, for example ``'runs.x.policy={type:replay,replay_file_path:demo.hdf5}'`` or
  ``'runs.x.rollout_limit={num_steps:5}'``. Quote it, because the shell expands braces.

For remote policies, configure the policy client inside its Run like any other policy. Start the
policy server separately for local execution. OSMO can co-schedule a server for supported policy
types.

Follow :doc:`First Arena Experiment <../quickstart/arena_experiment>` for a complete local
example.
For OSMO setup and submission options, see :doc:`Multi-node Evaluation
<../example_workflows/multi_node_evaluation/multi_node_evaluation>`.

.. _experiment-shared-values:

Reuse values with ``shared``
----------------------------

``shared`` removes repetition. Arena applies these values to every Run, then places each Run's
own values on top. A Run only needs to declare what is different.

A Run whose ``environment.type`` differs from ``shared.environment.type`` does not inherit
``shared.environment``, and the same holds for ``policy``. Fields that both types have, such as
``embodiment`` or ``remote_host``, must then be set in the Run.

A Run that declares ``rollout_limit`` replaces the shared one as a whole, so a Run can switch from
a shared ``num_steps`` to ``num_episodes`` or ``num_episodes_per_rebuild`` without clearing the
shared field.

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

   * The only top-level fields are ``name``, ``kpi_exporters``, ``shared``, and ``runs``.
   * ``runs`` must contain at least one Run.
   * A Run name starts with a letter or underscore. It can then contain letters, numbers,
     underscores, or hyphens.
   * Each ``kpi_exporters`` name must be a registered KPI exporter and appear once.


Choosing a runner
-----------------

.. note::

   Use the **Experiment Runner** for policy evaluations defined in YAML. Use the **Environment
   Runner** (``isaaclab_arena/scripts/environment_runner.py``) to inspect and physically manipulate
   an environment without a policy. The **Policy Runner** is currently needed only for policy
   evaluation of an external environment loaded with ``--external_environment_class_path``.


Related concepts
----------------

* :doc:`Environment Design <environment/index>`
* :doc:`Policy Design <policy/index>`
* :doc:`Environment Variations <variations/index>`

.. toctree::
   :hidden:
   :maxdepth: 1

   experiment_record_replay
