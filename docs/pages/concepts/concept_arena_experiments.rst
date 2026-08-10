.. _arena-experiments-and-runs:

Arena Experiments and Runs
==========================

An Arena Experiment groups one or more named Runs into one evaluation. Each Run selects an
existing Environment Definition and combines it with a policy and the settings that control the
rollout.

The **Experiment Definition** is the YAML file that describes those Runs. It is the main interface
for evaluation: run the same definition locally with the Experiment Runner, or submit it to OSMO
for managed execution. An Experiment with a single Run is valid, so the interface also works for
small evaluations.

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

An Experiment Definition has two top-level mappings: ``shared`` and ``runs``. Values below
``shared`` are reused by every Run. Each key below ``runs`` is a Run name:

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

Runs keep their YAML order. The local Experiment Runner executes them in that order.


What a Run contains
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``environment``
     - Selects and configures the Environment Definition. ``type`` is a registered Python
       environment name or the path to an environment graph YAML file.
   * - ``policy``
     - Selects and configures the policy. ``type`` is a registered policy name or a dotted Python
       class path.
   * - ``environment_builder``
     - Controls how simulation is built, including ``num_envs``, environment spacing, and seeds.
   * - ``rollout_limit``
     - Stops the rollout after ``num_steps`` or ``num_episodes``. If neither is set, the policy
       must provide its own length.
   * - ``num_rebuilds``
     - Builds a fresh environment several times and combines the resulting metrics.
   * - ``variations``
     - Configures registered environment variations for this Run.

The selected environment and policy determine which fields are valid inside their mappings.
Arena validates the definition against those typed configurations and reports unknown fields.

``environment_builder.num_envs`` creates parallel simulated environments *inside one Run*. It
does not make the Runs themselves concurrent. When ``num_rebuilds`` is greater than one, a step
limit applies to every rebuild, while an episode limit is divided across the rebuilds.


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

This order matters when a shared value is also set by a Run:

.. list-table:: Effective ``num_steps`` values
   :header-rows: 1
   :widths: 60 20 20

   * - Configuration
     - ``baseline``
     - ``parallel_envs``
   * - YAML has shared ``num_steps: 50``
     - 50
     - 50
   * - ``parallel_envs`` declares ``num_steps: 100``
     - 50
     - 100
   * - CLI sets ``shared.rollout_limit.num_steps=75``
     - 75
     - 100
   * - CLI also sets ``runs.parallel_envs.rollout_limit.num_steps=125``
     - 75
     - 125

A ``shared.<path>=...`` override changes the shared value before Runs are merged. A value written
directly in a Run still wins. A ``runs.<name>.<path>=...`` override changes the final Run.

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

      Loads the Experiment once and executes its Runs in YAML order in one process and one
      SimulationApp. Every Run builds a fresh environment. The runner stops at the first failure
      unless ``--continue_on_error`` is set.

   .. grid-item-card:: Managed — OSMO
      :class-card: sd-shadow-sm

      Turns every Run into an independently scheduled group. Runs can execute at the same time
      when resources are available. OSMO then collects their outputs into one Experiment result.

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

Follow :doc:`Run an Arena Experiment
<../quickstart/first_experiments/running_an_arena_experiment>` for a complete local example.
For OSMO setup and submission options, see :doc:`Multi-node Evaluation
<../example_workflows/multi_node_evaluation/multi_node_evaluation>`.


.. _policy-runner-single-job:

Policy Runner — single job (single GPU and multi-GPU)
-----------------------------------------------------

For a saved or repeatable evaluation, use an Experiment Definition even when it contains only one
Run. The **Policy Runner** (``isaaclab_arena/evaluation/policy_runner.py``) remains useful for an
ad-hoc evaluation, focused debugging, external Python environments, and ``torchrun`` multi-GPU
execution.

The Policy Runner:

* runs one environment configuration with one policy;
* supports several parallel simulated environments with ``--num_envs``;
* runs for ``--num_steps`` or ``--num_episodes``;
* uses the policy's own length when the policy provides one;
* records per-episode results, computes available metrics, and builds an evaluation report; and
* can record viewport or policy-camera videos.

.. tab-set::

   .. tab-item:: Single GPU

      This example runs one environment with the zero-action policy:

      .. code-block:: bash

         python isaaclab_arena/evaluation/policy_runner.py \
           --viz kit \
           --policy_type zero_action \
           --num_steps 50 \
           --num_envs 1 \
           pick_and_place_maple_table \
           --embodiment droid_rel_joint_pos \
           --pick_up_object rubiks_cube_hot3d_robolab \
           --destination_location bowl_ycb_robolab \
           --hdr home_office_robolab

      The environment name is the command's subcommand. Put general runner options before it and
      environment-specific options after it.

   .. tab-item:: Multi-GPU

      Use ``torchrun`` with ``--distributed`` to start one Policy Runner process per GPU. Each
      process starts its own Isaac Sim instance and uses its local GPU:

      .. code-block:: bash

         python -m torch.distributed.run \
           --standalone \
           --nproc-per-node=2 \
           isaaclab_arena/evaluation/policy_runner.py \
           --policy_type zero_action \
           --num_steps 200 \
           --num_envs 10 \
           --distributed \
           --headless \
           pick_and_place_maple_table \
           --embodiment droid_rel_joint_pos \
           --pick_up_object rubiks_cube_hot3d_robolab \
           --destination_location bowl_ycb_robolab \
           --hdr home_office_robolab

      Here, ``--num_envs 10`` applies to every process, so two GPU processes simulate 20
      environments in total.

.. dropdown:: Use different objects in parallel environments
   :animate: fade-in

   Some environments accept ``--object_set``. This assigns different objects to parallel
   simulated environments:

   .. code-block:: bash

      python isaaclab_arena/evaluation/policy_runner.py \
        --viz kit \
        --policy_type zero_action \
        --num_steps 200 \
        --num_envs 4 \
        put_item_in_fridge_and_close_door \
        --embodiment gr1_joint \
        --object_set \
          ketchup_bottle_hope_robolab \
          ranch_dressing_hope_robolab \
          bbq_sauce_bottle_hope_robolab \
          mayonnaise_bottle_hope_robolab

   See :doc:`Homogeneous and Heterogeneous Placement
   <object_placement/homogeneous_and_heterogeneous_placement>` for how object sets are assigned.

``--policy_type`` accepts a registered policy name or a dotted Python class path. Registered
policies add their typed configuration fields as command-line flags. This includes connection
settings for remote policies. See :doc:`Running a Real Policy
<../quickstart/first_experiments/running_a_real_policy/index>` for complete examples.


Related concepts
----------------

* :doc:`Environment Design <environment/index>`
* :doc:`Policy Design <policy/index>`
* :doc:`Metrics Design <task/concept_metrics_design>`
* :doc:`Environment Variations <variations/index>`
