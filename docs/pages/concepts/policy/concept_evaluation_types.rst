Evaluation Types
=================

Isaac Lab Arena supports two main ways to run policy evaluation: a single-job
**policy runner** (single or multi-GPU) and a **sequential batch eval runner** for
multiple jobs in one process. This section summarizes when to use each and how
they work. Each section below links to the relevant concept docs:
:doc:`Policy Design <index>`,
:doc:`Environment Design <../concept_overview>`, and
:doc:`Metrics Design <../task/concept_metrics_design>`.

Both runners support a **server–client** setup, where simulation runs locally
(client) and policy inference runs in a separate process or machine (server).
This is the deployment used by ``Gr00tRemoteClosedloopPolicy``: the simulation
client ships observations to a `GR00T <https://github.com/NVIDIA/Isaac-GR00T>`_
policy server over the network, receives action chunks, and applies them in the
sim. The split lets a heavyweight model (e.g. GR00T N1.6) live on a dedicated
GPU while the simulation client runs on its own GPU, and is orthogonal to the
runner choice — pass the remote-policy class as ``--policy_type`` and add
``--remote_host`` / ``--remote_port`` flags. End-to-end commands (including
how to launch the GR00T server out of the
``submodules/Isaac-GR00T`` submodule) live in
:doc:`../../quickstart/first_experiments/running_a_real_policy/index` and the
example workflows.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 20 30 30 20

   * - Type
     - Use case
     - Entry point
     - Multi-GPU
   * - Policy runner
     - Single job, one env config, one policy
     - ``policy_runner.py``
     - Yes (torchrun)
   * - Sequential batch eval runner
     - Multiple jobs (env/policy combos) in sequence
     - ``eval_runner.py``
     - No

1. Policy runner — single job (single GPU and multi-GPU)
--------------------------------------------------------

The **policy runner** (``isaaclab_arena/evaluation/policy_runner.py``) runs one
evaluation job: one environment configuration and one policy. It is the right
choice for ad-hoc runs, debugging, or when you want to drive one scenario with
full control over CLI arguments.

**Design context:** For how policies are defined and integrated with environments,
see :doc:`Policy Design <index>`.

**Features:**

- Single environment configuration (scene, embodiment, task) and one policy.
- **Heterogeneous objects:** When the environment supports it, you can pass
  ``--object_set`` with a space-separated list of object names. Each parallel
  environment is assigned a different object from the set (e.g. env 0 gets the
  first object, env 1 the second, etc.). This allows evaluating one policy
  across multiple object types in a single run without changing the scene or
  task logic.
- Run length by **steps** (``--num_steps``) or **episodes** (``--num_episodes``);
  policies that define a length (e.g. ``policy.has_length()``) can override this.
- **Single GPU**: one process, one Isaac Sim instance.
- **Multi-GPU**: use ``torchrun`` with ``--distributed``; one process per GPU,
  each with its own Isaac Sim instance and device (e.g. ``cuda:0``, ``cuda:1``).
- Metrics are computed at the end if the environment registers metrics and are logged to the console.

**Single-GPU example**

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type <policy_type> \
     --num_steps 2000 \
     --num_envs 10 \
     <arena_environment> \
     --embodiment <embodiment> \
     --object <object>
     ...

**Heterogeneous objects example (single or multi-GPU)**

Use ``--object_set`` so each of the ``--num_envs`` parallel environments gets a
different object from the list. Object-to-environment mapping: with the default
deterministic assignment, environment :math:`i` gets the object at index
:math:`i \\mod n` in the list (where :math:`n` = ``len(object_set)``)—so when
``num_envs`` > ``len(object_set)`` the assignment **cycles** (no truncation or
error). If the object set is created with ``random_choice=True``, each environment
gets a randomly chosen object from the set. Some environments may require
``num_envs == len(object_set)``.

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type <policy_type> \
     --num_steps 2000 \
     --num_envs 4 \
     --enable_cameras \
     put_item_in_fridge_and_close_door \
     --embodiment gr1_joint \
     --object_set ketchup_bottle_hope_robolab ranch_dressing_hope_robolab bbq_sauce_bottle_hope_robolab mayonnaise_bottle_hope_robolab

**Multi-GPU example**

Use ``torch.distributed.run`` (or ``torchrun``) with ``--nproc_per_node=<num_gpus>``
and pass ``--distributed`` so each process uses a different GPU (via ``LOCAL_RANK``):

.. code-block:: bash

   python -m torch.distributed.run --nnode=1 --nproc_per_node=<num_gpus> \
     isaaclab_arena/evaluation/policy_runner.py \
     --policy_type <policy_type> \
     --num_steps 2000 \
     --num_envs 10 \
     --distributed \
     --headless \
     <arena_environment> \
     ...

**Policy runner CLI (relevant flags)**

- ``--policy_type``: Registered policy name or dotted path to policy class
  (e.g. ``module.submodule.ClassName``).
- ``--num_steps``: Total simulation steps (mutually exclusive with
  ``--num_episodes``).
- ``--num_episodes``: Total episodes (mutually exclusive with ``--num_steps``).
- ``--distributed``: Enable distributed mode; use with ``torchrun`` and set
  device per rank (e.g. ``cuda:{local_rank}``).

The remaining environment arguments come from the Arena environments CLI. For
registered policies, policy-specific flags are generated from their ``PolicyCfg``.

.. _sequential-batch-eval-runner:

2. Experiment eval runner — ordered Runs
----------------------------------------

The **Experiment eval runner** (``isaaclab_arena/evaluation/eval_runner.py``) executes an
ordered list of typed Runs in one process. Each Run can configure a different environment,
policy, environment builder, and rollout limit. This is suited for benchmarking many
configurations without launching multiple processes by hand. The simulation application
remains active between Runs.

**Design context:** For how environments are composed and how metrics are
defined and computed, see :doc:`Environment Design <../concept_overview>`
and :doc:`Metrics Design <../task/concept_metrics_design>`. Policies used by each Run
follow :doc:`Policy Design <index>`.

**Features:**

- One typed YAML Experiment file (``--experiment_config``) listing all Runs.
- Runs execute in declaration order. Each Run builds its environment and policy,
  executes its rollout, and tears down its resources before the next Run.
- By default, a failed Run stops the Experiment. Pass ``--continue_on_error`` to
  record the failure and continue with later Runs.
- Metrics are aggregated and printed at the end (e.g. via ``MetricsLogger``).
- **Distributed evaluation is not supported**: the Experiment eval runner
  runs in a single process. For multi-GPU, use multiple policy runner
  invocations (e.g. with ``torchrun``) or split the batch across machines.

.. todo::

    Experiment with distributed evaluation in the Experiment eval runner.

**Experiment format**

The YAML document contains a ``runs`` sequence. Each Run declares:

- ``name``: Unique Run name used for output and metrics.
- ``environment``: Registered environment ``type`` and its typed configuration fields.
- ``policy``: Registered policy ``type`` and its typed configuration fields.
- ``environment_builder`` (optional): Process-independent build values such as
  ``num_envs``, ``env_spacing``, and ``placement_seed``.
- ``rollout_limit`` (optional): Exactly one of ``num_steps`` or ``num_episodes``.
- ``num_rebuilds`` and ``variations`` (optional): Rebuild count and nested variation values.

**Example config structure**

.. code-block:: yaml

   runs:
   - name: gr1_open_microwave_cracker_box
     environment:
       type: gr1_open_microwave
       object: cracker_box
       embodiment: gr1_joint
     policy:
       type: zero_action
     environment_builder:
       num_envs: 4
     rollout_limit:
       num_steps: 500

   - name: gr1_open_microwave_sugar_box
     environment:
       type: gr1_open_microwave
       object: sugar_box
       embodiment: gr1_pink
     policy:
       type: zero_action
     environment_builder:
       num_envs: 10
     rollout_limit:
       num_steps: 500

**Running the Experiment eval runner**

.. code-block:: bash

   python isaaclab_arena/evaluation/eval_runner.py \
     --viz kit \
     --experiment_config path/to/experiment.yaml

If any Run enables cameras, also pass ``--enable_cameras`` so AppLauncher enables
process-level camera support before the typed Experiment is composed.

Legacy JSON Experiment configs and ``--eval_jobs_config`` remain available during
the migration period but emit deprecation warnings.

Choosing an evaluation type
---------------------------

- **One-off run, one setup**: use the **policy runner** (single or multi-GPU);
  use ``--object_set`` for heterogeneous objects in one run.
- **Many env/policy combinations in one go**: use the **Experiment eval runner**
  with typed YAML; use ``--object_set`` for heterogeneous objects in one Run.
