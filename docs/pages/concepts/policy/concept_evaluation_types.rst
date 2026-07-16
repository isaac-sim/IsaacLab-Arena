Evaluation Types
=================

Isaac Lab Arena supports two evaluation entry points: an **Experiment Runner**
for inline environment previews and configured typed Runs, and a **policy
runner** for compatibility workflows such as distributed execution. This
section summarizes when to use each and how they work. Each section below links
to the relevant concept docs:
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
   * - Experiment Runner
     - Quick registered-environment preview or configured typed Runs
     - ``experiment_runner.py``
     - No

1. Policy runner — single job (single GPU and multi-GPU)
--------------------------------------------------------

The **policy runner** (``isaaclab_arena/evaluation/policy_runner.py``) runs one
environment configuration and one policy. Use it when compatibility with
graph-spec or external environments, arbitrary policy CLI configuration, or
distributed execution is required.

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

.. _sequential-batch-experiment-runner:

2. Experiment Runner — inline preview and configured Runs
---------------------------------------------------------

The **Experiment Runner** (``isaaclab_arena/evaluation/experiment_runner.py``)
executes typed Arena Runs. It accepts exactly one input source:

- ``--environment NAME`` creates one inline ``preview`` Run for a registered
  environment.
- ``--experiment_config PATH`` loads named Runs from a typed YAML Experiment
  Definition or a legacy JSON configuration.

Inline environment preview
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use inline mode to build and inspect a registered environment without first
writing an Experiment YAML:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --environment pick_and_place_maple_table

The generated ``preview`` Run uses ``zero_action`` for 100 steps, one
environment, and one rebuild. Trailing Hydra overrides are relative to that
Run. Select an explicit limit with ``--num_steps`` or ``--num_episodes``, or
override the corresponding ``rollout_limit.*`` field:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --environment pick_and_place_maple_table \
     --num_envs 4 \
     environment.embodiment=droid_rel_joint_pos \
     environment.pick_up_object=mustard_bottle_hot3d_robolab \
     rollout_limit.num_steps=500 \
     num_rebuilds=2 \
     +variations.light.hdr_image.enabled=true

Use the ``environment.*``, ``environment_builder.*``,
``rollout_limit.*``, ``variations.*``, and ``num_rebuilds`` namespaces. The
leading ``+`` is required when adding entries to the initially empty
``variations`` mapping. Prefer shared builder flags such as ``--num_envs`` and
``--mimic`` when available; inline composition also copies them into matching
environment-specific fields. Use ``environment_builder.*`` for builder fields
without a shared CLI flag.

Add ``--enable_cameras`` when the preview needs camera sensors.
``--record_camera_video`` enables camera support automatically. Inline mode
supports registered environments and the zero-action policy. Use
``policy_runner.py`` for graph-spec or external environments, arbitrary policy
CLI configuration, and distributed execution.

Configured Experiments
^^^^^^^^^^^^^^^^^^^^^^

For repeatable or multi-Run evaluation, declare typed Runs in YAML:

.. code-block:: yaml

   runs:
     baseline:
       environment:
         type: pick_and_place_maple_table
         embodiment: droid_rel_joint_pos
       policy:
         type: zero_action
       rollout_limit:
         num_steps: 50

Run the definition with:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/getting_started_experiment.yaml

Configured-Experiment overrides use the complete named-Run path, for example
``runs.baseline.rollout_limit.num_steps=100``. Runs execute sequentially in one
Isaac Sim process. Metrics and reports use the same execution path in inline
and configured modes.

Legacy JSON configs and the ``--eval_jobs_config`` alias remain supported for
compatibility. New Experiment Definitions should use typed YAML. Distributed
evaluation is not supported by the Experiment Runner.

Choosing an evaluation type
---------------------------

- **Quickly inspect a registered environment**: use
  ``experiment_runner.py --environment NAME``.
- **Repeatable or multi-Run evaluation**: use ``experiment_runner.py`` with a
  typed YAML Experiment Definition.
- **Graph-spec or external environments, arbitrary policy CLI configuration,
  or multi-GPU execution**: use ``policy_runner.py``.
