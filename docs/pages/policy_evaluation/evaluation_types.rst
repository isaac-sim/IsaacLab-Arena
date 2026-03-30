Evaluation Types
=================

Isaac Lab Arena supports three main ways to run policy evaluation: a single-job
**policy runner** (single or multi-GPU), a **sequential batch eval runner** for
multiple jobs in one process, and a **server–client** setup for remote policies. This
section summarizes when to use each and how they work. Each section below links
to the relevant concept docs: :doc:`Policy Design <../concepts/concept_policy_design>`,
:doc:`Environment Design <../concepts/concept_environment_design>`,
:doc:`Metrics Design <../concepts/concept_metrics_design>`, and
:doc:`Remote Policies Design <../concepts/concept_remote_policies_design>`.

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
   * - Server–client
     - Policy runs in separate process/machine
     - Policy runner + remote server
     - Depends on client

1. Policy runner — single job (single GPU and multi-GPU)
--------------------------------------------------------

The **policy runner** (``isaaclab_arena/evaluation/policy_runner.py``) runs one
evaluation job: one environment configuration and one policy. It is the right
choice for ad-hoc runs, debugging, or when you want to drive one scenario with
full control over CLI arguments.

**Design context:** For how policies are defined and integrated with environments,
see :doc:`Policy Design <../concepts/concept_policy_design>`.

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

The rest of the arguments (environment, embodiment, object, etc.) come from the
Arena environments CLI and the policy’s own ``add_args_to_parser``.

.. _sequential-batch-eval-runner:

2. Sequential batch eval runner — batch jobs
--------------------------------------------

The **sequential batch eval runner** (``isaaclab_arena/evaluation/eval_runner.py``)
runs a **batch** of evaluation jobs sequentially in a single process. Each job can have
a different environment (scene/object/embodiment), policy type, policy config,
and length (steps or episodes). This is suited for benchmarking many
configurations (e.g. many objects or tasks) without launching multiple processes
by hand. Persistence of the simulation application is maintained between jobs.

**Design context:** For how environments are composed and how metrics are
defined and computed, see :doc:`Environment Design <../concepts/concept_environment_design>`
and :doc:`Metrics Design <../concepts/concept_metrics_design>`. Policies used per job
follow :doc:`Policy Design <../concepts/concept_policy_design>`.

**Features:**

- One JSON config file (``--eval_jobs_config``) listing all jobs.
- Jobs run one after another; each job builds its environment, creates the
  policy from the job config, runs ``rollout_policy``, then tears down the env
  before the next job.
- If a job fails, the runner continues with the next job and marks the failed
  job accordingly.
- Metrics are aggregated and printed at the end (e.g. via ``MetricsLogger``).
- **Distributed evaluation is not supported**: the sequential batch eval runner
  runs in a single process. For multi-GPU, use multiple policy runner
  invocations (e.g. with ``torchrun``) or split the batch across machines.

.. todo::

    Experiment with distributed evaluation in the sequential batch eval runner.

**Jobs config format**

The config file must be a JSON object with a ``"jobs"`` array. Each job is an
object with:

- ``name``: Unique job name (for logging and metrics).
- ``arena_env_args``: Environment arguments as a dict (e.g. ``environment``,
  ``num_envs``, ``object``, ``embodiment``, ``enable_cameras``, etc.). Converted
  internally to the same CLI-style list the policy runner uses.
- ``policy_type``: Same as policy runner (registered name or dotted class path).
- ``policy_config_dict``: Policy configuration (e.g. checkpoint path, model
  options). Used with ``PolicyBase.from_dict`` if the policy has a
  ``config_class``, otherwise converted to CLI args and ``from_args``.
- ``num_steps`` or ``num_episodes`` (optional): Simulation length for this job.
  If both are omitted, the runner uses the policy’s length if defined, or a CLI
  default (e.g. ``--num_steps``).

**Example config structure**

.. code-block:: json

   {
     "jobs": [
       {
         "name": "gr1_open_microwave_cracker_box",
         "arena_env_args": {
           "environment": "gr1_open_microwave",
           "object": "cracker_box",
           "embodiment": "gr1_joint",
           "num_envs": 4
         },
         "num_steps": 500,
         "policy_type": "zero_action",
         "policy_config_dict": {}
       },
       {
         "name": "gr1_sequential_static_manipulation_put_ranch_dressing_bottle_in_fridge_and_close_door",
         "arena_env_args": {
           "enable_cameras": true,
           "environment": "gr1_sequential_static_manipulation",
           "object": "ranch_dressing_hope_robolab",
           "embodiment": "gr1_joint"
         },
         "num_steps": 100,
         "policy_type": "isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy",
         "policy_config_dict": {
           "policy_config_yaml_path": "isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml",
           "policy_device": "cuda:0"
          }
       }
     ]
   }

**Running the sequential batch eval runner**

.. code-block:: bash

   python isaaclab_arena/evaluation/eval_runner.py \
     --eval_jobs_config path/to/eval_jobs_config.json \
     --num_steps 1000

If any job needs cameras, set ``enable_cameras: true`` in that job’s
``arena_env_args``; the sequential batch eval runner automatically enables camera support if any job requires it.

3. Server–client (remote policies)
----------------------------------

When the policy runs in a **separate process or machine** (e.g. a GPU server
with a large model), evaluation still uses the **policy runner** on the client
side, but the policy is a **client-side remote policy** that talks to a
**remote policy server** over the network.

**Design context:** For the full remote policy design, protocol, and how to
implement custom server-side and client-side policies, see
:doc:`Remote Policies Design <../concepts/concept_remote_policies_design>`.

**Features:**

- **Server**: Runs the model and a ``ServerSidePolicy`` (e.g. GR00T), often
  started via ``remote_policy_server_runner`` or a wrapper script (e.g.
  ``docker/run_gr00t_server.sh``). No Isaac Sim on the server.
- **Client**: Runs Isaac Lab Arena and the simulation; the policy is a
  ``ClientSidePolicy`` (e.g. ``ActionChunkingClientSidePolicy``) that packs
  observations, sends them to the server, and applies returned actions (with
  optional chunking or post-processing).
- **Protocol**: Server and client agree on an ``ActionProtocol`` (e.g. observation
  keys, action shape, chunk length). The protocol is negotiated at handshake;
  no policy logic lives in the protocol.

**Typical workflow**

1. Start the remote policy server (separate terminal or machine), e.g. with GR00T:

   .. code-block:: bash

      bash docker/run_gr00t_server.sh \
        --host 127.0.0.1 \
        --port 5555 \
        --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_policy.Gr00tRemoteServerSidePolicy \
        --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml

2. Run evaluation on the client with a client-side policy (i.e. ``ActionChunkingClientSidePolicy``) and remote connection. Within the ``Base`` container, run:

   .. code-block:: bash

      python isaaclab_arena/evaluation/policy_runner.py \
        --policy_type isaaclab_arena.policy.action_chunking_client.ActionChunkingClientSidePolicy \
        --remote_host 127.0.0.1 \
        --remote_port 5555 \
        --num_steps 2000 \
        --num_envs 10 \
        --remote_kill_on_exit \
        <arena_environment> \
        --embodiment <embodiment> \
        ...

The same **policy runner** is used as in the single-job case; only the policy
type and remote options change.

Choosing an evaluation type
---------------------------

- **One-off run, one setup**: use the **policy runner** (single or multi-GPU);
  use ``--object_set`` for heterogeneous objects in one run.
- **Many env/policy combinations in one go**: use the **sequential batch eval
  runner** with a jobs JSON; use ``--object_set`` for heterogeneous objects in one run.
- **Heavy model on another machine or process**: use **server–client** with the
  policy runner on the client and a remote policy server.
