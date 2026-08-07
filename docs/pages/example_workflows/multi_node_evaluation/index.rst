Multi-Node Evaluation
=====================

A thorough evaluation requires many rollouts.
For example, a moderately sized evaluation may have
**20 tasks, 2 policies, 100 episodes per task, for a total of 4000 rollouts.**
Arena uses `OSMO <https://developer.nvidia.com/osmo>`_ to distribute execution across a cluster,
in order to reduce the time it takes to run the evaluations.


Setting up OSMO
---------------

**Install the OSMO client:** Submitting an Experiment requires the OSMO command-line client, installed and authenticated
against your cluster. The `OSMO User Guide <https://nvidia.github.io/OSMO/main/user_guide/index.html>`_
covers this; follow its Getting Started pages in order:

- `System Requirements <https://nvidia.github.io/OSMO/main/user_guide/getting_started/system_requirements.html>`_ — check your environment is supported.
- `Install Client <https://nvidia.github.io/OSMO/main/user_guide/getting_started/install/index.html>`_ — install the ``osmo`` CLI.
- `Setup Profile <https://nvidia.github.io/OSMO/main/user_guide/getting_started/profile.html>`_ — point the client at your cluster.
- `Setup Credentials <https://nvidia.github.io/OSMO/main/user_guide/getting_started/credentials.html>`_ — authenticate with ``osmo login``.

Once ``osmo`` is installed and you can log in, you are ready to submit an Experiment.

**Set up a cluster:** The steps above assume you already have a cluster running OSMO. If you need to stand one up, see the
`OSMO Deployment Guide <https://nvidia.github.io/OSMO/main/deployment_guide/index.html>`_, which
covers deploying the OSMO service and connecting your own compute.


Specifying a multi-node Experiment
-----------------------------------

We use the same experiment files as for single-node evaluations to specify multi-node evaluations.
Arena takes care of executing the evaluation described in an experiment file across a cluster.

.. figure:: ../../../images/multinode_evaluation_highlevel.jpg
   :width: 100%
   :alt: Multi-node execution of an experiment YAML through OSMO.
   :align: center

   An Experiment YAML is submitted for multi-node execution. Arena schedules each Run as
   a parallel OSMO process, then collates per-env results into a shared output.

.. todo::

   Add a link here to the single node evaluation page.


How Arena maps an Experiment onto the Cluster
---------------------------------------------

Arena takes a simple approach to scheduling experiments across a cluster.

- **Parallel execution of Runs.** Each ``Run`` specifies a simulation environment and a policy, and is scheduled independently. Given enough resources, Runs execute in parallel; otherwise they are queued until resources are available.
- **Policy servers.** Each Run using a server-backed remote policy starts its own policy server and an experiment runner that uses it. For these Runs, executing everything in parallel requires ``2 × number of Runs`` GPUs.
- **One collected output.** A final task collects the per-Run results and uploads them to an output bucket.


Submit an Experiment
--------------------

Submit an Experiment with ``osmo/submit_arena_experiment.py``. Point ``--experiment_cfg`` at an
Experiment YAML file. The command uses an experiment file included in the Arena repository.

.. code-block:: bash

   python osmo/submit_arena_experiment.py \
     --experiment_cfg isaaclab_arena_environments/robolab/experiment_configs/robolab_2_tasks_pi0_and_cosmos.yaml \
     osmo.pool=isaac-dev-l40-03 \
     osmo.platform=ovx-l40 \
     osmo.workflow_name=robolab_2tasks_pi_and_cosmos_100ep

This submits the two robolab tasks, each evaluated with both the
`pi0.5 <https://github.com/Physical-Intelligence/openpi>`_ and
`cosmos-action-edge <https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID>`_ policies.
The policies are executed on
`L40 <https://www.nvidia.com/en-us/data-center/l40/>`_
nodes.

The submission script outputs:

.. code-block:: text

  Workflow submit successful.
  Workflow ID        - robolab_2tasks_pi_and_cosmos_100ep-1
  Workflow Overview  - https://us-west-2-aws.osmo.nvidia.com/workflows/robolab_2tasks_pi_and_cosmos_100ep-1
  Workflow Dashboard - https://ovx-l40-03.osmo.nvidia.com/dashboard/#/search?namespace=osmo-prod&q=92795d52950048e8

.. note::

  For these commands to work, you need a cluster running OSMO.
  Replace ``osmo.pool=isaac-dev-l40-03`` with your cluster name,
  and ``osmo.platform=ovx-l40`` with the model of available compute nodes.

In the OSMO dashboard, you can view the workflow and its runs:

.. figure:: ../../../images/teaser_page/multinode_evaluation/multinode_evaluation.png
   :width: 100%
   :alt: OSMO workflow view of a multi-node Arena experiment with parallel arena-run groups
   :align: center

   An OSMO workflow for a multi-node Arena Experiment. Each ``arena-run`` group runs an experiment
   runner and a policy server in parallel; a final collect task gathers the results.


Adjust the Experiment at submission time
----------------------------------------

You can tweak a Run at submission time, without editing its YAML file.
To view the available override-able values, use the ``--list_overrides`` flag.

.. code-block:: bash

   python osmo/submit_arena_experiment.py \
     --experiment_cfg isaaclab_arena_environments/robolab/experiment_configs/robolab_20_tasks_pi0_and_cosmos.yaml \
     --list_overrides

For example, to shorten the banana in bowl ``pi0.5`` run to four episodes:

.. code-block:: bash

   python osmo/submit_arena_experiment.py \
     --experiment_cfg isaaclab_arena_environments/robolab/experiment_configs/robolab_20_tasks_pi0_and_cosmos.yaml \
     experiment_cfg.runs.banana_in_bowl_pi0.rollout_limit.num_episodes=4

Of particular note is the ability to override the way the experiment is executed.
Overrides prefixed with ``osmo.`` set the workflow's scheduling and per-Run resources. The most
common ones are:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Override
     - Default
     - Meaning
   * - ``osmo.workflow_name``
     - ``arena-workflow``
     - Name the workflow appears under in OSMO.
   * - ``osmo.pool``
     - ``isaac-dev-l40s-04``
     - Target compute pool.
   * - ``osmo.platform``
     - ``ovx-l40s``
     - Hardware platform requested for each Run.
   * - ``osmo.gpus``
     - ``1``
     - GPUs requested per Run.
   * - ``osmo.cpus``
     - ``15``
     - CPU cores requested per Run.
   * - ``osmo.memory``
     - ``128Gi``
     - Memory requested per Run.
   * - ``osmo.storage``
     - ``200Gi``
     - Storage requested per Run.


Preview before submitting
-------------------------

Add ``--dry_run`` to render the OSMO workflow YAML and print it instead of submitting it.

.. code-block:: bash

   python osmo/submit_arena_experiment.py \
     --experiment_cfg isaaclab_arena_environments/robolab/experiment_configs/robolab_20_tasks_pi0_and_cosmos.yaml \
     osmo.workflow_name=alex_arena_20tasks_pi_cosmos_robolab_100ep \
     --dry_run
