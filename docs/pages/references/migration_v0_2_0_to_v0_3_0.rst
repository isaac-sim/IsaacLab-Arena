Migrate from v0.2.0 to v0.3.0
=============================

Arena v0.3.0 replaces several loosely typed evaluation and extension interfaces with typed
Environment, Policy, Run, and Experiment configurations. This guide covers the changes most
likely to affect repositories that import Arena or keep it as a submodule.

Evaluation entry point and configuration
----------------------------------------

The v0.2 ``eval_runner.py`` accepted a JSON ``jobs`` list:

.. code-block:: bash

   python isaaclab_arena/evaluation/eval_runner.py \
     --eval_jobs_config path/to/eval_jobs.json

.. code-block:: json

   {
     "jobs": [
       {
         "name": "maple",
         "arena_env_args": {
           "environment": "pick_and_place_maple_table",
           "num_envs": 1,
           "enable_cameras": true
         },
         "policy_type": "zero_action",
         "policy_config_dict": {},
         "num_steps": 100
       }
     ]
   }

In v0.3, use ``experiment_runner.py`` with a typed YAML Experiment Definition. Run names are
mapping keys, and environment, builder, policy, and rollout settings are separate:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config path/to/experiment.yaml

.. code-block:: yaml

   runs:
     maple:
       environment:
         type: pick_and_place_maple_table
         enable_cameras: true
       environment_builder:
         num_envs: 1
       policy:
         type: zero_action
       rollout_limit:
         num_steps: 100

Move fields as follows:

* ``jobs`` list → ``runs`` mapping.
* ``job.name`` → key below ``runs``.
* ``arena_env_args.environment`` → ``environment.type``.
* environment-specific arguments → ``environment``.
* ``num_envs``, seed, language instruction, and other builder settings →
  ``environment_builder``. Continue to select the process device with ``--device``.
* ``policy_type`` and ``policy_config_dict`` → one typed ``policy`` mapping.
* ``num_steps`` or ``num_episodes`` → ``rollout_limit``.

``experiment_runner.py`` temporarily accepts legacy JSON and converts it through
``legacy_eval_config.py``. New configurations should use YAML; Hydra overrides and the complete
typed Experiment interface are available only on that path. The deprecated
``--eval_jobs_config`` spelling remains an alias for ``--experiment_config`` during this
transition.

See :doc:`Arena Experiments <../concepts/concept_arena_experiments>`,
:doc:`First Arena Experiment <../quickstart/arena_experiment>`, and the checked-in
``isaaclab_arena_environments/experiment_configs/getting_started_experiment.yaml`` example.
The corresponding legacy example is
``isaaclab_arena_environments/eval_jobs_configs/getting_started_jobs_config.json``.

Custom environments
-------------------

Registered v0.2 environments commonly parsed ``argparse.Namespace`` directly:

.. code-block:: python

   class MyEnvironment(ExampleEnvironmentBase):
       name = "my_environment"

       def get_env(self, args_cli: argparse.Namespace):
           embodiment = make_embodiment(args_cli.embodiment)
           return make_environment(embodiment)

       @staticmethod
       def add_cli_args(parser):
           parser.add_argument("--embodiment", default="franka")

In v0.3, declare a typed ``ArenaEnvironmentCfg`` and build through
``ArenaEnvironmentFactory``:

.. code-block:: python

   from dataclasses import dataclass

   from isaaclab_arena.assets.register import register_environment
   from isaaclab_arena.environments.arena_environment_factory import (
       ArenaEnvironmentCfg,
       ArenaEnvironmentFactory,
   )


   @dataclass
   class MyEnvironmentCfg(ArenaEnvironmentCfg):
       embodiment: str = "franka"


   @register_environment
   class MyEnvironment(ArenaEnvironmentFactory[MyEnvironmentCfg]):
       name = "my_environment"

       def build(self, cfg: MyEnvironmentCfg):
           embodiment = make_embodiment(cfg.embodiment)
           return make_environment(embodiment)

The typed configuration is shared by Experiment YAML, generated CLI flags, and direct Python
construction. ``ExampleEnvironmentBase`` remains as a deprecated compatibility shim, but new and
migrated environments should use the typed factory. See :doc:`Environment Definitions
<../concepts/environment/environment_definition>` and
``isaaclab_arena_environments/pick_and_place_maple_table_environment.py`` for complete typed
examples.

Policy and remote-policy APIs
-----------------------------

Policy configuration is also typed. Replace policy-owned argparse hooks or untyped dictionaries
with a ``PolicyCfg`` dataclass passed to ``PolicyBase``:

.. code-block:: python

   @dataclass
   class MyPolicyCfg(PolicyCfg):
       policy_device: str = "cuda:0"


   @register_policy
   class MyPolicy(PolicyBase[MyPolicyCfg]):
       name = "my_policy"

       def __init__(self, config: MyPolicyCfg):
           super().__init__(config)

       def get_action(self, env, observation):
           ...

The v0.2 ``isaaclab_arena.remote_policy`` client/server framework was removed. Remote model
integrations now own their server process and transport while exposing an Arena ``PolicyBase``
client. Chunked OpenPI-protocol clients can reuse ``RemoteChunkReplayPolicy`` and an
``EmbodimentAdapter``; GR00T uses its native policy client. Use the current OpenPI, GR00T,
Cosmos, or DreamZero integration as the closest template rather than importing the removed
``RemotePolicyConfig``, ``PolicyClient``, ``PolicyServer``, or ``ServerSidePolicy`` classes from
``isaaclab_arena.remote_policy``, or ``ClientSidePolicy`` from ``isaaclab_arena.policy``.

For GR00T specifically, replace the in-process
``isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy`` with
``Gr00tRemoteClosedloopPolicy``, configure ``remote_host`` and ``remote_port``, and start the
GR00T server separately. Compare
``isaaclab_arena_environments/eval_jobs_configs/droid_pnp_srl_gr00t_jobs_config.json`` with
``isaaclab_arena_environments/experiment_configs/droid_pnp_srl_gr00t_experiment.yaml``.

See :doc:`Policy <../concepts/policy/index>`,
:doc:`OpenPI <../quickstart/running_a_real_policy/openpi>`, and
:doc:`GR00T <../quickstart/running_a_real_policy/gr00t>`.

Additional evaluation changes
-----------------------------

Variation keys now belong below each Run's ``variations`` mapping. Graph YAML paths are selected
through ``environment.type``. Local Experiment outputs are grouped below a timestamped
``outputs`` directory, and OSMO submission uses typed YAML through
``osmo/submit_arena_experiment.py``.

Continue to v0.3.1
------------------

When updating beyond v0.3.0, continue with :doc:`Migrate from v0.3.0 to v0.3.1
<migration_v0_3_0_to_v0_3_1>`.
