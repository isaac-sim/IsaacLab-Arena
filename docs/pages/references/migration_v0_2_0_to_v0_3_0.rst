Migrate from v0.2.0 to v0.3.0
=============================

Arena v0.3.0 replaces several loosely typed evaluation and extension interfaces with typed
Environment, Policy, Run, and Experiment configurations. This guide covers the changes most
likely to affect repositories that import Arena or keep it as a submodule.

Quick access
------------

Use this table to jump to the parts that apply to your project.

.. list-table::
   :header-rows: 1

   * - If your v0.2 project...
     - Go to
   * - Installs Arena or pins its Isaac Lab submodule
     - :ref:`Installation and prerequisites <migration-v0-2-installation>`
   * - Calls Arena's teleoperation, recording, replay, or Mimic scripts
     - :ref:`Isaac Lab imitation-learning interoperation <migration-v0-2-il-interop>`
   * - Runs JSON evaluation jobs
     - :ref:`Evaluation entry point and configuration <migration-v0-2-evaluation>`
   * - Constructs ``ArenaEnvBuilder`` directly
     - :ref:`Direct environment-builder construction <migration-v0-2-builder>`
   * - Defines or registers a custom environment
     - :ref:`Custom environments <migration-v0-2-custom-environments>`
   * - Uses an orchestrator or overrides ``modify_env_cfg()``
     - :ref:`Environment customization hooks <migration-v0-2-customization>`
   * - Defines custom metrics
     - :ref:`Custom metrics <migration-v0-2-metrics>`
   * - Defines a policy or uses Arena's old remote-policy framework
     - :ref:`Policy and remote-policy APIs <migration-v0-2-policies>`

.. _migration-v0-2-installation:

Installation and prerequisites
------------------------------

Arena v0.3.0 requires Python 3.12 and moves package metadata from ``setup.py`` to
``pyproject.toml``. After checking out v0.3.0, update the Isaac Lab submodule as well; the v0.3.0
imitation-learning interoperation described below depends on the pinned Isaac Lab revision:

.. code-block:: bash

   git submodule update --init --recursive

Existing Docker workflows remain supported except for the v0.2 GR00T image selected with
``./docker/run_docker.sh -g``. That option was removed; use the standard Arena container and run
the GR00T server separately as described under :ref:`Policy and remote-policy APIs
<migration-v0-2-policies>`. Native installation with ``uv`` is an optional new route, not a
required migration:

.. code-block:: bash

   uv sync --extra dev
   source .venv/bin/activate

The committed ``uv.lock`` and ``.python-version`` select the supported dependencies and Python
version. Use the source-based setup above; Arena does not publish a standalone package containing
the complete workflow scripts. Native ``uv`` installations also omit the optional
``isaaclab_arena_curobo`` package, so use Docker when cuRobo reachability validation is required.
See :doc:`Installation <../quickstart/installation>` for the complete Docker and native setup.

.. _migration-v0-2-il-interop:

Isaac Lab imitation-learning interoperation
-------------------------------------------

Arena's v0.2 imitation-learning entry points were removed. In v0.3.0, invoke the corresponding
Isaac Lab scripts and register Arena environments through
``isaaclab_arena.environments.isaaclab_interop.environment_registration_callback``:

.. list-table::
   :header-rows: 1

   * - v0.2 Arena script
     - v0.3.0 Isaac Lab script
   * - ``isaaclab_arena/scripts/imitation_learning/teleop.py``
     - ``submodules/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py``
   * - ``isaaclab_arena/scripts/imitation_learning/record_demos.py``
     - ``submodules/IsaacLab/scripts/tools/record_demos.py``
   * - ``isaaclab_arena/scripts/imitation_learning/replay_demos.py``
     - ``submodules/IsaacLab/scripts/tools/replay_demos.py``
   * - ``isaaclab_arena/scripts/imitation_learning/annotate_demos.py``
     - ``submodules/IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py``
   * - ``isaaclab_arena/scripts/imitation_learning/generate_dataset.py``
     - ``submodules/IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py``

Replace the positional Arena environment name with ``--task`` and add ``--external_callback`` to
each Isaac Lab command. For example:

.. code-block:: bash

   python submodules/IsaacLab/scripts/tools/record_demos.py \
     --dataset_file /datasets/my_demos.hdf5 \
     --num_demos 10 \
     --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
     --task my_environment

Keep script-specific options such as ``--mimic`` where needed. Replace the v0.2
``--teleop_device <device>`` environment option with ``--arena_teleop_device <device>``. Do not
pass Isaac Lab's own ``--teleop_device`` option for an Arena OpenXR device; it selects Isaac Lab's
legacy device path. XR is no longer enabled from the device name, so also pass ``--xr`` for XR
devices. For an environment defined outside Arena, pass
``--external_environment_class_path package.module:EnvironmentClass``; the registration callback
loads and registers it before Isaac Lab resolves ``--task``. See the
:doc:`imitation-learning workflows <../example_workflows/imitation_learning/index>` and
:doc:`external environments <../arena_in_your_repo/external_environments>` for complete commands.

.. _migration-v0-2-evaluation:

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
Local Experiment results are now grouped below a timestamped ``outputs`` directory.

.. _migration-v0-2-builder:

Direct environment-builder construction
---------------------------------------

Code that constructs ``ArenaEnvBuilder`` directly must replace its ``argparse.Namespace`` with an
``ArenaEnvBuilderCfg``:

.. code-block:: python

   # v0.2
   args = argparse.Namespace(num_envs=4, env_spacing=3.0, seed=42, solve_relations=True, ...)
   builder = ArenaEnvBuilder(arena_env, args)
   env_cfg = builder.compose_manager_cfg()

.. code-block:: python

   # v0.3.0
   from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

   builder_cfg = ArenaEnvBuilderCfg(
       num_envs=4,
       env_spacing=3.0,
       seed=42,
       solve_relations=True,
   )
   builder = ArenaEnvBuilder(arena_env, builder_cfg)
   env_cfg, env_kwargs = builder.compose_manager_cfg()

``compose_manager_cfg()`` now returns both the Isaac Lab environment configuration and keyword
arguments that must be forwarded when creating the environment. The builder does this
automatically when calling ``builder.make_registered()``. If custom code materializes the
configuration first, pass both values back:

.. code-block:: python

   env = builder.make_registered(env_cfg, env_kwargs)

.. _migration-v0-2-custom-environments:

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
       _legacy_argparse_cfg_type = MyEnvironmentCfg

       def build(self, cfg: MyEnvironmentCfg):
           embodiment = make_embodiment(cfg.embodiment)
           return make_environment(embodiment)

The typed configuration is shared by Experiment YAML, generated CLI flags, and direct Python
construction. The temporary ``_legacy_argparse_cfg_type`` attribute is required while the
argparse-based runners remain supported.

Replace imports from ``isaaclab_arena.assets.asset_registry`` with
``isaaclab_arena.assets.registries``. The latter defines ``AssetRegistry``, ``DeviceRegistry``,
``RetargeterRegistry``, ``PolicyRegistry``, and ``HDRImageRegistry``.

The decorator above is for environments imported and registered as part of Arena. For a class
loaded with ``--external_environment_class_path``, follow the external-environment interface and
do not decorate the class: the loader performs registration. In v0.3.0, Experiment YAML cannot
load an arbitrary external Python environment by class path; use a registered environment name or
an environment graph YAML instead.

``ExampleEnvironmentBase`` remains as a deprecated compatibility shim for the external loader,
but new in-tree environments should use the typed factory. See :doc:`Environment Definitions
<../concepts/environment/environment_definition>`,
:doc:`External Environments <../arena_in_your_repo/external_environments>`, and
``isaaclab_arena_environments/pick_and_place_maple_table_environment.py`` for complete typed
examples.

.. _migration-v0-2-customization:

Environment customization hooks
-------------------------------

The v0.2 ``OrchestratorBase`` interface and the ``modify_env_cfg()`` hooks on ``Scene``,
``EmbodimentBase``, and ``TaskBase`` were removed. Move cross-component construction into the
environment factory's ``build()`` method. Move final changes to the composed Isaac Lab
configuration into the environment-level ``env_cfg_callback``:

.. code-block:: python

   def configure_env(env_cfg):
       env_cfg.decimation = 4
       return env_cfg


   def make_environment(scene, embodiment, task):
       return IsaacLabArenaEnvironment(
           name="my_environment",
           scene=scene,
           embodiment=embodiment,
           task=task,
           env_cfg_callback=configure_env,
       )

An inherited ``modify_env_cfg()`` method is no longer called, so migrate the customization rather
than leaving the old override in place.

.. _migration-v0-2-metrics:

Custom metrics
--------------

Custom ``MetricBase`` implementations now describe offline computation with
``get_metric_term_cfg()``. Replace the metric instance's ``compute_metric_from_recording()`` method
with a module-level function and return it in a ``MetricTermCfg``:

.. code-block:: python

   from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg


   def compute_my_metric(recorded_metric_data, scale):
       return scale * compute_value(recorded_metric_data)


   class MyMetric(MetricBase):
       name = "my_metric"
       recorder_term_name = "my_metric_data"

       def get_metric_term_cfg(self):
           return MetricTermCfg(
               compute_metric_func=compute_my_metric,
               params={"scale": 2.0},
               recorder_term_name=self.recorder_term_name,
           )

``compute_metric_func`` receives the recorded data as its first argument and the entries in
``params`` as keyword arguments. Keep implementing ``get_recorder_term_cfg()`` to define the data
recorded for the metric.

.. _migration-v0-2-policies:

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
GR00T server separately. Pass the checkpoint to the server with ``--model-path``; the v0.3.0
policy configuration no longer has ``model_path``, and an old YAML key is ignored after a warning.
Compare the ``release/0.2.0`` version of
``isaaclab_arena_environments/eval_jobs_configs/droid_pnp_srl_gr00t_jobs_config.json`` with the
v0.3.0
``isaaclab_arena_environments/experiment_configs/droid_pnp_srl_gr00t_experiment.yaml``.

See :doc:`Policy <../concepts/policy/index>`,
:doc:`OpenPI <../quickstart/running_a_real_policy/openpi>`, and
:doc:`GR00T <../quickstart/running_a_real_policy/gr00t>`.

Continue to v0.3.1
------------------

When updating beyond v0.3.0, continue with :doc:`Migrate from v0.3.0 to v0.3.1
<migration_v0_3_0_to_v0_3_1>`.
