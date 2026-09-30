Your Own Tasks and Embodiments
==============================

The :doc:`previous page <external_environments>` showed how to define an
environment in your repository, external to the Isaac Lab - Arena source tree.
Typical users will also want to define their own tasks and embodiments.
This page describes how that is done, in an external repo.

Defining a Custom Task
----------------------

A custom task is defined by subclassing ``TaskBase`` and implementing the required methods.
The code below shows how to define a simple task that succeeds after a fixed number of steps.
This task can be passed to the ``ArenaEnvBuilder`` to create an environment
(see :ref:`putting_it_all_together` below for an example).

.. code-block:: python

   # my_package/isaaclab_arena_environments/my_environment_with_task.py

   class SuccessAfterNStepsTask(TaskBase):
       """Minimal task: the episode succeeds after a fixed number of steps."""

       def __init__(self, num_steps_for_success: int = 50, episode_length_s: float = 10.0):
           super().__init__(
               episode_length_s=episode_length_s,
               task_description=f"Succeed after {num_steps_for_success} steps",
           )
           self.num_steps_for_success = num_steps_for_success

       def get_termination_cfg(self):
           n = self.num_steps_for_success
           success = TerminationTermCfg(func=lambda env, n=n: env.episode_length_buf >= n)
           return SuccessAfterNStepsTerminationsCfg(success=success)

       def get_metrics(self) -> list[MetricBase]:
           return [SuccessRateMetric()]


   @configclass
   class SuccessAfterNStepsTerminationsCfg:
       time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
       success: TerminationTermCfg = MISSING


Defining a Custom Embodiment
----------------------------

Custom embodiments are defined by subclassing ``EmbodimentBase`` and implementing
the required methods.
In this example, we keep things simple by subclassing an existing embodiment
(``FrankaIKEmbodiment``) and overriding some joint PD gains to produce a more compliant
arm:

.. code-block:: python

   # my_package/isaaclab_arena_environments/my_environment_with_task.py


   @register_asset
   class SoftFrankaIKEmbodiment(FrankaIKEmbodiment):
       """Franka IK embodiment with halved joint PD gains."""

       name = "franka_ik_soft"

       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           for actuator_name in ("panda_shoulder", "panda_forearm"):
               actuator = self.scene_config.robot.actuators[actuator_name]
               actuator.stiffness = 200.0
               actuator.damping = 40.0

The ``@register_asset`` decorator registers the class with the
``AssetRegistry`` under the name ``"franka_ik_soft"``, so it can be fetched
the same way as any built-in embodiment (see :ref:`putting_it_all_together` below for an example).

.. _putting_it_all_together:

Putting It All Together
-----------------------

Here we create an example environment, like in :doc:`external_environments`, but this time
that composes the custom task and custom embodiment.

.. code-block:: python

   # my_package/isaaclab_arena_environments/my_environment_with_task.py

   class ExternalFrankaTableWithTaskEnvironment(ExampleEnvironmentBase):

       name: str = "franka_table_with_task"

       def get_env(self, args_cli: argparse.Namespace):
           from isaaclab_arena.assets.object_reference import ObjectReference
           from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
           from isaaclab_arena.relations.relations import IsAnchor, On
           from isaaclab_arena.scene.scene import Scene

           background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
           light = self.asset_registry.get_asset_by_name("light")()
           pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
           embodiment = self.asset_registry.get_asset_by_name("franka_ik_soft")()

           table_reference = ObjectReference(
               name="table",
               prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
               parent_asset=background,
           )
           table_reference.add_relation(IsAnchor())
           pick_up_object.add_relation(On(table_reference))

           scene = Scene(assets=[background, table_reference, pick_up_object, light])

           task = SuccessAfterNStepsTask(
               num_steps_for_success=50,
               episode_length_s=10.0,
           )

           return IsaacLabArenaEnvironment(
               name=self.name,
               embodiment=embodiment,
               scene=scene,
               task=task,
           )

       @staticmethod
       def add_cli_args(parser: argparse.ArgumentParser) -> None:
           parser.add_argument("--object", type=str, default="cracker_box")

External environments can be used in Isaac Lab - Arena workflows by using a particular
CLI syntax. For example, a zero-action policy can be run with an
externally-defined environment like this:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --num_episodes 1 \
     --external_environment_class_path my_package.isaaclab_arena_environments.my_environment:ExternalFrankaTableWithTaskEnvironment \
     franka_table_with_task \
     --object tomato_soup_can

So the flag ``external_environment_class_path`` is used to specify the (fully qualified) path to the
external environment module and class. The environment name is then specified as the
first non flag argument to the policy runner, and any additional arguments are passed to the
environment's ``add_cli_args()`` method.

.. note::

    The environment above is actually located in ``isaaclab_arena_examples/external_environments/advanced.py``.
    So this environment is located in the Isaac Lab - Arena source-tree, but isn't included
    in the built in environments, so must be called through the external environment syntax.
    This is done to demonstrate how this would be done in an external codebase.

    The environment can be run with:

    .. code-block:: bash

        python isaaclab_arena/evaluation/policy_runner.py \
          --viz kit \
          --policy_type zero_action \
          --num_episodes 1 \
          --external_environment_class_path isaaclab_arena_examples.external_environments.advanced:ExternalFrankaTableWithTaskEnvironment \
          franka_table_with_task \
          --object tomato_soup_can
