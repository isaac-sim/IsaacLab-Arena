Your Own Tasks and Embodiments
==============================

The :doc:`previous page <external_environments>` showed how to define an
external environment that composes built-in assets.  This page goes further:
you will learn how to define a **custom task** and a **custom embodiment
variant** in an external file and wire them into an environment.

The full example lives in
``isaaclab_arena_examples/external_environments/advanced.py``.

Defining a Custom Task
----------------------

A task controls the episode lifecycle: termination conditions, metrics, and
(optionally) extra scene objects, events, rewards, and viewer settings.
Create one by subclassing ``TaskBase``:

.. code-block:: python

   from dataclasses import MISSING

   import isaaclab.envs.mdp as mdp_isaac_lab
   from isaaclab.managers import TerminationTermCfg
   from isaaclab.utils import configclass

   from isaaclab_arena.embodiments.common.arm_mode import ArmMode
   from isaaclab_arena.metrics.metric_base import MetricBase
   from isaaclab_arena.metrics.success_rate import SuccessRateMetric
   from isaaclab_arena.tasks.task_base import TaskBase


   class SuccessAfterNStepsTask(TaskBase):
       """Minimal task: the episode succeeds after a fixed number of steps."""

       def __init__(self, num_steps_for_success: int = 50, episode_length_s: float = 10.0):
           super().__init__(
               episode_length_s=episode_length_s,
               task_description=f"Succeed after {num_steps_for_success} steps",
           )
           self.num_steps_for_success = num_steps_for_success

       def get_scene_cfg(self):
           return None

       def get_termination_cfg(self):
           n = self.num_steps_for_success
           success = TerminationTermCfg(func=lambda env, n=n: env.episode_length_buf >= n)
           return SuccessAfterNStepsTerminationsCfg(success=success)

       def get_events_cfg(self):
           return None

       def get_mimic_env_cfg(self, arm_mode: ArmMode):
           raise NotImplementedError

       def get_metrics(self) -> list[MetricBase]:
           return [SuccessRateMetric()]


   @configclass
   class SuccessAfterNStepsTerminationsCfg:
       time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
       success: TerminationTermCfg = MISSING

The key methods to implement are:

* ``get_termination_cfg()`` — returns the termination conditions.  The example
  marks the episode as successful once ``episode_length_buf`` reaches the
  configured step count.
* ``get_metrics()`` — returns the list of metrics to compute.  Here we use the
  built-in ``SuccessRateMetric``.
* ``get_scene_cfg()`` and ``get_events_cfg()`` — return ``None`` when the task
  does not add extra scene objects or randomisation events beyond what the
  embodiment already provides.

Defining a Custom Embodiment
----------------------------

Embodiments encapsulate everything about a specific robot configuration:
articulation USD, actuator gains, action space, observations, and events.
You can create a variant of an existing embodiment by subclassing it and
overriding the parts you want to change.

The example below subclasses ``FrankaIKEmbodiment`` and halves the joint PD
gains on the shoulder and forearm actuator groups, producing a more compliant
arm:

.. code-block:: python

   from isaaclab_arena.assets.register import register_asset
   from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment


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
the same way as any built-in embodiment:

.. code-block:: python

   embodiment = self.asset_registry.get_asset_by_name("franka_ik_soft")()

Putting It All Together
-----------------------

The environment class composes the custom task, the custom embodiment, and a
scene — exactly like the basic example, but now using the pieces defined above:

.. code-block:: python

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

Running the Example
-------------------

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --num_steps 100 \
     --external_environment_class_path \
     isaaclab_arena_examples.external_environments.advanced:ExternalFrankaTableWithTaskEnvironment \
     franka_table_with_task

.. note::

   Like the basic example, this file lives inside the Isaac Lab Arena source
   tree (``isaaclab_arena_examples/external_environments/advanced.py``) but is
   **not** included in the built-in environments.  It is loaded through the
   ``--external_environment_class_path`` flag to demonstrate how an external
   codebase would use the same mechanism.
