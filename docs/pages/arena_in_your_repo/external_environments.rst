Defining Environments in Your Repository
========================================

This section describes how to write a new environment, within your own repository
(i.e. not in the Isaac Lab - Arena source tree).

To write your own environment, define an ``ArenaEnvironmentCfg`` dataclass and subclass
``ArenaEnvironmentFactory``. Set a unique ``name`` and implement ``build()``.
Below is an example of a custom environment that places a single object on a table.

.. code-block:: python

   # my_package/isaaclab_arena_environments/my_environment.py

   from dataclasses import dataclass

   from isaaclab_arena.environments.arena_environment_factory import (
       ArenaEnvironmentCfg,
       ArenaEnvironmentFactory,
   )


   @dataclass
   class ExternalFrankaTableEnvironmentCfg(ArenaEnvironmentCfg):
       object: str = "cracker_box"


   class ExternalFrankaTableEnvironment(
       ArenaEnvironmentFactory[ExternalFrankaTableEnvironmentCfg]
   ):

       name: str = "franka_table"
       # Keep direct argparse-based runners available.
       _legacy_argparse_cfg_type = ExternalFrankaTableEnvironmentCfg

       def build(self, cfg: ExternalFrankaTableEnvironmentCfg):
           from isaaclab_arena.assets.object_reference import ObjectReference
           from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
           from isaaclab_arena.relations.relations import IsAnchor, On
           from isaaclab_arena.scene.scene import Scene
           from isaaclab_arena.tasks.no_task import NoTask

           # Grab some assets from the registry.
           background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
           light = self.asset_registry.get_asset_by_name("light")()
           pick_up_object = self.asset_registry.get_asset_by_name(cfg.object)()
           embodiment = self.asset_registry.get_asset_by_name("franka_ik")()

           # Position the assets
           table_reference = ObjectReference(
               name="table",
               prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
               parent_asset=background,
           )
           table_reference.add_relation(IsAnchor())
           pick_up_object.add_relation(On(table_reference))

           # Compose the scene
           scene = Scene(assets=[background, table_reference, pick_up_object, light])

           # Create the environment
           isaaclab_arena_environment = IsaacLabArenaEnvironment(
               name=self.name,
               embodiment=embodiment,
               scene=scene,
               task=NoTask(),
           )
           return isaaclab_arena_environment


Register the Environment Package
--------------------------------

Publish the factory from the external package's ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."isaaclab_arena.environments"]
   franka_table = "my_package.isaaclab_arena_environments.my_environment:ExternalFrankaTableEnvironment"

The entry-point name must equal the factory's ``name``. Install the external package in the same
environment as Arena so Python can discover its package metadata:

.. code-block:: bash

   uv pip install --editable ./my_package

Arena loads installed environment entry points together with its built-in factories. It also
registers Gym placeholders for them, so Isaac Lab's ``train``, ``play``, ``teleop record``, and
``teleop replay`` commands can select an external environment by name without
``--external_callback``. Training and playback additionally require the environment to provide an
RL task and policy configuration.

For an RL-capable external factory, launch training with the registered name:

.. code-block:: bash

   uv run isaaclab train \
     --rl_library rsl_rl \
     --task franka_table \
     --num_envs 64 \
     --max_iterations 100

Evaluate the Environment
------------------------

Reference the registered name from a typed Experiment YAML:

.. code-block:: yaml

   runs:
     external_franka:
       environment:
         type: franka_table
         object: tomato_soup_can
       policy:
         type: zero_action
       rollout_limit:
         num_episodes: 1

Then run it through the canonical Arena command:

.. code-block:: bash

   uv run isaaclab arena evaluate \
     --experiment-config external_franka_experiment.yaml \
     --viz kit

Direct argparse-based runners can also resolve the installed factory:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --num_steps 50 \
     franka_table \
     --object tomato_soup_can

.. image:: ../../images/externally_defined_environment.png
   :width: 50%
   :alt: External environment example
   :align: center

Legacy Class-Path Loading
-------------------------

``--external_environment_class_path`` remains available for legacy factories based on
``ExampleEnvironmentBase``. That compatibility path supports direct runners but does not provide
the typed configuration needed by Experiment YAML. Migrate to ``ArenaEnvironmentFactory[Cfg]`` and
an installed entry point to use the unified commands without an explicit callback.

Next Steps
----------

The example above uses a built-in task (``NoTask``) and a built-in embodiment
(``franka_ik``).  To learn how to define your own custom tasks and embodiment
variants, see :doc:`external_tasks_and_embodiments`.
