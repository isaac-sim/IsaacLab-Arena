Defining Environments in Your Repository
========================================

You can define an environment in your own package and run it with Arena's runners, without
changing Arena. Like a built-in environment, it has two classes:

* A config dataclass that subclasses ``ArenaEnvironmentCfg``. Its fields are the values a user can
  set. Each environment needs its own config class.
* A factory class that subclasses ``ArenaEnvironmentFactory[YourCfg]``, sets a unique ``name``,
  and builds the environment in ``build(cfg)``.

This example places one object on a table:

.. code-block:: python

   # my_package/isaaclab_arena_environments/my_environment.py

   from __future__ import annotations

   from dataclasses import dataclass
   from typing import TYPE_CHECKING

   from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

   if TYPE_CHECKING:
       from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


   @dataclass
   class ExternalFrankaTableEnvironmentCfg(ArenaEnvironmentCfg):
       """Configure the external Franka table environment."""

       object: str = "cracker_box"
       """Registered name of the object placed on the table."""


   class ExternalFrankaTableEnvironment(ArenaEnvironmentFactory[ExternalFrankaTableEnvironmentCfg]):

       name: str = "franka_table"

       def build(self, cfg: ExternalFrankaTableEnvironmentCfg) -> IsaacLabArenaEnvironment:
           """Build the environment from its config."""
           from isaaclab_arena.assets.object_reference import ObjectReference
           from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
           from isaaclab_arena.relations.relations import IsAnchor, On
           from isaaclab_arena.scene.scene import Scene
           from isaaclab_arena.tasks.no_task import NoTask

           # Grab some assets from the registry.
           background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
           light = self.asset_registry.get_asset_by_name("light")()
           pick_up_object = self.asset_registry.get_asset_by_name(cfg.object)()
           embodiment = self.asset_registry.get_asset_by_name("franka_ik")(enable_cameras=cfg.enable_cameras)

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

To run it, pass the factory class to ``--external_environment_class_path``, followed by the
environment name and its flags. For example, with a zero-action policy:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --num_steps 50 \
     --external_environment_class_path my_package.isaaclab_arena_environments.my_environment.ExternalFrankaTableEnvironment \
     franka_table \
     --object tomato_soup_can

The class path is the module path and the class name, joined by a dot as in a Python import. For a
class ``MyEnvironment`` in ``my_package/my_environment.py``, it is
``my_package.my_environment.MyEnvironment``. The module must be importable, for example because your
package is installed or on ``PYTHONPATH``.

The flag works the same way with the Environment Runner
(``isaaclab_arena/scripts/environment_runner.py``) and with Isaac Lab scripts that register Arena
environments through
``--external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback``.
Isaac Lab scripts take the environment name with ``--task`` instead of as a positional argument.

.. note::

    The environment above is actually located in ``isaaclab_arena_examples/external_environments/basic.py``.
    So this environment is located in the Isaac Lab - Arena source-tree, but isn't included
    in the built in environments, so must be called through the external environment syntax.
    This is done to demonstrate how this would be done in an external codebase.

    The environment can be run with:

    .. code-block:: bash

        python isaaclab_arena/evaluation/policy_runner.py \
          --viz kit \
          --policy_type zero_action \
          --num_steps 50 \
          --external_environment_class_path isaaclab_arena_examples.external_environments.basic.ExternalFrankaTableEnvironment \
          franka_table \
          --object tomato_soup_can

    which results in an environment like the one below:

    .. image:: ../../images/externally_defined_environment.png
       :width: 50%
       :alt: External environment example
       :align: center

Next Steps
----------

The example above uses a built-in task (``NoTask``) and a built-in embodiment
(``franka_ik``).  To learn how to define your own custom tasks and embodiment
variants, see :doc:`external_tasks_and_embodiments`.
