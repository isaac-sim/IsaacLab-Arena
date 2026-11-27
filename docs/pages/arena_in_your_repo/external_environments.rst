Defining Environments in Your Repository
========================================

You can define an environment in your own package and run it with Arena's runners, without
changing Arena. Like a built-in environment, it has two classes:

* A config dataclass that subclasses ``ArenaEnvironmentCfg``. Its fields are the values a user can
  set. Each environment needs its own config class.
* A factory class that subclasses ``ArenaEnvironmentFactory[YourCfg]``, sets a unique ``name``,
  and builds the environment in ``build(cfg)``.

Arena registers the factory when it loads it by its class path, so it doesn't need the
``@register_environment`` decorator that the built-in environments use.

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

To run it, pass the factory class to ``--external_environment_class_path``, followed by the
environment name and its flags. For example, with a zero-action policy:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --num_steps 50 \
     --external_environment_class_path my_package.isaaclab_arena_environments.my_environment:ExternalFrankaTableEnvironment \
     franka_table \
     --object tomato_soup_can

The ``--external_environment_class_path`` flag takes the module path and the class name of the
factory, separated by a colon. The module must be importable, for example because your package is
installed or on ``PYTHONPATH``. The runner imports the class and registers it under its ``name``.
The environment name is the first positional argument, and the flags after it set the fields of the
environment's config.

The flag works the same way with the Environment Runner
(``isaaclab_arena/scripts/environment_runner.py``) and with Isaac Lab scripts that register Arena
environments through
``--external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback``.
Isaac Lab scripts take the environment name with ``--task`` instead of as a positional argument.

Use a ``name`` that no other environment uses. If another class already owns the name, the runner
stops with an error that names that class.

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
          --external_environment_class_path isaaclab_arena_examples.external_environments.basic:ExternalFrankaTableEnvironment \
          franka_table \
          --object tomato_soup_can

    which results in an environment like the one below:

    .. image:: ../../images/externally_defined_environment.png
       :width: 50%
       :alt: External environment example
       :align: center

Config Fields and Flags
-----------------------

Each field of the config becomes a flag with the field's name, such as ``--object``.
Flags work for fields of these types:

* ``bool``, ``int``, ``float``, and ``str``. A ``bool`` field that defaults to ``False`` becomes a
  switch such as ``--rl_training_mode``. Other ``bool`` fields get ``--name`` and ``--no-name``.
* A ``Literal`` whose values are all ``str`` or all ``int``, such as ``Literal["left", "right"]``.
* A string-valued enum, such as ``class Mode(str, Enum)``.
* An optional of these types, such as ``str | None``.
* A list of ``int``, ``float``, or ``str``, such as ``list[str]``. The flag takes zero or more values.

Other field types are not supported. Some, such as ``tuple[float, float]`` or ``dict[str, float]``,
stop the runner with an error. Others get a flag that rejects or misreads the value, for example
``list[bool]`` reads ``False`` as ``True``.

Fields named ``enable_cameras``, ``num_envs``, and ``mimic`` get no flag of their own. They take the
value of the runner's flag with the same name. A field named ``auto`` gets no flag.

Migrating from ``ExampleEnvironmentBase``
-----------------------------------------

``ExampleEnvironmentBase`` and its ``get_env(args_cli)`` and ``add_cli_args(parser)`` methods have
been removed. To migrate an environment defined in your repository:

#. Move each flag declared in ``add_cli_args()`` into a field of a new dataclass that subclasses
   ``ArenaEnvironmentCfg``. Use the flag's name as the field name, its default as the field
   default, and a type from the list above.
#. Replace the ``ExampleEnvironmentBase`` base class with ``ArenaEnvironmentFactory[YourCfg]``.
#. Rename ``get_env(self, args_cli)`` to ``build(self, cfg)``, and read values from ``cfg`` instead
   of ``args_cli``. If ``get_env()`` read ``args_cli.num_envs`` or ``args_cli.mimic``, add a field
   with that name to the config.
#. Delete ``add_cli_args()``. The flags are now generated from the config fields.

Commands that use ``--external_environment_class_path`` then keep working without changes.

Before:

.. code-block:: python

   import argparse

   from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


   class ExternalFrankaTableEnvironment(ExampleEnvironmentBase):

       name: str = "franka_table"

       def get_env(self, args_cli: argparse.Namespace):
           ...
           pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
           ...

       @staticmethod
       def add_cli_args(parser: argparse.ArgumentParser) -> None:
           parser.add_argument("--object", type=str, default="cracker_box")

After:

.. code-block:: python

   from dataclasses import dataclass

   from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory


   @dataclass
   class ExternalFrankaTableEnvironmentCfg(ArenaEnvironmentCfg):
       object: str = "cracker_box"


   class ExternalFrankaTableEnvironment(ArenaEnvironmentFactory[ExternalFrankaTableEnvironmentCfg]):

       name: str = "franka_table"

       def build(self, cfg: ExternalFrankaTableEnvironmentCfg):
           ...
           pick_up_object = self.asset_registry.get_asset_by_name(cfg.object)()
           ...

Next Steps
----------

The example above uses a built-in task (``NoTask``) and a built-in embodiment
(``franka_ik``).  To learn how to define your own custom tasks and embodiment
variants, see :doc:`external_tasks_and_embodiments`.
