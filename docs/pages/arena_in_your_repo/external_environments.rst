Defining Environments in Your Repository
========================================

This section describes how to write a new environment, within your own repository
(i.e. not in the Isaac Lab - Arena source tree).

An environment in your repository uses the same contract as the built-in environments:

* A configuration dataclass that subclasses ``ArenaEnvironmentCfg``. Its fields are the values a
  user can set. Each environment needs its own configuration class.
* A factory class that subclasses ``ArenaEnvironmentFactory[YourCfg]``, sets a unique ``name``,
  and implements ``build(cfg)``.
* The ``@register_environment`` decorator, which registers the factory and its configuration
  with the ``EnvironmentRegistry`` when the module is imported.

Below is an example of a custom environment that places a single object on a table.

.. code-block:: python

   # my_package/isaaclab_arena_environments/my_environment.py

   from __future__ import annotations

   from dataclasses import dataclass
   from typing import TYPE_CHECKING

   from isaaclab_arena.assets.register import register_environment
   from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

   if TYPE_CHECKING:
       from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


   @dataclass
   class ExternalFrankaTableEnvironmentCfg(ArenaEnvironmentCfg):
       """Configure the external Franka table environment."""

       object: str = "cracker_box"
       """Registered name of the object placed on the table."""


   @register_environment
   class ExternalFrankaTableEnvironment(ArenaEnvironmentFactory[ExternalFrankaTableEnvironmentCfg]):

       name: str = "franka_table"

       def build(self, cfg: ExternalFrankaTableEnvironmentCfg) -> IsaacLabArenaEnvironment:
           """Build the environment from its typed configuration."""
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

Evaluating the Environment in an Experiment
-------------------------------------------

An :doc:`Arena Experiment <../concepts/concept_arena_experiments>` selects an environment from your
repository by the dotted class path of its factory, ``module_path.ClassName``, in
``environment.type``. The other values under ``environment`` set the fields of the environment's
configuration.

.. code-block:: yaml

   # my_experiment.yaml
   runs:
     franka_table:
       environment:
         type: my_package.isaaclab_arena_environments.my_environment.ExternalFrankaTableEnvironment
         object: tomato_soup_can
       policy:
         type: zero_action
       rollout_limit:
         num_steps: 50

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py --experiment_config my_experiment.yaml

The module must be importable, for example because your package is installed or on
``PYTHONPATH``. The Experiment Runner imports it, which runs ``@register_environment``, and then
checks that the decorator registered this class under its ``name``. You can override the
environment values on the command line like any other Run value, for example
``runs.franka_table.environment.object=mustard_bottle``.

Use a ``name`` that no other environment uses. If the class is not decorated with
``@register_environment``, loading stops with an error that asks for the decorator. If another
class already owns the name, the error names that class.

.. note::

    The environment above is actually located in ``isaaclab_arena_examples/external_environments/basic.py``.
    So this environment is located in the Isaac Lab - Arena source-tree, but isn't included
    in the built in environments, so must be selected by its class path.
    This is done to demonstrate how this would be done in an external codebase.

    The Experiment below evaluates it in the Run ``franka_table``, and the environment from
    :doc:`external_tasks_and_embodiments` in the Run ``franka_table_with_task``:

    .. literalinclude:: ../../../isaaclab_arena_examples/external_environments/external_environments_experiment.yaml
       :language: yaml
       :lines: 6-

    This command runs both Runs, one after the other:

    .. code-block:: bash

        python isaaclab_arena/evaluation/experiment_runner.py \
          --viz kit \
          --experiment_config isaaclab_arena_examples/external_environments/external_environments_experiment.yaml

    The first Run, ``franka_table``, shows the environment below:

    .. image:: ../../images/externally_defined_environment.png
       :width: 50%
       :alt: External environment example
       :align: center

Using the Environment with Command-Line Scripts
-----------------------------------------------

The Environment Runner (``isaaclab_arena/scripts/environment_runner.py``) and Isaac Lab scripts
that register Arena environments through
``--external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback``
select the environment on the command line. They import the factory with the
``--external_environment_class_path`` flag:

.. code-block:: bash

   python isaaclab_arena/scripts/environment_runner.py \
     --external_environment_class_path my_package.isaaclab_arena_environments.my_environment:ExternalFrankaTableEnvironment \
     franka_table \
     --object tomato_soup_can

The flag takes the module path and the class name separated by a colon, ``module_path:ClassName``.
An Experiment uses the dotted form ``module_path.ClassName`` instead. The environment name is the
first positional argument, and the flags after it set the fields of the environment's
configuration. Isaac Lab scripts take the environment name with ``--task`` instead of as a
positional argument. The flag imports the module and checks the name in the same way as an
Experiment.

Configuration Fields and Flags
------------------------------

Each field of the configuration becomes a flag with the field's name, such as ``--object``.
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
value of the runner's flag with the same name. A field named ``auto`` gets no flag. In an
Experiment, set these fields under ``environment``. A ``num_envs`` field is not filled from
``environment_builder.num_envs``.

Migrating from ``ExampleEnvironmentBase``
-----------------------------------------

``ExampleEnvironmentBase`` and its ``get_env(args_cli)`` and ``add_cli_args(parser)`` methods have
been removed. To migrate an environment defined in your repository:

#. Move each flag declared in ``add_cli_args()`` into a field of a new dataclass that subclasses
   ``ArenaEnvironmentCfg``. Use the flag's name as the field name, its default as the field
   default, and a type from the list above.
#. Replace the ``ExampleEnvironmentBase`` base class with ``ArenaEnvironmentFactory[YourCfg]``, and
   decorate the class with ``@register_environment``.
#. Rename ``get_env(self, args_cli)`` to ``build(self, cfg)``, and read values from ``cfg`` instead
   of ``args_cli``. If ``get_env()`` read ``args_cli.num_envs`` or ``args_cli.mimic``, add a field
   with that name to the configuration.
#. Delete ``add_cli_args()``. The flags are now generated from the configuration fields.

Commands that use ``--external_environment_class_path`` then keep working without changes. To
evaluate the environment with a policy, select it in an Experiment as shown in
`Evaluating the Environment in an Experiment`_.

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

   from isaaclab_arena.assets.register import register_environment
   from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory


   @dataclass
   class ExternalFrankaTableEnvironmentCfg(ArenaEnvironmentCfg):
       object: str = "cracker_box"


   @register_environment
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
