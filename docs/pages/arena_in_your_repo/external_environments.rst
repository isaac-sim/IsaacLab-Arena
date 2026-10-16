Defining Environments in Your Repository
========================================

This section describes how to write a new environment, within your own repository
(i.e. not in the Isaac Lab - Arena source tree).

An environment in your repository uses the same contract as the built-in environments:

* A configuration dataclass that subclasses ``ArenaEnvironmentCfg`` and declares the values a
  user can set. Each environment needs its own configuration class.
* A factory class that subclasses ``ArenaEnvironmentFactory[YourCfg]``, sets a unique ``name``,
  and implements ``build(cfg)``.
* The ``@register_environment`` decorator, which registers the factory and its configuration
  with the ``EnvironmentRegistry``.

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

An :doc:`Arena Experiment <../concepts/concept_arena_experiments>` selects an environment from
your repository by the dotted Python class path of its factory:

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

The Experiment Runner imports the module, so it must be importable, for example by installing
your package or adding it to ``PYTHONPATH``. Importing the module runs ``@register_environment``,
and the remaining ``environment`` values are composed into ``ExternalFrankaTableEnvironmentCfg``.
You can override them on the command line like any other Run value, for example
``runs.franka_table.environment.object=mustard_bottle``.

.. note::

    The environment above is actually located in ``isaaclab_arena_examples/external_environments/basic.py``.
    So this environment is located in the Isaac Lab - Arena source-tree, but isn't included
    in the built in environments, so must be selected by its class path.
    This is done to demonstrate how this would be done in an external codebase.

    The Experiment below runs it, together with the environment from
    :doc:`external_tasks_and_embodiments`:

    .. literalinclude:: ../../../isaaclab_arena_examples/external_environments/external_environments_experiment.yaml
       :language: yaml
       :lines: 6-

    .. code-block:: bash

        python isaaclab_arena/evaluation/experiment_runner.py \
          --viz kit \
          --experiment_config isaaclab_arena_examples/external_environments/external_environments_experiment.yaml

    which results in an environment like the one below:

    .. image:: ../../images/externally_defined_environment.png
       :width: 50%
       :alt: External environment example
       :align: center

Using the Environment with Command-Line Scripts
-----------------------------------------------

Scripts that select an environment by name on the command line import a factory from your
repository with ``--external_environment_class_path module_path:ClassName``. These include the
Environment Runner and Isaac Lab scripts that register Arena environments through
``--external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback``.
The fields of the environment's configuration become command-line flags. Flags can be generated
for fields of type ``bool``, ``int``, ``float``, ``str``, an ``Enum`` or ``Literal``, an optional
of these, or a list of these:

.. code-block:: bash

   python isaaclab_arena/scripts/environment_runner.py \
     --external_environment_class_path my_package.isaaclab_arena_environments.my_environment:ExternalFrankaTableEnvironment \
     franka_table \
     --object tomato_soup_can

Migrating from ``ExampleEnvironmentBase``
-----------------------------------------

``ExampleEnvironmentBase`` and its ``get_env(args_cli)`` and ``add_cli_args(parser)`` methods have
been removed. To migrate an environment defined in your repository:

#. Move each flag declared in ``add_cli_args()`` into a field of a new dataclass that subclasses
   ``ArenaEnvironmentCfg``, using the flag's default as the field default. ``enable_cameras`` is
   already declared by ``ArenaEnvironmentCfg``. Do not reuse ``ArenaEnvironmentCfg`` or the
   configuration of another environment; each environment needs its own configuration class.
#. Replace the ``ExampleEnvironmentBase`` base class with ``ArenaEnvironmentFactory[YourCfg]`` and
   decorate the class with ``@register_environment``.
#. Rename ``get_env(self, args_cli)`` to ``build(self, cfg)`` and read values from ``cfg`` instead
   of ``args_cli``.
#. If ``get_env`` read values of the shared runner flags, such as ``args_cli.num_envs``, add a
   field with the same name to the configuration. Command-line scripts fill ``num_envs`` and
   ``mimic`` fields from their shared flags. In an Experiment, set the field under ``environment``;
   it is not filled from ``environment_builder.num_envs``.
#. Delete ``add_cli_args()``. Command-line flags are generated from the configuration fields.
#. Replace Policy Runner commands that used ``--external_environment_class_path`` with an
   Experiment that selects the factory by its class path, as shown above. Note that the
   Experiment uses a dotted path (``module_path.ClassName``), while the command-line flag uses
   ``module_path:ClassName``.

Use a ``name`` that no built-in environment uses. When the Experiment Runner or
``--external_environment_class_path`` imports your module, a factory whose name is already
registered by another class is rejected.

Next Steps
----------

The example above uses a built-in task (``NoTask``) and a built-in embodiment
(``franka_ik``).  To learn how to define your own custom tasks and embodiment
variants, see :doc:`external_tasks_and_embodiments`.
