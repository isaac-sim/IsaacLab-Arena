Installing IsaacLab-Arena in Your Repository
============================================

The recommended way to consume IsaacLab-Arena from an external project is to include it as an
**unmodified git submodule** and extend it purely through its registration API — without editing
any file inside the Arena source tree.

This is the currently recommended integration pattern until IsaacLab-Arena is available as a
published ``pip`` package, at which point the submodule will be replaced by a simple
``pip install isaaclab_arena``. The environment and asset extension patterns below will
remain unchanged.


Repository Layout
-----------------

A typical external repository looks like this:

.. code-block:: text

   my_project/
   ├── submodules/
   │   └── IsaacLab-Arena/          ← unmodified Arena submodule
   ├── my_package/
   │   ├── pyproject.toml
   │   ├── isaaclab_arena_environments/
   │   │   ├── __init__.py
   │   │   └── my_environment.py    ← custom environment class
   ├── docker/
   │   └── Dockerfile
   └── .gitmodules

Add the submodule with:

.. code-block:: bash

   git submodule add git@github.com:isaac-sim/IsaacLab-Arena.git submodules/IsaacLab-Arena


Dockerfile
----------

Your base image must already have **Isaac Sim** and **Isaac Lab** installed (e.g.
``nvcr.io/nvidia/isaac-sim:6.0.0``).

Copy the submodule into the image and install Arena
before your own package run ``pip install -e``.

.. code-block:: dockerfile

   # Base image must have Isaac Sim
   # e.g. FROM nvcr.io/nvidia/isaac-sim:6.0.0

   # Image must have Isaac Lab installed
   # e.g. RUN /isaaclab/isaaclab.sh -i

   # Copy the submodule into the image and install Arena
   COPY submodules/IsaacLab-Arena /opt/arena
   RUN /isaac-sim/python.sh -m pip install -e /opt/arena

   # Install your package after Arena is in place
   COPY my_package /workspace/my_package
   RUN /isaac-sim/python.sh -m pip install -e /workspace/my_package

See Arena's own `Dockerfile
<https://github.com/isaac-sim/IsaacLab-Arena/blob/main/docker/Dockerfile.isaaclab_arena>`_
for a complete reference, including Isaac Lab installation and optional GR00T dependencies.

.. note::

   Note that above we have assumed that Arena is being installed inside a Docker container.
   Of course, if you have an non-Docker environment that satisfies the prerequisites, you can
   install Arena directly into that environment, without the need for Docker.

.. Defining a Custom, Externally-Defined Environment
.. -------------------------------------------------

.. This section describes how to write a new environment, outside of the Isaac Lab - Arena source tree.

.. To write your own environment, subclass ``ExampleEnvironmentBase``, set a unique ``name``,
.. and implement ``get_env()`` and ``add_cli_args()``.
.. Below is an example of a custom environment that places a single object on a table.

.. .. code-block:: python

..    # my_package/isaaclab_arena_environments/my_environment.py

..    import argparse

..    from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


..    class ExternalFrankaTableEnvironment(ExampleEnvironmentBase):

..        name: str = "franka_table"

..        def get_env(self, args_cli: argparse.Namespace):
..            from isaaclab_arena.assets.object_reference import ObjectReference
..            from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
..            from isaaclab_arena.relations.relations import IsAnchor, On
..            from isaaclab_arena.scene.scene import Scene
..            from isaaclab_arena.tasks.no_task import NoTask

..            # Grab some assets from the registry.
..            background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
..            light = self.asset_registry.get_asset_by_name("light")()
..            pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
..            embodiment = self.asset_registry.get_asset_by_name("franka_ik")()

..            # Position the assets
..            table_reference = ObjectReference(
..                name="table",
..                prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
..                parent_asset=background,
..            )
..            table_reference.add_relation(IsAnchor())
..            pick_up_object.add_relation(On(table_reference))

..            # Compose the scene
..            scene = Scene(assets=[background, table_reference, pick_up_object, light])

..            # Create the environment
..            isaaclab_arena_environment = IsaacLabArenaEnvironment(
..                name=self.name,
..                embodiment=embodiment,
..                scene=scene,
..                task=NoTask(),
..            )
..            return isaaclab_arena_environment

..        @staticmethod
..        def add_cli_args(parser: argparse.ArgumentParser) -> None:
..            parser.add_argument("--object", type=str, default="cracker_box")

.. External environments can be used in Isaac Lab Arena workflows by using a particular
.. CLI syntax. For example, a zero-action policy can be run with an
.. externally-defined environment like this:

.. .. code-block:: bash

..    python isaaclab_arena/evaluation/policy_runner.py \
..      --policy_type zero_action \
..      --num_steps 50 \
..      --external_environment_class_path my_package.isaaclab_arena_environments.my_environment:ExternalFrankaTableEnvironment \
..      franka_table \
..      --object tomato_soup_can

.. So the flag ``external_environment_class_path`` is used to specify the (fully qualified) path to the
.. external environment module and class. The environment name is then specified as the
.. first non flag argument to the policy runner, and any additional arguments are passed to the
.. environment's ``add_cli_args()`` method.

.. .. note::

..     The environment above is actually located in ``isaaclab_arena_examples/external_environments/basic.py``.
..     So this environment is located in the Isaac Lab Arena source-tree, but isn't included
..     in the built in environments, so must be called through the external environment syntax.
..     This is done to demonstrate how this would be done in an external codebase.

..     The environment can be run with:

..     .. code-block:: bash

..         python isaaclab_arena/evaluation/policy_runner.py \
..           --visualizer kit \
..           --policy_type zero_action \
..           --num_steps 50 \
..           --external_environment_class_path isaaclab_arena_examples.external_environments.basic:ExternalFrankaTableEnvironment \
..           franka_table \
..           --object tomato_soup_can

..     which results in an environment like the one below:

..     .. image:: ../../images/externally_defined_environment.png
..        :width: 50%
..        :alt: External environment example
..        :align: center
