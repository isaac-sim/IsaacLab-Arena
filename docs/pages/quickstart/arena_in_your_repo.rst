Using IsaacLab-Arena in Your Own Repository
============================================

The recommended way to consume IsaacLab-Arena from an external project is to include it as an
**unmodified git submodule** and extend it purely through its registration API — without editing
any file inside the Arena source tree.

This is the currently recommended integration pattern until IsaacLab-Arena is available as a
published pip package, at which point the submodule will be replaced by a simple
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
   │   └── scripts/
   │       └── run_datagen.py       ← entry script
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

   COPY submodules/IsaacLab-Arena /opt/arena
   RUN /isaac-sim/python.sh -m pip install -e /opt/arena

   # Install your package after Arena is in place
   COPY my_package /workspace/my_package
   RUN /isaac-sim/python.sh -m pip install -e /workspace/my_package

See Arena's own `Dockerfile
<https://github.com/isaac-sim/IsaacLab-Arena/blob/main/docker/Dockerfile.isaaclab_arena>`_
for a complete reference, including Isaac Lab installation and optional GR00T dependencies.


Defining a Custom Environment
------------------------------

Subclass ``ExampleEnvironmentBase``, set a unique ``name``, and implement ``get_env()``:

.. code-block:: python

   # my_package/isaaclab_arena_environments/my_environment.py

   import argparse
   from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase
   from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
   from isaaclab_arena.scene.scene import Scene
   from isaaclab_arena.tasks.no_task import NoTask

   class MyEnvironment(ExampleEnvironmentBase):

       name: str = "my_environment"

       def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
           background = self.asset_registry.get_asset_by_name(args_cli.background)()
           embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
               enable_cameras=args_cli.enable_cameras
           )
           scene = Scene(assets=[background])
           return IsaacLabArenaEnvironment(
               name=self.name,
               embodiment=embodiment,
               scene=scene,
               task=NoTask(),
           )

       @staticmethod
       def add_cli_args(parser: argparse.ArgumentParser) -> None:
           parser.add_argument("--background", type=str, default="kitchen")
           parser.add_argument("--embodiment", type=str, default="droid_abs_joint_pos")


Wiring into the Arena CLI
--------------------------

In your entry script, inject the class into ``ExampleEnvironments`` before parsing arguments:

.. code-block:: python

   # my_package/scripts/run_datagen.py

   from isaaclab_arena_environments.cli import (
       ExampleEnvironments,
       get_arena_builder_from_cli,
       get_isaaclab_arena_environments_cli_parser,
   )
   from my_package.isaaclab_arena_environments.my_environment import MyEnvironment

   # Register the custom environment
   ExampleEnvironments[MyEnvironment.name] = MyEnvironment

   # Parse args and build
   parser = get_isaaclab_arena_environments_cli_parser()
   args_cli = parser.parse_args()
   arena_builder = get_arena_builder_from_cli(args_cli)
   env = arena_builder.make_registered()
   env.reset()

Run it with:

.. code-block:: bash

   python my_package/scripts/run_datagen.py my_environment --background kitchen
