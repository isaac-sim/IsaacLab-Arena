First Arena Environment
=======================

After setting up the docker container and installing ``isaaclab_arena``, learn how
to compose your first simple IsaacLab Arena environment by combining assets, scenes, and tasks.

Once within the docker container, run the following command to compile your first IsaacLab Arena environment:

.. code-block:: bash

    python isaaclab_arena/examples/compile_env_notebook.py


The compiled environment will spawn in an Isaac Sim instance and run for some steps with zero actions.
You should see the following scene:

.. image:: ../../images/franka_kitchen.png
   :align: center
   :width: 100%


Code Explanation
----------------

The following script demonstrates how to create the simple kitchen environment from
above with a Franka robot and a cracker box object using the ``isaaclab_arena`` API.


.. dropdown:: Create a Simple Environment
   :animate: fade-in

   .. code-block:: python

       import torch
       import tqdm

       import pinocchio  # noqa: F401
       from isaaclab.app import AppLauncher

       # Launch the Isaac Sim application
       print("Launching simulation app")
       simulation_app = AppLauncher()

       from isaaclab_arena.assets.asset_registry import AssetRegistry
       from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
       from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
       from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
       from isaaclab_arena.scene.scene import Scene
       from isaaclab_arena.tasks.dummy_task import DummyTask
       from isaaclab_arena.utils.pose import Pose

       # Step 1: Initialize and get the assets from the registry
       asset_registry = AssetRegistry()

       background = asset_registry.get_asset_by_name("kitchen")()
       embodiment = asset_registry.get_asset_by_name("franka")()
       cracker_box = asset_registry.get_asset_by_name("cracker_box")()
       cracker_box.set_initial_pose(
           Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
       )

       # Step 2: Create a scene with the assets
       scene = Scene(assets=[background, cracker_box])

       # Step 3: Create a task
       task = DummyTask()

       # Step 4: Create the IsaacLab Arena environment
       isaaclab_arena_environment = IsaacLabArenaEnvironment(
           name="my_first_arena_env",
           embodiment=embodiment,
           scene=scene,
           task=task,
           teleop_device=None,
       )

       # Step 5: Build and compile the environment
       args_cli = get_isaaclab_arena_cli_parser().parse_args([])
       env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
       env = env_builder.make_registered()
       env.reset()

       # Step 6: Run the simulation with zero actions
       NUM_STEPS = 1000
       for _ in tqdm.tqdm(range(NUM_STEPS)):
           with torch.inference_mode():
               actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
               env.step(actions)

Step-by-Step Breakdown
-----------------------

**1. Initialize and interac with the Asset Registry**

.. code-block:: python

    asset_registry = AssetRegistry()

The ``AssetRegistry`` provides access to all available assets including robots, objects, and backgrounds. It automatically discovers registered assets through the registration system.

See :doc:`../concepts/concept_assets_design` for details on asset architecture.

.. code-block:: python

    background = asset_registry.get_asset_by_name("kitchen")()
    embodiment = asset_registry.get_asset_by_name("franka")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()

**2. Compose the Scene**

.. code-block:: python

    scene = Scene(assets=[background, cracker_box])

See :doc:`../concepts/concept_scene_design` for scene composition details.

**3. Create a Task**

A task defines the objective, success criteria, and behavior logic for the environment. For this example, we use the ``DummyTask``.

.. code-block:: python

    task = DummyTask()

See :doc:`../concepts/concept_tasks_design` for task creation details.

**4. Create the IsaacLab Arena Environment**

.. code-block:: python

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="my_first_arena_env",
        embodiment=embodiment,
        scene=scene,
        task=DummyTask(),
        teleop_device=None,
    )

This puts everything together into an ``IsaacLabArenaEnvironment`` object.

See :doc:`../concepts/concept_environment_design` for environment composition details.

**5. Build the Environment**

.. code-block:: python

    args_cli = get_isaaclab_arenaena_cli_parser().parse_args([])
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()

The ``ArenaEnvBuilder`` compiles the high-level environment description into Isaac Lab configurations.
See :doc:`../concepts/concept_environment_compilation` for compilation details.

**6. Run the Simulation**

.. code-block:: python

    for _ in range(NUM_STEPS):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

This is just a standard Isaac Lab simulation loop with zero actions.

Next Steps
----------

Now that you have created your first environment, explore:

- :doc:`../concepts/concept_tasks_design` - Create custom tasks with rewards and terminations
- :doc:`../concepts/concept_assets_design` - Discover available assets and create custom ones
- :doc:`../concepts/concept_affordances_design` - Add interactive behaviors to objects

Explore pre-built example environments in ``isaaclab_arena/examples/example_environments/`` for more complex scenarios.

To move on to data generation and training (Imitation Learning), please refer to the :doc:`../example_workflows/locomanipulation/index` or
:doc:`../example_workflows/static_manipulation/index` pages.
