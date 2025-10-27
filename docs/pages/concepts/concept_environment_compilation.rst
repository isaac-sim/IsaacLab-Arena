Environment Compilation Design
================================

Environment compilation transforms modular Isaac Arena Environment components into complete Isaac Lab Arena environment configurations.
The system handles configuration merging, environment registration, and integration with Isaac Lab's architecture.

Core Architecture
-----------------

Compilation uses the ``ArenaEnvBuilder`` class:

.. code-block:: python

   class ArenaEnvBuilder:
       """Compose Isaac Arena → Isaac Lab configs"""

       def __init__(self, arena_env: IsaacArenaEnvironment, args: argparse.Namespace):
           self.arena_env = arena_env
           self.args = args

       def compose_manager_cfg(self) -> IsaacArenaManagerBasedRLEnvCfg:
           """Combine configurations from all components."""
           scene_cfg = combine_configclass_instances(
               "SceneCfg",
               self.DEFAULT_SCENE_CFG,
               self.arena_env.scene.get_scene_cfg(),
               self.arena_env.embodiment.get_scene_cfg(),
               self.arena_env.task.get_scene_cfg()
           )

The builder transforms Isaac Arena environment definitions into Isaac Lab's configuration format through systematic component integration.

Compilation in Detail
---------------------

**Configuration Merging**
   Systematic combination of component configurations:

   - **Scene Configuration**: Merges default settings, scene assets, embodiment physics, and task-specific elements
   - **Observation Configuration**: Extracts sensor data and state information from embodiment
   - **Action Configuration**: Defines control interfaces from embodiment specifications
   - **Event Configuration**: Combines reset and randomization logic from embodiment, scene, and task
   - **Termination Configuration**: Merges success/failure conditions from task and scene components
   - **Metrics Configuration**: Automatic recorder manager setup for performance evaluation
   - **XR Configuration**: XR device locations for teleop integration (optional)
   - **Teleop Device Configuration**: Teleop device configuration from embodiment
   - **Recorder Manager Configuration**: Recorder manager configuration for performance evaluation

**Environment Modes**
   Support for different Isaac Lab environment types:

   - **Standard Mode**: Full environment with observations, actions, events, terminations, and metrics
   - **Mimic Mode**: Mimic environment with subtask definitions

Environment Integration
-----------------------

.. code-block:: python

   # Create Isaac Arena environment definition
   environment = IsaacArenaEnvironment(
       name="kitchen_manipulation",
       embodiment=franka_embodiment,
       scene=kitchen_scene,
       task=pick_and_place_task,
       teleop_device=keyboard_device
   )

   # Compile to Isaac Lab environment
   env_builder = ArenaEnvBuilder(environment, args)

   # Register and create executable environment
   env = env_builder.make_registered()

   # Alternative: get both environment and configuration
   env, cfg = env_builder.make_registered_and_return_cfg()

Usage Examples
--------------

**Standard Environment Compilation**

.. code-block:: python

   # Build standard RL environment
   args.mimic = False
   env_builder = ArenaEnvBuilder(arena_environment, args)
   env = env_builder.make_registered()

   # Environment ready for training/evaluation
   obs, _ = env.reset()
   actions = policy.get_action(env, obs)
   obs, rewards, terminated, truncated, info = env.step(actions)

**Mimic Environment Compilation**

.. code-block:: python

   # Build demonstration generation environment
   args.mimic = True
   env_builder = ArenaEnvBuilder(arena_environment, args)
   env = env_builder.make_registered()

   # Environment configured for mimic data generation
   mimic_env.generate_demonstrations()

**Configuration Inspection**

.. code-block:: python

   # Examine compiled configuration before registration
   env_builder = ArenaEnvBuilder(arena_environment, args)
   cfg = env_builder.compose_manager_cfg()

   print(f"Scene objects: {list(cfg.scene.keys())}")
   print(f"Action space: {cfg.actions}")
   print(f"Observation space: {cfg.observations}")

**Runtime Parameter Override**

.. code-block:: python

   # Apply runtime configuration changes
   name, cfg = env_builder.build_registered()
   cfg = parse_env_cfg(
       name,
       device="cuda:0",
       num_envs=1024,
       use_fabric=True
   )
   env = gym.make(name, cfg=cfg).unwrapped
