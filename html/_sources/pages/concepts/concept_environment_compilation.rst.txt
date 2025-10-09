Environment Compilation Design
================================

Environment compilation in Isaac Arena transforms modular Isaac Arena components into complete Isaac Lab environment configurations. The compilation system handles configuration merging, environment registration, and seamless integration with Isaac Lab's manager-based architecture while supporting both standard and mimic environment modes.

Core Architecture
-----------------

The compilation system is built around the ``ArenaEnvBuilder`` class that orchestrates the transformation process:

.. code-block:: python

   class ArenaEnvBuilder:
       """Compose Isaac Arena → Isaac Lab configs"""

       DEFAULT_SCENE_CFG = InteractiveSceneCfg(num_envs=4096, env_spacing=30.0, replicate_physics=False)

       def __init__(self, arena_env: IsaacArenaEnvironment, args: argparse.Namespace):
           self.arena_env = arena_env
           self.args = args

       def compose_manager_cfg(self) -> IsaacArenaManagerBasedRLEnvCfg:
           """Combine configurations from all components."""
           # Merge scene configurations from multiple sources
           scene_cfg = combine_configclass_instances(
               "SceneCfg",
               self.DEFAULT_SCENE_CFG,
               self.arena_env.scene.get_scene_cfg(),
               self.arena_env.embodiment.get_scene_cfg(),
               self.arena_env.task.get_scene_cfg()
           )

The builder takes an Isaac Arena environment definition and transforms it into Isaac Lab's configuration format through systematic component integration.

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

**Environment Modes**
   Support for different Isaac Lab environment types:

   - **Standard Mode**: Full RL environment with observations, actions, events, terminations, and metrics
   - **Mimic Mode**: Demonstration generation environment with subtask decomposition and constraint handling
   - **Teleop Integration**: Optional human input device configuration for demonstration collection
   - **XR Support**: Extended reality camera configurations for immersive teleoperation

**Registration Process**
   Automatic Gymnasium environment registration and configuration parsing:

   - **Dynamic Registration**: Creates unique environment IDs and entry points
   - **Configuration Parsing**: Applies runtime parameters (device, num_envs, fabric settings)
   - **Entry Point Selection**: Chooses appropriate Isaac Lab environment class based on mode
   - **Environment Creation**: Instantiates configured environments ready for execution

**Configuration Validation**
   Automatic conflict resolution and compatibility checking:

   - **Component Compatibility**: Ensures scene, embodiment, and task configurations are compatible
   - **Isaac Lab Integration**: Validates configurations meet Isaac Lab manager requirements
   - **Default Handling**: Provides sensible defaults for optional configuration elements

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

The compilation system provides seamless transformation from Isaac Arena's modular component architecture to Isaac Lab's manager-based environment system, enabling rapid deployment of complex simulation scenarios while maintaining full compatibility with existing Isaac Lab workflows.
