Environment Design
==================

Environments in Isaac Arena are the top-level abstraction that brings together all components needed for simulation and learning. They provide a unified interface for combining embodiments, scenes, tasks, and teleoperation devices into complete, executable simulation experiences that integrate seamlessly with Isaac Lab's manager-based environment system.

Core Architecture
-----------------

Isaac Arena environments follow a compositional design pattern built around four primary components:

.. code-block:: python

   @configclass
   class IsaacArenaEnvironment:
       """Describes an environment in Isaac Arena."""

       name: str = MISSING
       embodiment: EmbodimentBase = MISSING
       scene: Scene = MISSING
       task: TaskBase = MISSING
       teleop_device: TeleopDeviceBase | None = None

Each component contributes specific aspects to the final environment configuration, enabling modular composition and reuse across different scenarios.

Environment Components
----------------------

**Embodiment**
   Defines the robot or agent and its capabilities:

   - **Robot Configuration**: Physical properties, actuators, sensors
   - **Action Space**: Control interface (joint positions, end-effector poses, etc.)
   - **Observation Space**: Sensor data and state information
   - **Event Space**: Resets, randomization, interventions
   - **Mimic Configuration**: Mimic environment configurations
   - **XR Configuration**: XR camera device locations
   - **Specialized Behaviors**: WBC, IK controllers, camera integration

**Scene**
   Manages the collection of assets in the environment:

   - **Asset Composition**: Background environments, objects, props
   - **Spatial Layout**: Asset positioning and relationships
   - **Physics Properties**: Contact detection, collision handling
   - **Visual Elements**: Lighting, materials, visual effects
   - **Event Space**: Resets, randomization, interventions
   - **Termination Conditions**: Success and failure criteria

**Task**
   Defines the objective and success criteria:

   - **Termination Conditions**: Success and failure criteria
   - **Scene Configuration**: Scene elements and properties
   - **Event Management**: Resets, randomization, interventions
   - **Metrics**: Performance measurement and evaluation
   - **Mimic Configuration**: Mimic environment configurations

**Teleop Device** *(Optional)*
   Enables human operator control and interaction:

   - **Input Devices**: VR controllers, spacemouse, keyboards
   - **Control Mapping**: Device inputs to robot actions
   - **Retargeters**: Specifies the retargeters to use for the teleop device

Environment Builder System
---------------------------

The ``ArenaEnvBuilder`` class orchestrates the composition of Isaac Arena environments into Isaac Lab configurations:

.. code-block:: python

   class ArenaEnvBuilder:
       """Compose Isaac Arena → IsaacLab configs"""

       def compose_manager_cfg(self) -> IsaacArenaManagerBasedRLEnvCfg:
           # Combine scene configurations
           scene_cfg = combine_configclass_instances(
               "SceneCfg",
               self.DEFAULT_SCENE_CFG,
               self.arena_env.scene.get_scene_cfg(),
               self.arena_env.embodiment.get_scene_cfg(),
               self.arena_env.task.get_scene_cfg(),
           )

           # Extract other configurations
           observation_cfg = self.arena_env.embodiment.get_observation_cfg()
           actions_cfg = self.arena_env.embodiment.get_action_cfg()
           events_cfg = combine_configclass_instances(...)
           termination_cfg = combine_configclass_instances(...)

The builder automatically handles configuration merging, ensuring compatibility and proper integration between components.

Configuration Composition
--------------------------

Environment configurations are built through systematic combination of component contributions:

**Scene Configuration**
   Combines physical scene elements from all components:

   - Base scene settings (environment count, spacing, physics)
   - Scene assets (backgrounds, objects, props)
   - Robot configuration (embodiment contribution)
   - Task-specific elements

**Observation Configuration**
   Combines observation configurations from all components:

   - Robot observation (embodiment contribution like joint states)

**Event Configuration**
   Combines event configurations from all components:

   - Robot event (embodiment contribution like robot resets)
   - Scene event (scene contribution like randomization)
   - Task-specific elements

**Termination Configuration**
   Combines termination configurations from all components:

   - Task-specific elements (task contribution like success/failure)
   - Scene termination (scene contribution like objects falling out of bounds)

**Action Configuration**

   - Robot action (embodiment contribution like joint positions)

**XR Configuration**

   - Robot XR (embodiment contribution like XR camera device locations)

**Teleop Device Configuration**

   - Teleop device contribution

**Metrics Configuration**

   - Task-specific elements (task contribution like metrics)

**Mimic Configuration**

   - Task-specific elements (task contribution such as subtask definitions)

**Manager Configuration Assembly**
   Assembles Isaac Lab manager configurations:

   - **Observation Manager**
   - **Action Manager**
   - **Event Manager**
   - **Scene Manager**
   - **Termination Manager**
   - **XR Manager**
   - **Mimic Manager**
   - **Teleop Manager**
   - **Recorder Manager**

Environment Types
-----------------

Isaac Arena supports different environment configurations for various use cases:

**Standard Environments**
   For data collection, training and evaluation:

   - Full manager-based environment with observations, actions, events, terminations
   - Metrics collection and performance monitoring
   - Support for parallel environment execution

   and we add the following for mimic environments:

   - Mimic environment configuration
        - Subtask decomposition for complex behaviors
        - Integration with embodiment and task specific mimic classes


Example Environments
--------------------

Isaac Arena includes several pre-configured example environments demonstrating different use cases:

**Kitchen Pick and Place**
   Manipulation task in a realistic kitchen environment:

   - **Scene**: Kitchen background with appliances and furniture
   - **Task**: Pick up objects and place them in drawers
   - **Embodiments**: Franka arm, G1 humanoid, GR1T2 humanoid
   - **Objects**: Various YCB and custom objects

**Microwave Opening**
   Affordance-based interaction task:

   - **Scene**: Packing table with microwave appliance
   - **Task**: Open microwave door using affordance system
   - **Embodiments**: G1 humanoid with WBC control
   - **Interaction**: Openable affordance with threshold-based success

**Laboratory Manipulation**
   Industrial environment with precision tasks:

   - **Scene**: Galileo laboratory environment
   - **Task**: Complex manipulation sequences
   - **Embodiments**: Multiple robot configurations
   - **Evaluation**: Comprehensive metrics and success rates

Environment Creation Workflow
-----------------------------

Creating new environments follows a structured process:

1. **Define Components**
   Create or select embodiment, scene assets, and task definition.

2. **Compose Environment**
   Instantiate ``IsaacArenaEnvironment`` with chosen components.

3. **Build Configuration**
   Use ``ArenaEnvBuilder`` to generate Isaac Lab configuration.

4. **Register and Execute**
   Register with Gymnasium and create executable environment.

.. code-block:: python

   # Create components
   embodiment = asset_registry.get_asset_by_name("franka")()
   background = asset_registry.get_asset_by_name("kitchen")()
   objects = [asset_registry.get_asset_by_name("cracker_box")()]

   # Set poses
   objects[0].set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1)))

   # Compose environment
   scene = Scene(assets=[background] + objects)
   task = PickAndPlaceTask(objects[0], destination, background)

   environment = IsaacArenaEnvironment(
       name="custom_task",
       embodiment=embodiment,
       scene=scene,
       task=task
   )

   # Build and execute
   env_builder = ArenaEnvBuilder(environment, args)
   env = env_builder.make_registered()

Simulation Parameters
---------------------

Isaac Arena environments use optimized simulation parameters for robot learning:

**Timing Configuration**
   - **Physics Timestep**: 1/200 Hz (5ms) for stable simulation
   - **Decimation**: 4 steps per environment step
   - **Episode Length**: 30 seconds default duration
   - **Render Interval**: 2 steps for smooth visualization

**Parallelization Settings**
   - **Environment Count**: Up to 4096 parallel environments
   - **Environment Spacing**: 30m separation for collision avoidance
   - **Physics Replication**: Configurable for memory vs. performance tradeoffs

**Performance Optimizations**
   - Texture loading optimization for faster startup
   - Fabric acceleration for tensor operations
   - GPU-accelerated physics when available

Usage Examples
--------------

**Command Line Usage**:

.. code-block:: bash

   # Kitchen manipulation with Franka arm
   python isaac_arena/examples/policy_runner.py --policy_type zero_action kitchen_pick_and_place --object cracker_box --embodiment franka

   # Humanoid microwave opening
   python isaac_arena/examples/policy_runner.py --policy_type zero_action gr1_open_microwave --embodiment g1_wbc_joint

   # Laboratory task with teleoperation
   python isaac_arena/examples/policy_runner.py --teleop_device spacemouse galileo_pick_and_place --object mustard_bottle

**Programmatic Usage**:

.. code-block:: python

   from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
   from isaac_arena.environments.compile_env import ArenaEnvBuilder
   from isaac_arena.scene.scene import Scene

   # Create environment
   environment = IsaacArenaEnvironment(
       name="experiment",
       embodiment=embodiment,
       scene=scene,
       task=task,
       teleop_device=None
   )

   # Build and run
   builder = ArenaEnvBuilder(environment, args)
   env = builder.make_registered()

   obs, _ = env.reset()
   for _ in range(1000):
       actions = policy(obs)
       obs, _, _, _ = env.step(actions)

Integration with Isaac Lab
--------------------------

Isaac Arena environments integrate seamlessly with Isaac Lab's ecosystem:

**Manager-Based Architecture**
   Leverages Isaac Lab's manager system for observation, action, event, and termination handling.

**Configuration System**
   Uses Isaac Lab's configuration classes and automatic composition mechanisms.

**Parallel Execution**
   Supports Isaac Lab's efficient parallel environment execution with thousands of environments.

**Extension Compatibility**
   Works with Isaac Lab extensions for sensors, assets, and specialized functionality.


The environment design in Isaac Arena provides a powerful and flexible framework for creating complex robot simulation scenarios. By composing modular components through a standardized interface, it enables rapid development of new environments while maintaining consistency and reusability across different use cases.
