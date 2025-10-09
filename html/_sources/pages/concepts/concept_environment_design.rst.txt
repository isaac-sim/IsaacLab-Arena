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

   class ArenaEnvBuilder:
       """Compose Isaac Arena → Isaac Lab configs."""

       def compose_manager_cfg(self) -> IsaacArenaManagerBasedRLEnvCfg:
           # Combine configurations from all components
           scene_cfg = combine_configclass_instances(...)
           observation_cfg = self.arena_env.embodiment.get_observation_cfg()
           actions_cfg = self.arena_env.embodiment.get_action_cfg()

Each component contributes specific aspects to the final environment configuration, with automatic integration handled by the environment builder system.

Environments in Detail
----------------------

**Component Integration**
   Four primary components combine to create complete environments:

   - **Embodiment**: Robot configuration, action/observation spaces, specialized behaviors
   - **Scene**: Physical asset layout, spatial relationships, physics properties
   - **Task**: Objectives, success criteria, termination conditions, performance metrics
   - **Teleop Device**: Human input interfaces for demonstration and control (optional)

**Configuration Composition**
   Systematic combination of component contributions:

   - **Scene Configuration**: Physical elements from scene, embodiment, and task components
   - **Observation Configuration**: Sensor data and state information from embodiment
   - **Action Configuration**: Control interfaces defined by embodiment
   - **Event Configuration**: Resets, randomization from all components
   - **Termination Configuration**: Success/failure conditions from tasks
   - **Metrics Configuration**: Performance evaluation from tasks

**Environment Builder System**
   ``ArenaEnvBuilder`` orchestrates composition into Isaac Lab configurations:

   - **Configuration Merging**: Automatic combination with conflict resolution
   - **Manager Assembly**: Creates Isaac Lab observation, action, event, termination managers

Environment Integration
-----------------------

.. code-block:: python

   # Component creation
   embodiment = asset_registry.get_asset_by_name("franka")()
   background = asset_registry.get_asset_by_name("kitchen")()
   pick_object = asset_registry.get_asset_by_name("cracker_box")()
   pick_object.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1)))

   # Environment composition
   scene = Scene(assets=[background, pick_object])
   task = PickAndPlaceTask(pick_object, destination, background)

   environment = IsaacArenaEnvironment(
       name="manipulation_task",
       embodiment=embodiment,
       scene=scene,
       task=task,
       teleop_device=None
   )

   # Build and execute
   env_builder = ArenaEnvBuilder(environment, args)
   env = env_builder.make_registered()

Usage Examples
--------------

**Kitchen Manipulation**

.. code-block:: python

   franka = asset_registry.get_asset_by_name("franka")()
   kitchen = asset_registry.get_asset_by_name("kitchen")()
   cracker_box = asset_registry.get_asset_by_name("cracker_box")()

   scene = Scene(assets=[kitchen, cracker_box])
   task = PickAndPlaceTask(cracker_box, destination, kitchen)

   environment = IsaacArenaEnvironment(
       name="kitchen_manipulation",
       embodiment=franka,
       scene=scene,
       task=task
   )

**Affordance Interaction**

.. code-block:: python

   g1 = asset_registry.get_asset_by_name("g1_wbc_joint")()
   microwave = asset_registry.get_asset_by_name("microwave")()

   task = OpenDoorTask(microwave, openness_threshold=0.8)
   environment = IsaacArenaEnvironment(embodiment=g1, task=task, scene=scene)

**Command Line Usage**

.. code-block:: bash

   # Basic manipulation
   python policy_runner.py --policy_type zero_action kitchen_pick_and_place --embodiment franka

   # Teleop demonstration
   python policy_runner.py --teleop_device spacemouse kitchen_pick_and_place --object mustard_bottle

The environment design provides a powerful framework for creating complex robot simulation scenarios through modular component composition, enabling rapid development while maintaining consistency and reusability across different use cases.
