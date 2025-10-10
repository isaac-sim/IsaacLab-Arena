Environment Design
==================

Environments are the top-level abstraction that combines all components for simulation and learning. They unify embodiments, scenes, tasks, and teleoperation devices into complete simulation experiences that integrate with Isaac Lab.

Core Architecture
-----------------

Environments use a compositional design with four primary components:

.. code-block:: python

   @configclass
   class IsaacArenaEnvironment:
       name: str = MISSING
       embodiment: EmbodimentBase = MISSING
       scene: Scene = MISSING
       task: TaskBase = MISSING
       teleop_device: TeleopDeviceBase | None = None

   class ArenaEnvBuilder:
       """Compose IsaacLab Arena → Isaac Lab configs."""
       def compose_manager_cfg(self) -> IsaacArenaManagerBasedRLEnvCfg:
           # Combine configurations from all components
           scene_cfg = combine_configclass_instances(...)
           observation_cfg = self.arena_env.embodiment.get_observation_cfg()
           actions_cfg = self.arena_env.embodiment.get_action_cfg()

Each component contributes to the final environment configuration, with automatic integration handled by the environment builder.

Environments in Detail
----------------------

**Component Integration**
   Four primary components combine to create complete environments:

   - **Embodiment**: Robot configuration, action/observation spaces, specialized behaviors
   - **Scene**: Physical asset layout, spatial relationships, physics properties
   - **Task**: Objectives, success criteria, termination conditions, performance metrics
   - **Teleop Device**: Human input interfaces for demonstration and control (optional)

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
