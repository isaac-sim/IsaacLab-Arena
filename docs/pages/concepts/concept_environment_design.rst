Environment Design
==================

IsaacLab Arena environment contains the main components used to create a manager-based RL environment.
This includes the embodiment, scene, task and teleop device.

Core Architecture
-----------------

.. code-block:: python

   @configclass
   class IsaacArenaEnvironment:
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

Each component contributes to the final environment configuration, with automatic integration handled by the environment builder.

Creating an Environment Example
--------------------------------

.. code-block:: python

   # Component creation
   embodiment = asset_registry.get_asset_by_name("franka")()
   background = asset_registry.get_asset_by_name("kitchen")()
   pick_object = asset_registry.get_asset_by_name("cracker_box")()
   pick_object.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1)))

   # Environment composition
   scene = Scene(assets=[background, pick_object])
   task = PickAndPlaceTask(pick_object, destination, background)

   arena_environment = IsaacArenaEnvironment(
       name="manipulation_task",
       embodiment=embodiment,
       scene=scene,
       task=task,
       teleop_device=None
   )

   # Build and execute
   env_builder = ArenaEnvBuilder(arena_environment, args)
   env = env_builder.make_registered()

To see how the manager-based RL environment configuration is compiled, please refer to the :doc:`./concept_environment_compilation` page.
