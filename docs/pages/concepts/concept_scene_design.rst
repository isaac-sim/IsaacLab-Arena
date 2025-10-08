Scene Design
============

Scenes in Isaac Arena manage collections of assets that define the physical environment for simulation. They provide a unified interface for composing backgrounds, objects, and interactive elements into complete environments that integrate seamlessly with Isaac Lab's scene system.

Core Architecture
-----------------

The scene system is built around the ``Scene`` class that manages asset collections and configuration generation:

.. code-block:: python

   class Scene:
       def __init__(self, assets: list[Asset] | None = None):
           self.assets: dict[str, Asset] = {}
           self.observation_cfg = None
           self.events_cfg = None
           self.termination_cfg = None

       def add_asset(self, asset: Asset):
           """Add single asset to scene."""
           self.assets[asset.name] = asset

       def get_scene_cfg(self) -> Any:
           """Generate Isaac Lab scene configuration from assets."""

Scenes automatically aggregate asset configurations into Isaac Lab-compatible scene configurations while maintaining asset relationships and poses.

Asset Composition
-----------------

Scenes compose different asset types into cohesive environments:

**Background Assets**
   Environmental foundations like kitchens, laboratories, or outdoor spaces that provide base geometry and context.

**Interactive Objects**
   Manipulable items with physics properties - YCB objects, tools, appliances - that can be picked, moved, or activated.

**Functional Elements**
   Specialized objects with affordances like doors, drawers, buttons that support specific interaction behaviors.

**Object References**
   Access to objects embedded within larger scene assets without separate spawning.

Scene Construction Patterns
---------------------------

Common patterns for scene creation:

**Asset Selection and Positioning**

.. code-block:: python

   # Get assets from registry
   background = asset_registry.get_asset_by_name("kitchen")()
   objects = [asset_registry.get_asset_by_name("cracker_box")()]
   microwave = asset_registry.get_asset_by_name("microwave")()

   # Set initial poses
   objects[0].set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1)))
   microwave.set_initial_pose(Pose(position_xyz=(0.8, 0.0, 0.23)))

**Scene Assembly**

.. code-block:: python

   # Compose scene from assets
   scene = Scene(assets=[background] + objects + [microwave])

   # Alternative: sequential addition
   scene = Scene()
   scene.add_asset(background)
   scene.add_assets(objects)

Configuration Integration
-------------------------

Scenes integrate with the broader environment system:

**Environment Composition**

.. code-block:: python

   isaac_arena_environment = IsaacArenaEnvironment(
       name="manipulation_task",
       embodiment=embodiment,
       scene=scene,           # Contains all physical assets
       task=task,
       teleop_device=teleop_device
   )

**Automatic Configuration**
   Scene configurations are automatically merged with embodiment and task configurations during environment building.


Usage Examples
--------------

**Pick and Place Scene**

.. code-block:: python

   # Kitchen manipulation environment
   background = asset_registry.get_asset_by_name("kitchen")()
   pick_object = asset_registry.get_asset_by_name("mustard_bottle")()
   pick_object.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1)))

   # Reference destination within kitchen
   destination = ObjectReference(
       name="kitchen_drawer",
       prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
       parent_asset=background
   )

   scene = Scene(assets=[background, pick_object])

**Affordance Interaction Scene**

.. code-block:: python

   # Microwave opening task
   packing_table = asset_registry.get_asset_by_name("packing_table")()
   microwave = asset_registry.get_asset_by_name("microwave")()
   microwave.set_initial_pose(Pose(position_xyz=(0.8, 0.0, 0.23)))

   scene = Scene(assets=[packing_table, microwave])

Scene Configuration Management
------------------------------

Scenes support optional configuration overrides:

**Custom Configurations**
   Override observation, event, or termination configurations for scene-specific behaviors.

**Dynamic Asset Management**
   Add or modify assets at runtime for procedural scene generation.

**Configuration Aggregation**
   Automatic merging of asset configurations ensures consistent Isaac Lab integration without manual configuration management.

The scene system in Isaac Arena provides efficient composition of simulation environments while maintaining clean separation between physical layout and task logic.
