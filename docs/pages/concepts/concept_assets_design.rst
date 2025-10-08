Assets Design
=============

Assets in Isaac Arena are the fundamental building blocks of simulation environments, representing all physical and logical entities that can be spawned and interacted with in the simulation. The asset system provides a unified framework for managing robots, objects, backgrounds, and other scene elements through a hierarchical architecture with standardized interfaces.

Core Architecture
-----------------

The asset system is built around a hierarchical class structure that provides increasing levels of specialization:

.. code-block:: python

   class Asset:
       """Base class for all assets."""

       def __init__(self, name: str, tags: list[str] | None = None, **kwargs):
           self._name = name
           self.tags = tags

       @property
       def name(self) -> str:
           return self._name

The base ``Asset`` class provides essential properties (name and tags) and supports cooperative multiple inheritance, enabling complex compositions through mixins like affordances.

Asset Hierarchy
---------------

**Asset (Base Class)**
   Provides fundamental properties for identification and categorization:

   - **Name**: Unique identifier for the asset
   - **Tags**: Categorization labels for discovery and filtering
   - **Cooperative Inheritance**: Supports mixin patterns for extensibility

**ObjectBase (Physical Assets)**
   Extends Asset for physical objects that can be spawned in simulation:

   - **Object Types**: BASE, RIGID, ARTICULATION
   - **Pose Management**: Initial positioning and orientation
   - **Configuration Generation**: Isaac Lab compatible configurations
   - **Physics Integration**: Contact sensors and collision detection

**Specialized Asset Classes**
   Concrete implementations for specific use cases:

   - **Object**: General purpose spawnable objects (boxes, tools, etc.)
   - **Background**: Scene environments and static geometry
   - **ObjectReference**: References to objects within larger scene assets
   - **EmbodimentBase**: Robot and agent assets with control interfaces

Asset Types and Physics
-----------------------

Assets support different physics representations based on their intended behavior:

**BASE Assets**
   Static, non-physics objects used for visual elements and references.

   - No physics simulation
   - Used for backgrounds, markers, and visual elements
   - Lightweight and performant

**RIGID Assets**
   Solid objects with rigid body physics simulation.

   - Single rigid body with contact dynamics
   - Suitable for boxes, tools, furniture
   - Supports contact sensors and collision detection

**ARTICULATION Assets**
   Multi-body objects with joints and articulated physics.

   - Complex mechanical structures with joints
   - Supports actuators and joint control
   - Used for robots, doors, drawers, appliances

Asset Registration System
-------------------------

Assets are automatically discovered and registered through a decorator-based system:

.. code-block:: python

   @register_asset
   class CrackerBox(LibraryObject):
       """A cracker box object."""

       name = "cracker_box"
       tags = ["object"]
       usd_path = "omniverse://isaac-dev.ov.nvidia.com/.../003_cracker_box.usd"

The registration system provides several discovery mechanisms:

**By Name**
   Direct lookup of specific assets by their unique name.

**By Tags**
   Find all assets matching specific categories (e.g., "object", "background", "embodiment").

**Random Selection**
   Select random assets from a category for procedural generation.

Configuration Generation
------------------------

Assets automatically generate Isaac Lab compatible configurations based on their type:

.. code-block:: python

   def get_cfgs(self) -> dict[str, Any]:
       if self.object_type == ObjectType.RIGID:
           object_cfg = self._generate_rigid_cfg()
       elif self.object_type == ObjectType.ARTICULATION:
           object_cfg = self._generate_articulation_cfg()
       elif self.object_type == ObjectType.BASE:
           object_cfg = self._generate_base_cfg()

       return {self.name: object_cfg}

This abstraction allows assets to be used consistently across different environments while handling the complexity of Isaac Lab configuration generation internally.

Asset Libraries
---------------

Isaac Arena provides pre-built asset libraries for common objects and environments:

**Object Library**
   Ready-to-use objects for manipulation tasks:

   - **YCB Objects**: Cracker box, mustard bottle, sugar box, tomato soup can
   - **Custom Objects**: Power drill, kettles, pots, spray cans
   - **Interactive Objects**: Microwave, toaster (with affordances)
   - **Utility Objects**: Tables, sorting bins, exhaust pipes

**Background Library**
   Complete scene environments for different contexts:

   - **Kitchen**: Residential kitchen with appliances and furniture
   - **Galileo**: Laboratory/industrial environment
   - **Packing Table**: Simple table surface for manipulation tasks

**Embodiment Assets**
   Robot and agent assets with specialized control interfaces (covered in Embodiment Design documentation).

Asset Integration with Affordances
----------------------------------

Assets can be enhanced with interaction capabilities through affordance mixins:

.. code-block:: python

   @register_asset
   class Microwave(LibraryObject, Openable):
       """A microwave oven with opening capability."""

       name = "microwave"
       tags = ["object", "openable"]
       object_type = ObjectType.ARTICULATION

       # Affordance parameters
       openable_joint_name = "microjoint"
       openable_open_threshold = 0.5

This composition pattern allows assets to inherit both physical properties and interaction behaviors seamlessly.

Scene Composition
-----------------

Assets are composed into scenes through the Scene class, which manages collections of related assets:

.. code-block:: python

   # Create individual assets
   background = asset_registry.get_asset_by_name("kitchen")()
   microwave = asset_registry.get_asset_by_name("microwave")()
   cracker_box = asset_registry.get_asset_by_name("cracker_box")()

   # Set initial poses
   microwave.set_initial_pose(Pose(position_xyz=(0.8, 0.0, 0.23)))
   cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1)))

   # Compose into scene
   scene = Scene(assets=[background, microwave, cracker_box])

Environment Integration
----------------------

Assets integrate into complete environments through composition with embodiments and tasks:

.. code-block:: python

   isaac_arena_environment = IsaacArenaEnvironment(
       name="microwave_opening",
       embodiment=G1WBCJointEmbodiment(),
       scene=scene,  # Contains our assets
       task=OpenDoorTask(microwave, openness_threshold=0.8),
       teleop_device=None,
   )

The environment builder automatically combines asset configurations from all components to create the final Isaac Lab environment configuration.

Object References
-----------------

For objects that exist within larger scene assets, ObjectReference provides access without separate spawning:

.. code-block:: python

   # Reference an object within the kitchen scene
   destination_drawer = ObjectReference(
       name="kitchen_drawer",
       prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
       parent_asset=kitchen_background,
       object_type=ObjectType.RIGID,
   )

This enables interaction with scene elements that are part of larger USD files without requiring separate asset definitions.

Usage Examples
--------------

**Asset Discovery and Creation**:

.. code-block:: python

   # Get asset registry
   asset_registry = AssetRegistry()

   # Create assets by name
   kitchen = asset_registry.get_asset_by_name("kitchen")()
   franka = asset_registry.get_asset_by_name("franka")()

   # Find assets by category
   objects = asset_registry.get_assets_by_tag("object")
   random_object = asset_registry.get_random_asset_by_tag("object")()

**Environment Construction**:

.. code-block:: bash

   # Use specific objects in examples
   python isaac_arena/examples/policy_runner.py --object cracker_box --embodiment franka

   # Different background scenes
   python isaac_arena/examples/policy_runner.py kitchen_pick_and_place --object mustard_bottle
   python isaac_arena/examples/policy_runner.py galileo_pick_and_place --object sugar_box

Creating New Assets
-------------------

To add new assets to Isaac Arena:

1. **Define Asset Class**
   Inherit from appropriate base class (Object, Background, etc.)

2. **Set Core Properties**
   Specify name, tags, USD path, and object type

3. **Configure Physics**
   Implement configuration generation methods for desired physics type

4. **Add Affordances** *(Optional)*
   Mix in affordance classes for interaction capabilities

5. **Register Asset**
   Use ``@register_asset`` decorator for automatic discovery

6. **Test Integration**
   Verify asset works in scenes and environments

.. code-block:: python

   @register_asset
   class CustomTool(LibraryObject):
       """A custom tool object."""

       name = "custom_tool"
       tags = ["object", "tool"]
       usd_path = "path/to/custom_tool.usd"
       object_type = ObjectType.RIGID
       scale = (1.0, 1.0, 1.0)

Physical Properties and Pose Management
---------------------------------------

Assets provide comprehensive pose and physics management:

**Initial Pose Configuration**
   Set starting position and orientation for consistent spawning.

**Runtime Pose Queries**
   Get current pose information during simulation for task logic.

**Physics Configuration**
   Automatic generation of Isaac Lab physics configurations with appropriate contact sensors and collision properties.

**Scaling Support**
   Uniform and non-uniform scaling for procedural generation and size variations.

The asset system provides a powerful foundation for building complex simulated environments, enabling rapid composition of scenes from reusable, well-defined components while maintaining consistency and extensibility across different use cases.
