Assets Design
=============

Assets are the building blocks of simulation environments - robots, objects, backgrounds, and scene elements.
The asset system provides a registry system for managing these components.

Core Architecture
-----------------

The asset system uses a hierarchical class structure:

.. code-block:: python

   class Asset:
       """Base class for all assets."""
       def __init__(self, name: str, tags: list[str] | None = None, **kwargs):
           self._name = name
           self.tags = tags

   class ObjectBase(Asset):
       """Physical objects that can be spawned."""
       def __init__(self, object_type: ObjectType, prim_path: str, **kwargs):
           self.object_type = object_type
           self.prim_path = prim_path

       def get_cfgs(self) -> dict[str, Any]:
           """Generate Isaac Lab configurations."""

The system supports complex compositions through mixins like affordances while automatically generating Isaac Lab-compatible configurations.

Assets in Detail
----------------

**Asset Hierarchy**
   Three-tier architecture from generic to specialized:

   - **Asset**: Base class providing name, tags, and identification
   - **ObjectBase**: Adds physics types, pose management, and configuration generation
   - **Specialized Classes**: Object, Background, ObjectReference, EmbodimentBase for specific use cases

**Physics Types**
   Support for different simulation behaviors:

   - **BASE**: Static objects without physics (backgrounds, markers)
   - **RIGID**: Single rigid body physics (boxes, tools, furniture)
   - **ARTICULATION**: Multi-body with joints (robots, doors, appliances)

**Registration System**
   Automatic asset discovery through decorators:

   .. code-block:: python

      @register_asset
      class CrackerBox(LibraryObject):
          name = "cracker_box"
          tags = ["object"]
          usd_path = "path/to/cracker_box.usd"
          object_type = ObjectType.RIGID

**Discovery Mechanisms**
   Multiple ways to find and instantiate assets:

   - **By Name**: ``asset_registry.get_asset_by_name("cracker_box")()``
   - **By Tags**: ``asset_registry.get_assets_by_tag("object")``
   - **Random Selection**: ``asset_registry.get_random_asset_by_tag("object")()``

**Configuration Generation**
   Automatic Isaac Lab configuration creation based on object type and properties, handling physics setup, contact sensors, and collision detection internally.

Environment Integration
-----------------------

.. code-block:: python

   # Asset creation and positioning
   background = asset_registry.get_asset_by_name("kitchen")()
   pick_object = asset_registry.get_asset_by_name("cracker_box")()
   pick_object.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1)))

   # Scene composition
   scene = Scene(assets=[background, pick_object])

   # Environment integration
   environment = IsaacArenaEnvironment(
       name="manipulation_task",
       embodiment=embodiment,
       scene=scene,
       task=task
   )

Usage Examples
--------------

**Asset Discovery**

.. code-block:: python

   # Direct asset creation
   kitchen = asset_registry.get_asset_by_name("kitchen")()
   objects = asset_registry.get_assets_by_tag("object")

**Affordance Integration**

.. code-block:: python

   # Objects with interaction capabilities
   microwave = asset_registry.get_asset_by_name("microwave")()  # Has Openable mixin
   task = OpenDoorTask(microwave, openness_threshold=0.8)

**Object References**

.. code-block:: python

   # Reference existing scene elements
   destination = ObjectReference(
       name="kitchen_drawer",
       prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
       parent_asset=kitchen_background,
       object_type=ObjectType.RIGID
   )
