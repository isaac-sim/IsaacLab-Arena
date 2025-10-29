Scene Design
============

Scenes manage collections of assets that define the physical environment for simulation. They provide a unified interface for composing backgrounds, objects, and interactive elements.

Core Architecture
-----------------

Scenes use the ``Scene`` class that manages asset collections:

.. code-block:: python

   class Scene:
       def __init__(self, assets: list[Asset] | None = None):
           self.assets: dict[str, Asset] = {}

       def add_asset(self, asset: Asset):
           """Add single asset to scene."""
           self.assets[asset.name] = asset

       def get_scene_cfg(self) -> Any:
           """Generate Isaac Lab scene configuration from assets."""

Scenes automatically aggregate asset configurations into Isaac Lab-compatible scene configurations while maintaining asset relationships and spatial organization.

Scenes in Detail
----------------

The main method in the scene class is ``get_scene_cfg`` which returns a configclass containing all the scene elements.

It reads all the registered assets and their cfgs and combines them into a configclass.

.. code-block:: python

   def get_scene_cfg(self) -> Any:
         """Returns a configclass containing all the scene elements."""
         # Combine the configs into a configclass.
         fields: list[tuple[str, type, AssetCfg]] = []
         for asset in self.assets.values():
               for asset_cfg_name, asset_cfg in asset.get_cfgs().items():
                  fields.append((asset_cfg_name, type(asset_cfg), asset_cfg))
         NewConfigClass = make_configclass("SceneCfg", fields)
         new_config_class = NewConfigClass()
         return new_config_class


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
   environment = IsaacLabArenaEnvironment(
       name="manipulation_task",
       embodiment=embodiment,
       scene=scene,  # Physical environment layout
       task=task,
       teleop_device=teleop_device
   )

Usage Examples
--------------

**Kitchen Pick and Place**

.. code-block:: python

   background = asset_registry.get_asset_by_name("kitchen")()
   mustard_bottle = asset_registry.get_asset_by_name("mustard_bottle")()
   mustard_bottle.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1)))

   scene = Scene(assets=[background, mustard_bottle])

**Microwave Interaction**

.. code-block:: python

   kitchen = asset_registry.get_asset_by_name("kitchen")()
   microwave = asset_registry.get_asset_by_name("microwave")()
   microwave.set_initial_pose(Pose(position_xyz=(0.8, 0.0, 0.23)))

   scene = Scene(assets=[kitchen, microwave])

**Object References**

.. code-block:: python

   # Reference elements within larger scene assets
   destination = ObjectReference(
       name="kitchen_drawer",
       prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
       parent_asset=kitchen,
       object_type=ObjectType.RIGID
   )

   scene = Scene(assets=[kitchen, destination])
