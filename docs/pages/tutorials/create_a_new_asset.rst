Creating a New Asset
====================

Learn how to create assets using examples from the object library.

Overview
--------

Assets are objects you can spawn in environments. They can be simple rigid bodies or complex articulated objects with affordances.

Basic Asset Structure
---------------------

All assets inherit from ``Asset`` or ``LibraryObject`` for library objects:

.. code-block:: python

    from isaac_arena.assets.object import Object
    from isaac_arena.assets.object_base import ObjectType
    from isaac_arena.assets.register import register_asset
    from isaac_arena.utils.pose import Pose

    @register_asset
    class MyAsset(LibraryObject):
        name = "my_asset"
        tags = ["object"]
        usd_path = "path/to/your/asset.usd"
        object_type = ObjectType.RIGID  # or ARTICULATION

Simple Rigid Object Example
---------------------------

.. code-block:: python

    @register_asset
    class CrackerBox(LibraryObject):
        name = "cracker_box"
        tags = ["object"]
        usd_path = "omniverse://isaac-dev.ov.nvidia.com/NVIDIA/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"

        def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
            super().__init__(prim_path=prim_path, initial_pose=initial_pose)

Interactive Object with Affordance
----------------------------------

For interactive objects:

.. code-block:: python

    from isaac_arena.affordances.openable import Openable

    @register_asset
    class Microwave(LibraryObject, Openable):
        name = "microwave"
        tags = ["object", "openable"]
        usd_path = "omniverse://isaac-dev.ov.nvidia.com/Projects/isaac_arena/interactable_objects/microwave.usd"
        object_type = ObjectType.ARTICULATION

        openable_joint_name = "microjoint"
        openable_open_threshold = 0.5

        def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
            super().__init__(
                prim_path=prim_path,
                initial_pose=initial_pose,
                openable_joint_name=self.openable_joint_name,
                openable_open_threshold=self.openable_open_threshold,
            )

Key Components
--------------

**Required:**
   - ``name``: Unique identifier
   - ``tags``: Descriptive tags (e.g., ["object", "openable"])
   - ``usd_path``: Path to USD file

**Optional:**
   - ``object_type``: ``RIGID``, ``ARTICULATION``, or ``BASE`` (default: ``RIGID``)
   - ``scale``: Scaling factor (default: ``(1.0, 1.0, 1.0)``)

**Registration:**
   - Use ``@register_asset`` decorator
   - Assets become discoverable by name

**Affordance Integration:**
   - Use multiple inheritance: ``class MyObject(LibraryObject, Openable)``
   - Set affordance class attributes (joint names, thresholds)
   - Pass parameters to ``super().__init__()``

Object Types
------------

**RIGID**: Objects without joints (boxes, bottles, tools)
   - Fast physics simulation
   - Good for pickable objects

**ARTICULATION**: Objects with joints (doors, drawers, robots)
   - Supports interactions via affordances
   - Requires joint definitions in USD file

**BASE**: Basic assets without physics
   - For visual elements or references

Usage Example
-------------

.. code-block:: python

    # Create and place an object
    cracker_box = CrackerBox(initial_pose=Pose(position=[0, 0, 1]))

    # Use affordance if available
    microwave = Microwave()
    microwave.open(env, env_ids=None, percentage=0.8)
    is_open = microwave.is_open(env)

This pattern lets you quickly create assets that integrate with Isaac Arena's simulation and task systems.
