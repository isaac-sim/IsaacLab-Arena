Affordances
===========

An affordance is an interaction that an object makes available to the robot —
opening a door, pressing a button, turning a knob.
By attaching affordances to objects, Arena gives tasks a standard interface
to interact with them, regardless of the specific object.

.. figure:: ../../../images/affordances_objects.png
   :width: 100%
   :alt: Examples of Pressable and Openable objects
   :align: center

   Two examples of affordances. A drill and coffee machine are **Pressable**
   (``is_pressed()``, ``press()``); a microwave and cardboard box are **Openable**
   (``is_open()``, ``close()``).

How an object gets an affordance
---------------------------------

Affordances are added to an object through multiple inheritance.
The microwave inherits from both ``LibraryObject`` and ``Openable``,
and passes the joint name and threshold to the affordance constructor.
This excerpt from ``object_library.py`` omits imports:

.. code-block:: python

   @register_asset
   class Microwave(LibraryObject, Openable):
       name = "microwave"
       tags = ["object", "openable"]
       usd_path = LightwheelLazyPath(registry_type="fixtures", file_name="Microwave039", file_type="USD")
       object_type = ObjectType.ARTICULATION

       # Openable affordance parameters
       openable_joint_name = "microjoint"
       openable_threshold = 0.5  # open if normalized openness > threshold

       def __init__(
           self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
       ):
           super().__init__(
               instance_name=instance_name,
               prim_path=prim_path,
               initial_pose=initial_pose,
               openable_joint_name=self.openable_joint_name,
               openable_threshold=self.openable_threshold,
           )

The ``Openable`` mixin implements ``is_open()`` and ``close()`` using the joint
name forwarded through ``super().__init__()``. Declaring the class attributes
alone does not initialize the affordance. The USD asset must contain the named joint.

Why this matters
----------------

Because tasks are written against the affordance interface rather than a specific object,
the same task works with any object that has the right affordance.
``OpenDoorTask`` works with the microwave, a fridge, a cabinet — any ``Openable``.
This is what makes tasks modular and reusable across different scenes.
