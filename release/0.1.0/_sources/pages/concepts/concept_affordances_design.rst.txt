Affordances Design
===================

Affordances define what interactions objects can perform - opening doors, pressing buttons, manipulating objects.
They provide standardized interfaces that integrate with assets and tasks.

Core Architecture
-----------------

Affordances use the ``AffordanceBase`` abstract class and mixin pattern:

.. code-block:: python

   class AffordanceBase(ABC):
       @property
       @abstractmethod
       def name(self) -> str:
           pass

   class Openable(AffordanceBase):
       def open(self, env, env_ids, percentage=1.0):
           """Set opening percentage via joint control."""

       def is_open(self, env, threshold=None):
           """Query current open state."""

Affordances are mixin classes that combine with objects through multiple inheritance, operating on articulated objects via joint control.

Affordances in Detail
---------------------

**Openable Affordance**
   Provides opening and closing functionality for articulated objects:

   - **State Methods**: ``is_open()``, ``get_openness()`` for querying current state
   - **Control Methods**: ``open()``, ``close()`` for direct state manipulation
   - **Parameters**: ``openable_joint_name``, ``openable_open_threshold``
   - **Implementation**: Normalized joint position control (0.0-1.0 range)

**Pressable Affordance**
   Enables pressing and releasing functionality for button-like objects:

   - **State Methods**: ``is_pressed()`` for current button state
   - **Control Methods**: ``press()``, ``unpress()`` for state changes
   - **Parameters**: ``pressable_joint_name``, ``pressable_pressed_threshold``
   - **Implementation**: Threshold-based binary state determination

**Object Integration Pattern**
   Affordances combine with assets through multiple inheritance:

   .. code-block:: python

      @register_asset
      class Microwave(LibraryObject, Openable):
          name = "microwave"
          object_type = ObjectType.ARTICULATION
          openable_joint_name = "microjoint"
          openable_open_threshold = 0.5

**Task Integration**
   Affordances provide termination conditions and event handling for tasks:

   - **Termination**: Use ``is_open()`` for success criteria in ``OpenDoorTask``
   - **Events**: Reset object states with affordance methods during episode resets
   - **Metrics**: Track interaction progress through state queries

Environment Integration
-----------------------

.. code-block:: python

   # Create affordance-enabled object
   microwave = asset_registry.get_asset_by_name("microwave")()

   # Use in task definition
   task = OpenDoorTask(
       openable_object=microwave,
       openness_threshold=0.8,
       reset_openness=0.2
   )

   # Environment composition
   environment = IsaacLabArenaEnvironment(
       name="microwave_opening",
       embodiment=embodiment,
       scene=scene,
       task=task
   )

Usage Examples
--------------

**Microwave Opening Task**

.. code-block:: python

   microwave = asset_registry.get_asset_by_name("microwave")()
   task = OpenDoorTask(microwave, openness_threshold=0.8)

**Runtime State Querying**

.. code-block:: python

   # Check object state during simulation
   if microwave.is_open(env, threshold=0.7):
       print("Microwave is open enough")

   current_openness = microwave.get_openness(env)
