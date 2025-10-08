Affordances Design
===================

Affordances in Isaac Arena define object interaction capabilities, providing a standardized way to represent what actions can be performed on objects in the environment. Each affordance encapsulates specific behaviors like opening, pressing, or manipulating objects through well-defined interfaces.

Core Architecture
-----------------

The affordance system is built around the ``AffordanceBase`` abstract class, which defines a standard interface for object interactions:

.. code-block:: python

   class AffordanceBase(ABC):
       """Base class for affordances."""

       @property
       @abstractmethod
       def name(self) -> str:
           # Affordances are always combined with Assets
           # which have a "name" property
           pass

Affordances are designed as mixing classes that combine with objects to provide interaction capabilities. They operate on articulated objects through joint manipulation and state querying.

Available Affordances
---------------------

**Openable**
   Provides opening and closing functionality for objects like doors, drawers, and appliances.

   - **Methods**: ``open()``, ``close()``, ``is_open()``, ``get_openness()``
   - **Parameters**: ``openable_joint_name``, ``openable_open_threshold``
   - **Use Cases**: Microwave doors, cabinet drawers, refrigerators

**Pressable**
   Enables pressing and releasing functionality for buttons, switches, and similar objects.

   - **Methods**: ``press()``, ``unpress()``, ``is_pressed()``
   - **Parameters**: ``pressable_joint_name``, ``pressable_pressed_threshold``
   - **Use Cases**: Toaster buttons, light switches, control panels

Affordance Integration
----------------------

Affordances integrate with objects through multiple inheritance, combining object properties with interaction capabilities:

.. code-block:: python

   @register_asset
   class Microwave(LibraryObject, Openable):
       """A microwave oven with opening capability."""

       name = "microwave"
       tags = ["object", "openable"]
       object_type = ObjectType.ARTICULATION

       # Openable affordance parameters
       openable_joint_name = "microjoint"
       openable_open_threshold = 0.5

       def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
           super().__init__(
               prim_path=prim_path,
               initial_pose=initial_pose,
               openable_joint_name=self.openable_joint_name,
               openable_open_threshold=self.openable_open_threshold,
           )

Affordance Operations
---------------------

**State Querying**
   Affordances provide methods to query the current state of objects:

   - ``get_openness()`` - Returns normalized joint position (0.0 to 1.0)
   - ``is_open()`` - Returns boolean based on threshold comparison
   - ``is_pressed()`` - Returns boolean for pressable objects

**State Manipulation**
   Affordances enable direct control of object states:

   - ``open(percentage=1.0)`` - Set opening percentage
   - ``close(percentage=0.0)`` - Set closing percentage
   - ``press(pressed_percentage=1.0)`` - Set pressed state
   - ``unpress(unpressed_percentage=1.0)`` - Release pressed state

**Environment Integration**
   All affordance methods operate on Isaac Lab environments:

   .. code-block:: python

      # Query object state
      openness = microwave.get_openness(env)
      is_open = microwave.is_open(env, threshold=0.7)

      # Manipulate object state
      microwave.open(env, env_ids, percentage=0.8)
      toaster.press(env, env_ids, pressed_percentage=1.0)

Task Integration
----------------

Affordances seamlessly integrate with task definitions to provide interaction-based objectives:

**Termination Conditions**
   Use affordance state queries for success conditions:

   .. code-block:: python

      class OpenDoorTask(TaskBase):
          def make_termination_cfg(self):
              success = TerminationTermCfg(
                  func=self.openable_object.is_open,
                  params={"threshold": self.openness_threshold}
              )
              return TerminationsCfg(success=success)

**Event Handling**
   Reset object states using affordance methods:

   .. code-block:: python

      self.reset_door_state = EventTermCfg(
          func=openable_object.close,
          mode="reset",
          params={"percentage": reset_openness}
      )

**Metrics and Monitoring**
   Track interaction progress through affordance states:

   .. code-block:: python

      class DoorMovedRateMetric(MetricBase):
          def compute_metric_from_recording(self, data):
              openness_data = self.openable_object.get_openness(env)
              return self._compute_movement_rate(openness_data)

Object Reference System
-----------------------

For objects that exist within scene assets, Isaac Arena provides reference-based affordances:

.. code-block:: python

   class OpenableObjectReference(ObjectReference, Openable):
       """References an openable object within a parent scene."""

       def __init__(self, openable_joint_name: str, openable_open_threshold: float = 0.5, **kwargs):
           super().__init__(
               openable_joint_name=openable_joint_name,
               openable_open_threshold=openable_open_threshold,
               object_type=ObjectType.ARTICULATION,
               **kwargs,
           )

This enables interaction with objects that are part of larger scene assets (e.g., kitchen appliances within a kitchen scene).

Usage Examples
--------------

**Basic Object Interaction**:

.. code-block:: bash

   python isaac_arena/examples/policy_runner.py --policy_type zero_action gr1_open_microwave --object microwave

**Task-Specific Usage**:

.. code-block:: python

   # Create openable object
   microwave = Microwave(prim_path="{ENV_REGEX_NS}/microwave")

   # Create task using affordance
   task = OpenDoorTask(
       openable_object=microwave,
       openness_threshold=0.8,
       reset_openness=0.0
   )

   # Use in environment
   environment = IsaacArenaEnvironment(
       name="microwave_opening",
       embodiment=G1WBCJointEmbodiment(),
       scene=KitchenScene(),
       task=task,
   )

Joint-Based Implementation
--------------------------

Affordances operate through normalized joint positions, making them compatible with various object designs:

- **Normalization**: Joint positions are mapped to 0.0-1.0 range regardless of actual joint limits
- **Thresholds**: Configurable thresholds define state boundaries (e.g., "open" vs "closed")
- **Flexibility**: Works with different joint types (revolute, prismatic) and orientations

Creating New Affordances
------------------------

To add new affordance types:

1. **Create affordance class** inheriting from ``AffordanceBase``
2. **Define joint parameters** for the specific interaction type
3. **Implement state methods** for querying and manipulation
4. **Add threshold logic** for boolean state determination
5. **Create object integration** through multiple inheritance
6. **Test with tasks** to ensure proper integration

The affordance system provides a powerful abstraction for object interactions, enabling complex manipulation tasks while maintaining consistency across different object types and interaction modalities.
