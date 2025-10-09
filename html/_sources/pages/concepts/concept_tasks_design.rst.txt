Tasks Design
============

Tasks in Isaac Arena define objectives, success criteria, and behavior logic for environments. They provide configurations for termination conditions, event handling, metrics collection, and demonstration generation, integrating seamlessly with Isaac Lab's manager-based environment system while remaining embodiment and scene agnostic.

Core Architecture
-----------------

The task system is built around the ``TaskBase`` abstract class that defines task interface requirements:

.. code-block:: python

   class TaskBase(ABC):
       @abstractmethod
       def get_scene_cfg(self) -> Any:
           """Additional scene configurations for the task."""

       @abstractmethod
       def get_termination_cfg(self) -> Any:
           """Success and failure termination conditions."""

       @abstractmethod
       def get_events_cfg(self) -> Any:
           """Reset and randomization event handling."""

       @abstractmethod
       def get_metrics(self) -> list[MetricBase]:
           """Performance evaluation metrics."""

       @abstractmethod
       def get_mimic_env_cfg(self, embodiment_name: str) -> Any:
           """Demonstration generation configuration."""

Tasks encapsulate all task-specific logic while maintaining clear separation of concerns between objectives and physical layout, enabling reuse across different embodiments and scenes.

Tasks in Detail
---------------

**Configuration Components**
   Tasks contribute to multiple Isaac Lab manager configurations:

   - **Scene Configuration**: Additional sensors and physics components (contact sensors, object interactions)
   - **Termination Configuration**: Success and failure conditions defining episode completion
   - **Event Configuration**: Reset and randomization logic for consistent episode initialization
   - **Metrics Integration**: Performance evaluation and data collection systems
   - **Mimic Configuration**: Demonstration generation with subtask decomposition

**Termination System**
   Success and failure conditions that define episode completion:

   - **Success Termination**: Task completion criteria (object placement, door opening threshold)
   - **Failure Termination**: Safety conditions (object dropping, timeout constraints)
   - **Physics-based Conditions**: Contact forces, object velocities, spatial relationships
   - **Threshold Logic**: Configurable parameters for task-specific success criteria

**Event Management**
   Reset and randomization logic for consistent episode initialization:

   - **Pose Resets**: Return objects to initial configurations between episodes
   - **State Resets**: Reset affordance states (door openness, button positions)
   - **Randomization**: Procedural variation in object poses and scene properties
   - **Episode Triggers**: Automatic handling of environment resets and state management

**Available Tasks**
   Built-in task implementations for common scenarios:

   - **PickAndPlaceTask**: Move objects between locations with contact-based success detection
   - **OpenDoorTask**: Affordance-based interaction with openable objects and thresholds
   - **G1LocomanipPickAndPlaceTask**: Combined locomotion and manipulation for humanoid robots
   - **DummyTask**: Empty task template for custom objective development


Environment Integration
-----------------------

.. code-block:: python

   # Task construction with scene assets
   pick_object = asset_registry.get_asset_by_name("cracker_box")()
   destination = ObjectReference("kitchen_drawer", parent_asset=kitchen)

   task = PickAndPlaceTask(
       pick_up_object=pick_object,
       destination_location=destination,
       background_scene=kitchen
   )

   # Environment composition
   environment = IsaacArenaEnvironment(
       name="kitchen_manipulation",
       embodiment=embodiment,
       scene=scene,
       task=task,  # Defines objectives and success criteria
       teleop_device=teleop_device
   )

   # Automatic configuration integration
   env = env_builder.make_registered()  # Task configs merged automatically

Usage Examples
--------------

**Pick and Place Task**

.. code-block:: python

   pick_object = asset_registry.get_asset_by_name("mustard_bottle")()
   destination = ObjectReference("kitchen_drawer", parent_asset=kitchen)

   task = PickAndPlaceTask(pick_object, destination, kitchen)

**Affordance Interaction**

.. code-block:: python

   microwave = asset_registry.get_asset_by_name("microwave")()
   task = OpenDoorTask(microwave, openness_threshold=0.8, reset_openness=0.2)

**Humanoid Locomotion**

.. code-block:: python

   pick_object = asset_registry.get_asset_by_name("cracker_box")()
   destination_bin = asset_registry.get_asset_by_name("sorting_bin")()

   task = G1LocomanipPickAndPlaceTask(pick_object, destination_bin, galileo_scene)

The task system provides modular objective definitions that integrate seamlessly with embodiments and scenes while maintaining clear separation of concerns between physical layout and behavioral goals.
