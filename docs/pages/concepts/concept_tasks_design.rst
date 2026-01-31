Tasks Design
============

Tasks define objectives, success criteria, and behavior logic for environments. They provide configurations for termination conditions, event handling, metrics collection, and demonstration generation.

Core Architecture
-----------------

Tasks use the ``TaskBase`` abstract class:

.. code-block:: python

   class TaskBase(ABC):
       @abstractmethod
       def get_scene_cfg(self) -> Any:
           """Additional scene configurations."""

       @abstractmethod
       def get_termination_cfg(self) -> Any:
           """Success and failure conditions."""

       @abstractmethod
       def get_events_cfg(self) -> Any:
           """Reset and randomization handling."""

       @abstractmethod
       def get_metrics(self) -> list[MetricBase]:
           """Performance evaluation metrics."""

       @abstractmethod
       def get_mimic_env_cfg(self, arm_mode: ArmMode) -> Any:
           """Demonstration generation configuration."""

Tasks in Detail
---------------

**Configuration Components**
   Tasks contribute to multiple Isaac Lab manager configurations:

   - **Scene Configuration**: Additional sensors and physics components (contact sensors, object interactions)
   - **Termination Configuration**: Success and failure conditions defining episode completion
   - **Event Configuration**: Reset and randomization logic for consistent episode initialization
   - **Metrics Integration**: Performance evaluation and data collection systems
   - **Mimic Configuration**: Demonstration generation with subtask decomposition

**Available Tasks**
   Built-in task implementations for common scenarios:

   - **PickAndPlaceTask**: Move objects between locations with contact-based success detection
   - **OpenDoorTask**: Affordance-based interaction with openable objects and thresholds
   - **G1LocomanipPickAndPlaceTask**: Combined locomotion and manipulation for humanoid robots

**RL Task Design**
   RL tasks extend their IL counterparts by adding reinforcement learning components:

   - **Inheritance**: RL tasks inherit from IL task base classes (e.g., ``LiftObjectTaskRL`` extends ``LiftObjectTask``)
   - **Commands**: Introduce command managers that sample random target goals each episode
   - **Rewards**: Define dense reward functions for desired behaviors (reaching, grasping, lifting, target achievement)
   - **Observations**: Configure state observations to include object state, and goal commands
   - **Success Function**: Adaptive termination logic that supports multiple modes:

      - *Training mode*: Disables early success termination to maximize episode length
      - *Evaluation mode*: Uses command manager goals for success detection
      - *IL mode*: Falls back to fixed goal positions for teleoperation/demonstrations


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
   environment = IsaacLabArenaEnvironment(
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

**Humanoid Locomotion**

.. code-block:: python

   pick_object = asset_registry.get_asset_by_name("cracker_box")()
   destination_bin = asset_registry.get_asset_by_name("sorting_bin")()

   task = G1LocomanipPickAndPlaceTask(pick_object, destination_bin, galileo_scene)
