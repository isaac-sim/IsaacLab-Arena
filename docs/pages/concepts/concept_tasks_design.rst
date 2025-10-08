Tasks Design
============

Tasks in Isaac Arena define objectives, success criteria, and behavior logic for robot learning environments. They provide configurations for termination conditions, event handling, metrics collection, and mimic, integrating seamlessly with Isaac Lab's manager-based environment system.

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

Tasks encapsulate all task-specific logic while remaining embodiment and scene agnostic.

Task Components
---------------

Tasks coordinate multiple Isaac Lab manager configurations:

**Scene Configuration**
   Additional sensors and physics components required by the task:

   - Contact sensors for success detection for now

**Termination Configuration**
   Success and failure conditions that define episode completion:

   - **Success Termination**: Task completion criteria (object placement, door opening threshold)
   - **Failure Termination**: Safety conditions (object dropping, timeout)
   - **Physics-based Conditions**: Contact forces, object velocities, spatial relationships

**Event Configuration**
   Reset and randomization logic for consistent episode initialization:

   - **Pose Resets**: Return objects to initial configurations
   - **State Resets**: Reset affordance states (door openness, button positions)
   - **Randomization**: Procedural variation in object poses and properties

**Metrics Integration**
   Performance evaluation and data collection:

   - **Success Rate**: Binary task completion tracking
   - **Object Movement**: Spatial displacement and manipulation progress
   - **Task-specific Metrics**: Door movement rates, affordance interaction progress

**Mimic Integration**
   Mimic environment configurations for demonstration generation:

   - **Subtask Definitions**: Multi-stage manipulation sequences with object-relative control
   - **Embodiment Adaptation**: Different subtask configurations for various robot morphologies (Franka, G1 dual-arm)


Configuration Generation
------------------------

Tasks generate Isaac Lab configurations for environment integration:

**Scene Configuration**
   Additional sensors and physics components required by the task:

.. code-block:: python

   # Contact sensors for success detection
   contact_sensor = pick_up_object.get_contact_sensor_cfg(
       contact_against_prim_paths=[destination.get_prim_path()]
   )

**Termination Configuration**
   Success and failure condition specifications:

.. code-block:: python

   success = TerminationTermCfg(
       func=object_on_destination,
       params={
           "object_cfg": SceneEntityCfg("pick_up_object"),
           "contact_sensor_cfg": SceneEntityCfg("contact_sensor"),
           "force_threshold": 1.0,
           "velocity_threshold": 0.1
       }
   )

**Events Configuration**
   Reset behavior for consistent episode initialization:

.. code-block:: python

   reset_object_pose = EventTermCfg(
       func=set_object_pose,
       mode="reset",
       params={
           "pose": initial_pose,
           "asset_cfg": SceneEntityCfg("object_name")
       }
   )

**Mimic Configuration**
   Mimic environment configurations for demonstration generation:

.. code-block:: python

   return PickAndPlaceMimicEnvCfg(
       embodiment_name=embodiment_name,
       pick_up_object_name="pick_up_object",
       destination_location_name="destination_location"
   )

**Metrics Configuration**
   Performance evaluation and data collection:

.. code-block:: python

   return [SuccessRateMetric(), ObjectMovedRateMetric(self.pick_up_object)]

Environment Integration
-----------------------

Tasks integrate into environments through composition:

.. code-block:: python

   # Task construction with scene assets
   task = PickAndPlaceTask(
       pick_up_object=cracker_box,
       destination_location=kitchen_drawer,
       background_scene=kitchen
   )

   # Environment composition
   environment = IsaacArenaEnvironment(
       name="kitchen_manipulation",
       embodiment=embodiment,
       scene=scene,
       task=task,
       teleop_device=teleop_device
   )


Usage Examples
--------------

**Kitchen Pick and Place**

.. code-block:: python

   pick_object = asset_registry.get_asset_by_name("mustard_bottle")()
   destination = ObjectReference("kitchen_drawer", parent_asset=kitchen)

   task = PickAndPlaceTask(pick_object, destination, kitchen)

**Microwave Opening**

.. code-block:: python

   microwave = asset_registry.get_asset_by_name("microwave")()
   task = OpenDoorTask(microwave, openness_threshold=0.8)

**Custom Task Development**

.. code-block:: python

   class CustomTask(TaskBase):
       def get_termination_cfg(self):
           return CustomTerminationsCfg(success=custom_success_condition)

       def get_metrics(self):
           return [CustomMetric()]

The task system in Isaac Arena provides modular objective definitions that integrate seamlessly with embodiments and scenes while maintaining clear separation of concerns between physical layout and behavioral goals.
