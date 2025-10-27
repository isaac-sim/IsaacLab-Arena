Creating a New Task
===================

Learn how to create tasks using the ``PickAndPlaceTask`` as an example.

Overview
--------

Tasks define complete robotics scenarios. They combine scene elements, success conditions, reset behaviors, and metrics into executable environments.

Basic Structure
---------------

All tasks inherit from ``TaskBase`` and follow this pattern:

.. code-block:: python

    from isaac_arena.tasks.task_base import TaskBase
    from isaac_arena.assets.asset import Asset
    from isaac_arena.metrics.metric_base import MetricBase

    class MyTask(TaskBase):
        def __init__(self, required_assets: Asset):
            super().__init__()
            self.assets = required_assets
            self.scene_config = MySceneCfg()
            self.termination_cfg = self.make_termination_cfg()
            self.events_cfg = MyEventsCfg()

        def get_scene_cfg(self): return self.scene_config
        def get_termination_cfg(self): return self.termination_cfg
        def get_events_cfg(self): return self.events_cfg
        def get_metrics(self) -> list[MetricBase]: return [MyMetric()]
        def get_mimic_env_cfg(self, embodiment_name: str): return MyMimicEnvCfg()

Key Components
--------------

**1. Scene Configuration**
   - Define contact sensors for interaction detection
   - Configure additional scene elements and sensors
   - Set up collision detection between objects

**2. Termination Configuration**
   - Specify success conditions (task completion criteria)
   - Define failure conditions (timeouts, object dropping)
   - Configure termination thresholds and parameters

**3. Event Configuration**
   - Handle environment resets and object positioning
   - Set up randomization for training diversity
   - Configure initial poses and states

**4. Metrics**
   - Define performance measurements (success rate, efficiency)
   - Track task-specific behaviors (object movement, completion time)
   - Integrate with recording system for data collection

Example: Pick and Place Task
----------------------------

.. code-block:: python

    class PickAndPlaceTask(TaskBase):
        def __init__(self, pick_up_object: Asset, destination_location: Asset, background_scene: Asset):
            super().__init__()
            self.pick_up_object = pick_up_object
            self.destination_location = destination_location
            self.background_scene = background_scene

            # Configure contact detection
            self.scene_config = SceneCfg(
                pick_up_object_contact_sensor=self.pick_up_object.get_contact_sensor_cfg(
                    contact_against_prim_paths=[self.destination_location.get_prim_path()]
                )
            )
            self.events_cfg = EventsCfg(pick_up_object=self.pick_up_object)
            self.termination_cfg = self.make_termination_cfg()

        def make_termination_cfg(self):
            success = TerminationTermCfg(
                func=object_on_destination,
                params={
                    "object_cfg": SceneEntityCfg(self.pick_up_object.name),
                    "contact_sensor_cfg": SceneEntityCfg("pick_up_object_contact_sensor"),
                    "force_threshold": 1.0,
                    "velocity_threshold": 0.1
                }
            )
            return TerminationsCfg(success=success)

        def get_metrics(self) -> list[MetricBase]:
            return [SuccessRateMetric(), ObjectMovedRateMetric(self.pick_up_object)]

    @configclass
    class SceneCfg:
        pick_up_object_contact_sensor: ContactSensorCfg = MISSING

    @configclass
    class TerminationsCfg:
        time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=False)
        success: TerminationTermCfg = MISSING

Implementation Tips
-------------------

**Asset Integration**
   - Accept required assets (objects, tools, environments) as parameters
   - Use asset methods like ``get_contact_sensor_cfg()`` for sensor setup
   - Reference assets by name in scene entity configurations

**Termination Design**
   - Combine multiple termination conditions (success, failure, timeout)
   - Use contact sensors to detect object interactions
   - Set appropriate thresholds for physics-based detection

**Event Management**
   - Reset object poses to initial positions on reset
   - Handle cases where assets don't have defined poses
   - Use ``set_object_pose()`` function for consistent positioning

**Configuration Classes**
   - Use ``@configclass`` decorator for structured configuration
   - Mark required parameters with ``dataclasses.MISSING``
   - Separate concerns (scene, terminations, events) into distinct configs

**Metric Selection**
   - Include standard metrics like ``SuccessRateMetric``
   - Add task-specific metrics (object movement, interaction quality)
   - Consider multiple metrics for comprehensive evaluation

Usage in Environments
---------------------

Tasks integrate with embodiments and environments:

.. code-block:: python

    # Create task with required assets
    task = PickAndPlaceTask(
        pick_up_object=cracker_box,
        destination_location=table,
        background_scene=kitchen
    )

    # Task provides all environment configuration
    scene_cfg = task.get_scene_cfg()
    termination_cfg = task.get_termination_cfg()
    metrics = task.get_metrics()
