Metrics
=======

A metric records any quantity during simulation and reduces it to a scalar after the rollout.
Each task defines which metrics to collect via ``get_metrics()``. Because metrics are attached
to the task, they travel with it across all environment compositions — define them once and
they are automatically applied to every environment that uses that task, with no manual
wiring required.

.. code-block:: python

   task = PickAndPlaceTask(pick_object, destination, kitchen)
   task.get_metrics()
   # → [SuccessRateMetric(), ObjectMovedRateMetric(pick_object)]

Arena metrics
-------------

Arena ships with three metrics:

**SuccessRateMetric**
   The fraction of episodes in which the task's ``success`` termination condition was triggered.
   Used by all tasks.

**ObjectMovedRateMetric**
   The fraction of episodes in which an object exceeded a minimum linear velocity.
   Used as a proxy for whether the robot interacted with the object at all.
   Used by pick-and-place and lift tasks.

**RevoluteJointMovedRateMetric**
   The fraction of episodes in which a revolute joint moved by at least a minimum delta
   from its reset position. Used by open-door and close-door tasks.

How recording works
-------------------

Metrics are recorded in two phases.

During simulation, a ``RecorderTerm`` (from Isaac Lab) samples data at each step or episode
boundary and writes it to an HDF5 file. For example, ``SuccessRateMetric`` records the
success flag just before each episode resets; ``ObjectMovedRateMetric`` records the
object's velocity at every simulation step.

After the rollout completes, Arena reads the HDF5 file and calls
``metric.compute_metric_from_recording()`` on the collected data to produce a scalar result.
The results are printed and saved to a JSON file.

You do not need to manage any of this manually — it is all wired up automatically
when the environment is compiled from the task.

Writing a custom metric
-----------------------

Because the ``RecorderTerm`` has full access to the simulation environment, anything available
in Isaac Lab — object poses, joint states, velocities, sensor readings — can be recorded and
turned into a metric.

A custom metric requires two classes: a ``RecorderTerm`` that samples data during simulation,
and a ``MetricBase`` subclass that computes the final value from the recorded data.

The example below is ``ObjectMovedRateMetric`` from Arena's own metrics library. It records
the object's linear velocity at every step and computes the fraction of episodes in which
the object exceeded a velocity threshold:

.. code-block:: python

   import numpy as np
   from dataclasses import MISSING

   import warp as wp
   from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
   from isaaclab.utils import configclass
   from isaaclab_arena.assets.asset import Asset
   from isaaclab_arena.metrics.metric_base import MetricBase


   class ObjectVelocityRecorder(RecorderTerm):
       """Records the linear velocity of an object for each sim step of an episode."""

       def __init__(self, cfg, env):
           super().__init__(cfg, env)
           self.name = cfg.name
           self.object_name = cfg.object_name

       def record_post_step(self):
           object_linear_velocity = wp.to_torch(self._env.scene[self.object_name].data.root_link_vel_w)[:, :3]
           return self.name, object_linear_velocity


   @configclass
   class ObjectVelocityRecorderCfg(RecorderTermCfg):
       class_type: type[RecorderTerm] = ObjectVelocityRecorder
       name: str = "object_linear_velocity"
       object_name: str = MISSING


   class ObjectMovedRateMetric(MetricBase):
       name = "object_moved_rate"
       recorder_term_name = "object_linear_velocity"

       def __init__(self, object: Asset, object_velocity_threshold: float = 0.5):
           super().__init__()
           self.object = object
           self.object_velocity_threshold = object_velocity_threshold

       def get_recorder_term_cfg(self) -> RecorderTermCfg:
           return ObjectVelocityRecorderCfg(name=self.recorder_term_name, object_name=self.object.name)

       def compute_metric_from_recording(self, recorded_metric_data: list[np.ndarray]) -> float:
           object_moved_per_demo = []
           for object_velocity in recorded_metric_data:
               object_linear_velocity_magnitude = np.linalg.norm(object_velocity, axis=-1)
               object_moved = np.any(object_linear_velocity_magnitude > self.object_velocity_threshold)
               object_moved_per_demo.append(object_moved)
           return float(np.mean(object_moved_per_demo))

To use it, return it from your task's ``get_metrics()``:

.. code-block:: python

   class MyTask(TaskBase):
       def get_metrics(self) -> list[MetricBase]:
           return [
               SuccessRateMetric(),
               ObjectMovedRateMetric(self.pick_object),
           ]
