Creating a New Metric
=====================

Learn how to create metrics using the ``SuccessRateMetric`` as an example.

Overview
--------

Metrics measure performance during simulation. They have recorders that collect data and metric classes that compute final values.

Basic Structure
---------------

All metrics follow this two-part pattern:

.. code-block:: python

    from isaac_arena.metrics.metric_base import MetricBase
    from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg

    class MyRecorder(RecorderTerm):
        name = "my_data"

        def record_post_step(self):
            # Collect data each simulation step
            data = self._env.get_some_data()
            return self.name, data

    class MyMetric(MetricBase):
        name = "my_metric"
        recorder_term_name = MyRecorder.name

        def get_recorder_term_cfg(self) -> RecorderTermCfg:
            return MyRecorderCfg()

        def compute_metric_from_recording(self, recorded_metric_data: list[np.ndarray]) -> float:
            # Process recorded data into final metric
            return np.mean(recorded_metric_data)

Key Components
--------------

**1. Recorder Class**
   - Inherits from ``RecorderTerm`` and collects simulation data
   - Implements recording methods (``record_post_step``, ``record_pre_reset``)
   - Returns tuple of (name, data) for each recording

**2. Recorder Configuration**
   - Inherits from ``RecorderTermCfg`` and configures the recorder
   - Sets ``class_type`` to the recorder class
   - Defines additional parameters if needed

**3. Metric Class**
   - Inherits from ``MetricBase`` and processes recorded data
   - Defines metric name and associated recorder name
   - Computes final scalar value from accumulated data

**4. Recording Methods**
   - ``record_post_step()``: Records data after each step
   - ``record_pre_reset()``: Records data before reset
   - Return format: ``(recorder_name, data_tensor)``

Example: Success Rate Metric
----------------------------

.. code-block:: python

    class SuccessRecorder(RecorderTerm):
        name = "success"

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.first_reset = True

        def record_pre_reset(self, env_ids):
            if self.first_reset:
                self.first_reset = False
                return None, None  # Skip first reset

            success_results = torch.zeros(len(env_ids), dtype=bool, device=self._env.device)
            success_results |= self._env.termination_manager.get_term("success")[env_ids]
            return self.name, success_results

    @configclass
    class SuccessRecorderCfg(RecorderTermCfg):
        class_type: type[RecorderTerm] = SuccessRecorder

    class SuccessRateMetric(MetricBase):
        name = "success_rate"
        recorder_term_name = SuccessRecorder.name

        def get_recorder_term_cfg(self) -> RecorderTermCfg:
            return SuccessRecorderCfg()

        def compute_metric_from_recording(self, recorded_metric_data: list[np.ndarray]) -> float:
            if len(recorded_metric_data) == 0:
                return 0.0
            all_success_flags = np.concatenate(recorded_metric_data)
            return np.mean(all_success_flags)

Implementation Tips
-------------------

**Recording Patterns**
   - Use ``record_post_step()`` for continuous data (velocities, positions)
   - Use ``record_pre_reset()`` for episode outcomes (success, failure states)
   - Handle first reset to avoid recording invalid data

**Data Processing**
   - Process ``list[np.ndarray]`` where each array represents one episode
   - Use ``np.concatenate()`` to combine data across episodes
   - Return single ``float`` value as final metric

**Multi-Environment Support**
   - Recorders handle multiple parallel environments
   - Data tensors include batch dimension for all environments
   - Metrics aggregate across all environments and episodes

**Parameter Configuration**
   - Add parameters to recorder config for customization
   - Use ``dataclasses.MISSING`` for required parameters
   - Initialize metric classes with object references and thresholds

Usage in Environments
---------------------

Metrics integrate automatically when defined in task configurations:

.. code-block:: python

    # Metric instances are created and registered
    success_metric = SuccessRateMetric()

    # Recorders collect data during simulation
    # Metrics compute final values after episodes
    final_success_rate = success_metric.compute_metric_from_recording(recorded_data)
