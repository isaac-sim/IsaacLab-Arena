Metrics Design
==============

Metrics evaluate robot performance and task completion. The system integrates with Isaac Lab's recorder manager to capture simulation data and compute performance indicators.

Core Architecture
-----------------

Metrics use two-component architecture separating data collection from metric computation:

.. code-block:: python

   class MetricBase(ABC):
       name: str
       recorder_term_name: str

       @abstractmethod
       def get_recorder_term_cfg(self) -> RecorderTermCfg:
           """Define what data to record."""

       @abstractmethod
       def compute_metric_from_recording(self, recorded_metric_data: list[np.ndarray]) -> float:
           """Compute final metric from recorded data."""

Each metric has a **RecorderTerm** that collects data and a **MetricBase** implementation that processes recorded data into performance indicators.

Metrics in Detail
-----------------

**Data Collection Pipeline**
   Two-phase approach to performance evaluation:

   - **RecorderTerm Components**: Real-time data collection during simulation with configurable triggers
   - **Recording Modes**: Pre-reset, post-step, event-triggered, and continuous monitoring patterns
   - **Storage Format**: HDF5 format with episode organization and parallel environment support
   - **Data Extraction**: Access simulation state and extract relevant measurements

**Available Metrics**
   Built-in metrics for common evaluation scenarios:

   - **Success Rate Metric**: Binary task completion tracking across episodes
   - **Door Moved Rate Metric**: Interaction progress with openable objects via joint positions
   - **Object Moved Rate Metric**: Manipulation assessment through object velocity tracking
   - **Custom Metrics**: Extensible framework for task-specific performance indicators

**Integration Pattern**
   Metrics integrate through task definitions:

   .. code-block:: python

      class OpenDoorTask(TaskBase):
          def get_metrics(self) -> list[MetricBase]:
              return [
                  SuccessRateMetric(),
                  DoorMovedRateMetric(self.openable_object, reset_openness=self.reset_openness)
              ]

**Computation Workflow**
   Standardized evaluation process:

   - **Data Recording**: Capture relevant simulation data throughout execution
   - **Episode Completion**: Organize and store data when episodes terminate
   - **Metric Computation**: Post-simulation processing of recorded data
   - **Result Aggregation**: Combine multiple metrics into evaluation reports

Environment Integration
-----------------------

.. code-block:: python

   # Metric collection during environment execution
   env_builder = ArenaEnvBuilder(arena_environment, args)
   env = env_builder.make_registered()  # Metrics auto-configured from task

   # Execute episodes with automatic recording
   for episode in range(100):
       obs, _ = env.reset()
       done = False
       while not done:
           actions = policy(obs)
           obs, _, terminated, truncated, _ = env.step(actions)
           done = terminated or truncated

   # Compute final performance indicators
   metrics_results = compute_metrics(env)

Usage Examples
--------------

**Task-Specific Metrics**

.. code-block:: python

   # Pick and place evaluation
   task = PickAndPlaceTask(pick_object, destination, background)
   metrics = task.get_metrics()  # [SuccessRateMetric(), ObjectMovedRateMetric()]

   # Door opening evaluation
   task = OpenDoorTask(microwave, openness_threshold=0.8)
   metrics = task.get_metrics()  # [SuccessRateMetric(), DoorMovedRateMetric()]

**Results Analysis**

.. code-block:: python

   # Performance evaluation across environments
   print(f"Success Rate: {metrics_results['success_rate']:.2%}")
   print(f"Object Moved Rate: {metrics_results['object_moved_rate']:.2%}")
