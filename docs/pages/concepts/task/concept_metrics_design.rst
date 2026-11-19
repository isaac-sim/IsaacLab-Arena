Metrics
=======

A metric is a quantity that is reported at the end of an evaluation, that quantifies
some aspect of a policy's performance on the task.
In Isaac Lab Arena, metrics are task-specific. A task defines the metrics that should
be calculated during the evaluation of that task.
This design means that metrics can be defined once in the task and
they are automatically applied to every environment that uses that task.

Under the hood, a metric does two things:

* **Recording** — The metric records the quantity during each episode of policy execution.
* **Computing** The metric computes the final value from the recorded data, typically by reducing the recordings to a scalar.

In the code below we show how to inspect the metrics defined by the ``PickAndPlaceTask``.

.. code-block:: python

   task = PickAndPlaceTask(pick_object, destination, kitchen)
   task.get_metrics()
   # → [SuccessRateMetric(), ObjectDisplacedRateMetric(pick_object)]

Example: ObjectPeakSpeedExceededRateMetric
------------------------------------------

Here we provide an example of how the ``ObjectPeakSpeedExceededRateMetric`` is implemented.
The ``ObjectPeakSpeedExceededRateMetric`` implements the recording and computing steps described above
in the following way:

* **Recording** During simulation ``ObjectPeakSpeedExceededRateMetric`` records the
  object's velocity vector (a 3D vector) at every simulation step.
* **Computing** After the rollout completes, the ``ObjectPeakSpeedExceededRateMetric`` process
  the velocity history for each episode. It first computes the magnitude of the velocity
  vector for each step in an episode.
  Then it checks if the magnitude is greater than a threshold. If it is, the episode
  counts as exceeded.
  Finally, it computes the fraction of episodes in which the peak speed exceeded the threshold.
  Note this measures peak speed, not displacement: a slow carry is not counted.

You do not need to manage any of this manually — it is all wired up automatically
in the Arena Environment Builder when the environment is compiled from the task.
