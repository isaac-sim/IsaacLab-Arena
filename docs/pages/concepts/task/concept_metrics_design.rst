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

Example: ObjectDisplacedRateMetric
----------------------------------

Here we provide an example of how the ``ObjectDisplacedRateMetric`` is implemented.
The ``ObjectDisplacedRateMetric`` implements the recording and computing steps described above
in the following way:

* **Recording** During simulation ``ObjectDisplacedRateMetric`` records the
  object's world-frame position (a 3D vector) after every environment step.
  Deformable objects report their mean nodal position.
* **Computing** After the rollout completes, the ``ObjectDisplacedRateMetric`` processes
  the position history for each episode. It first computes the distance of each recorded
  position from the first recorded position.
  Then it checks if any distance is greater than a threshold (2 cm by default). If it is, the
  episode counts as displaced, even if the object is later put back.
  Finally, it computes the fraction of episodes in which the object was displaced.
  Note this does not depend on speed: a slow carry is counted.

Other tasks, such as ``GoalPoseTask``, report ``ObjectPeakSpeedExceededRateMetric`` instead, which
counts an episode when the object's linear speed exceeds a threshold at any step.

You do not need to manage any of this manually — it is all wired up automatically
in the Arena Environment Builder when the environment is compiled from the task.
