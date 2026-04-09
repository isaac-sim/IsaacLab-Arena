Metrics
=======

A metric is a quantity that is reported at the end of an evaluation, that quantifies
some aspect of a policy's performance on the task.

In Isaac Lab Arena, metrics are attached task-specific. A task defines metrics that should
be calculated during the complete of that task, while that task is being used.
This design means that metrics can be defined once in the task and
they are automatically applied to every environment that uses that task.

Under the hood, a metric does two things:
* **Recording** — The metric records the quantity during each episode of policy execution.
* **Recordering** The metric records the quantity during each episode of policy execution.
* **Computing** The metric computes the final value from the recorded data, typically by reducing the recordings to a scalar.

In the code below we show how to inspect the metrics defined by the ``PickAndPlaceTask``.

.. code-block:: python

   task = PickAndPlaceTask(pick_object, destination, kitchen)
   task.get_metrics()
   # → [SuccessRateMetric(), ObjectMovedRateMetric(pick_object)]

Example: ObjectMovedRateMetric
------------------------------

Here we provide an example of how the ``ObjectMovedRateMetric`` is implemented.
The ``ObjectMovedRateMetric`` implements the recording and computing steps described above
in the following way:

* **Recording** During simulation ``ObjectMovedRateMetric`` records the
  object's velocity vector (a 3D vector) at every simulation step.
* **Computing** After the rollout completes, the ``ObjectMovedRateMetric`` process
  the velocity history for each episode. It first computes the magnitude of the velocity
  vector for each step in an episode.
  Then it checks if the magnitude is greater than a threshold. If it is, the object is
  considered moved in this episode.
  Finally, it computes the fraction of episodes in which the object was moved.

You do not need to manage any of this manually — it is all wired up automatically
in the Arena Environment Builder when the environment is compiled from the task.
