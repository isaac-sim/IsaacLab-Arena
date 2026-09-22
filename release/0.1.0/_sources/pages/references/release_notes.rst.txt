Release Notes
=============

v0.1.0
------

This initial release of Isaac Lab Arena delivers the first version of the
composable task definition API.
Also included are example workflows for static manipulation tasks and loco-manipulation
tasks including GR00T GN1.5 finetuning and evaluation.

Key features of this release include:

- **Composable Task Definition:** Base-class definition for ``Task``, ``Embodiment``, and ``Scene``
  that can be subclassed to create new tasks, embodiments, and scenes.
  ``ArenaEnvBuilder`` for converting ``Scene``, ``Embodiment``, and ``Task`` into an
  Isaac Lab runnable environment.
- **Metrics:** Mechanism for adding task-specific metrics which are reported during evaluation.
- **Isaac Lab Mimic Integration:** Integration with Isaac Lab Mimic to automatically generate Mimic definitions for
  available tasks.
- **Example Workflows:** Two example workflows for static manipulation tasks and loco-manipulation tasks.
- **GR00T GN1.5 Integration:** Integration with GR00T GN1.5 including a example workflows for finetuning and evaluating
  the model on the static and loco-manipulation workflows.

Known limitations:

- **Number of Environments/Tasks:** This initial is intended to validation the composable task
  definition API, and comes with a limited set of tasks and workflows.
- **Loco-manipulation GR00T GN1.5 finetuning:** GR00T GN1.5 finetuning for loco-manipulation
  requires a large amount of GPU resources. (Note that static manipulation finetuning can be
  performed on a single GPU.)
