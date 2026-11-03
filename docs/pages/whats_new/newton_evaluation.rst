Newton-Based Evaluation Workflows
=================================

Arena environments can run on the Newton physics backend as well as PhysX. The upcoming release expands
Newton support for objects and robots, so contact-rich and deformable tasks can be previewed,
teleoperated, recorded, and evaluated with the same Arena workflows.

What's included
---------------

- **Newton-backed robot control:** DROID control runs on Newton, and embodiments expose backend
  hooks.
- **Contact-rich and deformable tasks:** Example workflows cover gear insertion (rigid bodies) and
  deformable pick and place, which can be compared between Newton and PhysX.
- **Backend selection in YAML:** Select the backend with ``default_physics_backend`` or
  ``--presets``, and tune it with ``env_cfg_override`` (simulation timestep, control decimation,
  solver, and collision settings).

Try it
------

- :doc:`../example_workflows/newton/gear_insertion`: preview and teleoperate a rigid-body task,
  record demonstrations, and replay the saved dataset.
- :doc:`../example_workflows/newton/deformable_pick_and_place`: compare Newton and PhysX
  deformable-body execution, including closed-loop OpenPI evaluation.
- :doc:`../concepts/environment/physics_backend_selection`: how Arena selects the physics backend.

For the full list of changes in each release, see :doc:`../references/release_notes`.
