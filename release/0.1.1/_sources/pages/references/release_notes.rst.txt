Release Notes
=============


v0.1.1
------

This release includes bug fixes, documentation improvements, CI and infrastructure
updates, and several API and workflow enhancements over v0.1.0.

**Features and improvements**

- **Object configuration:** Object configuration is now created as soon as an asset is
  called, so users can edit object properties before a scene is created (#239).
- **Scene export:** Added support for saving a scene to a flattened USD file (#237).
  Scene export now correctly handles double-precision poses and adds contact reporters
  when exporting rigid objects (#242).
- **Parallel environment evaluation:** Enabled parallel environment evaluation for
  GR00T policy runner, with documentation for closed-loop GR00T workflows (#231, #236).
- **Episode length:** Increased episode length for loco-manipulation to support
  rollout through box drop (#235).
- **Microwave example:** Increased reset openness for the microwave example (#311).

**Bug fixes**

- **Reference object poses:** Fixed reference object poses so they correctly account
  for the parent object’s initial pose; poses are now relative and composed at compile
  time (#232).
- **IsaacLab-to-stage path conversion:** Fixed a bug when the asset name appeared twice
  in the prim path (replaced both instances instead of one) (#241).
- **qpsolvers:** Patched breakage with Isaac Lab 2.3 due to ``qpsolvers`` upgrade by
  pinning to 4.8.1 (#252).
- **Parallel eval:** Removed comments that were breaking the parallel eval run
  commands (#262).

**Documentation**

- **Multi-versioned docs:** Documentation is now versioned so users can read docs that
  match their release (#272, #300).
- **Links and structure:** Updated README docs link to the public location (#270),
  corrected doc pointers (#301), and added release warnings (#303).
- **Installation:** Private Omniverse/Nucleus access is described on a separate page
  to clarify it is not required for normal installation (#261).

**Infrastructure and CI**

- **Runners:** Release 0.1.1 CI runners moved from local (Zurich) to AWS (#433).
- **CI workflow:** Added YAML anchors to reduce repetition in the CI workflow (#245).
- **Contribution guide:** Added signoff requirements for external contributions (#238).
- **Docker:** Fixed Dockerfile pip usage and added SSL certificate support for
  Lightwheel SDK (#449).
- **Tests:** Finetuned GR00T locomanip model is now generated on the fly in tests
  instead of mounting a pre-finetuned models directory, improving public CI
  compatibility and testing the fine-tuning pipeline (#247).

**Assets and tests**

- **G1 WBC:** Updated G1 WBC embodiment file paths to use S3 (#251).
- **Test assets:** Removed internal or custom-only assets from tests: custom cracker
  box (#234), custom USD in ObjectReference test (#240), internal asset from USD
  utils test (#244). ObjectReference test now composes USD on the fly via scene
  export (#240).


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
