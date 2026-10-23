Tabletop Pick and Place: Egg and Tray
====================================

Load ``isaaclab_arena_environments/maple_table_top/tabletop_pick_place_egg_tray.yaml``
to build a Franka pick-and-place environment. Three eggs start on the maple table. The task moves ``egg`` into the egg tray;
``egg_2``, ``egg_3``, a nut, and pizza are distractors.
The environment is defined directly in YAML and does not require an LLM call.
Complete the shared :ref:`agentic-env-gen-prerequisites` first.

Pinned assets
-------------

All four assets come from ``PalatialSim/palatial-nv-asset-library`` at revision
``69a87329036cec0440403351c12230970c4e67d8`` and use the CC-BY-4.0 license.
Credit Palatial Platforms when sharing rendered examples or derived assets.

.. list-table::
   :header-rows: 1

   * - Asset
     - Dataset entry
     - Role
   * - Egg Tray
     - `D00290 <https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library/blob/69a87329036cec0440403351c12230970c4e67d8/D00290/simready_usd/D00290.usd>`_
     - Destination
   * - Egg
     - `D00287 <https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library/blob/69a87329036cec0440403351c12230970c4e67d8/D00287/simready_usd/D00287.usd>`_
     - Pick-up object
   * - Nut
     - `D00554 <https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library/blob/69a87329036cec0440403351c12230970c4e67d8/D00554/simready_usd/D00554.usd>`_
     - Distractor
   * - Pizza
     - `D00802 <https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library/blob/69a87329036cec0440403351c12230970c4e67d8/D00802/simready_usd/D00802.usd>`_
     - Distractor

The YAML contains revision-pinned asset URLs. When building the environment,
``SimReadyUsdObject`` downloads each asset package and creates a cached PhysX
wrapper, requiring exactly one enabled rigid body and no articulation root.

CLI
---

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
       --mode build \
       --env_spec isaaclab_arena_environments/maple_table_top/tabletop_pick_place_egg_tray.yaml \
       --num_envs 1 --num_steps 20

GUI
---

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/gui_runner.py \
       --env_spec isaaclab_arena_environments/maple_table_top/tabletop_pick_place_egg_tray.yaml

The YAML opens in the editor. Use the visualization and simulation-preview
controls to inspect the scene.

Cache and reproducibility
-------------------------

Packages use the Hugging Face cache. Validated wrapper files are stored in
``~/.cache/isaaclab_arena/simready-central``. The checked-in YAML keeps the
pinned URLs, so it can be opened on another machine without editing cache paths.
If a Docker host cache points to an unmounted directory, set ``HF_HOME`` to a
writable, mounted location before running either interface.

This example uses PhysX. A zero-action run checks construction and stepping;
it does not establish successful egg placement by a policy.
