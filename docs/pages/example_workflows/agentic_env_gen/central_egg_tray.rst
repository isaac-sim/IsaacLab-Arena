SimReady Central: Egg and Tray
=============================

Generate a Franka pick-and-place environment using four single-rigid-body assets
from SimReady Central. The egg starts on the maple table; its destination is the
egg tray. A nut and pizza are distractors. Complete the shared
:ref:`agentic-env-gen-prerequisites` first.

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

The preset looks up these exact revisions in Central and requires indexed body
count equal to one. It then downloads each asset package and verifies the USD
contains one enabled rigid body and no articulation root. An unavailable or
incompatible pinned asset fails generation rather than selecting a substitute.

CLI
---

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
       --mode resolve --example central-egg-tray \
       --out_dir /tmp/central-egg-tray

Select a configured inference endpoint with ``--inference_endpoint`` if needed.
The runner prints the returned assets with their source URLs, local USD paths,
and ``used_in_spec`` flags. It saves the generated YAML and a sibling
``.assets.json`` report. Check that all four entries have ``used_in_spec: true``.
The LLM determines the final layout, so review the generated spec before use.

Build the generated environment and run a short zero-action check:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
       --mode build --env_spec /tmp/central-egg-tray/central_egg_tray.yaml \
       --num_envs 1 --num_steps 20

For a different pinned asset list, use ``--enable_simready_search`` with
``--simready_source simready-central`` and repeat ``--simready_asset_url URL``.
Both revision-pinned Hugging Face ``blob`` and ``resolve`` URLs are accepted.

GUI
---

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/gui_runner.py

Open the displayed browser URL, click **Load Central egg-tray example**, choose
your inference endpoint, and click **Generate spec**. The preset fills the prompt,
enables Central search, and populates the four pinned URLs.

The **Returned SimReady assets** table shows the source URLs and cached USD paths.
Use **Download asset report** to save the same report format as the CLI. The
generated YAML appears in the editor; the visualization panel renders the assets.
Use the simulation-preview controls to inspect the resulting scene.

To reopen the CLI result with its saved asset report:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/gui_runner.py \
       --env_spec /tmp/central-egg-tray/central_egg_tray.yaml

Cache and reproducibility
-------------------------

Packages use the Hugging Face cache, and the generated spec references local
wrapper files in ``~/.cache/isaaclab_arena/simready-central``. Keep those files
and the package cache available when reopening the YAML. Regenerate on a new
machine from the pinned preset rather than copying cache paths. If a Docker host
cache is a symlink to an unmounted directory, set ``HF_HOME`` to a writable,
mounted location before running either interface.

This example uses PhysX. A zero-action run checks construction and stepping; it
does not establish successful egg placement by a policy.
