Offline Clutter Placement
=========================

Use ``ClutterOn`` to generate piles without choosing drop heights manually.
The offline generator saves settled poses for reproducible starting layouts.

Declare and generate clutter
----------------------------

In an environment YAML, mark the support as an anchor and add a clutter relation:

.. code-block:: yaml

   relations:
   - kind: is_anchor
     subject: table
   - kind: clutter_on
     subject: cube_0
     reference: table
     params:
       spread: 0.2
       random_yaw: true

Run the example inside the Arena container:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/generate_clutter_scene.py \
       env_spec=isaaclab_arena_environments/clutter/clutter_scene.yaml \
       output=outputs/clutter/episodes.jsonl num_envs=4 num_layouts=100 \
       seed=42 settle.timeout_s=15 --viz none

Generation settings use Hydra ``key=value`` overrides, including nested
``settle`` fields. Isaac Lab launcher options retain their ``--flag`` syntax.
Use ``presets=newton`` to select Newton; otherwise the environment selects the
physics backend. Custom assets can register through
``'register=[my_package:register_components]'``.

Python-defined environments use the same generator. After starting
``SimulationApp``, pass an ``IsaacLabArenaEnvironment`` containing ``ClutterOn``
relations:

.. code-block:: python

   from isaaclab_arena.scripts.generate_clutter_scene import ClutterGenerationCfg, generate_scene

   generate_scene(arena_env, ClutterGenerationCfg(output="outputs/clutter/episodes.jsonl"))

The command-line entry point loads the YAML into this same environment type.
Recording uses runtime scene keys, so Python environments need no graph node IDs.

How it works
------------

``ClutterOn`` → ``ObjectPlacer`` release poses → physics → rest and containment checks → JSONL.

Running the environment directly also produces release poses; objects drop when
physics starts. Use offline generation when initial poses must already be settled.

The generator accepts only releases that pass the source environment's
``placement_validators``. ``no_overlap`` and ``on_relation`` are always required.
For clutter, ``on_relation`` checks containment and minimum height, so an object
above the support passes without touching it. After the drop, the generator
checks rest, full-support containment and passive-body drift. It retries rejected
layouts and restores the original scene and robot targets on success or failure.
Output is written only after every requested layout passes; existing files are
never overwritten.

By default, the generator samples poses every 0.4 simulated seconds. A layout
is at rest when every recorded object moves at most 2 mm and rotates at most 2°
per sample interval for two consecutive intervals. Movement resets the quiet
count. A trial that does not settle within 10 simulated seconds is rejected.
Rest alone is insufficient: containment and passive-body checks must also pass.

The tool example uses three Robolab hammers and a clamp. These images show one
accepted layout before and after physics settling:

.. figure:: ../../../images/clutter/release.png
   :width: 640px
   :alt: Three hammers and a clamp suspended above a table before settling.

   Solver-generated release poses, before physics advances.

.. figure:: ../../../images/clutter/settled.png
   :width: 640px
   :alt: The same hammers and clamp resting on the table after settling.

   Settled poses accepted by the rest and containment checks.

Each JSONL line stores placement data using the episode variations envelope. The placement lives under
``variations["scene.relation_placement"]`` with a unique ``layout_id``,
``source: "settled"`` and ``poses`` keyed by runtime scene key (the asset instance name).
Positions are in metres in the local environment frame; quaternions use xyzw.
These are placement records, without episode outcomes or other run metadata.
Load them with ``--placement_layouts outputs/clutter/episodes.jsonl`` or set
``placement_layouts_path`` in the environment YAML. See :doc:`relations` for
replay selection and reset behavior. Remove any companion layout setting before
generating new clutter; generation requires the source relations.

Controls
--------

- ``spread``: fraction of each support dimension used for releases, centered on
  the support. For example, 0.2 uses the central 20% of width and depth.
  Settled containment uses the full support.
- ``clearance_m``: minimum release clearance above the support.
- ``gap_m``: initial gap between object bounds, increased to the solver's collision
  clearance when larger. Subsequent solving uses the solver's clearance.
- ``random_yaw``: sample world-Z yaw while preserving authored roll and pitch.
- ``num_envs`` / ``num_layouts``: parallel environments / total saved layouts.
  Omitting ``num_layouts`` saves one per environment.
- ``attempts`` / ``settle.timeout_s``: retry count / simulated seconds per trial.
- ``settle.move_thresh_m``, ``settle.turn_thresh_deg``,
  ``settle.required_quiet_windows``: motion thresholds and consecutive quiet polls
  required for rest. Lower thresholds or more quiet polls are stricter.
- ``settle.containment_margin_m`` / ``settle.fall_through_tolerance_m``: permitted
  overhang / penetration below the support. ``settle.passive_move_thresh_m`` and
  ``settle.passive_turn_thresh_deg`` limit support, neighbor and robot-link drift.

Limits
------

Supports must be fixed ``IsAnchor`` assets, static or kinematic in physics, with
untilted quarter-turn rotations. Clutter members must be dynamic rigid objects
with gravity enabled and ``ClutterOn`` as their only spatial relation; rotation
markers are allowed. Other placement must already be resolved to fixed anchors.
Object sets must be expanded to concrete objects. Concave supports need a fixed
reference for their usable surface. Robot joint configurations are not saved,
reachability is not certified, and pose-changing variations must be disabled for
reproducible generation.
