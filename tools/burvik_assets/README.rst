BURVIK asset tools
=================

These source scripts generate and inspect the local BURVIK assets used by the
Arena assembly environment. USD files, meshes, textures, Blender scenes, videos,
and recorded trajectories stay outside version control. The committed
``artifacts/ikea_burvik/specification.json`` contains design dimensions and
estimated physical parameters, without mesh or texture payloads.

The default output directory is ``artifacts/ikea_burvik`` in this checkout.
``BURVIK_ASSET_DIR`` overrides it for the Blender and standalone simulation tools.
In Arena, configure ``BurvikAssemblyEnvironmentCfg.asset_directory`` and
``robot_usd_path`` to select other local asset locations. The imported YAM USD
and its payload layers must be provided separately.

Generate assets with Blender's Python runtime, which supplies ``bpy``,
``mathutils``, and ``pxr``::

    blender --background --python tools/burvik_assets/build_blender.py
    blender --background --python tools/burvik_assets/package_assets.py

``build_blender.py`` preserves the current scene, creates a separate scene with
six editable parts and explicit convex collision proxies, exports the individual
USDs, and writes the local Blender source. It calls ``recover_handle.py`` to
export triangulated handle geometry. ``package_assets.py`` checks the exported
schemas and writes an assembled preview with ideal part-to-part retention joints.
To run through Blender MCP, execute the script with ``__file__`` set to its
absolute filename so its sibling helper and output directory resolve correctly.

For an existing Blender source, ``triangulate_visuals.py`` refreshes visual
meshes while preserving authored collision and mass properties::

    blender --background artifacts/ikea_burvik/burvik_source.blend \
        --python tools/burvik_assets/triangulate_visuals.py

The historical standalone part-displacement demo and its output audit run inside
the checkout's Arena container, as the host user::

    /isaac-sim/python.sh tools/burvik_assets/simulate_assembly.py --video
    /isaac-sim/python.sh tools/burvik_assets/audit_delivery.py

``audit_arena_export.py RECORDING_DIRECTORY`` checks the exported USD and report
from the Arena part-displacement demo. The robot pose policy uses a different
trajectory format and writes its own validation report. Its runner, replay
renderer, control interface, and measured validation results are documented in
``isaaclab_arena_environments/burvik_assembly/README.rst``.

Part dimensions follow the supplied visual references. Mass, inertia, friction,
and connector clearances are estimates. Regenerating assets is separate from
rerunning the assembly validation; changed geometry must be validated again.

The worked example in
``docs/pages/example_workflows/burvik_asset_to_policy/index.rst`` includes the
original prompts, measured results, time/token accounting, and three visual
checkpoints. On the host, ``python tools/burvik_assets/make_workflow_media.py``
rebuilds its documentation PNGs and small GIFs from the original local recordings.
Only these documentation excerpts are committed; full recordings remain local.
