# Prepare workbench USD assets

These commands rebuild the assets used by the example checkout's
`scene_bin_shelf.yaml`. They edit an anonymous copy of each source USD,
then export a flattened package with local material/texture dependencies.
The source assets and existing output packages are never overwritten.

Run Python commands inside this checkout's Arena container, as your host user,
using `isaaclab.sh -p`. Mount the dataset directory with
`./docker/run_docker.sh -d /path/to/datasets` before running them.
USD processing requires the runtime's `pxr`, NumPy, SciPy, PyYAML and
Hugging Face libraries; it does not start the simulator.

## Generate assets

From `/workspaces/isaaclab_arena` inside the container:

```bash
SCRIPTS=isaaclab_arena_environments/workbench_cleanup/scripts
OUTPUT=/datasets/workbench/my_new_asset_build
isaaclab.sh -p "$SCRIPTS/prepare_assets.py" \
  --spec "$SCRIPTS"/specs/*.yaml --output-dir "$OUTPUT/assets"
```

Each YAML is the complete specification for one output asset. To generate just
one asset, pass just that file. There is no shared asset specification to edit.
An output package contains its USD, dependencies, copied recipe, and
`manifest.json` with source-layer hashes, recipe/output hashes, operation
results, and validation results. Existing package directories cause an error;
choose a new output directory for a new build. Earlier successful packages in
a multi-asset invocation remain available if a later recipe fails.

| Recipe | Output and purpose |
| --- | --- |
| `workbench.yaml` | Separate tabletop, shelf, board and four blue containers; preserve independent drawer and door articulations. |
| `oscilloscope_button.yaml` | Keep the power button movable, weld the other controls, fix the base, set the button spring and create the screen-off material. |
| `socket_set_open.yaml` | Empty case with the lid driven toward 105 degrees. |
| `slotted_screwdriver.yaml` | Preserve tool geometry and configure rigid-body damping/contact properties. |
| `small_wrench.yaml` | Create a 0.75-scale combination wrench. |
| `white_bin.yaml` | Package the PalatialSim bin with a white material. |

The workbench, oscilloscope and hand tools come from revision-pinned NVIDIA
SimReady Central packages. The case comes from the pinned ForgeCAD Central
package used by the reference scene. The white bin uses the pinned PalatialSim
`palatial-nv-asset-library` package for `D00814`. Original layers are loaded from
the Hugging Face cache, or downloaded as a complete package when absent.
The reference `/datasets/workbench/assets/D00814.usd` is byte-for-byte identical
to that pinned PalatialSim USD. No recipe reads from the reference asset folder;
that folder is used only for optional validation.
Source asset licenses still apply to generated packages; the repository's
script license does not replace them.

## Inspect and author another asset

Start with one YAML file:

```yaml
asset:
  id: my_tool
  usd_path: /datasets/originals/my_tool.usd
  variants:
    Physics: physics
    PhysX: Enabled
output: my_tool.usdc
operations:
  - kind: edit_prims
    params:
      select:
        schema: PhysicsRigidBodyAPI
      apply_apis: [PhysxRigidBodyAPI]
      attributes:
        physxRigidBody:linearDamping: {type: float, value: 2.0}
validation:
  required_prims: [Geometry]
```

`usd_path` accepts a local USD (relative to its YAML, or absolute) or a pinned
Central URL. The output default prim is `/Asset`; operation prim paths are
relative to that root, and `.` means the root itself. Unknown operations,
unknown fields, missing selections, and missing required prims fail the build.

Supported ordered operations:

| `kind` | `params` |
| --- | --- |
| `edit_prims` | `select`, optional `attributes`, `apply_apis`, `remove_apis`, `type_name`, `active`. Select by root-relative `paths` globs, `exclude` globs, applied `schema`, or exact prim `type`. Globs can match descendants across `/`. |
| `partition_mesh` | `prim_path`, `groups: {part_name: [component_ids]}`, optional `remainder` and `geometry_sha256`. Preserve topology, UVs, normals, material subsets and collider settings. |
| `fixed_articulations` | `groups: {name: {frame_mesh: path, joints: [paths]}}`. Move a partitioned frame and its attached bodies into separate fixed-base articulations while preserving geometry and joint frames. |
| `configure_articulation` | Optional `root`, `fixed_base_body`, `movable_joints`, `self_collisions`. Omitting `movable_joints` keeps all joints; supplying it welds every other joint under the root. |
| `preview_material` | `prim_path`, typed `inputs`, optional `bind_to: [paths]`. Create a UsdPreviewSurface material; bindings override descendant materials. |
| `bake_scale` | `scale: [x, y, z]`. Put scale below the spawn root under `BakedScale`, preserving it when the scene replaces root transforms. Descendant reference paths gain that prefix. |

Attribute/input values use `{type: USD_type_name, value: YAML_value}`. Vectors,
matrices and arrays are supported; quaternions use `[w, x, y, z]`. Values use
USD units: distances use the source stage's metres-per-unit; angular limits and
drive targets are degrees, and angular stiffness/damping are per degree.
Scene joint positions remain SI units (metres and radians).

Mesh partitioning requires static polygon geometry and splits complete connected
components; it does not cut connected surfaces. Fixed articulation extraction
supports independent prismatic/revolute joints connected to one frame per group,
not chains or scaled/sheared/animated body frames. It rejects unsupported layouts.
Generic attribute and material edits can be used independently for assets that
do not need those operations.

## Tests

Keep test outputs outside the repository:

```bash
isaaclab.sh -p -m pytest -q "$SCRIPTS/tests" \
  --basetemp="$OUTPUT/tests/pytest" --junitxml="$OUTPUT/tests/pytest.xml"
isaaclab.sh -p "$SCRIPTS/tests/smoke_workbench_assets.py" \
  --assets-dir "$OUTPUT/assets" --output "$OUTPUT/tests/physics.json" \
  --device cpu --viz none
```

The focused pytest tests use synthetic assets. The separate physics smoke test
loads all six generated packages in Isaac Sim PhysX, verifies the four
articulations, checks drawer displacement at the scene's 1.35 scale, and tests
button spring return and the open-case lid drive over 480 steps. Gravity is
disabled to isolate articulation behavior; this is not a placement or policy
evaluation. `--physics ovphysx` selects the alternative PhysX backend.
