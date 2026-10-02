# Blender source for Return to Service

This directory contains the original parametric geometry for the vacuum service
cell. All nonrobot geometry, texture images, visual labels and collision shapes
are authored in Blender. The Arena scene reuses the existing DROID embodiment.
No downloaded models, product logos or third-party textures are required.

## Generate the assets

Use Blender 4.5 LTS with its bundled NumPy and USD Python modules. The script
loads only the local Blender tooling; it does not import the Arena environment
registry or require Isaac Sim inside Blender.

```bash
blender --background --python \
  isaaclab_arena_environments/return_to_service/asset_source/build_assets.py \
  -- --output-dir ~/.cache/isaaclab_arena/return_to_service/assets --render
```

For interactive authoring through Blender MCP, execute the same entry point:

```python
import importlib
import runpy
import sys

# Reload source edits when this Blender session already ran an earlier build.
for module_name in (
    "asset_source.uv",
    "asset_source.geometry",
    "asset_source.case_geometry",
    "asset_source.components",
    "asset_source.fits",
    "asset_source.validate_exports",
):
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

builder = runpy.run_path(
    "/absolute/path/to/IsaacLab-Arena/isaaclab_arena_environments/"
    "return_to_service/asset_source/build_assets.py",
    run_name="return_to_service_builder",
)
builder["build"]("/absolute/path/to/generated/assets", render=True)
```

Each invocation creates a separate scene. Existing scenes and unsaved user work
are retained. Only the new scene is written to `return_to_service.blend`; the
current Blender file is never overwritten. Repeated builds are supported even
when earlier scenes remain open. Generated models and textures belong outside
the repository and should be distributed as a versioned asset bundle.

Blender adds numeric suffixes to object names when an earlier scene contains
the same names. The manifest records the actual exported paths, including
instrument meshes; consumers must use those paths instead of hardcoding
Blender names. Geometry and seeded textures reproduce across builds, while
the exact internal prim names can differ in a populated interactive session.
Use a fresh background process for a release bundle with consistent names.

## Files and responsibilities

| Source | Responsibility |
| --- | --- |
| `geometry.py` | Rounded manufacturing primitives, original seeded PBR maps, UV completion, mesh lettering, hollow rings, USD export and manifest serialization. |
| `uv.py` | Shared, scale-aware tolerances for degenerate surface and texture-coordinate areas. |
| `components.py` | Dimensioned part designs and their local assembly, grasp, instrument and placement coordinates. |
| `case_geometry.py` | Case base, lid and passive latch geometry, including mounting hardware and local grasp frames. |
| `fits.py` | Measurements of exported guide and extraction clearances, axial seating contact, support heights and footprints, case packing, and continuous hinged travel. |
| `build_assets.py` | Isolated scene creation, complete bundle generation, studio lighting and an illustrative preview arrangement. |
| `validate_exports.py` | USD topology, units, default prim, collision, texture-resolution, UV-coordinate and instrument-visibility contract checks. |

The output includes one USD per rigid part, `lighting.usda`, `manifest.json`,
`validation_report.json`, `LICENSE.md`, texture PNGs and a standalone `.blend` scene. `--render`
also creates `workcell_preview.png`. The preview is an asset-authoring scene;
the Arena scene owns the evaluation layout, DROID placement, joints and sensors.

The builder copies Arena's license into the output. It first checks adjacent
Arena `.dist-info` metadata, then the repository root after confirming the
project identity in `pyproject.toml`. When running the absolute script path
from an installed wheel, retain its metadata directory. This lookup does not
import Arena. The checkout command above remains the primary asset-generation
workflow.

## Coordinate and physics contract

Every USD has default prim `/Asset`, meter units and Z up. Parts are modeled in
their own local frames, with bounds and suggested mass in the manifest. The
bench origin lies on the work surface. Most loose parts have their underside
near Z zero; the manifest records small rim or foot extensions explicitly.
The vacuum's longitudinal axis is +X. Its removable cup and filter extract
along +X, while the battery extracts along -X.

The motor body supports the filter on three axial guide rails. Removing the
cup exposes the filter's grip tab while the filter remains supported; the
guides allow the cup to slide away for emptying. Their nominal side and bottom
clearances are 3 mm and 0.5 mm respectively. Battery guides have 3 mm side
clearance in the vacuum body and 4 mm in the tester. The cup has a 37 mm bore
radius and a 4 mm wall. Its bore clears the retained filter, grip tab and guide
rails by at least 3 mm throughout axial extraction. A broad annular body
shoulder contacts the cup's rear lip at the nominal socket, providing a real
mechanical seating surface. Both debris-cavity metadata representations follow
the bore dimensions. Flat skids beneath the cup and both filter variants keep
the cylindrical parts from freely rolling on their supports. Their undersides
remain level with the existing cup lip and filter end caps. The cradle's passive
rail supports the cup at its nominal socket height and ends 8 mm behind the
cup after a 100 mm extraction, leaving room for a subsequent orientation
correction. Two 1 mm high kerbs along the lower filter guide form a 20 mm wide
channel around the 18 mm skid. They provide 1 mm lateral clearance per side
and 0.5 mm vertical engagement at the authored socket, increasing to 1 mm
after the filter settles. The short kerbs clear the filter's complete ring
caps during axial travel. Tall kerbs would obstruct those caps as they pass.
Lateral and yaw tolerance are coupled; the controller must align the filter
with the body axis before extracting or inserting it. These shallow guides
allow axial travel and are not vertical retention: lift or roll can disengage
them, so released-filter stability still requires physical validation.
Both filter variants declare `support_footprint_bounds` matching their actual
skid collision boxes, including the incompatible variant's lower underside.
The body's `filter_capture_bounds` defines a volume inside the physical guide.
Runtime seating requires the complete transformed skid to fit this volume,
in addition to socket position, orientation and model compatibility. The
capture sides remain 0.1 mm inside the kerbs and its ends 0.5 mm inside the
support. Its lower plane allows 50 micrometers of seating penetration; its
upper plane preserves at least 0.28 mm of vertical engagement under the
runtime's five-degree orientation gate. These bounds describe capture of a
seated filter, not an arbitrary extraction corridor.
These features add ordinary collision geometry, with no retention force,
constraint or mass change. Brush tufts have physical collision supports; the crevice
tool and nozzle have rubber neck rests level
with their connector undersides. These supports prevent the accessories from
pitching on a flat tray or case floor. The airflow adapter is a hollow U bridge whose two male
stems face -X and enter parallel cup/tester sockets together. Extract it along
+X before lifting; perpendicular or opposed sockets would trap a rigid bridge.

The adapter publishes `straight_stem_x_range = [-0.010, 0.020]` and
`tube_outer_radius = 0.0105` under its affordances. The range is the displacement
along adapter +X from each connector origin (`cup_connector` or
`tester_connector`) to the two endpoints of that connector's straight stem.
It describes geometry independently of the nominal `insertion_depth`.
Both receiving parts publish `airflow_port_inner_radius = 0.012`: on the cup
it belongs to `airflow_adapter_socket`, and on the tester to `airflow_port`.
All four fields use meters and share their values with the primitive builders.

The removable inlet obstruction has a 30 mm plug supported inside the inlet
and an exterior pull tab. Export validation checks both shapes against every
cup collider, their connection, and the estimated center of mass relative to
the supported interval. The nominal radial clearance is 1.5 mm and the tab
clears the inlet end by 2 mm. Contact and friction support the loose part;
simulation tests check that its fault assignment survives settling and resets.

For each receiving port, the two transformed stem-centerline endpoints bound
the straight segment's radial offset by convexity. Its solid radial envelope
is that bound plus `tube_outer_radius`. The nominal radial clearance is
1.5 mm; a controller must retain a positive margin within the declared port
radius. This local geometric check supplements the full adapter and gripper
clearance checks; it does not certify arbitrary tilted sweeps or tracking.

The case lid is exceptional: its origin is the hinge pivot. The closed pose,
hinge axis, opening angle and latch origin are recorded under the case-base
affordances. The case is intentionally tall enough to contain the vacuum's
handle. Keep its full opening sweep clear when changing the workcell layout.
`case_base.affordances.lid_handle_grasp` is expressed in the `case_lid` link
frame and describes the physical pull-bar center.
`case_base.affordances.latch_grasp` is expressed in the `case_latch` pivot
frame and describes the configured DROID TCP directly. Each carries an
explicit `frame` tag. These two position fields have different meanings;
apply the documented pull-bar offset only to the lid grasp.
The lid has an offset exterior pull bar with two mounting arms. Its authored
grasp frame points +Z outward along lid -Y and puts the jaw axis along lid +X.
Approaching from the front keeps the palm outside the falling lid's sweep;
the offset along X leaves the central latch's approach clear. For DROID, place
the TCP 25 mm outward from the physical bar center,
at `[0.160, -0.330, 0.075]` in the lid frame, with tool quaternion
`[sqrt(0.5), sqrt(0.5), 0, 0]` in XYZW order. Its 40 mm approach follows
the negative tool X direction. Full-object extents, including the pull bar,
come from the exported `bounds_min` and `bounds_max` fields.

The latch pivot is at `[0, -0.220, 0.140]` in the case-base frame. Two exterior
folded-steel supports leave the central lever clear through its 100-degree
travel. The 45 gram latch has a friction-rivet hinge, visible washers,
a raised hook and an opposed 46 mm gripping flange. Its closed hook captures
the raised lid catch with a 0.5 mm vertical gap. The manifest records
static and dynamic resisting torques of 0.012 and 0.010 N m under
`case_base.affordances.latch`; the runtime authors these as angular joint-axis
friction efforts. They are passive resistance, with no target-angle spring or
motor. The runtime's separate retaining constraint approximates the lid catch.
Export validation checks hook capture and continuous latch/lid sweeps, using
sampled box separation minus a bound on all intervening vertex motion.

The latch TCP is approximately `[0, -0.01749425, 0.11801185]` in the pivot
frame, with tool quaternion `[0.43045933, 0.43045933, -0.56098553, 0.56098553]`
in XYZW order. `contact_center_xyz` separately records the physical flange
center `[0, -0.025, 0.090]`, 29 mm along tool +X from the TCP. The tool frame
includes a 15-degree tilt to clear the hook. Apply this quaternion directly
as the tool orientation relative to the latch; it is already the configured
DROID tool frame. `approach_offset_case_xyz` is the exception to the grasp's
local frame: `[0, 0, 0.100]` is expressed in the **case-base frame**, placing
the approach 100 mm above the target. The released-hand retreat follows
negative tool X. A controller must preserve these frame distinctions when
composing world poses or adapting the grasp to another embodiment.

Visual meshes contain manufactured edge radii, hollow shells, filter pleats,
keyed rails, grip ribs, terminals, fasteners, labels and material variation.
All texture maps are deterministic images authored in Blender. Their node
graphs use USD-compatible image-based base color and roughness inputs.
`Geometry.finish` preserves complete authored UV maps and projects every face
when any nondegenerate face lacks mapped area. Whole-mesh projection is safe
for these uniform microstructure textures, including text sides and bevels.
Custom meshes, including filter pleats, pass through the same finish step as
built-in primitives.

Invisible USD boxes and cylinders carry `UsdPhysics.CollisionAPI`. Hollow
chambers use segmented wall proxies; they are never replaced by a solid convex
hull across their opening. Visual geometry is separate from collision geometry.
No source asset carries `RigidBodyAPI`: the runtime selects static, kinematic
and articulated ownership. Loose-part grasp tabs use 20–32 mm opposed pinch faces, within
the existing Robotiq gripper's 85 mm opening.

Loose-part `grasp` positions target DROID's configured end-effector point. The
grasp frame has +Z pointing outward from the contact surface and Y along the
jaw closing direction. Align DROID's +X approach axis with the grasp frame's
-Z axis. Positions account for the closed fingers
extending beyond their end-effector point, allowing pad contact while clearing
surrounding housings and tray rims. Recheck this clearance when adapting the
assets to another gripper.

Manifest affordances include assembly sockets, grasp positions, stock and tool
parking poses, case packing poses and cavity bounds. A placement may be an XYZ
list, denoting identity orientation, or a dictionary containing `position_xyz`
and `rotation_xyzw`. Both are expressed in the owning asset's local frame.
The compatible spare filter is rotated +90 degrees about the rack's Z axis;
its orientation is part of `spare_rack.affordances.stock_poses.filter_spare`.
All tray `interior_bounds` start at the physical floor's upper surface,
6 mm above the tray origin. Loose-part starting heights allow settling onto
that surface. The preview places the parking tray at world XY `[0.310, 0.005]`,
matching the evaluation scene's selected layout. These are part of the
asset contract, not policy observations. Work-order and part labels provide
visible model-compatibility information; hidden component condition belongs
to the task's functional model.

Diagnostic labels are actual Blender-authored meshes. Each tester advertises
`status_paths` for `idle`, `run`, `pass`, `fail` and `invalid`. Initially only
`idle` is visible. The runtime changes visibility on these existing prims;
it does not generate replacement visual geometry. The displayed nominal
readings represent the documented functional model, not a fluid or battery
electrochemistry simulation.

Numeric meshes are separately indexed by `reading_paths`, with keys formatted
using Python's `g` format (for example `"20"`, `"13.5"`, `"3.465"`). They cover
all discrete voltages and airflow values produced by the supplied fault model.
Initially every numeric mesh is hidden. Runtime code selects the measured
value independently of PASS/FAIL and hides the number when no valid reading is
available. Adding a new functional-model reading requires regenerating the
corresponding authored numeric mesh.

## Validation boundaries

Every complete build validates that meshes have finite vertices and consistent
topology, all referenced textures resolve, and each image material's UV reader
resolves to finite coordinates with the correct interpolation and nonzero
mapped area on every nondegenerate face. The surface-area cutoff is float32
machine precision times each mesh's squared bounding-box diagonal, with an
absolute floor of `1e-16` square meters; this excludes numerical bevel slivers
while retaining visible label sides. Reports include excluded face counts
and their total surface area. The build also checks that collision shapes
are present and hidden, units and root paths agree, and status visibility is
initialized correctly. Per-asset reports count the textured meshes validated.
Collision primitives retain their manufacturing feature names under
`arena:feature`. Cross-part fit checks use their actual transformed USD bounds
and the manifest's sockets: tester/body battery channels, filter guides, the
incompatible filter's exclusion, unchanged support heights and coplanar tool
supports. Flat-skid checks require full horizontal support, the designed
vertical clearance, and no change to the existing part underside. Continuous
cup extraction permits sliding contact with the cradle, while its endpoint
requires at least 6 mm clearance before lifting or correcting orientation.
Oriented-box separation also checks the filter and its guides against
both cup-wall rings over a continuous 100 mm extraction; their hollow geometry
cannot be represented by a single axis-aligned bound. The axial seating check
requires coincident shoulder/lip planes and transverse overlap for every lip
segment, so missing physical seating contact cannot pass on position alone.
Nominal loose-part packing checks use exported visual bounds, with
at least 8 mm between neighboring items and 10 mm from brush to case side.
The report records measured clearances and required minima. These are local
assembly checks, not a complete swept-path or contact-stability proof.
The keyed filter guide checks every simultaneous axial, lateral and vertical
translation over 75 mm extraction, the full channel slack and the 0.5 mm
settlement interval. The skid may contact the kerbs; every other filter
feature must retain at least 0.5 mm clearance, and the extracted filter must
clear the kerbs by at least 10 mm. The cup also clears them by at least 5 mm
through every combination of 0–3 mm prelift and 0–100 mm axial extraction.
These translation-volume certificates retain each collision box's orientation;
they do not certify tilted filter poses or arbitrary gripper trajectories.
The airflow display is recessed from the cup's extraction corridor. A continuous
100 mm translation check requires at least 5 mm clearance between the cup's
collision shapes and display housing. Tester placement for this check follows
the two bridge connectors and port metadata, independently of the Arena scene.
These checks do not establish robot reachability, contact stability,
grasp success or task solvability; those require the Arena runtime checks.

The assets and their source use the repository's Apache-2.0 license.
