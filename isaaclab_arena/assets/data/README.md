# Fixed deformable cube topology

These geometry-only USDs freeze the simulation node ordering and tetrahedron
connectivity used by the deformable pick-and-place example. Keep them unchanged
for existing demonstrations; a topology change requires a new asset and new recordings.

Both were baked from the previous `MeshCuboidCfg(size=(0.15, 0.04, 0.04))`
definitions using Isaac Lab `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`
and `isaaclab_arena/scripts/bake_deformable_mesh.py`:

| Asset | Original edge refinement | Simulation nodes |
| --- | --- | --- |
| `deformable_cube_newton.usda` | 8 | 69 |
| `deformable_cube_physx.usda` | 4 (default) | 21 |

The generated files contain one visual Mesh and one simulation TetMesh, in meters.
Backend properties and materials remain in `deformable_object_library.py`.
The meshing operation is nondeterministic: regenerating from the same settings
does not reproduce these files. The library now loads the baked files directly.

See the Deformable Pick and Place workflow documentation for recording, replay,
and baking another registered volume deformable.
