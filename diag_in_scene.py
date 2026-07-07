# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fast CPU repro of the box_can_in_bin scene placement solve (no SimulationApp).

Reproduces exactly what ArenaEnvBuilder does at build time: takes the objects-with-relations from the
graph spec and runs the placement solver, but without booting Isaac Sim (which is what makes the real
scene test take ~10 min). Prints geometry, the derived crate cavity, and per-check validation results
so we can see WHICH check fails and why.
"""

import copy
import pathlib

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import isaaclab_arena_environments
from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache


def _fmt(vals):
    return "[" + ", ".join(f"{v:.4f}" for v in vals) + "]"


def main():
    spec_path = str(
        pathlib.Path(isaaclab_arena_environments.__file__).parent / "robolab" / "box_can_in_bin_scene.yaml"
    )
    arena_env = ArenaEnvGraphSpec.from_yaml(spec_path).to_arena_env()
    objects = arena_env.scene.get_objects_with_relations()
    placer_params = arena_env.placer_params

    print("=" * 80)
    print("OBJECTS + GEOMETRY (local bbox)")
    print("=" * 80)
    for o in objects:
        bb = o.get_bounding_box()
        lo = bb.min_point[0].tolist()
        hi = bb.max_point[0].tolist()
        ext = [hi[i] - lo[i] for i in range(3)]
        rels = [type(r).__name__ + (f"->{r.parent.name}" if hasattr(r, "parent") else "") for r in o.get_relations()]
        print(f"{o.name:24s} ext={_fmt(ext)} lo={_fmt(lo)} hi={_fmt(hi)}")
        print(f"{'':24s} rels={rels} pose={o.get_initial_pose()}")

    crate = next(o for o in objects if o.name == "crate")
    mm = WarpMeshAndSphereCache(device="cpu")
    cm = mm.get_collision_mesh(crate)
    print("\ncrate collision mesh:", "None" if cm is None else f"watertight={cm.is_watertight} bounds={cm.bounds.tolist()}")
    cav = mm._derive_cavity_trimesh(crate)
    if cav is None:
        print("crate cavity: None")
    else:
        print(f"crate cavity: watertight={cav.is_watertight} bounds={cav.bounds.tolist()} extents={cav.extents.tolist()}")

    print("\n" + "=" * 80)
    print("PLACEMENT SOLVE")
    print("=" * 80)
    pp = copy.copy(placer_params)
    pp.apply_positions_to_objects = False
    pp.verbose = True
    pp.solver_params = copy.copy(pp.solver_params)
    pp.solver_params.verbose = True
    print("clearance_m:", pp.solver_params.clearance_m)
    placer = ObjectPlacer(params=pp)
    results = placer.place(objects, num_envs=1)

    for r in results:
        print("\nsuccess:", r.success)
        print("failed checks:", r.validation_results.get_failed_validation_check_names)
        crate_pos = r.positions[crate]
        print("crate pos:", _fmt(crate_pos))
        for o in objects:
            pos = r.positions.get(o)
            if pos is None:
                print(f"  {o.name:24s} -> None")
                continue
            rel = [pos[i] - crate_pos[i] for i in range(3)]
            print(f"  {o.name:24s} -> {_fmt(pos)}  (rel to crate {_fmt(rel)})")

    _probe_pairs(results[0], objects, placer)


def _probe_pairs(result, objects, placer):
    """For the final layout, print sphere-vs-mesh penetration depth for every non-skip pair."""
    import torch

    from isaaclab_arena.relations.warp_sdf_kernels import clamp_sdf_sentinel, mesh_sdf

    positions = result.positions
    orientations = result.orientations
    mm = placer._get_cpu_mesh_manager()
    print("\n" + "=" * 80)
    print("PAIR PENETRATION PROBE (final layout) — negative margin => overlap")
    print("=" * 80)
    for a, b in placer._non_skip_pairs(positions):
        a_mesh = mm.get_collision_mesh(a)
        b_mesh = mm.get_collision_mesh(b)
        if a_mesh is None or b_mesh is None:
            print(f"  {a.name} vs {b.name}: (no mesh, AABB fallback)")
            continue
        worst = None
        for src, src_mesh, tgt, tgt_mesh in [(a, a_mesh, b, b_mesh), (b, b_mesh, a, a_mesh)]:
            spheres = mm.get_query_spheres(src_mesh, obj=src)
            warp_mesh = mm.get_warp_mesh(tgt_mesh, obj=tgt)
            src_pos = torch.tensor(positions[src], dtype=torch.float32)
            tgt_pos = torch.tensor(positions[tgt], dtype=torch.float32)
            centers = placer._centers_in_target_frame(spheres[:, :3], src, tgt, src_pos, tgt_pos, orientations)
            sdf = clamp_sdf_sentinel(mesh_sdf(centers, warp_mesh))
            margin = (sdf - spheres[:, 3]).min().item()  # <0 => a sphere penetrates target mesh
            worst = margin if worst is None else min(worst, margin)
        flag = "  <-- OVERLAP" if worst < 0 else ""
        print(f"  {a.name:22s} vs {b.name:22s} worst_margin={worst:+.4f}{flag}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
