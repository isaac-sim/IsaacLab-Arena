# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Joint robot-edge + object-placement solver POC (no Isaac Sim).

Mirrors ``relation_solver_visualization_notebook.py`` in spirit: uses
``DummyObject`` and the existing relation loss strategies, but adds a
fourth-batch-element parameterization where each env represents the
robot standing at one of the four tabletop edges. The only learnable
robot DOF per env is a scalar ``fraction`` along that edge — yaw is a
constant cardinal rotation matrix per env. Objects' world XYZ are
optimized simultaneously, with three loss terms:

  * ``On(table)`` — keeps the objects on the tabletop (existing strategy).
  * ``Reach`` — a distance-to-reachable EDT sampled in robot-base frame
    (new; built once from ``tools/franka_reach_top_down.npz``).
  * Pairwise ``NoCollision`` — keeps objects from overlapping.

After Adam converges, the best of the four envs is selected by
per-env loss. No cuRobo, no Kit — pure torch + numpy + scipy.

Run:

    /isaac-sim/python.sh isaaclab_arena_examples/relations/joint_reach_solver_poc.py \\
        --reach_npz tools/franka_reach_top_down.npz \\
        --out_png tools/poc_joint_reach_solver.png
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Rectangle
from scipy.ndimage import distance_transform_edt

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.relation_loss_strategies import NoCollisionLossStrategy, OnLossStrategy
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


EDGE_NAMES = ("x_min", "x_max", "y_min", "y_max")


# ---------------------------------------------------------------------------
# Reachability field — built once from the npz, sampled per Adam iter.
# ---------------------------------------------------------------------------


@dataclass
class ReachField:
    """Distance-to-reachable EDT in robot-base frame.

    ``field[i, j, k]`` is the Euclidean distance (meters) from voxel
    ``(i, j, k)`` to the nearest reachable voxel. Zero everywhere
    inside the reachable set, positive outside.
    """

    field: torch.Tensor          # (Nx, Ny, Nz)
    grid_origin: torch.Tensor    # (3,) min corner in base frame (meters)
    voxel_size: torch.Tensor     # (3,) per-axis voxel spacing (meters)
    grid_max: torch.Tensor       # (3,) max corner in base frame (meters)

    @classmethod
    def from_npz(cls, path: Path, device: torch.device) -> "ReachField":
        npz = np.load(path)
        success = np.asarray(npz["success"], dtype=bool)
        x = np.asarray(npz["x"], dtype=np.float32)
        y = np.asarray(npz["y"], dtype=np.float32)
        z = np.asarray(npz["z"], dtype=np.float32)
        dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
        dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
        dz = float(z[1] - z[0]) if len(z) > 1 else 1.0
        # scipy's distance_transform_edt computes distance from non-zero
        # voxels to the nearest zero voxel. Pass ~success so zeros are
        # the reachable set; output is the distance field outside it.
        d_voxels = distance_transform_edt(~success, sampling=(dx, dy, dz))
        field = torch.tensor(d_voxels, dtype=torch.float32, device=device)
        grid_origin = torch.tensor([x[0], y[0], z[0]], dtype=torch.float32, device=device)
        voxel_size = torch.tensor([dx, dy, dz], dtype=torch.float32, device=device)
        grid_max = torch.tensor([x[-1], y[-1], z[-1]], dtype=torch.float32, device=device)
        return cls(field=field, grid_origin=grid_origin, voxel_size=voxel_size, grid_max=grid_max)

    def sample(self, p_local: torch.Tensor, cap_m: float = 0.5) -> torch.Tensor:
        """Trilinear EDT sample at ``p_local`` of shape ``(B, 3)``.

        Out-of-grid queries clamp to the grid edge and add a Euclidean
        penalty in meters so gradients still point back into the grid.
        Returns ``(B,)``.
        """
        Nx, Ny, Nz = self.field.shape
        max_idx = torch.tensor(
            [Nx - 1, Ny - 1, Nz - 1], dtype=p_local.dtype, device=p_local.device
        )
        idx_f = (p_local - self.grid_origin) / self.voxel_size
        idx_clamped = idx_f.clamp(min=torch.zeros_like(max_idx), max=max_idx)
        oob_dist = ((idx_f - idx_clamped) * self.voxel_size).norm(dim=-1)

        i0 = idx_clamped.floor().long()
        i1 = (i0 + 1).clamp(max=max_idx.long())
        frac = idx_clamped - i0.float()
        f000 = self.field[i0[:, 0], i0[:, 1], i0[:, 2]]
        f100 = self.field[i1[:, 0], i0[:, 1], i0[:, 2]]
        f010 = self.field[i0[:, 0], i1[:, 1], i0[:, 2]]
        f110 = self.field[i1[:, 0], i1[:, 1], i0[:, 2]]
        f001 = self.field[i0[:, 0], i0[:, 1], i1[:, 2]]
        f101 = self.field[i1[:, 0], i0[:, 1], i1[:, 2]]
        f011 = self.field[i0[:, 0], i1[:, 1], i1[:, 2]]
        f111 = self.field[i1[:, 0], i1[:, 1], i1[:, 2]]
        fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
        c00 = f000 * (1 - fx) + f100 * fx
        c01 = f001 * (1 - fx) + f101 * fx
        c10 = f010 * (1 - fx) + f110 * fx
        c11 = f011 * (1 - fx) + f111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        d_in_grid = c0 * (1 - fz) + c1 * fz
        return (d_in_grid + oob_dist).clamp(max=cap_m)

    def slice_xy(self, z_local: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Top-down EDT slice at a fixed local z, for plotting."""
        z_idx_f = float((z_local - self.grid_origin[2].item()) / self.voxel_size[2].item())
        Nz = self.field.shape[2]
        z_idx = max(0, min(Nz - 1, int(round(z_idx_f))))
        slice_xy = self.field[:, :, z_idx].cpu().numpy()
        x_axis = np.linspace(
            self.grid_origin[0].item(), self.grid_max[0].item(), self.field.shape[0]
        )
        y_axis = np.linspace(
            self.grid_origin[1].item(), self.grid_max[1].item(), self.field.shape[1]
        )
        return x_axis, y_axis, slice_xy


# ---------------------------------------------------------------------------
# Edge constants (one row per cardinal edge of the rectangular table).
# ---------------------------------------------------------------------------


def edge_constants(
    table_xy_min: tuple[float, float],
    table_xy_max: tuple[float, float],
    device: torch.device,
):
    """Return per-edge constants used to map fraction -> world base XY.

    Heading conventions match ``_EDGE_ROTATION_XYZW`` in
    ``placement_proposer.py`` — the robot's +X (forward) points across
    the table toward the centre.
    """
    x0, y0 = table_xy_min
    x1, y1 = table_xy_max
    # 2D rotation that maps base-frame XY into world-frame XY.
    # Inverse (world->base) is the transpose, used by the reach loss.
    R_x_min = torch.tensor([[1.0, 0.0], [0.0, 1.0]])           # heading +x
    R_x_max = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])         # heading -x
    R_y_min = torch.tensor([[0.0, -1.0], [1.0, 0.0]])          # heading +y
    R_y_max = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])          # heading -y
    R_edge = torch.stack([R_x_min, R_x_max, R_y_min, R_y_max], dim=0).to(device)

    corner_a = torch.tensor(
        [
            [x0, y0],  # x_min: along y from y0 -> y1
            [x1, y0],  # x_max: along y
            [x0, y0],  # y_min: along x
            [x0, y1],  # y_max: along x
        ],
        dtype=torch.float32,
        device=device,
    )
    corner_b = torch.tensor(
        [
            [x0, y1],
            [x1, y1],
            [x1, y0],
            [x1, y1],
        ],
        dtype=torch.float32,
        device=device,
    )
    n_out = torch.tensor(
        [
            [-1.0, 0.0],
            [+1.0, 0.0],
            [0.0, -1.0],
            [0.0, +1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    return R_edge, corner_a, corner_b, n_out


# ---------------------------------------------------------------------------
# Joint solver — one Adam loop, batched across 4 edges.
# ---------------------------------------------------------------------------


@dataclass
class SolveResult:
    best_idx: int
    loss_history: np.ndarray              # (n_iter, B)
    final_positions: np.ndarray           # (B, n_obj, 3) world XYZ
    base_xy: np.ndarray                   # (B, 2)
    fraction: np.ndarray                  # (B,)
    offset_m: np.ndarray                  # (B,)
    object_names: list[str]


def joint_solve(
    table_anchor: DummyObject,
    objects: list[DummyObject],          # all DummyObjects, including table
    reach_targets: list[DummyObject],    # subset that gets the reach loss
    on_targets: dict[DummyObject, DummyObject],  # child -> parent for On(...) losses
    reach_field: ReachField,
    table_xy_min: tuple[float, float],
    table_xy_max: tuple[float, float],
    offset_range: tuple[float, float] = (0.05, 0.40),
    base_z: float = 0.0,
    fraction_range: tuple[float, float] = (0.30, 0.70),
    cap_m: float = 0.5,
    reach_weight: float = 50.0,
    # NOTE: robot_footprint_half models the *Franka panda's static base at
    # object height*, not the full Franka-on-stand AABB. With this POC the
    # robot's panda_link0 origin sits at world z=0 and objects sit at
    # z ≈ 0.05 m on the table, so:
    #
    #   * The stand below z=0 (extending down to the floor) is irrelevant —
    #     it cannot overlap any on-table object.
    #   * The arm sweep above z ≈ 0.30 m is also irrelevant for this POC's
    #     soft-collision purpose; modeling it would need a separate
    #     forward-extending AABB that tracks where the EE will pass over
    #     the table during a top-down grasp (future work).
    #
    # What's left at object height is panda_link0 — the cylindrical mounting
    # plate (~0.22 m diameter, ~0.13 m tall) plus a small slack for joint 1.
    # AABB half-extents for that volume: (0.12, 0.12, 0.15), giving a
    # 24 × 24 × 30 cm static collider centered on the panda base.
    #
    # The exact stand+panda bbox is in the USD at
    # {ISAACLAB_NUCLEUS_DIR}/Arena/assets/robot_library/
    # franka_panda_hand_on_stand.usd (referenced by
    # embodiments/franka/franka.py:50). I attempted to query it via
    # AppLauncher + compute_aabb(...) but the docker session segfaulted
    # at sim startup; the production path should derive these numbers
    # programmatically at scene-build time, similar to how
    # _BACKGROUND_TABLETOP_ANCHOR reads table bboxes via
    # get_world_bounding_box().
    robot_footprint_half: tuple[float, float, float] = (0.12, 0.12, 0.15),
    robot_clearance_m: float = 0.02,
    obj_obj_collision_slope: float = 10000.0,
    # NOTE: the robot-vs-object NoCollision is intentionally a SOFT term
    # (slope ≪ obj_obj's 10000). It's a tie-breaker, not a hard constraint:
    # when the reach loss prefers an object position that happens to clip the
    # robot footprint, we'd rather accept a small overlap than push the
    # object outside the reachable workspace. With the current Franka-on-
    # stand geometry the term almost never actually engages — the stand sits
    # OUTSIDE the table by `offset_m`, so robot-vs-object overlap is
    # geometrically near-impossible for objects that satisfy On(table). The
    # term is wired in defensively for: (a) objects that drift off the table
    # edge during optimization, (b) future work modeling the arm sweep
    # volume as an additional forward-extending AABB at object height —
    # that's where soft-vs-hard actually distinguishes solutions.
    robot_obj_collision_slope: float = 200.0,
    init_jitter_xy: tuple[float, float] = (0.30, 0.20),
    n_iter: int = 600,
    lr: float = 0.01,
    robot_lr_mult: float = 5.0,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
) -> SolveResult:
    torch.manual_seed(seed)
    R_edge, corner_a, corner_b, n_out = edge_constants(table_xy_min, table_xy_max, device)
    R_inv = R_edge.transpose(1, 2)
    B = R_edge.shape[0]
    frac_min, frac_max = fraction_range
    off_min, off_max = offset_range

    # Build per-env initial positions (jittered for diversity).
    optimizable_objs = [o for o in objects if o is not table_anchor]
    n_obj = len(objects)

    init_pos = torch.zeros(B, n_obj, 3, dtype=torch.float32, device=device)
    obj_idx = {o: i for i, o in enumerate(objects)}
    for o in objects:
        p = o.get_initial_pose().position_xyz
        init_pos[:, obj_idx[o], :] = torch.tensor(p, dtype=torch.float32, device=device)
    # Stratified-uniform jitter on the optimizable starts so each env explores
    # a different region of the table. Without this the loss landscape inside
    # the (On ∩ Reach) plateau is flat-zero, and Adam leaves objects wherever
    # they spawned — clustering them all near the seeded center.
    rng = torch.Generator(device=device).manual_seed(seed)
    jitter_x, jitter_y = init_jitter_xy
    jitter_scale = torch.tensor([jitter_x, jitter_y, 0.0], device=device)
    for o in optimizable_objs:
        jitter = (
            torch.empty(B, 3, device=device).uniform_(-1.0, 1.0, generator=rng) * jitter_scale
        )
        init_pos[:, obj_idx[o], :] = init_pos[:, obj_idx[o], :] + jitter

    # Optimizable tensor: (B, n_optimizable, 3).
    opt_indices = [obj_idx[o] for o in optimizable_objs]
    optimizable_pos = init_pos[:, opt_indices, :].clone().detach().requires_grad_(True)
    raw_fraction = torch.zeros(B, dtype=torch.float32, device=device, requires_grad=True)
    raw_offset = torch.zeros(B, dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam(
        [
            {"params": [optimizable_pos], "lr": lr},
            {"params": [raw_fraction, raw_offset], "lr": lr * robot_lr_mult},
        ]
    )

    on_strategy = OnLossStrategy(slope=10.0)
    # Two separate strategies so robot-vs-object can stay SOFT (small slope)
    # while object-vs-object collisions remain hard (large slope). Reach is
    # the dominant term — keeping the robot constraint soft lets the optimizer
    # prefer "object slightly under the stand" over "object outside the reach
    # shell", which matches the physical priority for env generation.
    obj_obj_no_coll = NoCollisionLossStrategy(slope=obj_obj_collision_slope)
    robot_no_coll = NoCollisionLossStrategy(slope=robot_obj_collision_slope)

    table_world_bbox = table_anchor.get_world_bounding_box().to(device)

    def _get_pos(o: DummyObject) -> torch.Tensor:
        if o is table_anchor:
            p = init_pos[:, obj_idx[o], :]
            return p
        return optimizable_pos[:, optimizable_objs.index(o), :]

    history = []
    for _ in range(n_iter):
        optimizer.zero_grad()
        fraction = frac_min + (frac_max - frac_min) * torch.sigmoid(raw_fraction)
        offset = off_min + (off_max - off_min) * torch.sigmoid(raw_offset)
        base_xy = corner_a + fraction.unsqueeze(-1) * (corner_b - corner_a) + offset.unsqueeze(-1) * n_out

        total = torch.zeros(B, device=device)

        # 1. On(parent) for each child in on_targets.
        for child, parent in on_targets.items():
            child_pos = _get_pos(child)
            child_bbox = child.get_bounding_box().to(device)
            on_relation = On(parent, clearance_m=0.0, relation_loss_weight=1.0)
            parent_world_bbox = parent.get_world_bounding_box().to(device)
            total = total + on_strategy.compute_loss(on_relation, child_pos, child_bbox, parent_world_bbox)

        # 2. Reach loss for each reach target — sampled in robot base frame.
        for tgt in reach_targets:
            pos = _get_pos(tgt)
            delta_xy = pos[:, :2] - base_xy                      # (B, 2)
            local_xy = torch.einsum("bij,bj->bi", R_inv, delta_xy)
            local_z = pos[:, 2:3] - base_z
            p_local = torch.cat([local_xy, local_z], dim=-1)
            d_reach = reach_field.sample(p_local, cap_m=cap_m)   # (B,)
            total = total + reach_weight * d_reach

        # 3. Pairwise no-collision among optimizable objects.
        for i, a in enumerate(optimizable_objs):
            pos_a = _get_pos(a)
            bbox_a = a.get_bounding_box().to(device)
            for b in optimizable_objs[i + 1 :]:
                pos_b = _get_pos(b)
                bbox_b = b.get_bounding_box().to(device)
                # Forward direction only; symmetry doesn't matter much for POC.
                world_bbox_b = bbox_b.translated(pos_b.detach())
                total = total + obj_obj_no_coll.compute_loss(
                    clearance_m=0.02,
                    child_pos=pos_a,
                    child_bbox=bbox_a,
                    parent_world_bbox=world_bbox_b,
                )

        # 4. Robot footprint as a SOFT moving-anchor NoCollision.
        #
        # Soft because: when reach demands an object position that clips the
        # robot stand, we prefer the small clip over an unreachable placement.
        # See `robot_obj_collision_slope` field above for the rationale; with
        # default slope=200 vs reach_weight=50/m, a 5 cm reach violation
        # (loss ~2.5) outweighs a 5 cm³ stand overlap (loss ~0.025) by ~100×.
        #
        # Footprint dims are nominal (see field above) — for production use,
        # query the franka_panda_hand_on_stand.usd bbox at scene build time.
        #
        # Footprint is axis-aligned because R_edge is cardinal, so a per-env
        # AABB built from base_xy is exact (no rotation needed).
        hfx, hfy, hfz = robot_footprint_half
        half = torch.tensor([hfx, hfy, hfz], device=device, dtype=torch.float32)
        center_xyz = torch.cat([base_xy, torch.full((B, 1), base_z + hfz, device=device)], dim=-1)
        robot_min = center_xyz - half
        robot_max = center_xyz + half
        robot_world_bbox = AxisAlignedBoundingBox(min_point=robot_min, max_point=robot_max)
        for o in optimizable_objs:
            pos = _get_pos(o)
            bbox = o.get_bounding_box().to(device)
            total = total + robot_no_coll.compute_loss(
                clearance_m=robot_clearance_m,
                child_pos=pos,
                child_bbox=bbox,
                parent_world_bbox=robot_world_bbox,
            )

        history.append(total.detach().cpu().clone())
        loss_mean = total.mean()
        loss_mean.backward()
        optimizer.step()

    # Reconstruct full positions for all envs.
    with torch.no_grad():
        full = init_pos.clone()
        for j, o in enumerate(optimizable_objs):
            full[:, obj_idx[o], :] = optimizable_pos[:, j, :]
        fraction_final = (
            frac_min + (frac_max - frac_min) * torch.sigmoid(raw_fraction)
        ).cpu().numpy()
        offset_final = (
            off_min + (off_max - off_min) * torch.sigmoid(raw_offset)
        ).cpu().numpy()
        base_xy_final = (
            corner_a
            + torch.tensor(fraction_final, device=device).unsqueeze(-1) * (corner_b - corner_a)
            + torch.tensor(offset_final, device=device).unsqueeze(-1) * n_out
        ).cpu().numpy()
    loss_per_env_final = history[-1]
    best_idx = int(loss_per_env_final.argmin().item())

    return SolveResult(
        best_idx=best_idx,
        loss_history=torch.stack(history, dim=0).numpy(),
        final_positions=full.cpu().numpy(),
        base_xy=base_xy_final,
        fraction=fraction_final,
        offset_m=offset_final,
        object_names=[o.name for o in objects],
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _draw_topdown(
    ax,
    table_xy_min,
    table_xy_max,
    result,
    objects,
    env_idx,
    title,
    xlim,
    ylim,
):
    x0, y0 = table_xy_min
    x1, y1 = table_xy_max
    # Table footprint.
    ax.add_patch(
        Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2, edgecolor="navy", facecolor="lightsteelblue", alpha=0.4, label="table",
        )
    )
    # Robot bases for all 4 envs (faint), and the selected one (bold).
    # Heading vector for each env = world-frame direction of base-local +X.
    headings = {
        "x_min": (1.0, 0.0),
        "x_max": (-1.0, 0.0),
        "y_min": (0.0, 1.0),
        "y_max": (0.0, -1.0),
    }
    arrow_len = 0.18
    # Robot footprint half-extents at object height (matches solver default).
    # See `robot_footprint_half` in joint_solve(...) for the reasoning.
    half_x, half_y = 0.12, 0.12
    for b in range(result.base_xy.shape[0]):
        bx, by = result.base_xy[b]
        is_best = b == env_idx
        # Footprint AABB
        ax.add_patch(
            Rectangle(
                (bx - half_x, by - half_y), 2 * half_x, 2 * half_y,
                edgecolor=("crimson" if is_best else "gray"),
                facecolor="none",
                linestyle=("-" if is_best else "--"),
                linewidth=(1.8 if is_best else 1.0),
                alpha=(1.0 if is_best else 0.4),
            )
        )
        # Base center
        ax.add_patch(
            Circle(
                (bx, by), 0.045,
                edgecolor=("crimson" if is_best else "gray"),
                facecolor=("crimson" if is_best else "lightgray"),
                alpha=(1.0 if is_best else 0.35),
                linewidth=(2 if is_best else 1),
                label=("robot (this env)" if is_best else None),
            )
        )
        # Heading arrow (world-frame direction of robot's local +X / forward).
        hx, hy = headings[EDGE_NAMES[b]]
        ax.annotate(
            "",
            xy=(bx + hx * arrow_len, by + hy * arrow_len),
            xytext=(bx, by),
            arrowprops=dict(
                arrowstyle="-|>",
                color=("crimson" if is_best else "gray"),
                lw=(2.2 if is_best else 1.0),
                alpha=(1.0 if is_best else 0.4),
                mutation_scale=18,
            ),
        )
        ax.text(bx, by + (0.10 if hy >= 0 else -0.16), EDGE_NAMES[b], ha="center", fontsize=8, color="dimgray")

    # Final object positions for the selected env, drawn with their bbox.
    colors = ["forestgreen", "darkorange", "purple", "teal"]
    for i, obj in enumerate(objects):
        if obj.name == "table":
            continue
        pos = result.final_positions[env_idx, i]
        bbox = obj.get_bounding_box()
        sx, sy, _ = bbox.size[0].tolist()
        ax.add_patch(
            Rectangle(
                (pos[0] - sx / 2, pos[1] - sy / 2), sx, sy,
                edgecolor=colors[i % len(colors)],
                facecolor=colors[i % len(colors)],
                alpha=0.6,
                linewidth=2,
                label=obj.name,
            )
        )
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=7)


def _draw_edt_slice(ax, reach_field, z_local: float):
    x_axis, y_axis, slice_xy = reach_field.slice_xy(z_local)
    # Show D_out as a heatmap (low = reachable, high = far from reach).
    im = ax.imshow(
        slice_xy.T,
        origin="lower",
        extent=(x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]),
        cmap="viridis_r",
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="D_out (m)")
    ax.contour(
        x_axis, y_axis, slice_xy.T, levels=[1e-6], colors="white", linewidths=1.5,
    )
    ax.scatter([0.0], [0.0], s=120, c="red", marker="*", zorder=5, label="robot base")
    ax.set_xlabel("base-frame X (m)")
    ax.set_ylabel("base-frame Y (m)")
    ax.set_title(f"EDT slice at base-frame z={z_local:.2f} m\n(white contour = reachable boundary)")
    ax.legend(loc="upper left", fontsize=8)


def visualize(
    result: SolveResult,
    objects: list[DummyObject],
    reach_field: ReachField,
    table_xy_min: tuple[float, float],
    table_xy_max: tuple[float, float],
    z_slice_local: float,
    out_png: Path,
) -> None:
    # Common axis limits sized to include all 4 robot positions.
    pad = 0.25
    xlim = (table_xy_min[0] - pad, table_xy_max[0] + pad)
    ylim = (table_xy_min[1] - pad, table_xy_max[1] + pad)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # Row 0: 4 per-env top-downs (with the best one in column 0).
    ordered = [result.best_idx] + [b for b in range(4) if b != result.best_idx]
    for cell, env_idx in zip(
        [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]], ordered
    ):
        is_best = env_idx == result.best_idx
        prefix = "BEST — " if is_best else ""
        title = (
            f"{prefix}env {env_idx}: {EDGE_NAMES[env_idx]} "
            f"(frac={result.fraction[env_idx]:.2f}, off={result.offset_m[env_idx]:.2f}m)"
        )
        _draw_topdown(
            cell, table_xy_min, table_xy_max, result, objects, env_idx, title, xlim, ylim
        )

    # Row 1, col 1: EDT slice in robot-base frame.
    _draw_edt_slice(axes[1, 1], reach_field, z_slice_local)

    # Row 1, col 2: loss curves.
    ax_loss = axes[1, 2]
    for b in range(result.loss_history.shape[1]):
        ax_loss.plot(
            result.loss_history[:, b],
            label=f"{EDGE_NAMES[b]}{' (best)' if b == result.best_idx else ''}",
            linewidth=(2.5 if b == result.best_idx else 1.2),
            alpha=(1.0 if b == result.best_idx else 0.7),
        )
    ax_loss.set_xlabel("Adam iteration")
    ax_loss.set_ylabel("per-env total loss")
    ax_loss.set_title("Loss curves (4 edges in parallel)")
    ax_loss.set_yscale("symlog", linthresh=1e-3)
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    fig.suptitle("Joint robot-edge + object-placement solver POC", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"[poc] saved figure -> {out_png}")


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------


def _make_scene(
    table_size_xy: tuple[float, float] = (1.0, 0.6),
) -> tuple[DummyObject, list[DummyObject], dict, list[DummyObject], tuple, tuple]:
    """Build a tiny tabletop scene with a pick + destination.

    ``table_size_xy`` controls the rectangular footprint (default 1.0 m × 0.6 m).
    Shrink it (e.g. 0.4 × 0.3) to force the robot footprint to overlap with
    the table region, which exercises the robot-vs-object NoCollision term.
    """
    sx, sy = table_size_xy
    half_x, half_y = sx / 2.0, sy / 2.0
    cx, cy = 0.5, 0.0
    table_xy_min = (cx - half_x, cy - half_y)
    table_xy_max = (cx + half_x, cy + half_y)
    table_pose = (cx, cy, 0.025)
    table_local_bbox = AxisAlignedBoundingBox(
        min_point=(-half_x, -half_y, -0.025), max_point=(half_x, half_y, 0.025)
    )
    table = DummyObject(name="table", bounding_box=table_local_bbox)
    table.set_initial_pose(Pose(position_xyz=table_pose, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())

    pick_bbox = AxisAlignedBoundingBox(min_point=(-0.03, -0.03, 0.0), max_point=(0.03, 0.03, 0.06))
    pick = DummyObject(name="pick_obj", bounding_box=pick_bbox)
    pick.set_initial_pose(Pose(position_xyz=(0.5, 0.0, 0.05), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    pick.add_relation(On(table, clearance_m=0.0))

    dest_bbox = AxisAlignedBoundingBox(min_point=(-0.06, -0.06, 0.0), max_point=(0.06, 0.06, 0.04))
    dest = DummyObject(name="destination", bounding_box=dest_bbox)
    dest.set_initial_pose(Pose(position_xyz=(0.5, 0.1, 0.05), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    dest.add_relation(On(table, clearance_m=0.0))

    objects = [table, pick, dest]
    on_targets = {pick: table, dest: table}
    reach_targets = [pick, dest]
    return table, objects, on_targets, reach_targets, table_xy_min, table_xy_max


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reach_npz", type=str, required=True)
    parser.add_argument("--out_png", type=str, default="tools/poc_joint_reach_solver.png")
    parser.add_argument("--n_iter", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--reach_weight",
        type=float,
        default=50.0,
        help="Slope on the EDT reach loss (per meter). Reachability is the dominant term.",
    )
    parser.add_argument(
        "--cap_m",
        type=float,
        default=0.5,
        help="Clamp on EDT distance (m); prevents far-away objects from blowing up gradients.",
    )
    parser.add_argument(
        "--robot_obj_collision_slope",
        type=float,
        default=200.0,
        help=(
            "Slope on the robot-footprint-vs-object NoCollision (volume-product). "
            "Kept SOFT (≪ obj-obj slope of 10000) so reachability dominates — "
            "the optimizer prefers 'object slightly under the stand' over 'object "
            "outside the reach shell'."
        ),
    )
    parser.add_argument(
        "--obj_obj_collision_slope",
        type=float,
        default=10000.0,
        help="Slope on object-vs-object NoCollision. Kept hard so objects don't overlap.",
    )
    parser.add_argument(
        "--offset_range",
        type=float,
        nargs=2,
        default=[0.05, 0.40],
        metavar=("MIN", "MAX"),
        help="Min/max distance (m) of robot base outside the table edge.",
    )
    parser.add_argument(
        "--init_jitter_xy",
        type=float,
        nargs=2,
        default=[0.30, 0.20],
        metavar=("X", "Y"),
        help=(
            "Half-extent of uniform XY jitter applied to each object's initial position "
            "(per env). Larger values spread objects across the table — the loss landscape "
            "inside (On ∩ Reach) is flat, so Adam leaves them where they spawn."
        ),
    )
    parser.add_argument(
        "--table_size",
        type=float,
        nargs=2,
        default=[1.0, 0.6],
        metavar=("SX", "SY"),
        help=(
            "Rectangular table footprint (meters). Shrink to e.g. 0.4 0.3 to force the "
            "robot footprint to overlap with the table region — useful for stress-testing "
            "the soft robot-vs-object NoCollision term."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    reach = ReachField.from_npz(Path(args.reach_npz), device=device)
    print(
        f"[poc] EDT loaded: shape={tuple(reach.field.shape)} "
        f"voxel={reach.voxel_size.cpu().tolist()} max_d={reach.field.max().item():.3f} m"
    )

    table, objects, on_targets, reach_targets, txy0, txy1 = _make_scene(table_size_xy=tuple(args.table_size))

    result = joint_solve(
        table_anchor=table,
        objects=objects,
        reach_targets=reach_targets,
        on_targets=on_targets,
        reach_field=reach,
        table_xy_min=txy0,
        table_xy_max=txy1,
        offset_range=tuple(args.offset_range),
        base_z=0.0,
        cap_m=args.cap_m,
        reach_weight=args.reach_weight,
        obj_obj_collision_slope=args.obj_obj_collision_slope,
        robot_obj_collision_slope=args.robot_obj_collision_slope,
        init_jitter_xy=tuple(args.init_jitter_xy),
        n_iter=args.n_iter,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )

    print("[poc] per-env final loss:", result.loss_history[-1].tolist())
    print(
        f"[poc] best env: {EDGE_NAMES[result.best_idx]} "
        f"(frac={result.fraction[result.best_idx]:.3f}, "
        f"offset={result.offset_m[result.best_idx]:.3f} m, "
        f"base_xy=({result.base_xy[result.best_idx, 0]:.3f}, {result.base_xy[result.best_idx, 1]:.3f}))"
    )
    print("[poc] per-env (frac, offset_m): " + ", ".join(
        f"{EDGE_NAMES[b]}=({result.fraction[b]:.3f}, {result.offset_m[b]:.3f})"
        for b in range(4)
    ))
    for i, o in enumerate(objects):
        if o.name == "table":
            continue
        p = result.final_positions[result.best_idx, i]
        # Sanity: residual reach distance for each manipulable.
        base_xy = torch.tensor(result.base_xy[result.best_idx], device=device, dtype=torch.float32)
        # Use the same R_inv as the solver computed for the best env.
        from_edge = result.best_idx
        R_edge_all, *_ = edge_constants(txy0, txy1, device)
        R_inv = R_edge_all[from_edge].T
        delta_xy = torch.tensor(p[:2], device=device, dtype=torch.float32) - base_xy
        local_xy = R_inv @ delta_xy
        p_local = torch.cat([local_xy, torch.tensor([p[2]], device=device, dtype=torch.float32)])
        d = reach.sample(p_local.unsqueeze(0), cap_m=args.cap_m).item()
        print(f"[poc]   {o.name}: world={tuple(round(float(v), 3) for v in p)} "
              f"local_xy=({local_xy[0].item():+.3f}, {local_xy[1].item():+.3f}) reach_d={d:.4f} m")

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    visualize(result, objects, reach, txy0, txy1, z_slice_local=0.05, out_png=out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
