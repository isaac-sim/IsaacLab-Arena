# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Size the per-step camera-render cost: mean env.step time with/without the exterior-cam read.

Answers whether the RTX render is triggered by the data ACCESS (so a render cadence that skips the read
genuinely skips the render) or by env.step's render pass (so the lever is render_interval/resolution).

Run with cameras (access vs no-access in one boot):
    ./dev_run.sh isaaclab_arena/scripts/cap_bridge/m6_render_probe.py \
        --headless --num_envs 1 --enable_cameras --placement_seed 0 libero_object_packing --control joint_pos
Run physics-only floor (omit --enable_cameras):
    ./dev_run.sh isaaclab_arena/scripts/cap_bridge/m6_render_probe.py \
        --headless --num_envs 1 --placement_seed 0 libero_object_packing --control joint_pos
"""

from __future__ import annotations

import time

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

_CAM = "exterior_cam"
_DEPTH_DT = "distance_to_image_plane"


def main() -> None:
    args_parser = get_isaaclab_arena_cli_parser()
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        import torch

        from isaaclab_arena_environments.cli import (
            get_arena_builder_from_cli,
            get_isaaclab_arena_environments_cli_parser,
        )

        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_cli = args_parser.parse_args()

        env = get_arena_builder_from_cli(args_cli).make_registered()
        env.reset()
        unwrapped = env.unwrapped
        device = unwrapped.device
        action = torch.zeros(env.action_space.shape, device=device)
        cam = unwrapped.scene[_CAM] if _CAM in unwrapped.scene.sensors else None
        print(f"[probe] exterior_cam present: {cam is not None}")

        def timed(n: int, access: bool) -> float:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                env.step(action)
                if access and cam is not None:
                    _ = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
                    _ = cam.data.output[_DEPTH_DT][0].squeeze(-1).cpu().numpy()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n * 1000.0

        timed(20, cam is not None)  # warmup
        if cam is not None:
            ms_access = timed(150, True)
            ms_noaccess = timed(150, False)
            print(f"[probe] RESULT cameras_enabled: access={ms_access:.1f}ms/step  no_access={ms_noaccess:.1f}ms/step")
        else:
            ms_phys = timed(150, False)
            print(f"[probe] RESULT physics_only (no cameras): {ms_phys:.1f}ms/step")

        env.close()


if __name__ == "__main__":
    main()
