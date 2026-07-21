# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Phase-1 perception probe: render the CAP barrier env's exterior_cam (rgb + depth).

Verifies that make_cap_franka_environment(enable_cameras=True) attaches the
perception exterior_cam and produces a non-empty RGB-D frame -- the foundation of
the get_image ROS path. Does NOT touch the barrier lockstep; it just builds the
env, steps once, accesses the camera (which triggers the render), and reports the
frame. Run headless with cameras enabled:

    ./dev_run.sh isaaclab_arena/scripts/run_cap_barrier_render_probe.py \
        --headless --enable_cameras --device cuda:0
"""

from __future__ import annotations

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

_CAM = "exterior_cam"
_DEPTH_DT = "distance_to_image_plane"


def _probe(device: str) -> None:
    import numpy as np
    import torch

    from isaaclab_arena.integrations.cap_barrier.franka_env import (
        make_cap_franka_environment,
    )

    adapter = make_cap_franka_environment(device=device, enable_cameras=True)
    try:
        environment = adapter._environment
        unwrapped = adapter._unwrapped
        environment.reset()
        if _CAM not in unwrapped.scene.sensors:
            raise RuntimeError(f"{_CAM} sensor was not attached to the barrier scene")
        camera = unwrapped.scene[_CAM]
        action = torch.zeros(
            (1, unwrapped.action_manager.total_action_dim), device=unwrapped.device
        )
        environment.step(action)  # advance + trigger the RTX render via camera access below
        rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        depth = camera.data.output[_DEPTH_DT][0].squeeze(-1).detach().cpu().numpy()
        finite = float(np.isfinite(depth).mean())
        print(
            "CAP_RENDER_PROBE "
            f"rgb_shape={tuple(rgb.shape)} rgb_min={int(rgb.min())} rgb_max={int(rgb.max())} "
            f"depth_shape={tuple(depth.shape)} depth_finite_frac={finite:.3f} "
            f"depth_min={float(np.nanmin(depth)):.4f} depth_max={float(np.nanmax(depth)):.4f}",
            flush=True,
        )
        ok = (
            rgb.ndim == 3
            and rgb.shape[2] == 3
            and int(rgb.max()) > 0
            and depth.shape == rgb.shape[:2]
            and finite > 0.5
        )
        print("CAP_RENDER_PROBE_OK" if ok else "CAP_RENDER_PROBE_FAILED", flush=True)
    finally:
        adapter.close()


def main() -> None:
    parser = get_isaaclab_arena_cli_parser()
    args_cli = parser.parse_args()
    with SimulationAppContext(args_cli):
        _probe(args_cli.device)


if __name__ == "__main__":
    main()
