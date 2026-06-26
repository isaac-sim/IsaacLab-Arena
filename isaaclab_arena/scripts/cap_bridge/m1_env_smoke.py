# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""M1 smoke test for the GaP<->ILA bridge: env runs headless, holds a pose, reads joints + wrist cam.

Validates the milestone-1 gate from docs/ila_side_design.md: launch ``libero_object_packing`` headless,
reset/step holding a pose (zero action), read ``joint_pos``, render the wrist camera, and report the
step rate (50 Hz is a soft perf target, not pass/fail). No bridge, no joint commands yet (that is M2).

Run:

    ./dev_run.sh isaaclab_arena/scripts/cap_bridge/m1_env_smoke.py \
        --headless --num_envs 1 --enable_cameras libero_object_packing
"""

from __future__ import annotations

import time

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

_WARMUP_STEPS = 15
_MEASURE_STEPS = 300


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

        robot = unwrapped.scene["robot"]
        sensor_keys = list(unwrapped.scene.sensors.keys())
        cam_name = next((k for k in sensor_keys if "cam" in k.lower()), None)
        print(f"[m1] scene sensors: {sensor_keys}")

        # Zero action = hold pose (the env still uses the relative IK term here; absolute joint
        # control is M2). Shape matches the wrapped env's action space.
        action = torch.zeros(env.action_space.shape, device=unwrapped.device)

        for _ in range(_WARMUP_STEPS):
            env.step(action)

        t0 = time.perf_counter()
        for _ in range(_MEASURE_STEPS):
            env.step(action)
        elapsed = time.perf_counter() - t0
        hz = _MEASURE_STEPS / elapsed

        q = robot.data.joint_pos[0, :7].cpu().numpy()
        joints_ok = bool(q.shape == (7,))
        print(f"[m1] joint_pos[:7] = {q}")

        cam_ok = False
        if cam_name is not None:
            output = unwrapped.scene[cam_name].data.output
            out_keys = list(output.keys())
            rgb = output["rgb"][0] if "rgb" in output else None
            rgb_shape = tuple(rgb.shape) if rgb is not None else None
            cam_ok = rgb is not None and rgb.numel() > 0
            print(f"[m1] camera '{cam_name}': outputs={out_keys} rgb_shape={rgb_shape}")
        else:
            print("[m1] no camera sensor found (did you pass --enable_cameras?)")

        print(
            f"[m1] RESULT step_rate={hz:.1f} Hz over {_MEASURE_STEPS} steps ({elapsed:.2f}s) "
            f"joints_ok={joints_ok} cam_ok={cam_ok}"
        )

        env.close()


if __name__ == "__main__":
    main()
