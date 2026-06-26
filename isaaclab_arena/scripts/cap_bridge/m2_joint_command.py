# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""M2 action round-trip for the GaP<->ILA bridge: command an absolute joint target, assert the arm lands.

Validates the milestone-2 gate from docs/ila_side_design.md: with the absolute joint-position action
term (``--control joint_pos`` -> scale=1.0, use_default_offset=False), commanding a known ``q`` drives
the arm to ``q`` -- NOT to ``0.5*q + default`` (the failure mode if the stock scaled/offset term leaks
through). Uses the bridge_smoke move target so M2 mirrors what M3 will command over the wire.

Run:

    ./dev_run.sh isaaclab_arena/scripts/cap_bridge/m2_joint_command.py \
        --headless --num_envs 1 --control joint_pos libero_object_packing
"""

from __future__ import annotations

import numpy as np

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

# The absolute 7-joint target the bridge_smoke graph commands via robot.move_to_joints.
_TARGET_Q = [0.3, -0.6, 0.0, -2.2, 0.0, 1.5, 0.785]
_TOL = 0.02  # matches the GaP-side move_to_joints tolerance
_SETTLE_STEPS = 250


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

        target = np.asarray(_TARGET_Q)
        action = torch.zeros(env.action_space.shape, device=unwrapped.device)
        action[..., :7] = torch.tensor(_TARGET_Q, device=unwrapped.device)
        if action.shape[-1] > 7:
            action[..., 7:] = 1.0  # binary gripper -> open (irrelevant to the arm gate)

        default_q = robot.data.default_joint_pos[0, :7].cpu().numpy()
        buggy_landing = 0.5 * target + default_q  # where a scale=0.5/use_default_offset term would land
        print(f"[m2] start  q = {robot.data.joint_pos[0, :7].cpu().numpy()}")
        print(f"[m2] target q = {target}")
        print(f"[m2] default q = {default_q}")
        print(f"[m2] (0.5q+default failure landing would be {buggy_landing})")

        for i in range(_SETTLE_STEPS):
            env.step(action)
            if (i + 1) % 50 == 0:
                q = robot.data.joint_pos[0, :7].cpu().numpy()
                print(f"[m2] step {i + 1}: max_err_to_target={np.abs(q - target).max():.4f}")

        q = robot.data.joint_pos[0, :7].cpu().numpy()
        err = np.abs(q - target)
        max_err = float(err.max())
        dist_to_buggy = float(np.abs(q - buggy_landing).max())
        lands_at_target = max_err < _TOL

        print(f"[m2] final  q = {q}")
        print(f"[m2] per-joint err = {err}")
        print(
            f"[m2] RESULT max_err={max_err:.4f} (tol={_TOL}) lands_at_target={lands_at_target} "
            f"dist_to_0.5q+offset={dist_to_buggy:.4f}"
        )

        env.close()


if __name__ == "__main__":
    main()
