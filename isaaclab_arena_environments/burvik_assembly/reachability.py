# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check BURVIK grasp-point reach using Isaac Lab's existing differential IK controller."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_apply

from isaaclab_arena_environments.burvik_assembly.geometry import (
    ASSEMBLY_ORDER,
    TCP_OFFSET,
    attachment_pose,
    grasp_point,
    initial_part_poses,
)


def check_reachability(env, specification):
    """Find joint-limited TCP solutions for pickup and insertion grasp points, then restore robots.

    These are positional IK checks, not collision-free grasp or trajectory proofs. The controller
    and Jacobians come from the live imported YAM articulations; no approximate reach sphere is used.
    """
    targets = {}
    for name, (position, rotation) in initial_part_poses().items():
        targets[f"pickup_{name}"] = position + rotation.apply(grasp_point(name))
    for name in ASSEMBLY_ORDER:
        local_p, local_r = attachment_pose(name, specification)
        if name == "handle":
            tray_p, tray_r = np.array([0.0, 0.06, 1.12]), Rotation.from_euler("y", 90, degrees=True)
        else:
            tray_p, tray_r = initial_part_poses()["tray"]
        targets[f"insert_{name}"] = tray_p + tray_r.apply(local_p + local_r.apply(grasp_point(name)))
    solutions = {name: [] for name in targets}
    for side in ("left", "right"):
        robot = env.scene[f"{side}_robot"]
        saved = robot.data.joint_pos.torch.clone()
        saved_velocity = robot.data.joint_vel.torch.clone()
        ids = [robot.joint_names.index(f"joint{i}") for i in range(1, 7)]
        body_id = robot.body_names.index("link_6")
        jacobian_id = body_id - 1 if robot.is_fixed_base else body_id
        joint_columns = [j + robot.num_base_dofs for j in ids]
        limits = robot.data.soft_joint_pos_limits.torch[:, ids]
        controller = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
            num_envs=env.num_envs,
            device=env.device,
        )
        offset = torch.tensor(TCP_OFFSET, device=env.device).expand(env.num_envs, -1)
        try:
            for name, target in targets.items():
                command = torch.tensor(target, dtype=torch.float32, device=env.device).expand(env.num_envs, -1)
                command = command + env.scene.env_origins
                best = None
                for seed in (
                    (0.0, 0.85, 0.6, 0.0, 0.0, 0.0),
                    (0.0, -0.6, 1.2, 0.0, 0.5, 0.0),
                    (0.7, 0.4, 1.0, 0.0, -0.6, 0.0),
                    (-0.7, 0.4, 1.0, 0.0, -0.6, 0.0),
                ):
                    q = saved.clone()
                    q[:, ids] = torch.tensor(seed, device=env.device)
                    robot.write_joint_position_to_sim_index(position=q)
                    controller.reset()
                    controller.set_command(command, ee_quat=robot.data.body_quat_w.torch[:, body_id])
                    for _ in range(160):
                        env.sim.forward()
                        robot.update(env.step_dt)
                        pose = robot.data.body_pose_w.torch[:, body_id]
                        delta = quat_apply(pose[:, 3:], offset)
                        tcp = pose[:, :3] + delta
                        error = torch.linalg.vector_norm(command - tcp, dim=-1)
                        if best is None or float(error[0]) < best[0]:
                            best = (float(error[0]), q[:, ids].clone())
                        if bool((error < 0.003).all()):
                            break
                        jacobian = robot.data.body_link_jacobian_w.torch[:, jacobian_id, :, joint_columns].clone()
                        # Shift the link-origin Jacobian to the actual finger-center TCP.
                        jacobian[:, :3] += torch.cross(
                            jacobian[:, 3:].transpose(1, 2), delta[:, None, :], dim=-1
                        ).transpose(1, 2)
                        current = robot.data.joint_pos.torch[:, ids]
                        desired = controller.compute(tcp, pose[:, 3:], jacobian, current)
                        desired = current + (desired - current).clamp(-0.12, 0.12)
                        q[:, ids] = torch.maximum(torch.minimum(desired, limits[..., 1]), limits[..., 0])
                        robot.write_joint_position_to_sim_index(position=q)
                    if best[0] < 0.003:
                        break
                solutions[name].append({"arm": side, "error_m": best[0], "joint_positions_rad": best[1][0].tolist()})
        finally:
            robot.write_joint_position_to_sim_index(position=saved)
            robot.write_joint_velocity_to_sim_index(velocity=saved_velocity)
            env.sim.forward()
            robot.update(env.step_dt)
    results = {}
    for name, candidates in solutions.items():
        best = min(candidates, key=lambda candidate: candidate["error_m"])
        results[name] = {"target_w": targets[name].tolist(), **best, "reachable": best["error_m"] < 0.005}
    return {
        "method": "Joint-limited positional differential IK at YAM finger-center TCP; collisions/path not certified.",
        "targets": results,
        "passed": all(item["reachable"] for item in results.values()),
    }
