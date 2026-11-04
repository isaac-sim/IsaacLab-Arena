# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise the supplied workbench recipes in physics, independently of scene registry classes."""

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def run(args):
    """Spawn all generated assets and check drawer motion, button return and the open-case drive."""
    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg

    cfg = sim_utils.SimulationCfg(device=args.device, dt=1 / 240, gravity=(0, 0, 0))
    report = {}
    with launch_simulation(cfg, args):
        from pxr import UsdGeom, UsdPhysics

        sim = sim_utils.SimulationContext(cfg)
        stage = sim.stage
        roots = {}
        for index, manifest_path in enumerate(sorted(args.assets_dir.glob("*/manifest.json"))):
            manifest = json.loads(manifest_path.read_text())
            name = manifest["id"]
            root = f"/World/{name}"
            roots[name] = root
            scale = (1.35,) * 3 if name == "workbench" else (1.0,) * 3
            spawn = sim_utils.UsdFileCfg(usd_path=str(manifest_path.parent / manifest["output"]), scale=scale)
            spawn.func(root, spawn, translation=(index * 5.0, 0, 0))
        assert set(roots) == {
            "workbench",
            "socket_set_open",
            "oscilloscope_button",
            "white_bin",
            "slotted_screwdriver",
            "small_wrench",
        }
        paths = {
            "drawer": roots["workbench"] + "/Articulations/DrawerCabinet",
            "door": roots["workbench"] + "/Articulations/DoorCabinet",
            "button": roots["oscilloscope_button"],
            "open_case": roots["socket_set_open"],
        }
        articulations = {}
        for name, path in paths.items():
            articulations[name] = Articulation(ArticulationCfg(prim_path=path, spawn=None, actuators={}))
        sim.reset()
        physics_scenes = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        assert physics_scenes == [cfg.physics_prim_path], physics_scenes
        for name, asset in articulations.items():
            report[name] = {"joints": asset.joint_names, "fixed_base": asset.is_fixed_base}
        assert articulations["drawer"].num_joints == 4
        assert articulations["door"].num_joints == 1
        assert articulations["button"].num_joints == 1
        assert articulations["open_case"].num_joints == 1
        assert all(articulations[name].is_fixed_base for name in ("drawer", "door", "button"))

        def set_joint(name, joint_name, value):
            asset = articulations[name]
            values = asset.data.joint_pos.torch.clone()
            values[:, asset.joint_names.index(joint_name)] = value
            asset.write_joint_position_to_sim_index(position=values)
            asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(values))

        drawer = articulations["drawer"]
        drawer_body = drawer.body_names.index("drawersA_01_obj_01")
        t_W_before = drawer.data.body_link_pose_w.torch[0, drawer_body, :3].clone()
        set_joint("drawer", "joint_drawersA_01_joint_00", 0.20)
        sim.forward()
        drawer.update(cfg.dt)
        t_W_after = drawer.data.body_link_pose_w.torch[0, drawer_body, :3]
        displacement = torch.linalg.vector_norm(t_W_after - t_W_before).item()
        report["drawer"]["commanded_displacement_m"] = 0.20
        report["drawer"]["measured_displacement_m"] = displacement
        assert abs(displacement - 0.20) < 0.002, report["drawer"]
        set_joint("button", "joint_button_01_joint_00", -0.002)
        set_joint("open_case", "case_lid", 1.5)
        set_joint("door", "joint_door_01_joint_00", 0.4)
        for _ in range(480):
            sim.step(render=False)
            for asset in articulations.values():
                asset.update(cfg.dt)
                assert torch.isfinite(asset.data.joint_pos.torch).all()
                assert torch.isfinite(asset.data.body_link_pose_w.torch).all()
        for name, asset in articulations.items():
            report[name]["final_joint_positions"] = asset.data.joint_pos.torch.tolist()
        button_position = articulations["button"].data.joint_pos.torch[0, 0].item()
        assert abs(button_position) < 0.0002, report["button"]
        case_position = articulations["open_case"].data.joint_pos.torch[0, 0].item()
        assert abs(case_position - np.deg2rad(105)) < 0.08, report["open_case"]
        report["spawned_packages"] = len(roots)
        report["steps"] = 480
        report["gravity"] = list(cfg.gravity)
        report["physics_backend"] = args.physics
        report["physics_scenes"] = physics_scenes
        report["rigid_bodies"] = sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse())
        report["meshes"] = sum(p.IsA(UsdGeom.Mesh) for p in stage.Traverse())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        sim.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physics", default="isaacsim_physx", choices=["isaacsim_physx", "ovphysx"])
    add_launcher_args(parser)
    args = parser.parse_args()
    assert not args.output.exists(), "Choose a new physics report path"
    run(args)


if __name__ == "__main__":
    main()
