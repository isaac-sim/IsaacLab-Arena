# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run scripted pick-and-place episodes on Newton and save the state seen by MDP2."""

import argparse
import json
import torch
from pathlib import Path

import isaaclab.sim as sim
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnv, MDP2RLEnvCfg
from isaaclab_experimental.mdp2 import mdp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from PIL import Image, ImageDraw

from isaaclab_arena.tasks.predicates.spatial import contact_force_is_upward_support, object_is_moving_slowly

PLATE_SIZE = (0.5, 0.35, 0.04)
DT = 1 / 60
WORLDS = 4


def settled(pose, velocity, remembered, rest_z, linear_limit, angular_limit):
    at_rest = (velocity[:, :3].norm(dim=1) < linear_limit) & (velocity[:, 3:].norm(dim=1) < angular_limit)
    rest_z.copy_(torch.where(at_rest & (remembered == 0), pose[:, 2], rest_z))
    remembered.logical_or_(at_rest)
    return at_rest.float()


def lifted(pose, rest_z, remembered, distance):
    return ((remembered > 0) & (pose[:, 2] > rest_z + distance)).float()


def placed(pose, plate_pose, velocity, forces, force_threshold, velocity_threshold, support_cone_half_angle_rad):
    # The fixture is a centered cube and an axis-aligned, fixed rectangular plate.
    offset = pose[:, :3] - plate_pose[:, :3]
    over_plate = (offset[:, 0].abs() <= PLATE_SIZE[0] / 2) & (offset[:, 1].abs() <= PLATE_SIZE[1] / 2)
    over_plate &= offset[:, 2] >= -PLATE_SIZE[2] / 2
    supported = contact_force_is_upward_support(forces[:, 0, 0], force_threshold, support_cone_half_angle_rad)
    return (over_plate & supported & object_is_moving_slowly(velocity[:, :3], velocity_threshold)).float()


def pick_and_place(name, rules):
    """Bind one Arena pick-and-place sequence to the cube, plate, and contact sensor."""
    cube = mdp2.Entity(name)
    rest = f"{name}_rest"
    up = f"{name}_lift"
    on_plate = f"{name}_place"
    quantities = {
        rest: mdp2.Quantity(
            mdp2.Term(torch=settled, state={"remembered": mdp2.State(), "rest_z": mdp2.State()}),
            inputs={"pose": cube.root_link_pose_w, "velocity": cube.root_com_vel_w},
            params=rules["rest"],
        ),
        up: mdp2.Quantity(
            mdp2.Term(torch=lifted),
            inputs={
                "pose": cube.root_link_pose_w,
                "rest_z": mdp2.term_state(f"quantities.{rest}", "rest_z"),
                "remembered": mdp2.term_state(f"quantities.{rest}", "remembered"),
            },
            params=rules["lift"],
        ),
        on_plate: mdp2.Quantity(
            mdp2.Term(torch=placed),
            inputs={
                "pose": cube.root_link_pose_w,
                "plate_pose": mdp2.Entity("plate").root_link_pose_w,
                "velocity": cube.root_com_vel_w,
                "forces": mdp2.Entity(f"contact_sensor_{name}").normal_force_matrix_w,
            },
            params=rules["place"],
        ),
    }
    stages = {}
    for label, quantity, hold, weight in zip(
        ("settle", "lift", "place"), (rest, up, on_plate), (1, 1, rules["hold"]), rules["weights"], strict=True
    ):
        stages[label] = mdp2.Milestone(getattr(mdp2.quantities, quantity), hold=hold, score=weight)
    return quantities, mdp2.sequence(**stages)


def cuboid(name, size, xyz, fixed=False):
    colors = {"plate": (0.48, 0.53, 0.55), "red": (0.9, 0.07, 0.03), "blue": (0.04, 0.22, 0.9), "yellow": (1, 0.8, 0.1)}
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim.CuboidCfg(
            size=size,
            rigid_props=sim.UsdPhysicsRigidBodyCfg(kinematic_enabled=fixed),
            collision_props=sim.UsdPhysicsCollisionCfg(),
            mass_props=sim.MassCfg(mass=0.1),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=colors[name]),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=xyz),
    )


def make_env(rules, with_camera):
    scene = InteractiveSceneCfg(num_envs=WORLDS, env_spacing=3.0)
    scene.plate = cuboid("plate", PLATE_SIZE, (0, 0, 0.22), fixed=True)
    quantities, children = {}, {}
    ends = {
        "success": mdp2.Termination(mdp.is_true, inputs={"value": mdp2.progress.done}),
        "time_out": mdp2.TimeLimit(episode_length_s=rules["timeout"]),
    }
    for name, definition in rules["objects"].items():
        x = {"red": -0.1, "blue": 0.1, "yellow": 0.0}[name]
        setattr(scene, name, cuboid(name, (0.04,) * 3, (x, 0, 0.26)))
        sensor = ContactSensorCfg(prim_path=f"{{ENV_REGEX_NS}}/{name}", filter_prim_paths_expr=["{ENV_REGEX_NS}/plate"])
        setattr(scene, f"contact_sensor_{name}", sensor)
        measurements, children[name] = pick_and_place(name, definition)
        quantities.update(measurements)
        ends[f"{name}_dropped"] = mdp2.Termination(
            mdp.root_height_below_minimum,
            inputs={"root_link_pose_w": mdp2.Entity(name).root_link_pose_w},
            params={"minimum_height": definition["drop_height"]},
        )
    groups, paths = {}, {}
    for group in rules["groups"]:
        names = group["children"]
        if len(names) == 1:
            name = names[0]
            groups[name] = children[name]
            paths[name] = name
        else:
            label = "either" if group["require"] == 1 else "best_two"
            groups[label] = mdp2.compose(require=group["require"], **{name: children[name] for name in names})
            paths.update({name: f"{label}.{name}" for name in names})
    if rules["final"] is not None:
        for child, final in zip(groups.values(), rules["final"], strict=True):
            child.final = final
    composition = mdp2.sequence(**groups) if rules["sequential"] else mdp2.compose(**groups)
    task = mdp2.compose(mdp2.TaskCfg(quantities=quantities, terminations=ends), composition)
    if with_camera:
        scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            width=448,
            height=240,
            data_types=["rgb"],
            spawn=sim.PinholeCameraCfg(focal_length=6.0, horizontal_aperture=5.4, clipping_range=(0.01, 5)),
            offset=CameraCfg.OffsetCfg(pos=(0, -0.7, 0.54), rot=(-0.83205, 0, 0, 0.55470), convention="ros"),
            renderer_cfg=NewtonWarpRendererCfg(),
        )
    return (
        MDP2RLEnv(
            MDP2RLEnvCfg(
                sim=sim.SimulationCfg(dt=DT / 4, device="cuda:0", physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())),
                decimation=4,
                seed=17,
                scene=scene,
                task=task,
                capture=False,
                episode_budget=WORLDS,
            )
        ),
        paths,
    )


def move_cube(env, name, worlds, height):
    ids = torch.tensor(worlds, device=env.device)
    cube = env.scene[name]
    poses = cube.data.root_link_pose_w.torch[ids].clone()
    poses[:, 2] = height
    if height < 0:
        poses[:, 0] += 0.5
    cube.write_root_pose_to_sim_index(root_pose=poses, env_ids=ids)
    cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=env.device), env_ids=ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rules", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gif", type=Path, help="Save a slowed-down camera view of world 0 with its milestone scores.")
    args = parser.parse_args()
    rules = json.loads(args.rules.read_text())
    env, paths = make_env(rules, args.gif is not None)
    names = rules["objects"]
    samples = []
    frames = []
    try:
        env.reset()
        for step in range(1, round(rules["timeout"] / DT) + 2):
            # Worlds 0 and 1 succeed in opposite orders, 2 drops red, 3 never finishes.
            if step == 15:
                move_cube(env, "red", [0, 2], 0.34)
                if "blue" in names:
                    move_cube(env, "blue", [1, 3], 0.34)
            if step == 22:
                move_cube(env, "red", [2], -0.2)
            if step == 42:
                move_cube(env, "red", [1], 0.34)
                if "blue" in names:
                    move_cube(env, "blue", [0], 0.34)
            if step == 66 and "yellow" in names:
                move_cube(env, "yellow", [0, 1], 0.34)
            if step == 70 and rules["sequential"]:
                # This world has reported success. Keep measuring while the other worlds finish.
                move_cube(env, "red", [0], 0.34)
            if step == 72 and rules["case"] == "single":
                # At timeout the object is back on the plate, partway through the four-step hold.
                move_cube(env, "red", [3], 0.34)
            _, _, terminated, truncated, _ = env.step(torch.zeros((WORLDS, 0), device=env.device))
            # One-episode budget per world keeps terminal state readable, including the finishing step.
            sample = {
                "step": step,
                "terminated": terminated.tolist(),
                "timeout": truncated.tolist(),
                "plate_pose": env.scene["plate"].data.root_link_pose_w.torch.tolist(),
                "score": env.state.arrays[mdp2.progress.score].torch.tolist(),
                "success": env.state.arrays[mdp2.progress.done].torch.tolist(),
                "root_holding": env.state.arrays[mdp2.progress.holding].torch.tolist(),
                "objects": {},
            }
            for name in names:
                data = env.scene[name].data
                sensor = env.scene[f"contact_sensor_{name}"].data
                sample["objects"][name] = {
                    "pose": data.root_link_pose_w.torch.tolist(),
                    "velocity": data.root_com_vel_w.torch.tolist(),
                    "force": sensor.normal_force_matrix_w.torch[:, 0, 0].tolist(),
                    "done_at": {
                        stage: env.state.arrays[mdp2.progress_field(f"{paths[name]}.{stage}", "done_at")].torch.tolist()
                        for stage in ("settle", "lift", "place")
                    },
                }
            samples.append(sample)
            if args.gif and (not frames or not samples[-2]["success"][0]):
                rgb = env.scene["camera"].data.output["rgb"].torch[0, :, :, :3]
                view = Image.fromarray(rgb.cpu().numpy())
                frame = Image.new("RGB", (view.width, view.height + 65), "#19202b")
                frame.paste(view)
                captions = [f"World 0   |   step {step}   |   task score {sample['score'][0]:.2f}"]
                for name in names:
                    reached = [stage for stage, ticks in sample["objects"][name]["done_at"].items() if ticks[0] >= 0]
                    captions.append(f"{name.title()}: {', '.join(reached) or 'waiting'}")
                ImageDraw.Draw(frame).multiline_text((10, view.height + 2), "\n".join(captions), fill="white")
                frames.append(frame)
            if env.episodes_completed == WORLDS:
                break
        assert env.episodes_completed == WORLDS
        records = {key: value.tolist() for key, value in env.episode_records().items()}
        with args.output.open("x") as stream:
            json.dump(
                {"rules": rules, "plate_size": PLATE_SIZE, "dt": DT, "steps": samples, "episodes": records}, stream
            )
        print(f"Newton completed {WORLDS} episodes in {len(samples)} control steps. Trace: {args.output}")
        if frames:
            with args.gif.open("xb") as stream:
                frames[0].save(stream, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
    finally:
        env.close()


if __name__ == "__main__":
    main()
