# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Capture passive resets of successive KitchenBench construction stages."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("spec")
parser.add_argument("--count", type=int, default=5)
parser.add_argument("--output-root", type=Path)
args = parser.parse_args()
app = AppLauncher({"headless": True, "enable_cameras": True}).app
import torch

from film_camera import make_camera
from PIL import Image

from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena_environments.cli import ensure_environments_registered

ensure_environments_registered()
path = Path(args.spec)
output = (args.output_root or path.parent / "frames") / path.stem
output.mkdir(parents=True, exist_ok=True)
arena = ArenaEnvGraphSpec.from_yaml(path).to_arena_env(enable_cameras=False)
if path.stem in ("01_kitchen", "02_wooden_bowl"):
    arena.embodiment = NoEmbodiment()
builder = ArenaEnvBuilder(arena, ArenaEnvBuilderCfg(num_envs=1, seed=42, placement_seed=42, resolve_on_reset=True))
cfg, kwargs = builder.build_env_cfg()
cfg.scene.film_camera = make_camera()
cfg.recorders = {}
cfg.episode_recorders = {}
env = builder.make_registered(cfg, kwargs)
records = []
try:
    with torch.inference_mode():
        env.reset()
        camera = env.unwrapped.scene["film_camera"]
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        if "robot" in env.unwrapped.scene.articulations:
            robot = env.unwrapped.scene["robot"]
            ids, _ = robot.find_joints("panda_joint[1-7]", preserve_order=True)
            actions[:, :7] = robot.data.default_joint_pos[:, ids]
        for _ in range(12):
            env.step(actions)
        for layout in range(args.count):
            env.reset()
            folder = output / f"layout_{layout:02d}"
            folder.mkdir(exist_ok=True)
            for _ in range(4):
                env.unwrapped.sim.render()
            trajectory = []
            for i in range(34):
                if i:
                    env.step(actions)
                camera.update(0, force_recompute=True)
                Image.fromarray(camera.data.output["rgb"][0, ..., :3].cpu().numpy()).save(
                    folder / f"frame_{i:03d}.jpg", quality=94
                )
                poses = {
                    name: obj.data.root_pose_w[0].cpu().tolist()
                    for name, obj in env.unwrapped.scene.rigid_objects.items()
                }
                poses.update({
                    name: obj.data.root_pose_w[0].cpu().tolist()
                    for name, obj in env.unwrapped.scene.articulations.items()
                })
                trajectory.append({"frame": i, "time_s": i * env.unwrapped.step_dt, "poses_W_xyzw": poses})
            records.append({"layout": layout, "trajectory": trajectory})
            (output / "captures.json").write_text(json.dumps(records, indent=2) + "\n")
            print(f"CAPTURED {path.stem} {layout+1}/{args.count}", flush=True)
finally:
    env.close()
    app.close()
