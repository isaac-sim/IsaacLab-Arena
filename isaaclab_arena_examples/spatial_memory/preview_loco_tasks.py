# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Export the G1 head-camera reset image and sampled geometry of each selected task without running a policy."""

import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image

from isaaclab_arena_environments.spatial_memory.loco_catalog import CHALLENGES, PILOT_CONTROLS


def main():
    from isaaclab.app import AppLauncher

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="+", choices=(*PILOT_CONTROLS, *CHALLENGES), default=(*PILOT_CONTROLS, *CHALLENGES)
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layout_seed", type=int, default=3847)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as recorder_directory, SimulationAppContext(args):
        from isaaclab_arena_environments.spatial_memory.episode_state import get_episode_state
        from isaaclab_arena_environments.spatial_memory.evaluation import make_spatial_memory_env

        for name in args.tasks:
            env = make_spatial_memory_env(
                name,
                1,
                args.layout_seed,
                observation_variant="mapped",
                enable_cameras=True,
                recorder_directory=recorder_directory,
            )
            try:
                env.reset()
                base = env.unwrapped
                camera = base.scene["robot_head_cam"]
                base.sim.step(render=True)
                base.scene.update(base.physics_dt)
                camera.update(base.step_dt, force_recompute=True)
                episode = get_episode_state(env)
                logic = episode.logic
                parts = []
                for part, T_E_part in zip(logic.part_specs(), episode.layout_poses_e[0], strict=True):
                    parts.append({"name": part.name, "size_xyz": part.size_xyz, "T_E_part": T_E_part.tolist()})
                record = {
                    "task": name,
                    "instruction": logic.task_description,
                    "layout_seed": args.layout_seed,
                    "parts": parts,
                }
                (args.output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
                rgb = camera.data.output["rgb"][0, ..., :3]
                Image.fromarray(rgb.cpu().numpy()).save(args.output / f"{name}.png")
            finally:
                env.close()


if __name__ == "__main__":
    main()
