# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render saved states from a successful physical rollout through the YAM top camera."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    args = parser.parse_args()
    output = args.recording.resolve()
    validation = json.loads((output / "validation.json").read_text())
    assert validation["passed"], "Render a successful physical rollout; playback does not validate task success"
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import get_app_launcher

    app = get_app_launcher(
        argparse.Namespace(headless=True, enable_cameras=True, visualizer=[], livestream=-1, device="cpu")
    ).app
    env = None
    writer = None
    code = 0
    try:
        import numpy as np
        import torch

        import imageio.v2 as imageio
        import omni.physx

        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_arena_environments.burvik_assembly_environment import (
            BurvikAssemblyEnvironment,
            BurvikAssemblyEnvironmentCfg,
        )

        torch.set_num_threads(1)
        config = BurvikAssemblyEnvironmentCfg(**validation["environment_config"])
        config.enable_top_camera = True
        arena = BurvikAssemblyEnvironment().build(config)
        env = ArenaEnvBuilder(
            arena,
            ArenaEnvBuilderCfg(
                device="cpu",
                solve_relations=False,
                disable_fabric=True,
                recorder_dataset_export_dir_path=str(output / "render_recordings"),
            ),
        ).make_registered()
        env.reset()
        base = env.unwrapped
        recording = np.load(output / "trajectory.npz")
        timestamps = recording["time"]
        fps = round(1.0 / float(np.median(np.diff(timestamps))))
        assert np.allclose(np.diff(timestamps), 1.0 / fps), "The saved trajectory must have uniform frame times"
        parts = validation["trajectory_part_names"]
        for side in ("left", "right"):
            assert base.scene[f"{side}_robot"].joint_names == validation["trajectory_joint_names"][side]
        writer = imageio.get_writer(
            str(output / "yam_top.mp4"),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            ffmpeg_params=["-movflags", "+faststart"],
        )
        for frame_index in range(len(timestamps)):
            # Playback writes recorded states. The physical policy rollout never uses these writes.
            for part_index, name in enumerate(parts):
                body = base.scene[name]
                pose = torch.as_tensor(recording["parts"][frame_index, part_index], device=base.device)[None]
                body.write_root_pose_to_sim_index(root_pose=pose)
            for side_index, side in enumerate(("left", "right")):
                robot = base.scene[f"{side}_robot"]
                positions = torch.as_tensor(recording["joints"][frame_index, side_index], device=base.device)[None]
                robot.write_joint_position_to_sim_index(position=positions)
            # Forward kinematics and rendering only: do not step dynamics, predicates, or policy.
            base.sim.forward()
            base.scene.update(1.0 / fps)
            # With Fabric disabled, explicitly publish tensor state to the USD renderer.
            for side in ("left", "right"):
                _ = base.scene[f"{side}_robot"].data.body_pose_w.torch
            omni.physx.get_physx_interface().update_transformations(False, True)
            base.sim.render()
            camera = base.scene["top_camera"]
            camera.update(1.0 / fps, force_recompute=True)
            frame = camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy().astype(np.uint8)
            if frame_index % fps == 0:
                for part_index, name in enumerate(parts):
                    actual = base.scene[name].data.root_pose_w.torch[0].cpu().numpy()
                    assert np.allclose(
                        actual, recording["parts"][frame_index, part_index], atol=1e-5
                    ), "Rendering changed the recorded physical state"
            writer.append_data(frame)
            if frame_index % fps == 0:
                imageio.imwrite(output / "latest_top.png", frame)
            if frame_index % (fps * 10) == 0:
                print(f"Rendered {frame_index}/{len(timestamps)} recorded frames", flush=True)
        imageio.imwrite(output / "final_top.png", frame)
        metadata = {
            "file": "yam_top.mp4",
            "camera": "/World/envs/env_0/top_camera",
            "fps": fps,
            "frames": len(timestamps),
            "source": "Saved states from the successful physical policy rollout",
            "physics_stepped_during_playback": False,
        }
        (output / "rendering.json").write_text(json.dumps(metadata, indent=2))
    except Exception:
        code = 1
        raise
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        app.close(exit_code=code)


if __name__ == "__main__":
    main()
