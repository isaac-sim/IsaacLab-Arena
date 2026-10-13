# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the state-based YAM pose policy and save measured task results."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/ikea_burvik/yam_policy")
    parser.add_argument("--stop-after-leg", type=int, default=0, choices=range(5))
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import get_app_launcher

    app = get_app_launcher(
        argparse.Namespace(headless=True, enable_cameras=args.video, visualizer=[], livestream=-1, device="cpu")
    ).app
    env = None
    writer = None
    policy = None
    stability_samples = []
    trajectory = []
    frames = 0
    code = 0
    report = {
        "passed": False,
        "control": "YAM absolute TCP pose actions and physical finger contacts",
        "diagnostic_stop_after_leg": args.stop_after_leg,
    }
    try:
        import numpy as np
        import torch

        torch.set_num_threads(1)

        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_arena_environments.burvik_assembly.geometry import PART_NAMES
        from isaaclab_arena_environments.burvik_assembly.scripted_policy import (
            BurvikYamAssemblyPolicy,
            BurvikYamAssemblyPolicyCfg,
        )
        from isaaclab_arena_environments.burvik_assembly_environment import (
            BurvikAssemblyEnvironment,
            BurvikAssemblyEnvironmentCfg,
        )

        environment_config = BurvikAssemblyEnvironmentCfg(
            enable_top_camera=args.video, episode_length_s=args.max_seconds + 10.0
        )
        report["environment_config"] = asdict(environment_config)
        arena = BurvikAssemblyEnvironment().build(environment_config)
        env = ArenaEnvBuilder(
            arena,
            ArenaEnvBuilderCfg(
                device="cpu",
                solve_relations=False,
                disable_fabric=True,
                recorder_dataset_export_dir_path=str(output / "arena_recordings"),
            ),
        ).make_registered()
        base = env.unwrapped
        report["trajectory_part_names"] = list(PART_NAMES)
        report["trajectory_joint_names"] = {side: base.scene[f"{side}_robot"].joint_names for side in ("left", "right")}
        base.configure_episode_limit(1)
        observation, _ = env.reset()
        policy = BurvikYamAssemblyPolicy(BurvikYamAssemblyPolicyCfg(stop_after_leg=args.stop_after_leg))
        repository = Path(__file__).resolve().parents[2]
        sources = [
            Path(__file__).with_name("scripted_policy.py"),
            Path(__file__).with_name("geometry.py"),
            Path(__file__).with_name("connections.py"),
            Path(__file__).with_name("task.py"),
            repository / "isaaclab_arena_environments/burvik_assembly_environment.py",
            repository / "isaaclab_arena/embodiments/yam/actions.py",
            repository / "isaaclab_arena/embodiments/yam/control.py",
            repository / "isaaclab_arena/embodiments/yam/kinematics.py",
            repository / "isaaclab_arena/embodiments/yam/yam.py",
        ]
        report["source_sha256"] = {
            str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        }
        for path in sources:
            snapshot = output / "source" / path.relative_to(repository)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(path.read_bytes())
        if args.video:
            import imageio.v2 as imageio

            writer = imageio.get_writer(
                str(output / "yam_top.mp4"),
                fps=24,
                codec="libx264",
                quality=8,
                macro_block_size=1,
                ffmpeg_params=["-movflags", "+faststart"],
            )
        for step in range(round(args.max_seconds / base.step_dt)):
            action = policy.get_action(env, observation)
            # Completion describes the state just observed; do not take an extra unrequested step.
            if policy.finished:
                break
            observation, _, terminated, truncated, _ = env.step(action)
            if step % 10 == 0:
                trajectory.append({
                    "time": (step + 1) * base.step_dt,
                    "actions": action[0].cpu().numpy().copy(),
                    "parts": np.array(
                        [base.arena_world.get_pose_w(name)[0].cpu().numpy().copy() for name in PART_NAMES]
                    ),
                    "joints": np.array([
                        base.scene[f"{side}_robot"].data.joint_pos.torch[0].cpu().numpy().copy()
                        for side in ("left", "right")
                    ]),
                })
            if writer is not None and step % 10 == 0:
                base.sim.render()
                camera = base.scene["top_camera"]
                camera.update(base.step_dt, force_recompute=True)
                frame = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
                writer.append_data(frame)
                frames += 1
                if step % 240 == 0:
                    imageio.imwrite(output / "latest_top.png", frame)
            if step % 240 == 0:
                state = {name: base.arena_world.get_pose_w(name)[0].tolist() for name in PART_NAMES}
                state["tcp"] = {
                    side: np.r_[policy.tcp(side)[0], policy.tcp(side)[1].as_quat()].tolist()
                    for side in ("left", "right")
                }
                state["targets"] = {side: np.r_[p, r.as_quat()].tolist() for side, (p, r) in policy.targets.items()}
                state["completion"] = base.progress_tracker.get_subtask_completion()[0].tolist()
                state["current_task_success"] = base.progress_tracker.is_complete().tolist()
                state["part_linear_speeds"] = {
                    name: float(base.scene[name].data.root_lin_vel_w.torch[0].norm()) for name in PART_NAMES
                }
                state["measured_settled"] = (
                    base.burvik_motion_measurement.settled.tolist()
                    if hasattr(base, "burvik_motion_measurement")
                    else None
                )
                state["joints"] = {
                    side: base.scene[f"{side}_robot"].data.joint_pos.torch[0].tolist() for side in ("left", "right")
                }
                state["phase"] = policy.phase
                state["time_s"] = step * base.step_dt
                (output / "latest_state.json").write_text(json.dumps(state, indent=2))
            if policy.phase.startswith("verify:"):
                stability_samples.append(
                    np.array([base.arena_world.get_pose_w(name)[0, :3].cpu().numpy().copy() for name in PART_NAMES])
                )
            if bool(terminated.any()) and not bool(base.progress_tracker.is_complete()[0]):
                raise RuntimeError(f"Arena reported task failure during {policy.phase}")
            if bool(truncated.any()):
                raise RuntimeError("Assembly episode timed out")
        assert policy.finished, f"Policy exceeded {args.max_seconds}s in {policy.phase}"
        completion = base.progress_tracker.get_subtask_completion()[0].tolist()
        report["completion"] = completion
        report["current_task_success"] = bool(base.progress_tracker.is_complete()[0])
        report["passed"] = report["current_task_success"] and args.stop_after_leg == 0
        if stability_samples:
            samples = np.array(stability_samples)
            drift = np.linalg.norm(samples - samples[0], axis=-1).max(axis=0)
            report["released_stability"] = {
                "duration_s": len(samples) * base.step_dt,
                "maximum_drift_m": dict(zip(PART_NAMES, drift.tolist())),
                "passed": bool((drift < 0.002).all()),
            }
            report["passed"] &= report["released_stability"]["passed"]
        if not args.stop_after_leg:
            assert report["passed"], f"Task incomplete: {completion}"
    except Exception:
        import traceback

        code = 1
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    finally:
        if policy is not None:
            report["transitions"] = policy.transitions
            report["policy_failure"] = policy.failure
        if env is not None:
            from isaaclab_arena_environments.burvik_assembly.connections import attachment_errors
            from isaaclab_arena_environments.burvik_assembly.geometry import ASSEMBLY_ORDER, attachment_pose

            base = env.unwrapped
            report["completion"] = base.progress_tracker.get_subtask_completion()[0].tolist()
            report["current_task_success"] = bool(base.progress_tracker.is_complete()[0])
            report["arena_events"] = [asdict(event) for event in base.progress_tracker.get_events()[0]]
            if policy is not None:
                report["seating_errors"] = {}
                for name in ASSEMBLY_ORDER:
                    p, r = attachment_pose(name, policy.spec)
                    distance, angle = attachment_errors(base, name, p.tolist(), r.as_quat().tolist())
                    report["seating_errors"][name] = {
                        "distance_m": float(distance[0]),
                        "angle_rad": float(angle[0]),
                    }
            report["bench_support_force_n"] = sum(
                base.scene[f"leg_{index}_bench_contact"].data.normal_force_matrix_w.torch[0, 0, 0]
                for index in range(1, 5)
            ).tolist()
            base.sim.stage.Export(str(output / "final_stage.usda"))
        if trajectory:
            np.savez_compressed(
                output / "trajectory.npz",
                **{key: np.array([item[key] for item in trajectory]) for key in trajectory[0]},
            )
        if writer is not None:
            report["video"] = {
                "file": "yam_top.mp4",
                "camera": "/World/envs/env_0/top_camera",
                "fps": 24,
                "frames": frames,
            }
        (output / "validation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        app.close(exit_code=code)


if __name__ == "__main__":
    main()
