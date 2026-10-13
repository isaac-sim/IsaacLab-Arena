# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the BURVIK displacement-guided demo through an actual Arena environment."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--record-cameras", nargs="+", choices=("overview", "viewport", "top"), default=["overview"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--check-reachability", action="store_true")
    parser.add_argument("--output", default="artifacts/ikea_burvik/arena_results")
    args = parser.parse_args()
    assert (
        args.smoke or args.num_envs == 1
    ), "The full scripted demo runs one environment; use --smoke for parallel resets."
    recording_cameras = list(dict.fromkeys(args.record_cameras)) if args.video else []
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import get_app_launcher

    launcher = get_app_launcher(
        argparse.Namespace(
            headless=True,
            enable_cameras=args.video,
            visualizer=["kit"] if "viewport" in recording_cameras else [],
            livestream=-1,
            device="cpu",
            width=1280,
            height=720,
        )
    )
    app = launcher.app
    env = None
    exit_code = 0
    try:
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_arena_environments.burvik_assembly_environment import (
            BurvikAssemblyEnvironment,
            BurvikAssemblyEnvironmentCfg,
        )

        factory = BurvikAssemblyEnvironment()
        arena = factory.build(
            BurvikAssemblyEnvironmentCfg(
                action_mode="joint_position",
                enable_scene_camera="overview" in recording_cameras,
                enable_top_camera="top" in recording_cameras,
            )
        )
        if "viewport" in recording_cameras:
            from isaaclab_visualizers.kit import KitVisualizerCfg

            configure_scene = arena.env_cfg_callback

            def configure_recording(env_cfg):
                env_cfg = configure_scene(env_cfg)
                env_cfg.sim.visualizer_cfgs = [
                    KitVisualizerCfg(
                        headless=True,
                        eye=env_cfg.viewer.eye,
                        lookat=env_cfg.viewer.lookat,
                        focal_length=55.0,
                        window_width=1280,
                        window_height=720,
                        enable_live_plots=False,
                    )
                ]
                return env_cfg

            arena.env_cfg_callback = configure_recording
        builder = ArenaEnvBuilder(
            arena,
            ArenaEnvBuilderCfg(
                num_envs=args.num_envs,
                device="cpu",
                solve_relations=False,
                disable_fabric=True,
            ),
        )
        env = builder.make_registered()
        print("[BURVIK ARENA] Built; resetting", flush=True)
        if not args.smoke:
            env.unwrapped.configure_episode_limit(1)
        env.reset()
        print(
            f"[BURVIK ARENA] Workbench pose: {env.unwrapped.arena_world.get_pose_w('workbench').tolist()}", flush=True
        )
        if args.check_reachability:
            from isaaclab_arena_environments.burvik_assembly.reachability import check_reachability

            root = Path(BurvikAssemblyEnvironmentCfg().asset_directory)
            result = check_reachability(env.unwrapped, json.loads((root / "specification.json").read_text()))
            (output / "reachability.json").write_text(json.dumps(result, indent=2))
            print(f"[BURVIK ARENA] Reachability passed: {result['passed']}", flush=True)
            assert result["passed"], "Some pickup/insertion targets are outside the joint-limited TCP workspace"
        demo = AssemblyDemo(env, output, video=args.video, recording_cameras=recording_cameras)
        demo.run(smoke=args.smoke)
    except Exception:
        import traceback

        exit_code = 1
        error = traceback.format_exc()
        print(error, flush=True)
        (output / "error.txt").write_text(error)
        raise
    finally:
        if env is not None:
            env.close()
        app.close(exit_code=exit_code)


class AssemblyDemo:
    """Keep dynamics active while guiding part velocities through measured socket targets."""

    def __init__(self, env, output, video=False, recording_cameras=None):
        import torch

        from isaaclab_arena_environments.burvik_assembly.geometry import PART_NAMES

        self.env = env
        self.base = env.unwrapped
        self.output = output
        self.dt = self.base.step_dt
        self.drives = {}
        self.attached = []
        self.phase = "Loose parts settling"
        self.steps = 0
        self.success = False
        self.report = {
            "control": "Scripted dynamic part velocities; YAM joint targets hold its parked pose.",
            "subtasks": [],
            "stability": {},
            "passed": False,
        }
        self.actions = torch.zeros(
            (self.base.num_envs, self.base.action_manager.total_action_dim), device=self.base.device
        )
        for side, offset in (("left", 0), ("right", 7)):
            robot = self.base.scene[f"{side}_robot"]
            ids = [robot.joint_names.index(f"joint{i}") for i in range(1, 7)]
            self.actions[:, offset : offset + 6] = robot.data.default_joint_pos.torch[:, ids]
        self.names = PART_NAMES
        self.cameras = {}
        self.writers = {}
        self.viewport = None
        self.recorded_frames = 0
        self.trajectory = []
        if video:
            import imageio.v2 as imageio
            from pxr import UsdGeom

            streams = {}
            for name in recording_cameras or ["overview"]:
                if name == "viewport":
                    self.viewport = next(v for v in self.base.sim.visualizers if v.cfg.visualizer_type == "kit")
                    camera_path = "/OmniverseKit_Persp"
                    camera = UsdGeom.Camera.Get(self.base.sim.stage, camera_path)
                    assert camera, "Kit viewport camera is missing"
                    camera.GetHorizontalApertureAttr().Set(36.0)
                    camera.GetVerticalApertureAttr().Set(20.25)
                    camera.GetFocalLengthAttr().Set(55.0)
                else:
                    sensor_name = "demo_camera" if name == "overview" else "top_camera"
                    self.cameras[name] = self.base.scene[sensor_name]
                    camera_path = f"{self.base.scene.env_prim_paths[0]}/{sensor_name}"
                filename = "burvik_arena_assembly.mp4" if name == "overview" else f"burvik_arena_{name}.mp4"
                self.writers[name] = imageio.get_writer(
                    str(output / filename),
                    fps=24,
                    codec="libx264",
                    quality=8,
                    macro_block_size=1,
                    ffmpeg_params=["-movflags", "+faststart"],
                )
                camera = UsdGeom.Camera.Get(self.base.sim.stage, camera_path)
                position = UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(0).ExtractTranslation()
                streams[name] = {
                    "file": filename,
                    "camera_prim": camera_path,
                    "fps": 24,
                    "resolution": [1280, 720],
                    "position_w_m": list(position),
                }
            self.report["video_streams"] = streams
            for _ in range(16):
                self.base.sim.render()
                if self.viewport is not None:
                    self.viewport.render_rgb_array()

    def pose(self, name):
        from scipy.spatial.transform import Rotation

        T_W_P = self.base.arena_world.get_pose_w(name)[0].cpu().numpy()
        return T_W_P[:3].copy(), Rotation.from_quat(T_W_P[3:])

    def step(self, seconds):
        import numpy as np
        import torch

        from isaaclab_arena_environments.burvik_assembly.connections import attachment_enabled

        for _ in range(round(seconds / self.dt)):
            for name, (target, target_r, limit) in list(self.drives.items()):
                if name != "tray" and bool(attachment_enabled(self.base, name)[0]):
                    self.drives.pop(name)
                    continue
                p, r = self.pose(name)
                gain = 60.0 if name in ("tray", "handle") else 18.0
                velocity = gain * (np.asarray(target) - p)
                velocity *= min(1.0, limit / max(np.linalg.norm(velocity), 1e-8))
                velocity[2] += 9.81 * self.dt
                angular = (60.0 if name in ("tray", "handle") else 16.0) * (target_r * r.inv()).as_rotvec()
                angular *= min(1.0, 1.5 / max(np.linalg.norm(angular), 1e-8))
                # Root velocity writes are COM velocities. Account for the offset from the part origin.
                body = self.base.scene[name]
                com = body.data.root_com_pos_w.torch[0].cpu().numpy()
                velocity += np.cross(angular, com - p)
                command = torch.tensor(np.r_[velocity, angular], dtype=torch.float32, device=self.base.device)
                body.write_root_velocity_to_sim_index(root_velocity=command.expand(self.base.num_envs, -1).contiguous())
            _, _, terminated, truncated, extras = self.env.step(self.actions)
            self.steps += 1
            if bool(terminated.any() or truncated.any()):
                state = extras["progress_tracking"]["states"][0]
                assert state.all_complete and not bool(
                    truncated.any()
                ), f"Unexpected termination in {self.phase}: {state}"
                self.success = True
                self.report["completion_time_s"] = self.steps * self.dt
                self.report["final_progress"] = {
                    "overall_score": state.overall_score,
                    "all_complete": state.all_complete,
                }
                # Preserve the final physics state by using Arena's one-episode budget (set after diagnostics).
            if self.steps % 10 == 0:
                self.trajectory.append({
                    "time_s": self.steps * self.dt,
                    "phase": self.phase,
                    "poses": {n: [*self.pose(n)[0], *self.pose(n)[1].as_quat()] for n in self.names},
                })
                if self.writers:
                    self.capture()
            if self.steps % 1200 == 0:
                print(f"[BURVIK ARENA] {self.steps * self.dt:.1f}s {self.phase}", flush=True)

    def camera_frame(self, name):
        """Read the current viewport or named YAM/overview sensor RGB frame."""
        if name == "viewport":
            return self.viewport.render_rgb_array()
        return self.cameras[name].data.output["rgb"].torch.cpu().numpy()[0, :, :, :3]

    def capture(self):
        import numpy as np

        from PIL import Image, ImageDraw, ImageFont

        self.base.sim.render()
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
        for name, writer in self.writers.items():
            rgb = self.camera_frame(name)
            assert rgb is not None and rgb.shape == (720, 1280, 3) and rgb.std() > 1.0, f"Invalid {name} frame"
            picture = Image.fromarray(rgb.astype(np.uint8))
            draw = ImageDraw.Draw(picture, "RGBA")
            draw.rectangle((0, 0, 1280, 100), fill=(13, 20, 30, 235))
            draw.text((25, 14), f"ARENA / YAM / BURVIK / {name.upper()} CAMERA", font=font, fill="white")
            draw.text(
                (25, 54),
                f"{self.phase} | {len(self.attached)}/5 attached | {self.steps * self.dt:.1f} s",
                font=font,
                fill="white",
            )
            draw.rectangle((0, 675, 1280, 720), fill=(13, 20, 30, 235))
            draw.text(
                (25, 683), "Scripted part guidance; robot grasp execution is not demonstrated", font=font, fill="white"
            )
            writer.append_data(np.asarray(picture))
            if self.steps % 240 == 0:
                filename = "latest_frame.png" if name == "overview" else f"latest_{name}.png"
                picture.save(self.output / filename)
        self.recorded_frames += 1

    def move(self, name, target, rotation, duration=1.5, speed=0.30):
        import numpy as np
        from scipy.spatial.transform import Rotation, Slerp

        p, r = self.pose(name)
        target = np.asarray(target)
        duration = max(
            duration,
            1.7 * np.linalg.norm(target - p) / speed,
            1.7 * np.linalg.norm((rotation * r.inv()).as_rotvec()) / 1.5,
        )
        slerp = Slerp([0, 1], Rotation.concatenate([r, rotation]))
        ticks = round(duration / self.dt)
        for tick in range(ticks):
            t = (tick + 1) / ticks
            s = t * t * (3 - 2 * t)
            self.drives[name] = (p + s * (target - p), slerp([s])[0], speed)
            self.step(self.dt)
        self.drives[name] = (target, rotation, speed)
        self.step(0.3)
        for _ in range(240):
            actual_p, actual_r = self.pose(name)
            if (
                np.linalg.norm(actual_p - target) < 0.002
                and np.linalg.norm((rotation * actual_r.inv()).as_rotvec()) < 0.02
            ):
                return
            self.step(self.dt)
        assert (
            np.linalg.norm(actual_p - target) < 0.004
        ), f"{name} blocked during {self.phase}: target={target}, actual={actual_p}"

    def seat(self, name, spec):
        from isaaclab_arena_environments.burvik_assembly.connections import attachment_enabled, attachment_errors
        from isaaclab_arena_environments.burvik_assembly.geometry import attachment_pose

        local_p, local_r = attachment_pose(name, spec)

        def target():
            tray_p, tray_r = self.pose("tray")
            p, r = tray_p + tray_r.apply(local_p), tray_r * local_r
            axis = r.apply([0, 0, 1])
            return p, r, axis * (1 if name == "handle" else -1)

        self.phase = f"Install {name}"
        print(f"[BURVIK ARENA] {self.phase}", flush=True)
        p, r = self.pose(name)
        self.move(name, p + [0, 0, 0.32], r)
        goal, goal_r, outward = target()
        pre = goal + outward * 0.085
        clearance = pre + [0, 0, 0.25]
        self.move(name, clearance, goal_r)
        self.move(name, pre, goal_r)
        for tick in range(720):
            goal, goal_r, outward = target()
            self.drives[name] = (goal + outward * 0.085 * (1 - (tick + 1) / 720), goal_r, 0.045)
            self.step(self.dt)
        for _ in range(480):
            if bool(attachment_enabled(self.base, name).all()):
                break
            goal, goal_r, _ = target()
            self.drives[name] = (goal, goal_r, 0.035)
            self.step(self.dt)
        self.step(0.5)
        assert bool(attachment_enabled(self.base, name).all()), f"{name} did not latch"
        distance, angle = attachment_errors(self.base, name, local_p.tolist(), local_r.as_quat().tolist())
        self.attached.append(name)
        self.report["subtasks"].append({
            "part": name,
            "time_s": self.steps * self.dt,
            "position_error_m": distance.tolist(),
            "angle_error_rad": angle.tolist(),
        })
        self.drives.pop(name, None)
        self.step(0.3)
        completion = self.base.progress_tracker.get_subtask_completion()[0].tolist()
        self.report["subtasks"][-1]["arena_subtask_completion"] = completion
        if name != "handle":
            expected = [index < len(self.attached) for index in range(5)]
            assert completion == expected, f"Unexpected Arena subtask order after {name}: {completion}"

    def stability(self, label, duration):
        import numpy as np
        from scipy.spatial.transform import Rotation

        assert not self.drives
        samples = []
        orientations = []
        velocities = []
        for _ in range(round(duration / 0.05)):
            self.step(0.05)
            samples.append([self.pose(name)[0] for name in self.names])
            orientations.append([self.pose(name)[1].as_quat() for name in self.names])
            velocities.append(
                [self.base.arena_world.get_root_linear_velocity_w(name)[0].cpu().numpy().copy() for name in self.names]
            )
        positions = np.array(samples)
        drift = np.linalg.norm(positions - positions[0], axis=-1).max(axis=0)
        orientations = np.array(orientations)
        angle_drift = []
        for index in range(len(self.names)):
            rotations = Rotation.from_quat(orientations[:, index])
            angle_drift.append(float(np.degrees((rotations * rotations[0].inv()).magnitude()).max()))
        speeds = np.linalg.norm(velocities, axis=-1).max(axis=0)
        self.report["stability"][label] = {
            "duration_s": duration,
            "part_guidance_active": False,
            "max_drift_m": dict(zip(self.names, drift.tolist())),
            "max_rotation_drift_deg": dict(zip(self.names, angle_drift)),
            "max_linear_speed_m_s": dict(zip(self.names, speeds.tolist())),
            "passed": bool((drift < 0.002).all() and max(angle_drift) < 1.0 and (speeds < 0.02).all()),
        }
        assert self.report["stability"][label]["passed"], f"Unstable {label}: {drift}"

    def run(self, smoke=False):
        import numpy as np
        from scipy.spatial.transform import Rotation

        from isaaclab_arena_environments.burvik_assembly.geometry import ASSEMBLY_ORDER, BENCH_HEIGHT
        from isaaclab_arena_environments.burvik_assembly_environment import BurvikAssemblyEnvironmentCfg

        spec = json.loads((Path(BurvikAssemblyEnvironmentCfg().asset_directory) / "specification.json").read_text())
        try:
            self.step(2.0)
            self.stability("loose_parts", 1.0)
            if smoke:
                if self.base.num_envs > 1:
                    self.check_partial_reset(spec)
                self.report["passed"] = True
                return
            tray_p, tray_r = self.pose("tray")
            self.drives["tray"] = (tray_p, tray_r, 0.3)
            for name in ASSEMBLY_ORDER[:-1]:
                self.seat(name, spec)
            self.phase = "Reorient for carry handle"
            self.move("tray", (0.0, 0.06, 1.24), tray_r, 2.0)
            side_rotation = Rotation.from_euler("y", 90, degrees=True)
            self.move("tray", (0.0, 0.06, 1.24), side_rotation, 3.0)
            self.move("tray", (0.0, 0.06, 1.12), side_rotation, 1.5)
            self.seat("handle", spec)
            self.phase = "Lower and release assembled table"
            self.move("tray", (0.0, 0.06, 1.26), side_rotation, 1.5)
            self.move("tray", (0.0, 0.06, 1.26), Rotation.identity(), 2.5)
            self.move("tray", (0.0, 0.06, BENCH_HEIGHT + 0.424), Rotation.identity(), 2.0, speed=0.10)
            self.drives.clear()
            self.step(2.0)
            self.stability("assembled_free", 5.0)
            assert self.success, "Arena did not terminate with task success"
            from dataclasses import asdict

            self.report["progress_events"] = [asdict(event) for event in self.base.progress_tracker.get_events()[0]]
            self.report["consecutive_step_summary"] = self.base.progress_tracker.get_consecutive_step_summary(0)
            self.report["final_subtask_completion"] = self.base.progress_tracker.get_subtask_completion().tolist()
            self.report["final_bench_support_force_N"] = sum(
                self.base.scene[f"leg_{i}_bench_contact"].data.normal_force_matrix_w.torch[0, 0, 0].tolist()[2]
                for i in range(1, 5)
            )
            from pxr import UsdPhysics

            self.report["all_parts_dynamic"] = all(
                not UsdPhysics.RigidBodyAPI(
                    self.base.sim.stage.GetPrimAtPath(f"{self.base.scene.env_prim_paths[0]}/{name}")
                )
                .GetKinematicEnabledAttr()
                .Get()
                for name in self.names
            )
            self.report["passed"] = True
            self.phase = "Task complete / 5 of 5 / released and stable"
            if self.writers:
                self.capture()
                from PIL import Image

                for name in self.writers:
                    filename = "assembled.png" if name == "overview" else f"assembled_{name}.png"
                    Image.fromarray(self.camera_frame(name)).save(self.output / filename)
            self.base.sim.stage.Export(str(self.output / "arena_assembled.usda"))
            print("[BURVIK ARENA] FIVE-STEP TASK AND RELEASED STABILITY PASSED", flush=True)
        finally:
            if self.writers:
                self.report["recorded_frames_per_stream"] = self.recorded_frames
            (self.output / "validation.json").write_text(json.dumps(self.report, indent=2))
            np.savez_compressed(self.output / "trajectory.npz", frames=np.array(self.trajectory, dtype=object))
            for writer in self.writers.values():
                writer.close()

    def check_partial_reset(self, specification):
        """Verify real connection isolation while resetting one of two live environments."""
        import torch

        from isaaclab.utils.math import combine_frame_transforms

        from isaaclab_arena_environments.burvik_assembly.connections import (
            attachment_enabled,
            engage_seated_connections,
        )
        from isaaclab_arena_environments.burvik_assembly.geometry import attachment_pose

        position, rotation = attachment_pose("leg_1", specification)
        transform = {"position": position.tolist(), "quaternion": rotation.as_quat().tolist()}
        T_W_T = self.base.arena_world.get_pose_w("tray")[1:2]
        p, q = combine_frame_transforms(
            T_W_T[:, :3],
            T_W_T[:, 3:],
            torch.tensor([transform["position"]], device=self.base.device),
            torch.tensor([transform["quaternion"]], device=self.base.device),
        )
        ids = torch.tensor([1], device=self.base.device)
        self.base.scene["leg_1"].write_root_pose_to_sim_index(root_pose=torch.cat((p, q), dim=-1), env_ids=ids)
        engage_seated_connections(self.base, None, {"leg_1": transform})
        assert attachment_enabled(self.base, "leg_1")[:2].tolist() == [False, True]
        previous_step = self.base.episode_length_buf[1].clone()
        self.base.reset(env_ids=torch.tensor([0], device=self.base.device))
        assert attachment_enabled(self.base, "leg_1")[:2].tolist() == [False, True]
        assert self.base.episode_length_buf[1] == previous_step
        self.base.reset()
        assert not attachment_enabled(self.base, "leg_1").any()
        self.report["partial_reset_isolation"] = True


if __name__ == "__main__":
    main()
