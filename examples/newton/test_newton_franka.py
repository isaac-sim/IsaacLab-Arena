# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""用 Newton 原生 API 检查固定基座 Franka 在重力下是否发生关节漂移。

该脚本不经过 Isaac Lab。默认配置保留机械臂真实质量和世界重力，使用与
Arena Franka IK 接近的关节 PD 参数，并启用 Newton/MuJoCo 原生重力补偿。

运行方法：

    .venv/bin/python examples/newton/test_newton_franka.py --viewer null
    .venv/bin/python examples/newton/test_newton_franka.py --no-gravity-compensation
    .venv/bin/python examples/newton/test_newton_franka.py --disable-pd --no-gravity-compensation
"""

import math
import numpy as np
import warnings

import newton
import newton.examples
import newton.utils
import warp as wp

FRAME_DT = 1.0 / 120.0
DEFAULT_SUBSTEPS = 2
DEFAULT_LOG_EVERY = 60
DEFAULT_NUM_FRAMES = 600
DRIFT_THRESHOLD_M = 1.0e-3
DRIFT_THRESHOLD_RAD = 1.0e-3

FRANKA_INITIAL_JOINT_POS = (
    0.0,
    -0.569,
    0.0,
    -2.810,
    0.0,
    3.037,
    0.741,
    0.04,
    0.04,
)
FRANKA_STIFFNESS = (400.0,) * 7 + (2000.0,) * 2
FRANKA_DAMPING = (80.0,) * 7 + (100.0,) * 2
FRANKA_EFFORT_LIMIT = (87.0,) * 4 + (12.0,) * 3 + (200.0,) * 2
FRANKA_ARMATURE = (1.0e-3,) * 7 + (0.0,) * 2


def _find_body(builder: newton.ModelBuilder, body_name: str) -> int:
    """Return the unique builder body whose label ends with ``body_name``."""
    body_ids = [index for index, label in enumerate(builder.body_label) if label.endswith(f"/{body_name}")]
    assert len(body_ids) == 1, f"应当唯一找到 {body_name}，实际匹配：{body_ids}。"
    return body_ids[0]


class Example:
    """Simulate a fixed-base Franka and report gravity-driven pose drift."""

    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True

        self.viewer = viewer
        self.num_frames = args.num_frames
        self.log_every = args.log_every
        self.sim_substeps = args.substeps
        self.sim_dt = FRAME_DT / self.sim_substeps
        self.sim_time = 0.0
        self.step_count = 0
        self.result_reported = False
        self.gravity_compensation = args.gravity_compensation
        self.pd_enabled = not args.disable_pd

        assert self.num_frames > 0, "--num-frames 必须为正整数。"
        assert self.log_every > 0, "--log-every 必须为正整数。"
        assert self.sim_substeps > 0, "--substeps 必须为正整数。"

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        franka_urdf = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
        builder.add_urdf(
            franka_urdf,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )

        self.ee_body = _find_body(builder, "fr3_hand_tcp")
        builder.joint_q[:9] = FRANKA_INITIAL_JOINT_POS
        builder.joint_target_q[:9] = FRANKA_INITIAL_JOINT_POS
        builder.joint_effort_limit[:9] = FRANKA_EFFORT_LIMIT
        builder.joint_armature[:9] = FRANKA_ARMATURE
        if self.pd_enabled:
            builder.joint_target_ke[:9] = FRANKA_STIFFNESS
            builder.joint_target_kd[:9] = FRANKA_DAMPING
        else:
            builder.joint_target_ke[:9] = [0.0] * 9
            builder.joint_target_kd[:9] = [0.0] * 9

        if self.gravity_compensation:
            body_gravcomp = builder.custom_attributes["mujoco:gravcomp"]
            if body_gravcomp.values is None:
                body_gravcomp.values = {}
            for body_index, label in enumerate(builder.body_label):
                if label.endswith("/base") or label.endswith("/fr3_link0"):
                    continue
                body_gravcomp.values[body_index] = 1.0

            joint_gravcomp = builder.custom_attributes["mujoco:jnt_actgravcomp"]
            if joint_gravcomp.values is None:
                joint_gravcomp.values = {}
            for dof_index in range(7):
                joint_gravcomp.values[dof_index] = True

        self.model = builder.finalize()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            iterations=100,
            ls_iterations=15,
            impratio=10.0,
            cone="elliptic",
            use_mujoco_contacts=False,
            update_data_interval=1,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        wp.copy(self.control.joint_target_q, self.model.joint_q)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.initial_joint_pos = self.state_0.joint_q.numpy()[:7].copy()
        self.initial_ee_pos = self.state_0.body_q.numpy()[self.ee_body, :3].copy()
        self.max_joint_drift = 0.0
        self.max_ee_drift = 0.0
        self.max_ee_drop = 0.0

        gravity = self.model.gravity.numpy()[0]
        target_modes = self.model.joint_target_mode.numpy()[:7]
        body_gravcomp = self.model.mujoco.gravcomp.numpy()
        joint_gravcomp = self.model.mujoco.jnt_actgravcomp.numpy()[:7]
        print(f"[INFO] Franka URDF: {franka_urdf}", flush=True)
        print(
            f"[INFO] gravity={gravity.tolist()}, frame_dt={FRAME_DT:.8f}, "
            f"substeps={self.sim_substeps}, sim_dt={self.sim_dt:.8f}",
            flush=True,
        )
        print(
            f"[INFO] PD={'开启' if self.pd_enabled else '关闭'}, "
            f"gravity_compensation={'开启' if self.gravity_compensation else '关闭'}",
            flush=True,
        )
        print(
            f"[INFO] arm target_mode={target_modes.tolist()}, "
            f"jnt_actgravcomp={joint_gravcomp.tolist()}, "
            f"gravcomp_nonzero={int(np.count_nonzero(body_gravcomp))}",
            flush=True,
        )
        print(
            f"[STATE] step=0000 t=0.0000 q={self.initial_joint_pos.tolist()} ee={self.initial_ee_pos.tolist()}",
            flush=True,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(1.5, -1.5, 1.2),
            pitch=-12.0,
            yaw=135.0,
        )
        self.graph = self._capture()

    def _simulate(self) -> None:
        """Advance one displayed frame with Newton MuJoCo-Warp substeps."""
        for _ in range(self.sim_substeps):
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _capture(self):
        """Capture the physics frame on CUDA and fall back to eager stepping."""
        if not wp.get_device().is_cuda:
            return None
        try:
            with wp.ScopedCapture() as capture:
                self._simulate()
            return capture.graph
        except Exception as exc:
            warnings.warn(f"CUDA graph capture 失败，改用 eager stepping：{exc}", stacklevel=2)
            return None

    def _read_drift(self) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """Read current joint/EE state and update maximum drift values."""
        joint_pos = self.state_0.joint_q.numpy()[:7].copy()
        ee_pos = self.state_0.body_q.numpy()[self.ee_body, :3].copy()
        joint_drift = float(np.max(np.abs(joint_pos - self.initial_joint_pos)))
        ee_drift = float(np.linalg.norm(ee_pos - self.initial_ee_pos))
        ee_drop = float(self.initial_ee_pos[2] - ee_pos[2])
        self.max_joint_drift = max(self.max_joint_drift, joint_drift)
        self.max_ee_drift = max(self.max_ee_drift, ee_drift)
        self.max_ee_drop = max(self.max_ee_drop, ee_drop)
        return joint_pos, ee_pos, joint_drift, ee_drift, ee_drop

    def _report_result(self) -> None:
        """Print an idempotent final gravity-drift verdict."""
        if self.result_reported:
            return
        self.result_reported = True
        joint_pos, ee_pos, joint_drift, ee_drift, ee_drop = self._read_drift()
        gravity_drift = self.max_joint_drift > DRIFT_THRESHOLD_RAD or self.max_ee_drift > DRIFT_THRESHOLD_M
        moved_down = self.max_ee_drop > DRIFT_THRESHOLD_M
        print(
            f"[FINAL] q={joint_pos.tolist()} ee={ee_pos.tolist()} "
            f"joint_drift={joint_drift:.8f} rad ee_drift={ee_drift:.8f} m ee_drop={ee_drop:.8f} m",
            flush=True,
        )
        print(
            f"[RESULT] 机械臂是否发生明显重力驱动漂移：{'是' if gravity_drift else '否'}；"
            f"末端是否明显下降：{'是' if moved_down else '否'}；"
            f"max_joint_drift={self.max_joint_drift:.8f} rad，"
            f"max_ee_drift={self.max_ee_drift:.8f} m，max_ee_drop={self.max_ee_drop:.8f} m。",
            flush=True,
        )

    def step(self) -> None:
        """Advance simulation and periodically print joint and EE drift."""
        if self.graph is None:
            self._simulate()
        else:
            wp.capture_launch(self.graph)

        self.step_count += 1
        self.sim_time += FRAME_DT
        joint_pos, ee_pos, joint_drift, ee_drift, ee_drop = self._read_drift()
        if self.step_count % self.log_every == 0 or self.step_count == self.num_frames:
            print(
                f"[STATE] step={self.step_count:04d} t={self.sim_time:.4f} "
                f"q={joint_pos.tolist()} ee={ee_pos.tolist()} "
                f"joint_drift={joint_drift:.8f} rad ee_drift={ee_drift:.8f} m "
                f"ee_drop={ee_drop:.8f} m",
                flush=True,
            )
        if self.step_count == self.num_frames:
            self._report_result()

    def render(self) -> None:
        """Render the current Newton state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        """Report the final result when invoked through ``--test``."""
        self._report_result()
        assert math.isfinite(self.max_joint_drift), "关节漂移结果不是有限数。"
        assert math.isfinite(self.max_ee_drift), "末端漂移结果不是有限数。"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=DEFAULT_NUM_FRAMES, gravity_compensation=True)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY, help="每隔多少帧打印一次状态。")
    parser.add_argument("--substeps", type=int, default=DEFAULT_SUBSTEPS, help="每个 1/120 s 帧的求解器子步数。")
    gravity_group = parser.add_mutually_exclusive_group()
    gravity_group.add_argument(
        "--gravity-compensation",
        dest="gravity_compensation",
        action="store_true",
        help="启用 Franka MuJoCo body gravcomp 和 arm joint actuatorgravcomp（默认）。",
    )
    gravity_group.add_argument(
        "--no-gravity-compensation",
        dest="gravity_compensation",
        action="store_false",
        help="关闭重力补偿，用作纯 PD 重力下沉对照。",
    )
    parser.add_argument("--disable-pd", action="store_true", help="关闭关节 PD，用作纯重力自由下落对照。")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
