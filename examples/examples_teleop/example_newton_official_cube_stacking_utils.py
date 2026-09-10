"""Arena Docker wrapper for Newton's official Franka cube-stacking example."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import warp as wp

import newton
from newton.examples.ik.example_ik_cube_stacking import Example as OfficialCubeStackingExample


def _make_collision_only_urdf(urdf_path: Path) -> Path:
    output_path = Path("/tmp/newton_official_franka_collision_only.urdf")
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


class ArenaDockerCubeStackingExample(OfficialCubeStackingExample):
    """Official Newton cube-stacking example with optional URDF visuals."""

    def __init__(self, viewer, args):
        self.load_visuals = args.load_visuals
        super().__init__(viewer, args)

    def build_franka_with_table(self):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        franka_urdf = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
        urdf_path = franka_urdf if self.load_visuals else _make_collision_only_urdf(franka_urdf)
        builder.add_urdf(
            urdf_path,
            xform=wp.transform(self.robot_base_pos, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            hide_visuals=not self.load_visuals,
            parse_visuals_as_colliders=False,
        )

        builder.joint_q[:9] = [
            -3.6802115e-03,
            2.3901723e-02,
            3.6804110e-03,
            -2.3683236e00,
            -1.2918962e-04,
            2.3922248e00,
            7.8549200e-01,
            0.05,
            0.05,
        ]

        builder.joint_target_pos[:9] = [
            -3.6802115e-03,
            2.3901723e-02,
            3.6804110e-03,
            -2.3683236e00,
            -1.2918962e-04,
            2.3922248e00,
            7.8549200e-01,
            1.0,
            1.0,
        ]

        builder.joint_target_ke[:9] = [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]
        builder.joint_target_kd[:9] = [450, 450, 350, 350, 200, 200, 200, 10, 10]
        builder.joint_effort_limit[:9] = [87, 87, 87, 87, 12, 12, 12, 100, 100]
        builder.joint_armature[:9] = [0.3] * 4 + [0.11] * 3 + [0.15] * 2

        gravcomp_attr = builder.custom_attributes["mujoco:jnt_actgravcomp"]
        if gravcomp_attr.values is None:
            gravcomp_attr.values = {}
        for dof_idx in range(7):
            gravcomp_attr.values[dof_idx] = True

        gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp_body.values is None:
            gravcomp_body.values = {}
        for body_idx in range(2, 14):
            gravcomp_body.values[body_idx] = 1.0

        shape_cfg = newton.ModelBuilder.ShapeConfig(margin=0.0, density=1000.0)
        shape_cfg.ke = 5.0e4
        shape_cfg.kd = 5.0e2
        shape_cfg.kf = 1.0e3
        shape_cfg.mu = 0.75

        builder.add_shape_box(
            body=-1,
            hx=0.4,
            hy=0.4,
            hz=0.5 * self.table_height,
            xform=wp.transform(self.table_pos, wp.quat_identity()),
            cfg=shape_cfg,
        )

        if self.use_mujoco_contacts:
            condim_attr = builder.custom_attributes["mujoco:condim"]
            if condim_attr.values is None:
                condim_attr.values = {}
            for shape_idx in range(builder.shape_count):
                if builder.shape_body[shape_idx] in (12, 13):
                    condim_attr.values[shape_idx] = 4

        return builder
