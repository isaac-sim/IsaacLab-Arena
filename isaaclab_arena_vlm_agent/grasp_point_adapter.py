# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Authored grasp and contact points as privileged VLM policy input."""

from __future__ import annotations

import numpy as np
from collections.abc import Mapping
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from isaaclab_arena_vlm_agent.observation_adapter import VLMObservationAdapter
from isaaclab_arena_vlm_agent.pose_utils import read_robot_root_poses_w


@dataclass
class RevolutePointMotionCfg:
    """Describe rotation about a pivot in a scene object or articulation body's frame B."""

    scene_key: str
    """Scene object or articulation containing the pivot."""

    pivot_B_m: tuple[float, float, float]
    """Pivot position in frame B, in meters."""

    axis_B: tuple[float, float, float]
    """Nonzero rotation axis expressed in frame B; normalized before use."""

    body_name: str | None = None
    """Articulation body defining B, or None to use the scene object's frame."""

    direction: int = 1
    """Positive or negative rotation about axis_B, following the right-hand rule."""

    def __post_init__(self):
        assert self.scene_key, "Set the pivot scene_key"
        _finite_vector3(self.pivot_B_m)
        _unit_vector(_finite_vector3(self.axis_B))
        assert self.direction in (-1, 1), "direction must be -1 or 1"


@dataclass
class GraspPointCfg:
    """Locate a candidate grasp or contact point in a scene object's local frame O."""

    scene_key: str
    """Rigid object or scene extra defining O; use an object reference for a moving articulation part."""

    offset_O_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Metric offset in O from its origin, or from the selected local bounding-box point."""

    aabb_fraction: tuple[float, float, float] | None = None
    """Optional coordinates from 0 (minimum) to 1 (maximum) within the spawned local geometry bounds."""

    normal_O: tuple[float, float, float] | None = None
    """Optional nonzero outward surface normal in O; normalized before use."""

    revolute_motion: RevolutePointMotionCfg | None = None
    """Optional pivot and rotation direction used to compute the point's live motion tangent."""

    def __post_init__(self):
        assert self.scene_key, "Set the grasp point scene_key"
        _finite_vector3(self.offset_O_m)
        if self.aabb_fraction is not None:
            fraction = _finite_vector3(self.aabb_fraction)
            assert np.all((fraction >= 0) & (fraction <= 1)), "aabb_fraction must lie within [0, 1]"
        if self.normal_O is not None:
            _unit_vector(_finite_vector3(self.normal_O))


class GraspPointObservationAdapter(VLMObservationAdapter):
    """Report authored surface points and optional directions in each commanded robot's root frame.

    Points follow the selected environment's live object poses. They are privileged geometric
    candidates; this adapter does not predict grasps or check IK, collisions, or grasp stability.
    """

    def __init__(self, observation_adapter: VLMObservationAdapter, grasp_points: Mapping[str, GraspPointCfg]):
        """Wrap an adapter and select named contact points.

        Args:
            observation_adapter: Adapter supplying images and other prompt values.
            grasp_points: Named points attached to scene objects or moving object references.
        """
        assert grasp_points, "GraspPointObservationAdapter requires at least one point"
        assert all(name for name in grasp_points), "Grasp points require nonempty names"
        self.observation_adapter = observation_adapter
        self.grasp_points = dict(grasp_points)

    def extract_vlm_input_from_observation(self, env, observation, env_id: int):
        """Return the wrapped input with current grasp-point geometry for one environment."""
        images, prompt_values = self.observation_adapter.extract_vlm_input_from_observation(env, observation, env_id)
        base = env.unwrapped
        roots = read_robot_root_poses_w(base, env_id)
        points = {}
        for name, cfg in self.grasp_points.items():
            t_W_point, vectors_W = _point_geometry_w(base.arena_world, cfg, env_id)
            points_in_roots = {}
            for asset_name, T_W_R in roots.items():
                rotation_R_W = Rotation.from_quat(T_W_R[3:]).inv()
                geometry = {"position_root_m": rotation_R_W.apply(t_W_point - T_W_R[:3]).tolist()}
                for vector_name, vector_W in vectors_W.items():
                    geometry[f"{vector_name}_root"] = rotation_R_W.apply(vector_W).tolist()
                points_in_roots[asset_name] = geometry
            points[name] = {"scene_key": cfg.scene_key, "points_in_robot_roots": points_in_roots}
        assistance = dict(prompt_values.get("simulator_ground_truth_assistance", {}))
        assert "grasp_points" not in assistance, "grasp points already exist in prompt values"
        assistance["grasp_points"] = points
        return images, {**prompt_values, "simulator_ground_truth_assistance": assistance}


def _point_geometry_w(world, cfg: GraspPointCfg, env_id: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return a configured point and its optional unit vectors in the world frame W."""
    T_W_O = world.get_pose_w(cfg.scene_key)[env_id].detach().cpu().numpy()
    rotation_W_O = Rotation.from_quat(T_W_O[3:])
    t_O_point = np.array(cfg.offset_O_m, dtype=float)
    if cfg.aabb_fraction is not None:
        bounds_O = world.get_aabb_in_local_frame(cfg.scene_key)
        min_O = bounds_O.min_point[env_id].detach().cpu().numpy()
        max_O = bounds_O.max_point[env_id].detach().cpu().numpy()
        t_O_point += min_O + np.asarray(cfg.aabb_fraction) * (max_O - min_O)
    t_W_point = T_W_O[:3] + rotation_W_O.apply(t_O_point)
    vectors_W = {}
    if cfg.normal_O is not None:
        vectors_W["outward_normal"] = rotation_W_O.apply(_unit_vector(np.asarray(cfg.normal_O)))
    if cfg.revolute_motion is not None:
        motion = cfg.revolute_motion
        poses_W_B = (
            world.get_pose_w(motion.scene_key)
            if motion.body_name is None
            else world.get_body_pose_w(motion.scene_key, motion.body_name)
        )
        T_W_B = poses_W_B[env_id].detach().cpu().numpy()
        rotation_W_B = Rotation.from_quat(T_W_B[3:])
        t_W_pivot = T_W_B[:3] + rotation_W_B.apply(motion.pivot_B_m)
        axis_W = rotation_W_B.apply(_unit_vector(np.asarray(motion.axis_B)))
        tangent_W = motion.direction * np.cross(axis_W, t_W_point - t_W_pivot)
        vectors_W["motion_tangent"] = _unit_vector(tangent_W)
    return t_W_point, vectors_W


def _finite_vector3(value) -> np.ndarray:
    """Return a finite three-component float vector."""
    vector = np.asarray(value, dtype=float)
    assert vector.shape == (3,) and np.isfinite(vector).all(), "Expected a finite three-component vector"
    return vector


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    """Return the direction of a finite, nonzero vector."""
    length = np.linalg.norm(vector)
    assert np.isfinite(length) and length > 1e-9, "Expected a nonzero direction vector"
    return vector / length
