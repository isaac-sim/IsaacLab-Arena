# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Translate measured physical workcell state into servicing-model evidence."""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pxr import Gf, Sdf, UsdGeom, UsdPhysics

from isaaclab_arena.geometry.containment import BoxRegion, ContainmentMeasurement, RegionContainment
from isaaclab_arena.tasks.predicates.spatial import relative_pose_matches
from isaaclab_arena.tasks.task_runtime import TaskRuntime

from .connectors import RetainedSocket, Socket, environment_prim_path, relative_pose
from .model import ServiceModel, ServiceSnapshot
from .scenarios import SCENARIOS
from .scene import DEBRIS_NAMES

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.gripper import ParallelJawGripper

    from .scene import ServiceLayout


@dataclass(slots=True)
class ServiceRuntimeCfg:
    """Bind geometry, evaluator-only scenarios, and the embodiment's gripper interface."""

    layout: ServiceLayout = field(repr=False)
    scenario_names: tuple[str, ...]
    gripper: ParallelJawGripper | None = None


class ServiceRuntime(TaskRuntime):
    """Advance physical latches and functional instruments once per control step."""

    def __init__(self, env, cfg: ServiceRuntimeCfg) -> None:
        self.env = env
        self.cfg = cfg
        self.layout = cfg.layout
        self.scenario_names = [cfg.scenario_names[i % len(cfg.scenario_names)] for i in range(env.num_envs)]
        self.models = [ServiceModel(SCENARIOS[name]) for name in self.scenario_names]
        self.statuses = [model.status for model in self.models]
        self.snapshots: list[ServiceSnapshot | None] = [None] * env.num_envs
        self._last_steps = [-1] * env.num_envs
        self._power_remaining = [0.0] * env.num_envs
        self._previous_flow_pressed = [False] * env.num_envs
        self.sockets = {}
        for name in ("cradle", "battery", "cup", "filter"):
            specification = self.layout.sockets[name]
            socket = Socket(
                name,
                specification.parent_name,
                specification.candidate_names,
                specification.pose_in_parent.position_xyz,
                specification.pose_in_parent.rotation_xyzw,
                position_tolerance_m=specification.position_tolerance_m,
                capture_bounds=specification.capture_bounds,
                candidate_bounds=specification.candidate_bounds,
            )
            self.sockets[name] = RetainedSocket(env, socket)
        self._case_locks = self._create_case_locks()
        self._display_states: dict[tuple[int, str], tuple[str, float | None]] = {}
        self._bounds_cache: dict[tuple[str, ...], torch.Tensor] = {}
        self.region_containment = RegionContainment(env.cfg.scene, unit_scale_regions=tuple(self.layout.regions))
        self.regions = {
            name: BoxRegion(name, self.layout.source_records[name]["affordances"]["interior_bounds"])
            for name in self.layout.regions
        }
        self._stock_targets = self._make_stock_targets()
        self._case_interior: torch.Tensor | None = None
        self._case_overlap_bounds: torch.Tensor | None = None
        self._case_region_definition = self._case_region_configuration()
        self._cup_inlet: torch.Tensor | None = None
        self._settled_cache: dict[str, list[bool]] | None = None
        self._velocity_names = tuple(env.scene.rigid_objects)
        self._case_names = tuple(
            name
            for name in self._velocity_names
            if name in self.layout.source_records and name not in ("cradle", "battery_tester", "airflow_tester")
        )

    def _make_stock_targets(self):
        """Keep return targets in their fixture frames so coherent layouts can move."""
        from isaaclab_arena.utils.pose import Pose

        parents = {
            "battery_original": "body",
            "filter_original": "body",
            "battery_spare": "spare_rack",
            "battery_decoy": "spare_rack",
            "filter_spare": "spare_rack",
            "filter_decoy": "spare_rack",
            "crevice_tool": "parking_tray",
            "brush_tool": "parking_tray",
            "airflow_adapter": "airflow_tester",
        }
        targets = {}
        for name, parent in parents.items():
            T_E_P = self.layout.initial_poses[parent].to_tensor(device="cpu").unsqueeze(0)
            T_E_O = self.layout.initial_poses[name].to_tensor(device="cpu").unsqueeze(0)
            T_P_O = relative_pose(T_E_P, T_E_O)[0].tolist()
            targets[name] = (parent, Pose(tuple(T_P_O[:3]), tuple(T_P_O[3:])))
        return targets

    def _create_case_locks(self):
        """Model the case's retaining catch only after its lid and latch physically close."""
        joints = []
        stage = self.env.sim.stage
        for env_path in self.env.scene.env_prim_paths:
            root = environment_prim_path(self.env, "case", env_path)
            closed_pose = self.layout.source_records["case"]["affordances"]["lid_closed_pose"]
            translation = closed_pose["position_xyz"]
            rotation = closed_pose["rotation_xyzw"]
            joint = UsdPhysics.FixedJoint.Define(stage, f"{env_path}/service_connectors/case_catch")
            joint.CreateJointEnabledAttr(False)
            joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{root}/base")])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{root}/lid")])
            joint.CreateLocalPos0Attr(Gf.Vec3f(*translation))
            joint.CreateLocalRot0Attr(Gf.Quatf(rotation[3], Gf.Vec3f(*rotation[:3])))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
            joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
            joint.CreateExcludeFromArticulationAttr(True)
            joints.append(joint)
        return joints

    def prepare_reset(self, env_ids) -> None:
        """Release connectors before placement and variation events move their bodies."""
        ids = list(range(self.env.num_envs)) if env_ids is None else [int(i) for i in env_ids]
        for socket in self.sockets.values():
            socket.reset(ids, initial=None)
        for env_id in ids:
            self._case_locks[env_id].GetJointEnabledAttr().Set(False)

    def reset(self, env_ids) -> None:
        """Reset episode state and physical mechanisms for selected environments only."""
        ids = list(range(self.env.num_envs)) if env_ids is None else [int(i) for i in env_ids]
        self._settled_cache = None
        initial = {"cradle": "body", "battery": "battery_original", "cup": "dust_cup", "filter": "filter_original"}
        for name, socket in self.sockets.items():
            socket.reset(ids, initial[name])
        tensor_ids = torch.tensor(ids, device=self.env.device, dtype=torch.int32)
        for name in (*self.layout.buttons, "case"):
            articulation = self.env.scene[name]
            position = articulation.data.default_joint_pos.torch[tensor_ids].clone()
            velocity = torch.zeros_like(position)
            articulation.write_joint_state_to_sim_index(position=position, velocity=velocity, env_ids=tensor_ids)
        for env_id in ids:
            scenario = SCENARIOS[self.scenario_names[env_id]]
            self.models[env_id] = ServiceModel(scenario)
            self.statuses[env_id] = self.models[env_id].status
            self.snapshots[env_id] = None
            self._last_steps[env_id] = -1
            self._power_remaining[env_id] = 0.0
            self._previous_flow_pressed[env_id] = False
            self._case_locks[env_id].GetJointEnabledAttr().Set(False)
            if not scenario.inlet_obstruction:
                obstruction = self.env.scene["obstruction"]
                pose = obstruction.data.default_root_state.torch[env_id : env_id + 1, :7].clone()
                from isaaclab.utils.math import quat_apply

                T_W_P = self.env.arena_world.get_pose_w("waste")[env_id : env_id + 1]
                bounds = pose.new_tensor(self.regions["waste"].bounds)
                center = bounds.mean(dim=0).unsqueeze(0)
                pose[:, :3] = T_W_P[:, :3] + quat_apply(T_W_P[:, 3:], center)
                selected = tensor_ids.new_tensor([env_id])
                obstruction.write_root_pose_to_sim_index(root_pose=pose, env_ids=selected)
                obstruction.write_root_velocity_to_sim_index(root_velocity=pose.new_zeros((1, 6)), env_ids=selected)
            self._show_instrument(env_id, "battery_tester", "idle")
            self._show_instrument(env_id, "airflow_tester", "idle")

    def _show_instrument(self, env_id: int, name: str, result: str, value: float | None = None) -> None:
        """Select the Blender-authored display corresponding to the public instrument result."""
        state = "run" if result == "running" else result
        key = (env_id, name)
        if self._display_states.get(key) == (state, value):
            return
        record = self.layout.source_records[name]
        paths = record["affordances"]["status_paths"]
        assert state in paths, f"Instrument {name} has no {state} display."
        env_path = self.env.scene.env_prim_paths[env_id]
        root = environment_prim_path(self.env, name, env_path)
        for label, source_path in paths.items():
            target_path = root + source_path.removeprefix(record.get("root_prim", "/Asset"))
            prim = self.env.sim.stage.GetPrimAtPath(target_path)
            assert prim.IsValid(), f"Missing authored instrument display: {target_path}"
            visibility = UsdGeom.Tokens.inherited if label == state else UsdGeom.Tokens.invisible
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(visibility)
        reading_paths = record["affordances"]["reading_paths"]
        matched = False
        for reading, source_path in reading_paths.items():
            visible = value is not None and math.isclose(float(reading), value, rel_tol=0.0, abs_tol=1e-4)
            matched |= visible
            target_path = root + source_path.removeprefix(record.get("root_prim", "/Asset"))
            prim = self.env.sim.stage.GetPrimAtPath(target_path)
            assert prim.IsValid(), f"Missing authored numeric display: {target_path}"
            visibility = UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(visibility)
        assert value is None or matched, f"Instrument {name} has no visual reading for {value}."
        self._display_states[key] = (state, value)

    def _buttons(self) -> dict[str, list[bool]]:
        readings = {}
        for name, button in self.layout.buttons.items():
            position = self.env.arena_world.get_joint_position(name, button.joint_name)
            readings[name] = (position <= button.pressed_position_m).tolist()
        return readings

    def _in_region(self, name: str, region_name: str) -> list[bool]:
        """Require all configured physical component shapes inside a workcell region."""
        world = self.env.arena_world
        region = self.regions[region_name]
        return self.region_containment.measure_region(
            name, world.get_pose_w(name), world.get_pose_w(region.parent_name), region
        ).contained.tolist()

    def _at_pose(self, name: str, parent: str, pose, distance_m: float = 0.012, angle_deg: float = 10.0):
        return relative_pose_matches(
            self.env,
            name,
            parent,
            pose.position_xyz,
            pose.rotation_xyzw,
            position_tolerance_m=distance_m,
            orientation_tolerance_rad=math.radians(angle_deg),
        ).tolist()

    def _seated_in_fixture(self, socket_name: str) -> list[str | None]:
        spec = self.layout.sockets[socket_name]
        seated: list[str | None] = [None] * self.env.num_envs
        for name in spec.candidate_names:
            matches = self._at_pose(name, spec.parent_name, spec.pose_in_parent, 0.004, 5.0)
            settled = self._settled(name)
            forces = self.env.scene[f"service_contact_{socket_name}_{name}"].data.normal_force_matrix_w
            assert forces is not None, "A fixture reading requires measured support contact."
            supported = (torch.linalg.vector_norm(forces.torch, dim=-1).amax(dim=(1, 2)) > 0.01).tolist()
            for env_id, match in enumerate(matches):
                if match and settled[env_id] and supported[env_id]:
                    assert seated[env_id] is None, "Fixture contains overlapping components."
                    seated[env_id] = name
        return seated

    def _stock_ready(self, used: set[str], env_id: int) -> bool:
        for name, (parent, target) in self._stock_targets.items():
            if name in used:
                continue
            if not self._at_pose(name, parent, target, distance_m=0.025, angle_deg=15.0)[env_id]:
                return False
            if not self._settled(name)[env_id]:
                return False
        return True

    def _settled(self, name: str) -> list[bool]:
        """Read all rigid-body velocities together and reuse them within this update."""
        if self._settled_cache is None:
            world = self.env.arena_world
            linear = torch.stack([world.get_root_linear_velocity_w(key) for key in self._velocity_names])
            angular = torch.stack([world.get_root_angular_velocity_w(key) for key in self._velocity_names])
            speed = torch.linalg.vector_norm(linear, dim=-1)
            angular_speed = torch.linalg.vector_norm(angular, dim=-1)
            values = ((speed < 0.03) & (angular_speed < 0.2)).tolist()
            self._settled_cache = dict(zip(self._velocity_names, values, strict=True))
        return self._settled_cache[name]

    def _airflow_coupled(self) -> list[bool]:
        """Require both ends of the removable adapter to mate with their actual ports."""
        from isaaclab_arena.utils.pose import Pose

        spec = self.layout.sockets["airflow_adapter"]
        cup_seated = self._at_pose("airflow_adapter", spec.parent_name, spec.pose_in_parent, 0.004, 5.0)
        adapter_port = self.layout.source_records["airflow_adapter"]["affordances"]["tester_connector"]
        tester_port = self.layout.source_records["airflow_tester"]["affordances"]["airflow_port"]
        root_target = tuple(tester_port[index] - adapter_port[index] for index in range(3))
        tester_seated = self._at_pose("airflow_adapter", "airflow_tester", Pose(root_target), 0.004, 5.0)
        settled = self._settled("airflow_adapter")
        return [cup and tester and still for cup, tester, still in zip(cup_seated, tester_seated, settled)]

    def _bounds(self, name: str):
        record = self.layout.source_records[name]
        return (record["bounds_min"], record["bounds_max"])

    def _bounds_tensor(self, names: tuple[str, ...], reference: torch.Tensor) -> torch.Tensor:
        """Cache immutable Blender bounds on the runtime pose device once per object group."""
        if names not in self._bounds_cache:
            self._bounds_cache[names] = reference.new_tensor([self._bounds(name) for name in names])
        return self._bounds_cache[names]

    def _cup_contents(self) -> tuple[list[bool], list[bool]]:
        """Detect blocked inlet and remaining debris from geometry, independent of orientation."""
        from isaaclab_arena.geometry.measurements import box_overlaps, sphere_overlaps_cylinder

        world = self.env.arena_world
        T_W_C = world.get_pose_w("dust_cup")
        geometry = self.layout.source_records["dust_cup"]["affordances"]
        if self._cup_inlet is None:
            self._cup_inlet = T_W_C.new_tensor(geometry["inlet_bounds"])
        obstructed = box_overlaps(
            relative_pose(T_W_C, world.get_pose_w("obstruction")),
            self._bounds_tensor(("obstruction",), T_W_C),
            self._cup_inlet,
        )
        T_W_D = torch.stack([world.get_pose_w(name) for name in DEBRIS_NAMES], dim=1)
        T_W_C_batch = T_W_C[:, None, :].expand_as(T_W_D)
        T_C_D = relative_pose(T_W_C_batch.reshape(-1, 7), T_W_D.reshape(-1, 7)).reshape_as(T_W_D)
        cavity = geometry["debris_cavity_cylinder"]
        debris_present = sphere_overlaps_cylinder(
            T_C_D,
            self._bounds_tensor(DEBRIS_NAMES, T_W_C),
            cavity["x_range"],
            cavity["center_yz"],
            cavity["radius"],
        ).any(dim=1)
        obstructed_values, debris_values = torch.stack((obstructed, debris_present)).tolist()
        return obstructed_values, debris_values

    def _case_region_configuration(self) -> tuple[tuple[tuple[float, ...], ...], float]:
        interior = self.layout.source_records["case"]["affordances"]["interior_bounds"]
        return tuple(tuple(corner) for corner in interior), self.layout.case_floor_contact_allowance_m

    def measure_case_containment(self, name: str, T_W_O: torch.Tensor, T_W_C: torch.Tensor) -> ContainmentMeasurement:
        """Measure a live component against the case's immutable geometry contract.

        Args:
            name: Configured component O to measure.
            T_W_O: Component-to-world poses, scalar or batched XYZ+XYZW.
            T_W_C: Live case-base-to-world poses on the same dtype/device and compatible batch.

        Returns:
            Primitive containment and six-face diagnostics. This geometry-only method never
            advances the evaluator or reads task success, attachment, or placement state.
        """
        assert (
            self._case_region_configuration() == self._case_region_definition
        ), "Case interior or floor allowance changed; construct a new runtime"
        interior, allowance = self._case_region_definition
        return self.region_containment.measure_in_frame(name, T_W_O, T_W_C, interior, floor_allowance_m=allowance)

    def _case_contents(self) -> tuple[dict[str, list[bool]], dict[str, list[bool]]]:
        """Measure primitive containment and conservative authored-envelope intrusion."""
        from isaaclab_arena.geometry.measurements import box_overlaps

        world = self.env.arena_world
        T_W_C = world.get_pose_w("case")
        if self._case_interior is None:
            self._case_interior = T_W_C.new_tensor(self._case_region_definition[0])
        if not self._case_names:
            return {}, {}
        T_W_O = torch.stack([world.get_pose_w(name) for name in self._case_names], dim=1)
        measured = [
            self.measure_case_containment(name, T_W_O[:, index], T_W_C) for index, name in enumerate(self._case_names)
        ]
        inside = torch.stack([item.contained for item in measured], dim=1)
        valid = torch.stack([item.valid_pose for item in measured], dim=1)
        if self._case_overlap_bounds is None:
            bounds = self._bounds_tensor(self._case_names, T_W_C)
            scales = []
            for name in self._case_names:
                scale = getattr(self.env.cfg.scene, name).spawn.scale
                scales.append((1.0, 1.0, 1.0) if scale is None else scale)
            self._case_overlap_bounds = bounds * T_W_C.new_tensor(scales)[:, None, :]
            assert bool(torch.isfinite(self._case_overlap_bounds).all()) and bool(
                (self._case_overlap_bounds[:, 1] > self._case_overlap_bounds[:, 0]).all()
            ), "Invalid conservative case-participant bounds"
        # Containment and overlap answer different questions. Preserve the whole-box SAT
        # exclusion of unexpected contents, and never interpret invalid poses as absence.
        T_C_O = torch.stack([item.T_R_O for item in measured], dim=1)
        overlaps = box_overlaps(T_C_O, self._case_overlap_bounds, self._case_interior) | ~valid
        inside_values, overlap_values = torch.stack((inside, overlaps)).transpose(1, 2).tolist()
        return dict(zip(self._case_names, inside_values, strict=True)), dict(
            zip(self._case_names, overlap_values, strict=True)
        )

    def update(self) -> None:
        """Sample the physical world once; repeated predicate reads do not advance tests."""
        steps = self.env.episode_length_buf.tolist()
        active = [i for i, step in enumerate(steps) if step != self._last_steps[i]]
        if not active:
            return
        self._settled_cache = None
        buttons = self._buttons()
        self.sockets["battery"].update(buttons["battery_release"])
        self.sockets["cup"].update(buttons["cup_release"])
        self.sockets["cradle"].update(buttons["cradle_release"])
        cups_seated = self.sockets["cup"].seated_candidates(require_stationary=False)
        cup_closed = [
            attached == "dust_cup" and seated == attached
            for attached, seated in zip(self.sockets["cup"].attached, cups_seated)
        ]
        for env_id, closed in enumerate(cup_closed):
            if not closed:
                self.sockets["filter"].detach(env_id, require_extraction=False)
        self.sockets["filter"].update([False] * self.env.num_envs, capture_allowed=cup_closed)
        batteries_seated = self.sockets["battery"].seated_candidates(require_stationary=False)
        filters_seated = self.sockets["filter"].seated_candidates(require_stationary=False)
        battery_connected = [False] * self.env.num_envs
        for name in self.sockets["battery"].socket.candidates:
            distances, _, _ = self.sockets["battery"].candidate_errors(name)
            battery_connected = [
                connected or distance < 0.015 for connected, distance in zip(battery_connected, distances.tolist())
            ]
        in_tester = self._seated_in_fixture("battery_tester")
        airflow_coupled = self._airflow_coupled()
        case_inside, case_overlaps = self._case_contents()
        packed = {}
        for name, target in (
            ("body", "body"),
            ("battery_original", "battery"),
            ("battery_spare", "battery"),
            ("battery_decoy", "battery"),
            ("crevice_tool", "crevice_tool"),
            ("brush_tool", "brush_tool"),
        ):
            matches = self._at_pose(name, "case", self.layout.packing_poses[target], distance_m=0.025, angle_deg=15.0)
            packed[name] = [match and contained for match, contained in zip(matches, case_inside[name])]
        waste = {
            name: self._in_region(name, "waste")
            for name in self.layout.initial_poses
            if name.startswith("debris_") or name == "obstruction"
        }
        battery_service = self._in_region("battery_original", "battery_service")
        filter_service = self._in_region("filter_original", "filter_service")
        obstructed, cup_debris = self._cup_contents()
        hinge = self.env.arena_world.get_joint_position("case", "hinge").tolist()
        latch = self.env.arena_world.get_joint_position("case", "latch").tolist()
        assert self.cfg.gripper is not None, "Bind the task to the robot's gripper before building."
        released = (self.cfg.gripper.get_opening_width_m(self.env.arena_world) > 0.07).tolist()
        settled = {name: self._settled(name) for name in case_inside}
        case_still = (self.env.scene["case"].data.joint_vel.torch.abs() < 0.1).all(dim=-1).tolist()
        for env_id in active:
            self._last_steps[env_id] = steps[env_id]
            dt = 0.0 if self.snapshots[env_id] is None else self.env.step_dt
            flow_pressed = buttons["airflow_test_button"][env_id]
            if flow_pressed and not self._previous_flow_pressed[env_id]:
                self._power_remaining[env_id] = 1.25
            else:
                self._power_remaining[env_id] = max(0.0, self._power_remaining[env_id] - dt)
            self._previous_flow_pressed[env_id] = flow_pressed
            battery = self.sockets["battery"].attached[env_id]
            if batteries_seated[env_id] != battery:
                battery = None
            filter_id = filters_seated[env_id]
            packed_battery = next(
                (name for name in ("battery_original", "battery_spare", "battery_decoy") if packed[name][env_id]), None
            )
            packed_tools = frozenset(name for name in ("crevice_tool", "brush_tool") if packed[name][env_id])
            used = {name for name in (battery, filter_id, packed_battery) if name is not None} | set(packed_tools)
            if battery_service[env_id]:
                used.add("battery_original")
            if filter_service[env_id]:
                used.add("filter_original")
            case_closed = abs(hinge[env_id]) < 0.04
            case_latched = abs(latch[env_id]) < 0.06
            self._case_locks[env_id].GetJointEnabledAttr().Set(case_closed and case_latched)
            expected_contents = {"body", "dust_cup", filter_id, packed_battery} | set(packed_tools)
            expected_contents.discard(None)
            correct_contents = all(
                case_inside[name][env_id] if name in expected_contents else not values[env_id]
                for name, values in case_overlaps.items()
            )
            all_settled = all(values[env_id] for values in settled.values()) and case_still[env_id]
            snapshot = ServiceSnapshot(
                installed_battery=battery,
                battery_connected=battery_connected[env_id],
                installed_filter=filter_id,
                cup_closed=cup_closed[env_id],
                inlet_obstructed=obstructed[env_id],
                cup_debris_present=cup_debris[env_id],
                debris_in_waste=all(values[env_id] for name, values in waste.items() if name.startswith("debris_")),
                battery_in_tester=in_tester[env_id],
                battery_test_pressed=buttons["battery_test_button"][env_id],
                vacuum_in_test_dock=airflow_coupled[env_id],
                airflow_test_pressed=flow_pressed,
                power_on=self._power_remaining[env_id] > 0 and battery is not None,
                vacuum_packed=packed["body"][env_id] and correct_contents,
                packed_battery=packed_battery,
                packed_accessories=packed_tools,
                case_closed=case_closed,
                case_latched=case_latched,
                removed_battery_in_service=battery_service[env_id],
                removed_filter_in_service=filter_service[env_id],
                obstruction_in_waste=waste["obstruction"][env_id],
                station_reset=self._stock_ready(used, env_id)
                and in_tester[env_id] is None
                and not any(values[env_id] for values in buttons.values()),
                objects_released=released[env_id] and all_settled,
            )
            self.snapshots[env_id] = snapshot
            self.statuses[env_id] = self.models[env_id].update(snapshot, dt)
            status = self.statuses[env_id]
            self._show_instrument(env_id, "battery_tester", status.battery_reading.result, status.battery_reading.value)
            self._show_instrument(env_id, "airflow_tester", status.airflow_reading.result, status.airflow_reading.value)
