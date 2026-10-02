# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Probe service evidence and mechanisms; diagnostic state writes are not robot rollouts."""

from isaaclab_arena.tests.utils.return_to_service import _pose_in_parent, _require_assets, _write_joint, _write_pose


def _make_environment():
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.utils.physics_backend import PhysicsBackend
    from isaaclab_arena_environments.return_to_service_environment import (
        ReturnToServiceEnvironment,
        ReturnToServiceEnvironmentCfg,
    )

    specification = ReturnToServiceEnvironment().build(
        ReturnToServiceEnvironmentCfg(scenarios=["healthy", "combined"], episode_length_s=60.0)
    )
    return ArenaEnvBuilder(
        specification, ArenaEnvBuilderCfg(num_envs=2, presets=PhysicsBackend.PHYSX)
    ).make_registered()


def _step_physics(env, count: int, closed: bool = False) -> None:
    import torch

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    actions[:, -1] = float(closed)
    with torch.no_grad():
        for _ in range(count):
            _, _, terminated, truncated, _ = env.step(actions)
            assert not terminated.any(), "Diagnostic probes must not manufacture a successful robot episode."
            assert not truncated.any(), "A focused runtime test unexpectedly exceeded its episode budget."


def _sample_probe(base, env_id: int) -> None:
    """Sample an injected physical-state probe without representing it as a policy action."""
    # Invalidate only this adapter cache; the environment's physical step counter
    # and ProgressTracker sequence must still advance exclusively through env.step.
    base.return_to_service._last_steps[env_id] = -1
    base.return_to_service.update()


def _write_velocity(base, name: str, env_id: int, velocity) -> None:
    import torch

    indices = torch.tensor([env_id], device=base.device, dtype=torch.int32)
    base.scene[name].write_root_velocity_to_sim_index(
        root_velocity=torch.tensor([velocity], device=base.device, dtype=torch.float32), env_ids=indices
    )


def _display_visibility(base, env_id: int, name: str) -> dict[str, str]:
    from pxr import UsdGeom

    from isaaclab_arena_environments.return_to_service.connectors import environment_prim_path

    runtime = base.return_to_service
    record = runtime.layout.source_records[name]
    root = environment_prim_path(base, name, base.scene.env_prim_paths[env_id])
    result = {}
    for label, source_path in record["affordances"]["status_paths"].items():
        path = root + source_path.removeprefix(record.get("root_prim", "/Asset"))
        result[label] = UsdGeom.Imageable(base.sim.stage.GetPrimAtPath(path)).ComputeVisibility()
    return result


def _test_workcell_gripper_tuning_preserves_droid_capabilities(_simulation_app) -> bool:
    import math
    import torch
    from copy import deepcopy
    from types import SimpleNamespace

    import warp as wp
    from isaaclab_physx.physics import PhysxCfg

    from isaaclab_arena.embodiments.droid.droid import DroidSceneCfg
    from isaaclab_arena_environments.return_to_service_environment import _configure_service_physics

    configured, independent = DroidSceneCfg(), DroidSceneCfg()
    original = deepcopy(configured.robot.to_dict())
    expected = deepcopy(original)
    expected["actuators"]["gripper"].update(stiffness=4.0, damping=1.0)
    gripper = configured.robot.actuators["gripper"]
    cfg = SimpleNamespace(scene=configured, sim=SimpleNamespace(physics=PhysxCfg()))
    assert _configure_service_physics(cfg, gripper_stiffness=4.0, gripper_damping=1.0) is cfg
    assert configured.robot.actuators["gripper"] is gripper
    assert configured.robot.to_dict() == expected, "Workcell tuning changed other robot configuration."
    assert independent.robot.to_dict() == original, "Workcell tuning leaked into another DROID configuration."

    env = _make_environment()
    try:
        env.reset()
        _step_physics(env, 2)
        robot = env.unwrapped.scene["robot"]
        driver = robot.joint_names.index("finger_joint")
        torch.testing.assert_close(
            robot.data.joint_stiffness.torch[:, driver], torch.full((2,), 4.0, device=robot.device)
        )
        torch.testing.assert_close(
            robot.data.joint_damping.torch[:, driver], torch.full((2,), 1.0, device=robot.device)
        )
        view = robot.root_view
        names = list(view.shared_metatype.dof_names)
        stiffness = wp.to_torch(view.get_dof_stiffnesses())
        damping = wp.to_torch(view.get_dof_dampings())
        for index, name in enumerate(names):
            expected_stiffness, expected_damping = (400.0, 80.0) if name.startswith("panda_joint") else (0.0, 0.0)
            if name == "finger_joint":
                expected_stiffness, expected_damping = 4.0, 1.0
            torch.testing.assert_close(stiffness[:, index], stiffness.new_full((2,), expected_stiffness))
            torch.testing.assert_close(damping[:, index], damping.new_full((2,), expected_damping))
        driver = names.index("finger_joint")
        forces = wp.to_torch(view.get_dof_max_forces())
        velocities = wp.to_torch(view.get_dof_max_velocities())
        torch.testing.assert_close(forces[:, driver], forces.new_full((2,), 16.5))
        torch.testing.assert_close(velocities[:, driver], velocities.new_full((2,), math.radians(500)))
        assert robot.cfg.actuators["gripper"].velocity_limit == 5.0
    finally:
        env.close()
    return True


def test_workcell_gripper_tuning_preserves_droid_capabilities():
    _require_assets()
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_workcell_gripper_tuning_preserves_droid_capabilities)


def _test_runtime_reset_and_observation_isolation(_simulation_app) -> bool:
    import torch

    from isaaclab_arena_environments.return_to_service.task import instrument_readings, service_condition

    env = _make_environment()
    try:
        env.reset()
        _step_physics(env, 15)
        base = env.unwrapped
        runtime = base.return_to_service
        assert all(not status.success for status in runtime.statuses)
        assert all(status.battery_test_count == 0 and status.airflow_test_count == 0 for status in runtime.statuses)
        assert runtime.snapshots[0].cup_debris_present
        assert runtime.snapshots[1].cup_debris_present
        assert not runtime.snapshots[0].inlet_obstructed
        assert runtime.snapshots[1].inlet_obstructed

        readings = instrument_readings(base, runtime.cfg)
        torch.testing.assert_close(readings, readings.new_tensor([[-1, 0, -1, 0], [-1, 0, -1, 0]]))
        before = tuple(runtime.statuses)
        for _ in range(4):
            service_condition(base, "success", runtime.cfg)
            instrument_readings(base, runtime.cfg)
            runtime.update()
        assert tuple(runtime.statuses) == before, "Repeated evidence reads must not advance time or test state."

        # The tester is empty. A measured button edge is an invalid attempt in env 1 only.
        _write_joint(base, "battery_test_button", "press", 1, -0.005)
        _sample_probe(base, 1)
        assert runtime.statuses[1].battery_test_count == 1
        assert runtime.statuses[1].battery_reading.result == "invalid"
        assert runtime.statuses[0].battery_test_count == 0
        second_model = runtime.models[1]
        second_status = runtime.statuses[1]
        second_snapshot = runtime.snapshots[1]
        second_visibility = _display_visibility(base, 1, "battery_tester")
        second_poses = {}
        for name in ("body", "battery_original", "dust_cup", "filter_original", "obstruction"):
            second_poses[name] = base.arena_world.get_pose_w(name)[1].clone()

        base._reset_idx(torch.tensor([0], device=base.device))
        runtime.update()
        assert runtime.models[1] is second_model
        assert runtime.statuses[1] is second_status
        assert runtime.snapshots[1] is second_snapshot
        assert _display_visibility(base, 1, "battery_tester") == second_visibility
        assert runtime.statuses[0].battery_test_count == 0
        assert runtime.statuses[0].battery_reading.result == "idle"
        assert _display_visibility(base, 0, "battery_tester")["idle"] == "inherited"
        for name, expected in second_poses.items():
            torch.testing.assert_close(base.arena_world.get_pose_w(name)[1], expected, atol=1e-6, rtol=0)
    finally:
        env.close()
    return True


def test_runtime_reset_and_observation_isolation():
    _require_assets()
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_runtime_reset_and_observation_isolation)


def _test_retention_release_and_capture_evidence(_simulation_app) -> bool:
    import math
    import torch

    env = _make_environment()
    try:
        env.reset()
        _step_physics(env, 15)
        base = env.unwrapped
        runtime = base.return_to_service
        battery_socket = runtime.sockets["battery"]
        assert runtime.sockets["cup"].socket.position_tolerance_m == 0.001
        for name in ("battery", "filter", "cradle"):
            assert runtime.sockets[name].socket.position_tolerance_m == 0.003

        # A velocity impulse must be resisted by the physical constraint, not only an attached flag.
        _write_velocity(base, "battery_original", 0, (-0.8, 0, 0, 0, 0, 0))
        _step_physics(env, 8)
        distance, _, _ = battery_socket.candidate_errors("battery_original")
        assert distance[0] < 0.006, "Enabled battery retention did not physically hold the part."
        assert battery_socket.attached == ["battery_original", "battery_original"]

        _write_joint(base, "battery_release", "press", 0, -0.005)
        _sample_probe(base, 0)
        assert battery_socket.attached[0] is None
        assert battery_socket.attached[1] == "battery_original"
        assert runtime.snapshots[0].battery_connected, "Releasing retention alone must not establish isolation."
        _write_velocity(base, "battery_original", 0, (-0.8, 0, 0.2, 0, 0, 0))
        _step_physics(env, 8)
        distance, _, _ = battery_socket.candidate_errors("battery_original")
        assert distance[0] > 0.025, "The released battery remained constrained to its socket."
        assert not runtime.snapshots[0].battery_connected

        # Correct pose alone is insufficient: the spare's filtered parent-contact history is empty.
        spec = runtime.layout.sockets["battery"]
        target = _pose_in_parent(base, spec.parent_name, spec.pose_in_parent, 0)
        target[2] += 0.002
        _write_pose(base, "battery_spare", 0, target)
        _sample_probe(base, 0)
        assert battery_socket.attached[0] is None, "An aligned, unsupported part was captured without contact."
        assert runtime.snapshots[0].installed_battery is None

        reversed_pose = target.clone()
        reversed_pose[3:] = target.new_tensor((0.0, 0.0, 1.0, 0.0))
        _write_pose(base, "battery_spare", 0, reversed_pose)
        _sample_probe(base, 0)
        assert battery_socket.seated_candidates()[0] is None
        assert runtime.snapshots[0].installed_battery is None

        _write_pose(base, "battery_spare", 0, target, velocity=(0, 0, 0, 0, 0, 2.0))
        _sample_probe(base, 0)
        assert battery_socket.attached[0] is None, "A spinning component must not capture a retention detent."

        # A disabled connector must still permit parent contact so an inserted part can recapture.
        # This is a fixture-mechanism diagnostic, not a robot insertion demonstration.
        _write_pose(base, "battery_spare", 0, target)
        _step_physics(env, 15)
        assert battery_socket.attached[0] == "battery_spare", "A stable, supported insertion failed to retain."
        assert runtime.snapshots[0].installed_battery == "battery_spare"

        # Button and case articulation state comes from physical joints, not a prescribed task stage.
        _write_joint(base, "cup_release", "press", 0, -0.005)
        _sample_probe(base, 0)
        assert not runtime.snapshots[0].cup_closed
        _write_joint(base, "case", "hinge", 0, 0.0)
        _write_joint(base, "case", "latch", 0, 0.0)
        _sample_probe(base, 0)
        assert runtime.snapshots[0].case_closed and runtime.snapshots[0].case_latched
        assert runtime._case_locks[0].GetJointEnabledAttr().Get()
        # The latch must remain closed under gravity after release. A single
        # injected angle snapshot cannot establish passive mechanical retention.
        _step_physics(env, 30)
        assert runtime.snapshots[0].case_closed and runtime.snapshots[0].case_latched
        assert abs(float(base.arena_world.get_joint_position("case", "latch")[0])) < 0.02
        _write_joint(base, "case", "latch", 0, 1.0)
        _sample_probe(base, 0)
        assert not runtime.snapshots[0].case_latched
        assert not runtime._case_locks[0].GetJointEnabledAttr().Get()
        _step_physics(env, 30)
        # The injected open angle is an initial state, not a servo target;
        # a passive latch may move toward its authored stop under gravity.
        latch_angle = float(base.arena_world.get_joint_position("case", "latch")[0])
        lower_deg, upper_deg = runtime.layout.source_records["case"]["affordances"]["latch"]["angle_limits_degrees"]
        assert math.isfinite(latch_angle)
        assert math.radians(lower_deg) - 1e-5 <= latch_angle <= math.radians(upper_deg) + 1e-5
        assert not runtime.snapshots[0].case_latched
        assert not runtime._case_locks[0].GetJointEnabledAttr().Get()
        assert not runtime.statuses[0].success
        assert torch.isfinite(base.arena_world.get_pose_w("body")).all()
    finally:
        env.close()
    return True


def test_retention_release_and_capture_evidence():
    _require_assets()
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_retention_release_and_capture_evidence)


def _assert_released_filter_supported(base, elapsed_s: float) -> None:
    """Check both workcells without confusing physical guidance with retained seating."""
    import torch

    from isaaclab_arena_environments.return_to_service.connectors import relative_pose
    from isaaclab_arena_environments.return_to_service.measurements import box_contained

    runtime = base.return_to_service
    specification = runtime.layout.sockets["filter"]
    socket = runtime.sockets["filter"]
    assert specification.capture_bounds is not None, "Filter dwell requires the authored physical guide volume."
    T_W_F = base.arena_world.get_pose_w("filter_original")
    T_B_F = relative_pose(base.arena_world.get_pose_w(specification.parent_name), T_W_F)
    inside = box_contained(T_B_F, specification.candidate_bounds["filter_original"], specification.capture_bounds)
    assert bool(
        inside.all()
    ), f"Released filter left its guide after {elapsed_s:.3f}s: contained={inside.tolist()}, T_B_F={T_B_F.tolist()}"
    assert (
        socket.seated_candidates(require_stationary=False) == ["filter_original"] * base.num_envs
    ), f"Released filter lost socket alignment after {elapsed_s:.3f}s."
    for env_id, snapshot in enumerate(runtime.snapshots):
        assert runtime.sockets["cup"].attached[env_id] is None and not snapshot.cup_closed
        assert socket.attached[env_id] is None, f"Environment {env_id} recaptured the released filter."
        for joint in socket._joints[env_id].values():
            assert not joint.GetJointEnabledAttr().Get(), "A retention joint masked passive filter instability."
        assert snapshot.installed_filter == "filter_original"
        assert not runtime.statuses[env_id].success, "A passive mechanism dwell cannot establish task success."
    assert runtime.cfg.gripper is not None
    hand_position = runtime.cfg.gripper.get_position_w(base.arena_world)
    hand_distance = torch.linalg.vector_norm(hand_position - T_W_F[:, :3], dim=-1)
    assert bool((hand_distance > 0.25).all()), "The reset hand must stay withdrawn during the passive-filter dwell."


def _test_released_filter_remains_supported_with_cup_present(_simulation_app) -> bool:
    """Check passive support with the released cup still physically present."""
    import math

    env = _make_environment()
    try:
        env.reset()
        _step_physics(env, 15)
        base = env.unwrapped
        runtime = base.return_to_service
        assert runtime.sockets["filter"].attached == ["filter_original"] * base.num_envs
        # Inject only the physical release button, as in the mechanism probes
        # above. Do not move or stabilize the cup, filter, body, or robot.
        for env_id in range(base.num_envs):
            _write_joint(base, "cup_release", "press", env_id, -0.005)
            _sample_probe(base, env_id)
        _assert_released_filter_supported(base, 0.0)
        dwell_steps = math.ceil(30.0 / base.step_dt)
        for step in range(dwell_steps):
            # Zero Cartesian actions hold the withdrawn reset hand open. Check
            # every control step; this is not a robot cup-removal demonstration.
            _step_physics(env, 1)
            _assert_released_filter_supported(base, (step + 1) * base.step_dt)
        assert dwell_steps * base.step_dt >= 30.0
    finally:
        env.close()
    return True


def test_released_filter_remains_supported_with_cup_present():
    """Require 30 seconds of released-filter support with the cup present."""
    _require_assets()
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_released_filter_remains_supported_with_cup_present)


def _test_adversarial_packing_and_fixture_evidence(_simulation_app) -> bool:
    env = _make_environment()
    try:
        env.reset()
        _step_physics(env, 15)
        base = env.unwrapped
        runtime = base.return_to_service

        spec = runtime.layout.sockets["battery_tester"]
        target = _pose_in_parent(base, spec.parent_name, spec.pose_in_parent, 0)
        _write_pose(base, "battery_spare", 0, target, velocity=(0, 0, 0, 0, 0, 2.0))
        _sample_probe(base, 0)
        assert runtime.snapshots[0].battery_in_tester is None, "A rotating battery cannot supply a stable load test."
        target[0] += 0.025
        _write_pose(base, "battery_spare", 0, target)
        _sample_probe(base, 0)
        assert runtime.snapshots[0].battery_in_tester is None, "Proximity to the tester does not establish seating."
        assert not runtime.snapshots[0].vacuum_in_test_dock, "The airflow test requires its removable coupling."

        # Install a synthetic packed arrangement only to exercise containment and inventory predicates.
        # These writes bypass manipulation and cannot validate robot-policy performance or task solvability.
        runtime.sockets["cradle"].detach(0)
        runtime.sockets["battery"].detach(0)
        body_pose = _pose_in_parent(base, "case", runtime.layout.packing_poses["body"], 0)
        body_pose[2] -= 0.003  # Put the skids on the case floor below the approach clearance.
        _write_pose(base, "body", 0, body_pose)
        for socket_name, name in (("cup", "dust_cup"), ("filter", "filter_original")):
            spec = runtime.layout.sockets[socket_name]
            _write_pose(base, name, 0, _pose_in_parent(base, spec.parent_name, spec.pose_in_parent, 0))
        for name, packing_name in (
            ("battery_original", "battery"),
            ("crevice_tool", "crevice_tool"),
            ("brush_tool", "brush_tool"),
        ):
            _write_pose(base, name, 0, _pose_in_parent(base, "case", runtime.layout.packing_poses[packing_name], 0))
        for name in ("debris_0", "debris_1", "debris_2", "obstruction"):
            pose = base.arena_world.get_pose_w(name)[0].clone()
            pose[:3] = pose.new_tensor(runtime.layout.regions["waste"].center_xyz) + base.scene.env_origins[0]
            _write_pose(base, name, 0, pose)
        spare_pose = runtime.layout.initial_poses["battery_spare"].to_tensor(base.device)
        spare_pose[:3] += base.scene.env_origins[0]
        _write_pose(base, "battery_spare", 0, spare_pose)
        _sample_probe(base, 0)
        assert runtime.snapshots[0].vacuum_packed, "The intended assembly must fit fully inside the case."
        assert not runtime.statuses[0].vacuum_verified, "Pose writes must not manufacture an airflow certificate."
        assert not runtime.statuses[0].success

        extra_pose = _pose_in_parent(base, "case", runtime.layout.packing_poses["crevice_tool"], 0)
        _write_pose(base, "battery_decoy", 0, extra_pose)
        _sample_probe(base, 0)
        assert not runtime.snapshots[0].vacuum_packed, "Extra unrequested case contents were accepted."
        decoy_pose = runtime.layout.initial_poses["battery_decoy"].to_tensor(base.device)
        decoy_pose[:3] += base.scene.env_origins[0]
        _write_pose(base, "battery_decoy", 0, decoy_pose)
        _write_velocity(base, "brush_tool", 0, (0, 0, 0.2, 0, 0, 0))
        _sample_probe(base, 0)
        assert not runtime.snapshots[0].objects_released, "A moving component must not satisfy final settled state."

        # Check the gripper release gate using actual actuator motion after a clean reset.
        env.reset()
        _step_physics(env, 40, closed=True)
        assert runtime.cfg.gripper.get_opening_width_m(base.arena_world)[0] < 0.07
        assert not runtime.snapshots[0].objects_released
    finally:
        env.close()
    return True


def test_adversarial_packing_and_fixture_evidence():
    _require_assets()
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_adversarial_packing_and_fixture_evidence)
