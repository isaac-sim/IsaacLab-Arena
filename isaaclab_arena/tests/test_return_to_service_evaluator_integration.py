# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate native scoring from synthetic physical fixtures, never a robot-policy success claim."""


class _ScoringFixture:
    """Own labeled fixture actuation while the native environment owns all evidence and time."""

    def __init__(self, output_dir, asset_root=None):
        import torch

        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_arena.tests.utils.return_to_service import _pose_in_parent, _write_joint, _write_pose
        from isaaclab_arena.utils.physics_backend import PhysicsBackend
        from isaaclab_arena_environments.return_to_service.scenarios import SCENARIOS
        from isaaclab_arena_environments.return_to_service_environment import (
            ReturnToServiceEnvironment,
            ReturnToServiceEnvironmentCfg,
        )

        self.scenario_names = list(SCENARIOS)
        self.scenarios = list(SCENARIOS.values())
        specification = ReturnToServiceEnvironment().build(
            ReturnToServiceEnvironmentCfg(
                scenarios=self.scenario_names, episode_length_s=120.0, enable_cameras=False, asset_root=asset_root
            )
        )
        self.env = ArenaEnvBuilder(
            specification,
            ArenaEnvBuilderCfg(
                num_envs=len(self.scenarios),
                presets=PhysicsBackend.PHYSX,
                recorder_dataset_export_dir_path=str(output_dir),
                recorder_dataset_filename="evaluator_fixture",
            ),
        ).make_registered()
        self.base = self.env.unwrapped
        self.actions = torch.zeros(self.env.action_space.shape, device=self.base.device)
        self.fixture_buttons = {}
        self.write_pose, self.write_joint, self.pose_in_parent = _write_pose, _write_joint, _pose_in_parent
        self.original_after_physics = self.base.recorder_manager.record_post_physics_decimation_step
        self.base.recorder_manager.record_post_physics_decimation_step = self.after_physics
        self.original_before_reset = self.base.recorder_manager.record_pre_reset
        self.terminal_samples = []
        self.observe_resets = False
        self.base.recorder_manager.record_pre_reset = self.before_reset
        self.selected_batteries = [s.spare_battery if s.weak_battery else s.original_battery for s in self.scenarios]
        self.selected_filters = [s.spare_filter if s.clogged_filter else s.original_filter for s in self.scenarios]

    def close(self):
        self.base.recorder_manager.record_post_physics_decimation_step = self.original_after_physics
        self.base.recorder_manager.record_pre_reset = self.original_before_reset
        self.env.close()

    def before_reset(self, env_ids, *args, **kwargs):
        from isaaclab_arena_environments.return_to_service.metrics import terminal_service_values

        if self.observe_resets:
            ids = range(self.base.num_envs) if env_ids is None else env_ids
            for env_id in ids:
                env_id = int(env_id)
                values = terminal_service_values(self.runtime.statuses[env_id])
                self.terminal_samples.append((env_id, values))
        return self.original_before_reset(env_ids, *args, **kwargs)

    def after_physics(self):
        self.original_after_physics()
        # Explicit test-fixture actuation holds only selected spring-return
        # buttons at measurement. No model/contact/cache/clock/retention writes.
        for name, ids in self.fixture_buttons.items():
            for env_id in ids:
                self.write_joint(self.base, name, "press", env_id, -0.005)

    def step(self, count=1, allow_success=False):
        import torch

        result = None
        with torch.no_grad():
            for _ in range(count):
                result = self.env.step(self.actions)
                assert not result[3].any(), "Evaluator fixture exceeded its finite episode budget."
                assert allow_success or not result[2].any(), "Fixture terminated before the final hold test."
        return result

    def wait_for(self, predicate, label, count=120):
        for _ in range(count):
            if predicate():
                return
            self.step()
        raise AssertionError(f"Fixture failed to establish {label} within {count} control steps.")

    def press(self, name, ids=None):
        self.fixture_buttons[name] = list(range(self.base.num_envs)) if ids is None else ids
        try:
            self.step()
        finally:
            del self.fixture_buttons[name]
        self.step(3)

    def place_socket(self, name, socket_name, env_id):
        socket = self.runtime.layout.sockets[socket_name]
        pose = self.pose_in_parent(self.base, socket.parent_name, socket.pose_in_parent, env_id)
        self.write_pose(self.base, name, env_id, pose)

    def stock(self, name, env_id):
        pose = self.runtime.layout.initial_poses[name].to_tensor(self.base.device)
        pose[:3] += self.base.scene.env_origins[env_id]
        self.write_pose(self.base, name, env_id, pose)

    def bin_pose(self, name, region_name, env_id, offset=(0.0, 0.0)):
        region = self.runtime.layout.regions[region_name]
        record = self.runtime.layout.source_records[name]
        pose = self.base.arena_world.get_pose_w(name)[env_id].clone()
        pose[3:] = pose.new_tensor((0, 0, 0, 1))
        center = 0.5 * (pose.new_tensor(record["bounds_min"]) + pose.new_tensor(record["bounds_max"]))
        pose[:3] = pose.new_tensor(region.center_xyz) + self.base.scene.env_origins[env_id] - center
        pose[0] += offset[0]
        pose[1] += offset[1]
        self.write_pose(self.base, name, env_id, pose)


def _reset(f):
    f.env.reset()
    f.runtime = f.base.task_runtime
    f.observe_resets = True
    f.step(15)
    runtime = f.runtime
    assert [model.scenario.name for model in runtime.models] == f.scenario_names
    assert all(status.battery_test_count == status.airflow_test_count == 0 for status in runtime.statuses)
    assert all(model.certificate is None for model in runtime.models)
    assert [snapshot.inlet_obstructed for snapshot in runtime.snapshots] == [s.inlet_obstruction for s in f.scenarios]


def _load_batteries(f):
    base = f.base
    runtime = f.runtime
    scenarios = f.scenarios
    wait_for = f.wait_for
    press = f.press
    place_socket = f.place_socket
    bin_pose = f.bin_pose

    # First inspect the original batteries. Real support contact and the
    # native one-second dwell determine each reading, including weak cells.
    press("battery_release")
    assert all(value is None for value in runtime.sockets["battery"].attached)
    for env_id in range(base.num_envs):
        place_socket("battery_original", "battery_tester", env_id)
    wait_for(lambda: all(s.battery_in_tester == "battery_original" for s in runtime.snapshots), "original load seating")
    assert all(not s.battery_connected for s in runtime.snapshots)
    press("battery_test_button")
    assert all(s.battery_reading.result == "running" for s in runtime.statuses)
    wait_for(lambda: all(s.battery_reading.result in ("pass", "fail") for s in runtime.statuses), "original load dwell")
    for scenario, status in zip(scenarios, runtime.statuses, strict=True):
        assert status.battery_reading.value == scenario.battery_voltage("battery_original")
        assert status.battery_reading.result == ("fail" if scenario.weak_battery else "pass")

    weak_ids = [i for i, scenario in enumerate(scenarios) if scenario.weak_battery]
    for env_id in weak_ids:
        bin_pose("battery_original", "battery_service", env_id)
        place_socket("battery_spare", "battery_tester", env_id)
    wait_for(
        lambda: all(runtime.snapshots[i].battery_in_tester == "battery_spare" for i in weak_ids),
        "replacement load seating",
    )
    press("battery_test_button", weak_ids)
    wait_for(lambda: all(s.battery_reading.result == "pass" for s in runtime.statuses), "replacement load dwell")
    assert [s.battery_test_count for s in runtime.statuses] == [2 if s.weak_battery else 1 for s in scenarios]


def _service_airway(f):
    from isaaclab.utils.math import quat_apply

    base = f.base
    runtime = f.runtime
    scenarios = f.scenarios
    step = f.step
    wait_for = f.wait_for
    press = f.press
    place_socket = f.place_socket
    bin_pose = f.bin_pose
    _write_pose = f.write_pose
    selected_filters = f.selected_filters

    # Isolate before opening, then use labeled state setup to substitute for
    # manipulation. Existing constraints are released only by measured buttons.
    press("cup_release")
    assert all(not s.cup_closed and not s.battery_connected for s in runtime.snapshots)
    for env_id, scenario in enumerate(scenarios):
        pose = base.arena_world.get_pose_w("case")[env_id].clone()
        interior = runtime.layout.source_records["case"]["affordances"]["interior_bounds"]
        cup_bottom = runtime.layout.source_records["dust_cup"]["bounds_min"][2]
        pose[:3] += quat_apply(pose[3:], pose.new_tensor((0.135, -0.041, interior[0][2] - cup_bottom)))
        pose[3:] = pose.new_tensor(runtime.layout.initial_poses["dust_cup"].rotation_xyzw)
        _write_pose(base, "dust_cup", env_id, pose)
        for name, offset in (
            ("debris_0", (-0.018, -0.018)),
            ("debris_1", (0.018, -0.018)),
            ("debris_2", (-0.018, 0.018)),
            ("obstruction", (0.020, 0.020)),
        ):
            bin_pose(name, "waste", env_id, offset)
        if scenario.clogged_filter:
            bin_pose("filter_original", "filter_service", env_id)
            place_socket("filter_spare", "filter", env_id)
    step(30)
    for env_id in range(base.num_envs):
        place_socket("dust_cup", "cup", env_id)
    wait_for(lambda: all(s.cup_closed for s in runtime.snapshots), "closed cup recapture")
    assert [s.installed_filter for s in runtime.snapshots] == selected_filters
    assert all(not s.cup_debris_present and not s.inlet_obstructed for s in runtime.snapshots)
    assert all(not s.dependency_violation for s in runtime.statuses)


def _airflow_test(f):
    base = f.base
    runtime = f.runtime
    wait_for = f.wait_for
    press = f.press
    place_socket = f.place_socket
    selected_batteries = f.selected_batteries
    selected_filters = f.selected_filters

    for env_id, name in enumerate(selected_batteries):
        place_socket(name, "battery", env_id)
    wait_for(lambda: [s.installed_battery for s in runtime.snapshots] == selected_batteries, "battery recapture")
    for env_id in range(base.num_envs):
        place_socket("airflow_adapter", "airflow_adapter", env_id)
    wait_for(lambda: all(s.vacuum_in_test_dock for s in runtime.snapshots), "two-port airflow coupling")
    press("airflow_test_button")
    assert all(s.airflow_reading.result == "running" for s in runtime.statuses)
    wait_for(lambda: all(s.vacuum_verified for s in runtime.statuses), "native airflow certificate")
    for env_id, model in enumerate(runtime.models):
        assert model.certificate.battery_id == selected_batteries[env_id]
        assert model.certificate.filter_id == selected_filters[env_id]
        assert model.status.airflow_reading.value == 22.0
    f.certificates = [model.certificate for model in runtime.models]


def _pack(f):
    base = f.base
    runtime = f.runtime
    step = f.step
    wait_for = f.wait_for
    press = f.press
    place_socket = f.place_socket
    stock = f.stock
    _write_pose = f.write_pose
    _write_joint = f.write_joint
    _pose_in_parent = f.pose_in_parent
    selected_batteries = f.selected_batteries
    selected_filters = f.selected_filters

    # Withdraw the test coupling, isolate the same certified battery, release
    # the cradle, and set a geometrically valid packed fixture once.
    for env_id in range(base.num_envs):
        stock("airflow_adapter", env_id)
    press("battery_release")
    for env_id, name in enumerate(selected_batteries):
        _write_pose(base, name, env_id, _pose_in_parent(base, "case", runtime.layout.packing_poses["battery"], env_id))
    step(3)
    assert all(not s.battery_connected for s in runtime.snapshots)
    press("cradle_release")
    assert all(value is None for value in runtime.sockets["cradle"].attached)
    for env_id, filter_name in enumerate(selected_filters):
        pose = _pose_in_parent(base, "case", runtime.layout.packing_poses["body"], env_id)
        pose[2] -= 0.003
        _write_pose(base, "body", env_id, pose)
        place_socket("dust_cup", "cup", env_id)
        place_socket(filter_name, "filter", env_id)
        for name in ("crevice_tool", "brush_tool"):
            _write_pose(base, name, env_id, _pose_in_parent(base, "case", runtime.layout.packing_poses[name], env_id))
        _write_joint(base, "case", "hinge", env_id, 0.0)
        _write_joint(base, "case", "latch", env_id, 1.0)
    wait_for(
        lambda: all(s.vacuum_packed and s.station_reset and s.objects_released for s in runtime.snapshots),
        "settled complete packed fixture",
    )
    assert [model.certificate for model in runtime.models] == f.certificates
    assert [s.installed_filter for s in runtime.snapshots] == selected_filters
    assert [s.packed_battery for s in runtime.snapshots] == selected_batteries
    assert all(s.installed_battery is None and s.cup_closed for s in runtime.snapshots)
    assert all(not s.dependency_violation for s in runtime.statuses)
    assert all(not s.success for s in runtime.statuses)


def _scoring(f):
    import numpy as np

    from isaaclab_arena_environments.return_to_service.metrics import SERVICE_METRIC_NAMES

    base = f.base
    runtime = f.runtime
    step = f.step
    _write_joint = f.write_joint

    # An unlatched case in env0 blocks that environment without borrowing
    # other environments' certificates or their consecutive-success holds.
    for env_id in range(1, base.num_envs):
        _write_joint(base, "case", "latch", env_id, 0.0)
    consecutive = [0] * base.num_envs
    completed = set()
    for _ in range(60):
        statuses_before = tuple(runtime.statuses)
        _, _, terminated, _, _ = step(allow_success=True)
        for env_id, ended in enumerate(terminated.tolist()):
            if env_id in completed:
                continue
            if ended:
                assert env_id != 0, "An unlatched environment borrowed another environment's success."
                assert consecutive[env_id] == 14
                assert statuses_before[env_id].success
                completed.add(env_id)
            else:
                consecutive[env_id] = consecutive[env_id] + 1 if runtime.statuses[env_id].success else 0
        if len(completed) == base.num_envs - 1:
            break
    assert completed == set(range(1, base.num_envs)), f"Completed environments: {sorted(completed)}"
    assert runtime.models[0].certificate == f.certificates[0] and not runtime.statuses[0].success

    # Start env0's hold, physically break it, then require all 15 steps again.
    _write_joint(base, "case", "latch", 0, 0.0)
    step(7)
    assert runtime.statuses[0].success
    _write_joint(base, "case", "latch", 0, 1.0)
    step()
    assert not runtime.statuses[0].success
    _write_joint(base, "case", "latch", 0, 0.0)
    seen = 0
    for _ in range(60):
        result = step(allow_success=True)
        if result[2][0]:
            assert seen == 14, "Success requires a fresh 15-step measured hold after physical interruption."
            completed.add(0)
            break
        seen = seen + 1 if runtime.statuses[0].success else 0
    assert completed == set(range(base.num_envs))
    metrics = base.compute_metrics()
    assert metrics.num_episodes == 8
    assert metrics.metric_data_entries["success_rate"].metric_value == 1.0
    samples = metrics.metric_data_entries["service_diagnostics"].recorded_data
    assert len(samples) == 8
    rows = np.concatenate(samples, axis=0)
    assert rows.shape == (8, len(SERVICE_METRIC_NAMES)) and np.isfinite(rows).all()
    assert sorted(env_id for env_id, _ in f.terminal_samples) == list(range(8))
    expected_rows = np.asarray([values for _, values in f.terminal_samples], dtype=np.float32)
    np.testing.assert_array_equal(rows, expected_rows)
    assert (rows[:, 0] == 1).all() and (rows[:, 3] == 0).all() and (rows[:, 5] == 0).all()
    assert sorted(rows[:, 1].tolist()) == [1] * 4 + [2] * 4
    assert (rows[:, 2] == 1).all() and (rows[:, 4] > 1).all()
    assert all(
        model.certificate is None for model in runtime.models
    ), "Autoreset must clear each completed certificate."
    assert all(not status.success for status in runtime.statuses)


def _test_all_scenarios_score_measured_fixtures(_simulation_app, output_dir, asset_root=None):
    fixture = _ScoringFixture(output_dir, asset_root)
    try:
        for phase in (_reset, _load_batteries, _service_airway, _airflow_test, _pack, _scoring):
            phase(fixture)
    finally:
        fixture.close()
    return True


def test_all_scenarios_score_measured_fixtures(tmp_path):
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
    from isaaclab_arena.tests.utils.return_to_service import _require_assets

    _require_assets()
    assert run_function_with_persistent_simulation_app(_test_all_scenarios_score_measured_fixtures, output_dir=tmp_path)
