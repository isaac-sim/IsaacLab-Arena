# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_catalog_layouts(simulation_app):
    import torch

    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.utils.class_path import import_class_from_path
    from isaaclab_arena_environments.cli import ensure_environments_registered
    from isaaclab_arena_environments.spatial_memory.g1_adapter import G1Adapter
    from isaaclab_arena_environments.spatial_memory.loco_catalog import CHALLENGES, PILOT_CONTROLS
    from isaaclab_arena_environments.spatial_memory.tasks.workcell import SurveyedWorkcell

    ensure_environments_registered()
    names = PILOT_CONTROLS + CHALLENGES
    assert len(PILOT_CONTROLS) == 3 and len(CHALLENGES) == 27 and len(set(names)) == 30
    registry = EnvironmentRegistry()
    for name in names:
        factory = registry.get_component_by_name(name)
        logic = import_class_from_path(factory.task_logic_path)(None, G1Adapter())
        assert logic.name == name
        if isinstance(logic, SurveyedWorkcell):
            fixtures = set(logic.part_names())
            assert set(logic.survey_targets) <= fixtures
            assert logic.approach[0] in fixtures
            assert set(logic.survey_targets).isdisjoint(m.fixture for m in logic.motions)
            for stage in logic.stages:
                assert stage.fixture in fixtures
                assert set(stage.contact_parts) <= set(logic.hand_touchable_part_names())
                if stage.released:
                    assert (stage.support or stage.fixture) in logic.supports
            positions = []
            for seed in range(12):
                sample = logic.sample_layout(torch.Generator().manual_seed(seed))
                assert torch.isfinite(sample.part_poses_e).all()
                positions.append(sample.layout_params["entry_y"])
            assert torch.stack(positions).max() - torch.stack(positions).min() > 1.5
    return True


def _test_workcell_approach_and_retreat(simulation_app):
    import math
    import torch

    from isaaclab_arena_environments.spatial_memory.episode_state import get_episode_state
    from isaaclab_arena_environments.spatial_memory.evaluation import make_spatial_memory_env
    from isaaclab_arena_environments.spatial_memory.tasks.workcell import heading

    env = make_spatial_memory_env("spatial_memory_service_trench", 2, 3847)
    try:
        env.reset()
        base = env.unwrapped
        episode = get_episode_state(base)
        logic = episode.logic
        active = torch.ones(2, device=base.device, dtype=torch.bool)
        name, x, y = logic.approach
        fixture = logic.reference_pose(base, name)
        mirror = episode.layout_params["mirror"]
        pelvis = fixture.clone()
        pelvis[:, :3] = logic._goal(fixture, mirror, (x - 0.04, y, 0))
        pelvis[1, :3] = logic._goal(fixture, mirror, (x - 0.04, y + 0.65, 0))[1]
        logic._check_approach(base, episode, active, pelvis)
        assert not logic.commitment_made(base, episode, active).any()
        episode.instruction_revealed[:] = True
        logic._check_approach(base, episode, active, pelvis)
        assert logic.commitment_made(base, episode, active).all()
        assert episode.task_state["wrong_entrance"].tolist() == [False, True]

        wall = logic.reference_pose(base, "end_wall")
        angle = heading(wall[:, 3:]) + math.pi
        pelvis[:, 5] = (angle / 2).sin()
        pelvis[:, 6] = (angle / 2).cos()
        for stage in logic.stages:
            pelvis[:, :3] = logic._goal(wall, mirror, stage.offset)
            for _ in range(math.ceil(stage.hold_s / base.step_dt)):
                logic._advance_operation(base, episode, active, pelvis)
        assert logic.success(base, episode).all()
        assert logic.extra_failures(base, episode).tolist() == [False, True]
        ids = torch.tensor([0], device=base.device)
        episode.reset(ids)
        assert episode.task_state["operation"].tolist() == [0, 3]
        episode.task_state["entered"][0] = True
        episode.task_state["operation"][0] = 1
        episode.task_state["previous_xy"][0] = logic._goal(wall, mirror, logic.stages[0].offset)[0, :2]
        pelvis[0] = wall[0]
        pelvis[0, :3] = logic._goal(wall, mirror, logic.stages[1].offset)[0]
        logic._advance_operation(base, episode, active, pelvis)
        assert logic.extra_failures(base, episode)[0], "Leaving the trench facing forward must fail."
    finally:
        env.close()
    return True


def test_catalog_layouts():
    assert run_function_with_persistent_simulation_app(_test_catalog_layouts)


def test_workcell_approach_and_retreat():
    assert run_function_with_persistent_simulation_app(_test_workcell_approach_and_retreat)


def _test_object_commitments(simulation_app):
    import math
    import torch

    from isaaclab_arena_environments.spatial_memory.episode_state import get_episode_state, write_part_poses
    from isaaclab_arena_environments.spatial_memory.evaluation import make_spatial_memory_env

    for task, item in (("floor_sled", "sled"), ("jig_rotation", "bar")):
        env = make_spatial_memory_env("spatial_memory_" + task, 2, 3847)
        try:
            env.reset()
            base = env.unwrapped
            episode = get_episode_state(base)
            logic = episode.logic
            active = torch.ones(2, device=base.device, dtype=torch.bool)
            episode.instruction_revealed[:] = True
            T_E_item = episode.layout_poses_e[:, episode.part_index[item]].clone()
            logic._check_approach(base, episode, active, None)
            assert not logic.commitment_made(base, episode, active).any()
            if task == "floor_sled":
                direction = base.arena_world.get_pose_e("bumper")[:, :2] - T_E_item[:, :2]
                direction[1] *= -1
                T_E_item[:, :2] += 0.23 * torch.nn.functional.normalize(direction, dim=1)
            else:
                yaw = episode.layout_params["fence_yaw"].clone()
                yaw[0] += math.pi / 2
                T_E_item[:, 5] = (yaw / 2).sin()
                T_E_item[:, 6] = (yaw / 2).cos()
                T_E_item[:, 2] += 0.12
            write_part_poses(base, [item], torch.arange(2, device=base.device), T_E_item[:, None])
            base.sim.forward()
            logic._check_approach(base, episode, active, None)
            assert logic.commitment_made(base, episode, active).all()
            assert episode.task_state["wrong_entrance"].tolist() == [False, True]
        finally:
            env.close()
    return True


def test_object_commitments():
    assert run_function_with_persistent_simulation_app(_test_object_commitments)
