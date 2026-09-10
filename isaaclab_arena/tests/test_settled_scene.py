# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Round-trip physics-settled clutter through the public YAML example."""

from pathlib import Path

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_settled_scene_round_trip(simulation_app):
    import tempfile
    import torch
    from argparse import Namespace

    import warp as wp

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.placement_events import get_placement_pool
    from isaaclab_arena_examples.relations.generate_clutter_scene import generate_scene

    source = Path(__file__).parents[2] / "isaaclab_arena_examples/relations/clutter_scene.yaml"
    with tempfile.TemporaryDirectory() as directory:
        args = Namespace(
            env_spec=source,
            output=Path(directory) / "scene.yaml",
            num_envs=2,
            seed=42,
            layouts_per_env=2,
            device="cuda:0",
            presets=None,
            register=[],
        )
        paths = generate_scene(args)
        assert len(paths) == 2
        specs = [ArenaEnvGraphSpec.from_yaml(path) for path in paths]
        assert all(not spec.relations for spec in specs)
        assert specs[0].objects[0].params["initial_pose"] != specs[1].objects[0].params["initial_pose"]
        spec = specs[0]
        env = ArenaEnvBuilder(spec.to_arena_env(), ArenaEnvBuilderCfg(num_envs=2)).make_registered()
        try:
            assert get_placement_pool(env) is None
            for _ in range(3):
                env.reset()
                for obj in spec.objects:
                    expected = obj.params["initial_pose"]
                    T_E_O = torch.tensor(
                        expected["position_xyz"] + expected["rotation_xyzw"], device=env.unwrapped.device
                    )
                    T_W_O = wp.to_torch(env.unwrapped.scene[obj.id].data.root_link_pose_w)[:, :7].clone()
                    T_W_O[:, :3] -= env.unwrapped.scene.env_origins
                    torch.testing.assert_close(T_W_O, T_E_O.expand_as(T_W_O), atol=2e-5, rtol=0)
        finally:
            env.close()
    return True


def test_settled_scene_round_trip():
    assert run_function_with_persistent_simulation_app(_test_settled_scene_round_trip)


def test_initial_pose_validation_and_reset_contract():
    import pytest

    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import _apply_initial_pose

    class Asset:
        def set_initial_pose(self, pose, create_reset_event=True):
            self.pose = pose
            self.reset = create_reset_event

    asset = Asset()
    _apply_initial_pose(asset, {"position_xyz": [1, 2, 3], "rotation_xyzw": [0, 0, 0, 1]})
    assert asset.pose.position_xyz == (1.0, 2.0, 3.0)
    assert asset.reset
    for bad in (
        {"position_xyz": [float("nan"), 0, 0]},
        {"position_xyz": [True, 0, 0]},
        {"rotation_xyzw": [0, 0, 0, 0]},
        {"rotation_xyzw": [0, 0, 1]},
        {"unknown": 1},
    ):
        with pytest.raises(AssertionError):
            _apply_initial_pose(asset, bad)


def test_export_rejects_invalid_layouts():
    import pytest

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.placement_validation import PlacementValidationResults
    from isaaclab_arena.relations.settled_scene import settled_scene_spec

    source = Path(__file__).parents[2] / "isaaclab_arena_examples/relations/clutter_scene.yaml"
    spec = ArenaEnvGraphSpec.from_yaml(source)
    for checks in (
        {},
        {"captured_objects_settled": False},
        {"captured_objects_settled": True, "clutter_contained": False},
        {
            "captured_objects_settled": True,
            "clutter_contained": True,
            "final_poses_validated": True,
            "on_relation": False,
        },
    ):
        layout = PlacementResult(PlacementValidationResults(checks), {}, 0.0, 1)
        with pytest.raises(AssertionError):
            settled_scene_spec(spec, layout, {})


def test_reference_path_uses_named_parent(monkeypatch):
    from types import SimpleNamespace

    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environment_spec import arena_env_graph_conversion_utils as conversion

    monkeypatch.setattr(conversion, "ObjectReference", lambda **kwargs: kwargs)
    ref = SimpleNamespace(id="floor", prim_path="placement_surface", params={}, object_type=ObjectType.BASE)
    parent = SimpleNamespace(name="source_bin")
    created = conversion._instantiate_object_reference(ref, parent)
    assert created["prim_path"] == "{ENV_REGEX_NS}/source_bin/placement_surface"
    assert created["parent_asset"] is parent


def test_export_uses_graph_identity_when_runtime_names_differ():
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.placement_validation import PlacementValidationResults
    from isaaclab_arena.relations.settled_scene import settled_scene_spec
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    source = Path(__file__).parents[2] / "isaaclab_arena_examples/relations/clutter_scene.yaml"
    spec = ArenaEnvGraphSpec.from_yaml(source)
    nodes = [spec.background, spec.embodiment, *spec.objects]
    mapping = {
        node.id: DummyObject(f"runtime_{index}", AxisAlignedBoundingBox(min_point=(0, 0, 0), max_point=(1, 1, 1)))
        for index, node in enumerate(nodes)
    }
    positions = {asset: (float(index), 0.0, 0.5) for index, asset in enumerate(mapping.values())}
    layout = PlacementResult(
        PlacementValidationResults({
            "captured_objects_settled": True,
            "clutter_contained": True,
            "final_poses_validated": True,
        }),
        positions,
        0.0,
        1,
    )
    result = settled_scene_spec(spec, layout, mapping)
    for node in [result.background, result.embodiment, *result.objects]:
        assert node.params["initial_pose"]["position_xyz"] == list(positions[mapping[node.id]])
    assert spec.relations and not result.relations
    assert all("initial_pose" not in node.params for node in spec.objects)
