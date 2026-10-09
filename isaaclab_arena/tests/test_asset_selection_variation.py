# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check asset selection through the ordinary Object and environment builder APIs."""

from __future__ import annotations

import json
import torch
from functools import wraps

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _with_simulation_app(test_function):
    @wraps(test_function)
    def run_test(*args, **kwargs):
        def run_in_app(simulation_app):
            test_function(*args, **kwargs)
            return True

        assert run_function_with_persistent_simulation_app(run_in_app)

    return run_test


def _make_asset_definition(name="small", width=0.1, mass=0.2):
    from isaaclab.sim import CuboidCfg, MassPropertiesCfg, RigidBodyPropertiesCfg

    return name, CuboidCfg(
        size=(width, width, width),
        mass_props=MassPropertiesCfg(mass=mass),
        rigid_props=RigidBodyPropertiesCfg(disable_gravity=True),
        activate_contact_sensors=True,
    )


def _make_object(name="pick_up_object", width=0.1, mass=0.2, **kwargs):
    from isaaclab_arena.assets.object import Object

    return Object(name=name, asset=_make_asset_definition(name, width, mass), **kwargs)


def _attach_selection(target, **cfg_options):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg

    variation = AssetSelectionVariation(
        asset_candidates=[_make_asset_definition("small", 0.1), _make_asset_definition("large", 0.2, 0.5)],
        cfg=AssetSelectionVariationCfg(enabled=True, **cfg_options),
    )
    target.add_variation(variation)
    return variation


def _make_builder(objects, num_envs=5, seed=19, hydra_overrides=None, **cfg_options):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    environment = IsaacLabArenaEnvironment(name="test_asset_selection_variation", scene=Scene(assets=objects))
    return ArenaEnvBuilder(
        environment,
        ArenaEnvBuilderCfg(num_envs=num_envs, seed=seed, solve_relations=False, **cfg_options),
        hydra_overrides=hydra_overrides,
    )


@_with_simulation_app
def test_default_disabled_selection_retains_the_original_object():
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation

    target = _make_object(width=0.7)
    variation = AssetSelectionVariation(asset_candidates=[_make_asset_definition("candidate", width=0.3)])
    target.add_variation(variation)

    assert not variation.enabled
    assert variation.cfg.sample_per_environment
    torch.testing.assert_close(target.get_bounding_box().size, torch.full((1, 3), 0.7))
    env_cfg, env_kwargs = _make_builder([target]).compose_manager_cfg()
    assert env_cfg.scene.pick_up_object.spawn.size == (0.7, 0.7, 0.7)
    assert target.asset_indices_by_env is None
    assert f"{target.name}.{variation.name}" not in env_kwargs["variation_recorder"]


@_with_simulation_app
def test_empty_object_receives_its_asset_from_selection():
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType

    target = Object(name="pick_up_object")
    assert target.object_type == ObjectType.RIGID
    assert target.object_cfg.spawn is None
    variation = _attach_selection(target)
    env_cfg, env_kwargs = _make_builder([target], num_envs=3).compose_manager_cfg()
    assert target.asset_indices_by_env == (0, 1, 0)
    assert len(env_cfg.scene.pick_up_object.spawn.assets_cfg) == 2
    record = env_kwargs["variation_recorder"][f"{target.name}.{variation.name}"]
    assert [record.sample_for_episode(env_id, 0) for env_id in range(3)] == ["small", "large", "small"]


@pytest.mark.parametrize("provider_state", ["missing", "disabled", "disabled_by_override"])
@_with_simulation_app
def test_empty_object_requires_an_enabled_asset_provider_after_overrides(provider_state):
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation

    target = Object(name="pick_up_object")
    overrides = None
    if provider_state == "disabled":
        target.add_variation(AssetSelectionVariation(asset_candidates=[_make_asset_definition()]))
    elif provider_state == "disabled_by_override":
        variation = _attach_selection(target)
        overrides = [f"{target.name}.{variation.name}.enabled=false"]

    with pytest.raises(AssertionError):
        _make_builder([target], hydra_overrides=overrides).compose_manager_cfg()
    assert target.object_cfg.spawn is None


@_with_simulation_app
def test_explicit_default_remains_available_when_hydra_disables_selection():
    target = _make_object(width=0.7)
    variation = _attach_selection(target)
    torch.testing.assert_close(target.get_bounding_box().size, torch.full((1, 3), 0.7))
    env_cfg, env_kwargs = _make_builder(
        [target], hydra_overrides=[f"{target.name}.{variation.name}.enabled=false"]
    ).compose_manager_cfg()
    assert not variation.enabled
    assert env_cfg.scene.pick_up_object.spawn.size == (0.7, 0.7, 0.7)
    assert target.asset_indices_by_env is None
    assert f"{target.name}.{variation.name}" not in env_kwargs["variation_recorder"]


@_with_simulation_app
def test_reusing_an_asset_definition_creates_independent_native_settings():
    from isaaclab_arena.assets.object import Object

    definition = _make_asset_definition("shared_definition", width=0.3, mass=0.5)
    first = Object(name="first", asset=definition)
    second = Object(name="second", asset=definition)
    first.spawn_cfg.size = (0.9, 0.9, 0.9)
    first.spawn_cfg.mass_props.mass = 2.0
    definition[1].mass_props.mass = 3.0
    assert second.spawn_cfg.size == (0.3, 0.3, 0.3)
    assert second.spawn_cfg.mass_props.mass == 0.5
    assert definition[1].size == (0.3, 0.3, 0.3)
    assert first.spawn_cfg.mass_props.mass == 2.0


@_with_simulation_app
def test_registry_definitions_keep_the_registry_name_as_the_recorded_candidate_id():
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg

    registry = AssetRegistry()
    definition = registry.get_asset_definition("sphere", instance_name="temporary_source")
    assert definition[0] == "sphere"
    second_definition = registry.get_asset_definition("sphere")
    definition[1].radius = 0.25
    assert second_definition[1].radius == 0.1
    target = Object(name="pick_up_object")
    variation = AssetSelectionVariation(asset_candidates=[definition], cfg=AssetSelectionVariationCfg(enabled=True))
    target.add_variation(variation)
    _, env_kwargs = _make_builder([target], num_envs=2).compose_manager_cfg()
    assert target.spawn_cfg.radius == 0.25
    record = env_kwargs["variation_recorder"][f"{target.name}.{variation.name}"]
    assert [record.sample_for_episode(env_id, 0) for env_id in range(2)] == ["sphere", "sphere"]


@pytest.mark.parametrize(
    "query",
    [
        lambda target: target.get_bounding_box(),
        lambda target: target.get_bounding_box_for_env(0),
        lambda target: target.get_bounding_box_per_env(3),
        lambda target: target.get_collision_mesh(),
        lambda target: target.get_contact_sensor_prim_path(),
    ],
    ids=["bounds", "environment_bounds", "all_bounds", "collision_mesh", "contact_path"],
)
@_with_simulation_app
def test_empty_object_rejects_geometry_queries_until_the_builder_resolves_its_asset(query):
    from isaaclab_arena.assets.object import Object

    target = Object(name="pick_up_object")
    _attach_selection(target)
    with pytest.raises(AssertionError):
        query(target)


@_with_simulation_app
def test_scene_export_rejects_unresolved_selection_and_resolved_multiple_assets(tmp_path):
    target = _make_object()
    _attach_selection(target)
    builder = _make_builder([target])
    output_path = tmp_path / "selected_scene.usda"
    with pytest.raises(AssertionError):
        builder.arena_env.scene.export_to_usd(output_path)
    assert not output_path.exists()

    builder.compose_manager_cfg()
    with pytest.raises(AssertionError, match="single-asset spawn configuration"):
        builder.arena_env.scene.export_to_usd(output_path)
    assert not output_path.exists()


@_with_simulation_app
def test_sequential_selection_copies_candidates_and_preserves_the_target():
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg

    initial_pose = Pose(position_xyz=(1.0, 2.0, 3.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    anchor = IsAnchor()
    target = _make_object(prim_path="{ENV_REGEX_NS}/Pickup", initial_pose=initial_pose, relations=[anchor])
    candidates = [_make_asset_definition("small", 0.1), _make_asset_definition("large", 0.2, 0.5)]
    variation = AssetSelectionVariation(asset_candidates=candidates, cfg=AssetSelectionVariationCfg(enabled=True))
    target.add_variation(variation)
    candidates[0][1].size = (9.0, 9.0, 9.0)
    candidates[0] = ("renamed_after_declaration", candidates[0][1])
    candidates[1][1].mass_props.mass = 99.0
    candidates.reverse()
    builder = _make_builder([target])
    env_cfg, env_kwargs = builder.compose_manager_cfg()

    assert type(target) is Object
    assert builder.arena_env.scene.assets["pick_up_object"] is target
    assert target.prim_path == "{ENV_REGEX_NS}/Pickup"
    assert target.initial_pose is initial_pose
    assert target.get_relations() == [anchor]
    assert target.asset_indices_by_env == (0, 1, 0, 1, 0)
    assert env_cfg.scene.pick_up_object.init_state.pos == initial_pose.position_xyz
    assert [spawn_cfg.mass_props.mass for spawn_cfg in target.spawn_cfg.assets_cfg] == [0.2, 0.5]
    expected_sizes = torch.tensor([[0.1] * 3, [0.2] * 3, [0.1] * 3, [0.2] * 3, [0.1] * 3])
    torch.testing.assert_close(target.get_bounding_box_per_env(5).size, expected_sizes)
    record = env_kwargs["variation_recorder"][f"{target.name}.{variation.name}"]
    for episode_index in (0, 1, 8):
        assert [record.sample_for_episode(env_id, episode_index) for env_id in range(5)] == [
            "small",
            "large",
            "small",
            "large",
            "small",
        ]


@_with_simulation_app
def test_shared_selection_broadcasts_one_choice_to_every_environment():
    target = _make_object()
    variation = _attach_selection(target, sample_per_environment=False)
    _, env_kwargs = _make_builder([target]).compose_manager_cfg()
    assert target.asset_indices_by_env == (0,) * 5
    torch.testing.assert_close(target.get_bounding_box_per_env(5).size, torch.full((5, 3), 0.1))
    record = env_kwargs["variation_recorder"][f"{target.name}.{variation.name}"]
    assert [record.sample_for_episode(env_id, 4) for env_id in range(5)] == ["small"] * 5


@_with_simulation_app
def test_each_fresh_sequential_build_starts_from_the_first_candidate():
    for num_envs in (3, 7):
        target = _make_object()
        _attach_selection(target)
        _make_builder([target], num_envs=num_envs).compose_manager_cfg()
        assert target.asset_indices_by_env == tuple(index % 2 for index in range(num_envs))


@_with_simulation_app
def test_same_asset_definition_can_supply_a_default_and_a_candidate():
    from isaaclab.sim import MultiAssetSpawnerCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg

    definition = _make_asset_definition("small_definition", width=0.3)
    target = Object(name="pick_up_object", asset=definition)
    variation = AssetSelectionVariation(asset_candidates=[definition], cfg=AssetSelectionVariationCfg(enabled=True))
    target.add_variation(variation)
    _, env_kwargs = _make_builder([target], num_envs=3).compose_manager_cfg()
    assert not isinstance(target.spawn_cfg, MultiAssetSpawnerCfg)
    assert target.asset_indices_by_env == (0, 0, 0)
    torch.testing.assert_close(target.get_bounding_box_per_env(3).size, torch.full((3, 3), 0.3))
    record = env_kwargs["variation_recorder"][f"{target.name}.{variation.name}"]
    assert [record.sample_for_episode(env_id, 2) for env_id in range(3)] == ["small_definition"] * 3


@_with_simulation_app
def test_random_selection_is_seeded_independently_of_other_objects_and_global_draws():
    from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg

    def build_assignments(names, seed):
        objects = []
        for name in names:
            target = _make_object(name)
            _attach_selection(target, sampler_cfg=ChoiceSamplerCfg())
            objects.append(target)
        _make_builder(objects, num_envs=32, seed=seed).compose_manager_cfg()
        return {target.name: target.asset_indices_by_env for target in objects}

    original = build_assignments(["pickup", "destination"], seed=19)
    torch.rand(100)
    reordered = build_assignments(["destination", "unrelated", "pickup"], seed=19)
    assert original == {name: reordered[name] for name in original}
    assert original["pickup"] != original["destination"]
    assert original != build_assignments(["pickup", "destination"], seed=20)


@_with_simulation_app
def test_hydra_can_enable_selection_and_choose_shared_sampling():
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation

    target = Object(name="pick_up_object")
    variation = AssetSelectionVariation(
        asset_candidates=[_make_asset_definition("small"), _make_asset_definition("large", 0.2)]
    )
    target.add_variation(variation)
    _make_builder(
        [target],
        hydra_overrides=[
            f"{target.name}.{variation.name}.enabled=true",
            f"{target.name}.{variation.name}.sample_per_environment=false",
        ],
    ).compose_manager_cfg()
    assert variation.enabled
    assert target.asset_indices_by_env == (0,) * 5


@pytest.mark.parametrize("candidate_names", [[], [""], ["same", "same"]], ids=["empty", "unnamed", "duplicate"])
@_with_simulation_app
def test_selection_requires_nonempty_unique_candidate_names(candidate_names):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation

    with pytest.raises(AssertionError):
        AssetSelectionVariation(asset_candidates=[_make_asset_definition(name) for name in candidate_names])


@pytest.mark.parametrize("candidate_kind", ["scene_object", "raw_config", "multi_asset", "multi_usd", "invalid_config"])
@_with_simulation_app
def test_selection_requires_named_single_asset_definitions(candidate_kind):
    from isaaclab.sim import MultiAssetSpawnerCfg, MultiUsdFileCfg

    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation

    if candidate_kind == "scene_object":
        candidate = _make_object("candidate")
    elif candidate_kind == "raw_config":
        candidate = _make_asset_definition()[1]
    elif candidate_kind == "multi_asset":
        candidate = ("candidate", MultiAssetSpawnerCfg(assets_cfg=[_make_asset_definition()[1]]))
    elif candidate_kind == "multi_usd":
        candidate = ("candidate", MultiUsdFileCfg(usd_path=["first.usda", "second.usda"]))
    else:
        candidate = ("candidate", None)
    with pytest.raises(AssertionError):
        AssetSelectionVariation(asset_candidates=[candidate])


@_with_simulation_app
def test_selection_cannot_be_attached_to_two_targets_or_duplicated_on_one_target():
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation

    target = _make_object()
    variation = _attach_selection(target)
    with pytest.raises(AssertionError):
        _make_object("second_target").add_variation(variation)
    with pytest.raises(AssertionError):
        target.add_variation(
            AssetSelectionVariation(asset_candidates=[_make_asset_definition("other")], name="other_selection")
        )


@_with_simulation_app
def test_a_second_builder_cannot_reuse_a_resolved_selection_graph():
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    target = _make_object()
    _attach_selection(target)
    builder = _make_builder([target])
    builder.compose_manager_cfg()
    with pytest.raises(AssertionError):
        ArenaEnvBuilder(builder.arena_env, builder.cfg).compose_manager_cfg()
    with pytest.raises(AssertionError):
        builder.compose_manager_cfg()


@pytest.mark.parametrize("mutation", ["size", "mass", "candidate_order", "selection_cfg"])
@_with_simulation_app
def test_builder_rejects_final_config_edits_that_invalidate_asset_selection(mutation):
    target = _make_object()
    variation = _attach_selection(target)
    builder = _make_builder([target])

    def change_selected_asset(env_cfg):
        selected_spawn = env_cfg.scene.pick_up_object.spawn
        if mutation == "size":
            selected_spawn.assets_cfg[0].size = (0.9, 0.9, 0.9)
        elif mutation == "mass":
            selected_spawn.assets_cfg[0].mass_props.mass = 10.0
        elif mutation == "candidate_order":
            selected_spawn.assets_cfg.reverse()
        else:
            variation.cfg.sample_per_environment = False
        return env_cfg

    builder.arena_env.env_cfg_callback = change_selected_asset
    with pytest.raises(AssertionError, match="changed"):
        builder.compose_manager_cfg()


@_with_simulation_app
def test_single_candidate_selection_rejects_spawn_changes_in_the_final_config_callback():
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg

    target = _make_object()
    target.add_variation(
        AssetSelectionVariation(
            asset_candidates=[_make_asset_definition("small")], cfg=AssetSelectionVariationCfg(enabled=True)
        )
    )
    builder = _make_builder([target])

    def resize_selected_object(env_cfg):
        env_cfg.scene.pick_up_object.spawn.size = (0.9, 0.9, 0.9)
        return env_cfg

    builder.arena_env.env_cfg_callback = resize_selected_object
    with pytest.raises(AssertionError, match="changed"):
        builder.compose_manager_cfg()


@pytest.mark.parametrize("candidate_count", [1, 2])
@_with_simulation_app
def test_builder_rejects_resolved_source_changes_even_when_the_final_config_is_unchanged(candidate_count):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg

    target = _make_object()
    definitions = [_make_asset_definition("small"), _make_asset_definition("large", width=0.2)]
    target.add_variation(
        AssetSelectionVariation(
            asset_candidates=definitions[:candidate_count], cfg=AssetSelectionVariationCfg(enabled=True)
        )
    )
    builder = _make_builder([target])

    def resize_source_object(env_cfg):
        source_spawn = target.spawn_cfg
        final_spawn = env_cfg.scene.pick_up_object.spawn
        if candidate_count > 1:
            source_spawn = source_spawn.assets_cfg[0]
            final_spawn = final_spawn.assets_cfg[0]
        source_spawn.size = (0.9, 0.9, 0.9)
        assert final_spawn.size == (0.1, 0.1, 0.1)
        return env_cfg

    builder.arena_env.env_cfg_callback = resize_source_object
    with pytest.raises(AssertionError, match="(?i)resolved asset settings changed"):
        builder.compose_manager_cfg()


@_with_simulation_app
def test_scene_export_rejects_source_changes_after_single_candidate_resolution(tmp_path):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg

    target = _make_object()
    target.add_variation(
        AssetSelectionVariation(
            asset_candidates=[_make_asset_definition("small")], cfg=AssetSelectionVariationCfg(enabled=True)
        )
    )
    builder = _make_builder([target])
    builder.compose_manager_cfg()
    target.spawn_cfg.size = (0.9, 0.9, 0.9)
    output_path = tmp_path / "changed_selected_asset.usda"
    with pytest.raises(AssertionError, match="(?i)resolved asset settings changed"):
        builder.arena_env.scene.export_to_usd(output_path)
    assert not output_path.exists()


@pytest.mark.parametrize("layout_source", ["in_memory", "file"])
@_with_simulation_app
def test_single_candidate_selection_rejects_cached_placement_before_sampling(tmp_path, layout_source):
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg

    target = _make_object()
    variation = AssetSelectionVariation(
        asset_candidates=[_make_asset_definition("small")], cfg=AssetSelectionVariationCfg(enabled=True)
    )
    target.add_variation(variation)
    builder = _make_builder([target])
    layouts = PlacementLayouts({target.name: [Pose.identity()]})
    if layout_source == "in_memory":
        builder.arena_env.placement_layouts = layouts
    else:
        layouts_path = tmp_path / "layouts.jsonl"
        layouts.write_episode_jsonl(layouts_path, source="test")
        builder.cfg.placement_layouts_path = str(layouts_path)

    with pytest.raises(AssertionError, match="placement replay"):
        builder.compose_manager_cfg()
    assert variation.selected_candidate_names is None
    assert target.asset_indices_by_env is None


@_with_simulation_app
def test_selection_rejects_another_enabled_build_time_variation_before_sampling():
    from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
    from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

    target = _make_object()
    selection = _attach_selection(target)

    class ResizeObjectVariation(BuildTimeVariationBase):
        def _realize_at_build_time(self, context=None):
            target.spawn_cfg.size = (0.9, 0.9, 0.9)

    target.add_variation(
        ResizeObjectVariation(cfg=VariationBaseCfg(enabled=True, sampler_cfg=ChoiceSamplerCfg()), name="resize_object")
    )
    with pytest.raises(AssertionError, match="another build-time variation"):
        _make_builder([target]).compose_manager_cfg()
    assert selection.selected_candidate_names is None
    assert target.spawn_cfg.size == (0.1, 0.1, 0.1)


@_with_simulation_app
def test_recorded_replay_explicitly_rejects_asset_selection(tmp_path):
    target = _make_object()
    variation = _attach_selection(target)
    recordings_path = tmp_path / "episodes.jsonl"
    recordings_path.write_text(json.dumps({"variations": {f"{target.name}.{variation.name}": "small"}}) + "\n")
    builder = _make_builder([target], recorded_variation_samples_path=str(recordings_path))
    with pytest.raises(AssertionError, match="[Rr]eplay"):
        builder.compose_manager_cfg()
