# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check registered policy construction, execution strategies, and client ownership."""

import torch
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from isaaclab_arena_vlm_agent import droid_adapter, vlm_agent_policy
from isaaclab_arena_vlm_agent.absolute_action_adapter import AbsoluteChunkActionAdapter, AbsoluteStepActionAdapter
from isaaclab_arena_vlm_agent.command_indices import AbsoluteCommandIndices, GripperCommandIndex, PoseCommandIndices
from isaaclab_arena_vlm_agent.vlm_agent_policy import (
    VLMChunkPolicy,
    VLMChunkPolicyCfg,
    VLMGoalPolicy,
    VLMGoalPolicyCfg,
    VLMStepPolicy,
    VLMStepPolicyCfg,
)

POLICIES = [
    (VLMStepPolicy, VLMStepPolicyCfg),
    (VLMChunkPolicy, VLMChunkPolicyCfg),
    (VLMGoalPolicy, VLMGoalPolicyCfg),
]


@pytest.fixture(autouse=True)
def gripper_measurements(monkeypatch):
    gripper = SimpleNamespace(
        get_closing_error_m=lambda env: torch.zeros(2),
        get_opening_width_m=lambda world: torch.zeros(2),
    )
    monkeypatch.setattr(droid_adapter, "_make_gripper", lambda: gripper)


def _config(cfg_type, modality="image_state"):
    options = {"max_steps_per_goal": 2} if cfg_type is VLMGoalPolicyCfg else {}
    cfg = cfg_type(system_prompt="Return absolute targets.", model="test-model", **options)
    cfg.observation_modality = modality
    if cfg_type is VLMChunkPolicyCfg:
        cfg.chunk_size = 2
    return cfg


@pytest.mark.parametrize("policy_type,cfg_type", POLICIES)
@pytest.mark.parametrize("modality", ["image", "state", "image_state"])
def test_configured_execution_and_partial_reset(monkeypatch, policy_type, cfg_type, modality):
    backend = Mock()
    replies = []
    for x in (1.0, 3.0, 5.0, 7.0):
        command = {"values": [x, 0, 0, 0, 0, 0, 1, 0]}
        if policy_type is VLMChunkPolicy:
            command = {"commands": [command, {"values": [x + 1, 0, 0, 0, 0, 0, 1, 0]}]}
        replies.append(command)
    backend.infer.side_effect = replies
    constructor = Mock(return_value=backend)
    monkeypatch.setattr(vlm_agent_policy, "InferenceBackend", constructor)
    indices = AbsoluteCommandIndices(
        8,
        (PoseCommandIndices(slice(0, 7), "robot", "base_link"),),
        (),
        (GripperCommandIndex(7, "gripper_action", 0.0, 1.0),),
    )
    for module in ("absolute_action_adapter", "pose_target_motion_generator"):
        monkeypatch.setattr(f"isaaclab_arena_vlm_agent.{module}.read_absolute_command_indices", lambda env: indices)
    observation_cfg = SimpleNamespace(
        camera_obs=SimpleNamespace(live_camera=SimpleNamespace(func=Mock(), params={"data_type": "rgb"}))
    )
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            num_envs=2,
            device="cpu",
            arena_world=SimpleNamespace(
                get_body_pose_in_root=lambda *args: torch.tensor([[0, 0, 0, 0, 0, 0, 1.0]] * 2)
            ),
            cfg=SimpleNamespace(observations=observation_cfg),
            observation_manager=SimpleNamespace(
                active_terms={"policy": ["joint_pos"]}, group_obs_concatenate={"policy": False}
            ),
            action_manager=SimpleNamespace(active_terms=[], total_action_dim=8, action=torch.zeros(2, 8)),
            episode_length_buf=torch.zeros(2, dtype=torch.long),
        )
    )
    observations = {
        "policy": {"joint_pos": torch.zeros(2, 1)},
        "camera_obs": {"live_camera": torch.zeros(2, 4, 4, 3, dtype=torch.uint8)},
    }
    policy = policy_type(_config(cfg_type, modality))
    policy.reset()
    policy.set_task_description("Move the robot")
    torch.testing.assert_close(policy.get_action(env, observations)[:, 0], torch.tensor([1.0, 3.0]))
    content = backend.infer.call_args.args[0].messages[1]["content"]
    assert any(item["type"] == "image_url" for item in content) == (modality != "state")
    assert any('"joint_pos"' in item.get("text", "") for item in content) == (modality != "image")
    assert backend.infer.call_args.args[0].messages[0]["content"] == "Return absolute targets."
    policy.reset(torch.tensor([0]))
    expected_second = {VLMStepPolicy: 7.0, VLMChunkPolicy: 4.0, VLMGoalPolicy: 3.0}[policy_type]
    torch.testing.assert_close(policy.get_action(env, observations)[:, 0], torch.tensor([5.0, expected_second]))
    assert backend.infer.call_count == (4 if policy_type is VLMStepPolicy else 3)
    if policy_type is VLMGoalPolicy:
        # Only env 1 has exhausted the two-step goal budget after the partial reset.
        torch.testing.assert_close(policy.get_action(env, observations)[:, 0], torch.tensor([5.0, 7.0]))
        assert backend.infer.call_count == 4
    assert constructor.call_args.kwargs["model"] == "test-model"
    policy.close()
    policy.close()
    backend.client.close.assert_called_once()
    assert not getattr(policy, "_chunks", {})
    assert not getattr(policy, "_goals", {})


@pytest.mark.parametrize("policy_type,cfg_type", POLICIES)
def test_injected_backend_remains_caller_owned(monkeypatch, policy_type, cfg_type):
    constructor = Mock()
    monkeypatch.setattr(vlm_agent_policy, "InferenceBackend", constructor)
    backend = Mock()
    policy = policy_type(_config(cfg_type), inference_backend=backend)
    policy.close()
    constructor.assert_not_called()
    backend.client.close.assert_not_called()


@pytest.mark.parametrize("policy_type,cfg_type", POLICIES)
def test_invalid_adapter_rejected_before_client_creation(monkeypatch, policy_type, cfg_type):
    constructor = Mock()
    monkeypatch.setattr(vlm_agent_policy, "InferenceBackend", constructor)
    cfg = _config(cfg_type)
    adapter = AbsoluteChunkActionAdapter(1) if policy_type is not VLMChunkPolicy else AbsoluteStepActionAdapter()
    with pytest.raises(AssertionError):
        policy_type(cfg, action_adapter=adapter)
    constructor.assert_not_called()


def test_invalid_schema_rejected_before_client_creation(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(vlm_agent_policy, "InferenceBackend", constructor)
    monkeypatch.setattr(vlm_agent_policy, "build_strict_schema", Mock(side_effect=RuntimeError("invalid schema")))
    with pytest.raises(RuntimeError, match="invalid schema"):
        VLMStepPolicy(_config(VLMStepPolicyCfg))
    constructor.assert_not_called()


def test_example_yaml_policy_selectors_and_round_trip(monkeypatch, tmp_path):
    from isaaclab_arena.assets import registries
    from isaaclab_arena.evaluation import arena_experiment_config_loader as loader
    from isaaclab_arena.hydra.typed_experiment_serializer import serialize_arena_experiment_to_yaml
    from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironmentCfg

    monkeypatch.setattr(registries, "ensure_assets_registered", lambda: None)
    monkeypatch.setattr(
        loader,
        "_registered_environment_cfg_types",
        lambda: {"pick_and_place_maple_table": PickAndPlaceMapleTableEnvironmentCfg},
    )
    for policy_type, cfg_type in POLICIES:
        class_path = f"isaaclab_arena_vlm_agent.vlm_agent_policy.{policy_type.__name__}"
        for selector in (policy_type.name, class_path):
            assert loader._resolve_policy_cfg_type_from_name_or_class_path(selector) is cfg_type
    root = Path(__file__).resolve().parents[2]
    experiment = loader.load_arena_experiment_from_config_file(
        root / "isaaclab_arena_environments/experiment_configs/droid_pnp_vlm_experiment.yaml",
        device="cpu",
        overrides=[
            "runs.droid_pnp_vlm_goal.policy.max_steps_per_goal=5",
            "runs.droid_pnp_vlm_chunk.policy.chunk_size=2",
        ],
    )
    assert len(experiment.runs) == 3
    for run, (policy_type, cfg_type) in zip(experiment.runs.values(), POLICIES, strict=True):
        assert isinstance(run.policy, cfg_type)
        assert run.environment_builder.device == "cpu"
        assert registries.PolicyRegistry().get_policy_type_for_cfg(run.policy) is policy_type
    assert experiment.runs["droid_pnp_vlm_goal"].policy.max_steps_per_goal == 5
    assert experiment.runs["droid_pnp_vlm_chunk"].policy.chunk_size == 2
    round_trip_path = tmp_path / "experiment.yaml"
    round_trip_path.write_text(serialize_arena_experiment_to_yaml(experiment))
    restored = loader.load_arena_experiment_from_config_file(round_trip_path, device="cpu")
    assert restored == experiment


@pytest.mark.parametrize("policy_type,cfg_type", POLICIES)
def test_preset_supplies_prompt_and_strategy(monkeypatch, policy_type, cfg_type):
    monkeypatch.setattr(vlm_agent_policy, "InferenceBackend", Mock())
    cfg = _config(cfg_type)
    cfg.system_prompt = None
    policy = policy_type(cfg)
    assert "base_link" in policy._system_prompt
    if policy_type is VLMChunkPolicy:
        assert "exactly 2 commands" in policy._system_prompt
        assert policy.action_adapter.chunk_size == 2
    else:
        assert "Return one command" in policy._system_prompt
    if policy_type is VLMGoalPolicy:
        assert "gripper_action" in policy.action_adapter.motion_generator.grippers
    policy.close()


@pytest.mark.parametrize(
    "options", [{"vlm_embodiment_adapter": "unknown"}, {"observation_modality": "unknown"}, {"chunk_size": 0}]
)
def test_invalid_flat_settings(options):
    with pytest.raises(AssertionError):
        VLMChunkPolicyCfg(**options)
