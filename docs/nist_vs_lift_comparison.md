# NIST Gear Insertion vs Lift Object — Arena Integration Comparison

This document shows the structural differences between the **Lift Object** task (the upstream reference) and the **NIST Gear Insertion** task, organized by integration layer. Use it to identify what needs to change to bring NIST into alignment.

---

## 1. Environment File (`isaaclab_arena_environments/`)

| Aspect | Lift (`lift_object_environment.py`) | NIST (`nist_assembled_gearmesh_osc_environment.py`) | Delta |
|--------|------|------|-------|
| **RL Framework** | `RLFramework.RSL_RL` | `RLFramework.RL_GAMES` | Different RL library |
| **Policy Config** | `f"{base_rsl_rl_policy.__name__}:RLPolicyCfg"` (Python class) | `"isaaclab_arena.policy.rl_policy:nist_gear_insertion_osc_rl_games.yaml"` (YAML string) | Lift uses a Python configclass; NIST uses a YAML file. The `rl_policy/` directory has no `__init__.py`, which breaks `importlib.import_module` in Lab's `load_cfg_from_registry`. |
| **Embodiment default** | `"franka_joint_pos"` | `"franka_ik"` | Different default robot mode |
| **Robot config override** | None (uses embodiment default) | Replaces `scene_config.robot` with `FRANKA_MIMIC_OSC_CFG`, overrides `ee_frame`, `initial_joint_pose`, `arm_action`, `gripper_action` | NIST deeply customizes the embodiment post-construction |
| **Observation overrides** | None (uses embodiment defaults) | Nulls out 6 default policy obs, adds custom `nist_gear_policy_obs` ObsTerm | NIST replaces the entire policy observation group |
| **Reward overrides** | None | Nulls out `action_rate` and `joint_vel` from embodiment | NIST replaces default regularizers with task-specific ones |
| **env_cfg_callback** | None | `mdp.assembly_env_cfg_callback` (sets PhysX `gpu_collision_stack_size`) | NIST needs custom sim config |
| **Extra imports** | Minimal (5 imports in `get_env`) | Heavy (10+ imports including OSC controller, frame transformer, custom action/obs) | NIST has significantly more complexity in the env file |

### Lift environment file (89 lines)

```python
# Key structure:
embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(concatenate_observation_terms=True)
# No overrides — uses embodiment defaults for robot, actions, observations, rewards

task = LiftObjectTaskRL(pick_up_object, background, embodiment, ...)

return IsaacLabArenaEnvironment(
    ...,
    rl_framework=RLFramework.RSL_RL,
    rl_policy_cfg=f"{base_rsl_rl_policy.__name__}:RLPolicyCfg",
)
```

### NIST environment file (216 lines)

```python
# Key structure:
embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(...)
embodiment.scene_config.robot = mdp.FRANKA_MIMIC_OSC_CFG.replace(...)     # Override robot
embodiment.scene_config.ee_frame = FrameTransformerCfg(...)               # Override EE frame
embodiment.set_initial_joint_pose([...])                                   # Override joint pose
embodiment.action_config.arm_action = NistGearInsertionOscActionCfg(...)  # Override action
embodiment.action_config.gripper_action = BinaryJointPositionActionCfg(...)
embodiment.observation_config.policy.joint_pos = None                     # Null out defaults
# ... null out 5 more default obs ...
embodiment.observation_config.policy.nist_gear_policy_obs = ObsTerm(...)  # Custom obs
embodiment.reward_config.action_rate = None                               # Null out defaults
embodiment.reward_config.joint_vel = None

task = NistGearInsertionTask(assembled_board, held_gear, background, ...)

return IsaacLabArenaEnvironment(
    ...,
    env_cfg_callback=mdp.assembly_env_cfg_callback,
    rl_framework=RLFramework.RL_GAMES,
    rl_policy_cfg="isaaclab_arena.policy.rl_policy:nist_gear_insertion_osc_rl_games.yaml",
)
```

---

## 2. Task Class (`isaaclab_arena/tasks/`)

| Aspect | Lift (`lift_object_task.py`) | NIST (`nist_gear_insertion_task.py`) | Delta |
|--------|------|------|-------|
| **Base class** | `TaskBase` (IL) / `LiftObjectTaskRL(LiftObjectTask)` (RL) | `TaskBase` (single class) | Lift has IL/RL split; NIST is RL-only |
| **Constructor args** | 8 params (RL version) | 35+ params | NIST is far more parameterized |
| **Observations** | `LiftObjectObservationsCfg` — 2 terms (target_pos, object_pos) in `task_obs` group | `_GearInsertionObservationsCfg` — 9 terms (gear_pos/quat, peg_pos, board_quat, peg_delta, joint_pos/vel, ee_pos/quat) in `task_obs` group | NIST has richer privileged observations |
| **Rewards** | 4 terms: reaching, lifting, goal_tracking, goal_tracking_fine | 10 terms: 3x keypoint_squashing, engagement_bonus, success_bonus, action_penalty, action_grad, contact_penalty, success_pred_error | NIST has a much more complex reward structure |
| **Terminations** | `time_out`, `object_dropped`, `success` (conditional on `rl_training_mode`) | `time_out`, `success` (conditional), `object_dropped`, `gear_dropped_from_gripper`, `gear_orientation_exceeded` | Similar pattern, NIST has more termination conditions |
| **Events** | None (uses embodiment defaults) | 13+ events: `place_gear`, `fixed_asset_pose`, 6x physics materials, `held_object_mass`, `robot_actuator_gains`, `robot_joint_friction` | NIST has extensive domain randomization |
| **Commands** | `LiftObjectCommandsCfg` with `UniformPoseCommandCfg` | None | Lift uses a command manager for goal poses; NIST has no commands |
| **Curriculum** | None | None | Same |
| **`rl_training_mode`** | Passed to termination; disables success termination | Same pattern | Aligned |

---

## 3. Training Scripts

| Aspect | Lift | NIST | Delta |
|--------|------|------|-------|
| **Script** | Isaac Lab's `submodules/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py` | Arena's custom `isaaclab_arena/scripts/reinforcement_learning/train_rl_games.py` | **Completely different scripts** |
| **Environment registration** | `--external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback` | `get_arena_builder_from_cli(args_cli)` → `arena_builder.build_registered()` | Lift uses Lab's callback mechanism; NIST builds directly |
| **Agent config source** | Auto-resolved from gym registry via `@hydra_task_config` decorator → `rsl_rl_cfg_entry_point` | Loaded manually from `--agent_cfg_path` CLI arg (YAML file) | Lift is auto-discovered; NIST is manual |
| **Hydra support** | Yes (override hyperparams via CLI: `agent.policy.activation=relu`) | No | NIST cannot use Hydra overrides |
| **Distributed training** | Built-in `--distributed` flag with multi-GPU rank handling | Not supported | Missing from NIST |
| **W&B tracking** | Not in RSL-RL Arena flow (but available in Lab RL-Games script) | Not supported | Missing from NIST |

### Lift training command

```bash
python submodules/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task lift_object \
  --rl_training_mode \
  --num_envs 4096 \
  --max_iterations 2000
```

### NIST training command

```bash
/isaac-sim/python.sh isaaclab_arena/scripts/reinforcement_learning/train_rl_games.py \
  --environment nist_assembled_gear_mesh_osc \
  --rl_training_mode \
  --num_envs 4096 \
  --max_iterations 1000 \
  --agent_cfg_path isaaclab_arena/policy/rl_policy/nist_gear_insertion_osc_rl_games.yaml
```

---

## 4. Evaluation / Play Scripts

| Aspect | Lift | NIST | Delta |
|--------|------|------|-------|
| **Script** | Arena's `isaaclab_arena/evaluation/policy_runner.py` (generic, framework-agnostic) | Arena's custom `isaaclab_arena/scripts/reinforcement_learning/play_rl_games.py` | Lift uses the generic policy runner; NIST has its own |
| **Policy class** | Uses `PolicyBase` subclass (e.g., RSL-RL policy loaded by policy runner) | Directly creates RL-Games `Runner` and calls `runner.run(play=True)` | NIST bypasses the policy abstraction |
| **NIST also has** | — | `isaaclab_arena/policy/rl_games_action_policy.py` (`RlGamesActionPolicy`) which IS a `PolicyBase` subclass and CAN work with `policy_runner.py` | This exists but isn't the primary eval path |

### Lift play command

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --task lift_object \
  --num_envs 50 \
  --checkpoint <path> \
  --policy_type rsl_rl
```

### NIST play command

```bash
/isaac-sim/python.sh isaaclab_arena/scripts/reinforcement_learning/play_rl_games.py \
  --environment nist_assembled_gear_mesh_osc \
  --num_envs 50 \
  --agent_cfg_path isaaclab_arena/policy/rl_policy/nist_gear_insertion_osc_rl_games.yaml \
  --checkpoint <path>
```

### NIST alternative (via generic policy runner)

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --task nist_assembled_gear_mesh_osc \
  --num_envs 50 \
  --policy_type rl_games \
  --checkpoint_path <path> \
  --agent_cfg_path isaaclab_arena/policy/rl_policy/nist_gear_insertion_osc_rl_games.yaml
```

---

## 5. Policy Configuration

| Aspect | Lift | NIST | Delta |
|--------|------|------|-------|
| **Format** | Python `@configclass` (`RslRlOnPolicyRunnerCfg` subclass) | YAML file | Different formats |
| **Location** | `isaaclab_arena_examples/policy/base_rsl_rl_policy.py` | `isaaclab_arena/policy/rl_policy/nist_gear_insertion_osc_rl_games.yaml` | Different packages |
| **Network** | MLP `[256, 128, 64]`, activation=elu | MLP `[512, 128, 64]` + LSTM (1024 units, 2 layers), activation=elu | NIST uses recurrent policy |
| **PPO params** | `gamma=0.98`, `lam=0.95`, `lr=1e-4`, `entropy_coef=0.006` | `gamma=0.995`, `tau=0.95`, `lr=1e-4`, `entropy_coef=0.0` | Different discount/entropy |
| **Horizon** | `num_steps_per_env=24` | `horizon_length=128`, `seq_length=128` | NIST uses longer rollouts (LSTM needs it) |
| **Obs groups** | `{"actor": ["policy", "task_obs"], "critic": ["policy", "task_obs"]}` | `{"obs": ["policy", "task_obs"], "states": ["policy", "task_obs"]}` + `central_value_config` | NIST uses asymmetric actor-critic with central value |
| **Resolvable by Lab** | Yes — Python class loaded via `load_cfg_from_registry` | Partially — YAML can be loaded but `isaaclab_arena.policy.rl_policy` has no `__init__.py` so `importlib.import_module` fails | **Blocker** for Lab standard path |

---

## 6. Custom MDP Components (NIST-only)

These files exist only for NIST and have no lift equivalent:

| File | Purpose |
|------|---------|
| `isaaclab_arena_environments/mdp/nist_gear_insertion_osc_action.py` | Custom OSC action with peg-relative targets, EMA smoothing, contact-aware dead zones |
| `isaaclab_arena_environments/mdp/nist_gear_insertion_observations.py` | 24-D policy observation (peg-relative EE pose, force feedback, prev actions) |
| `isaaclab_arena_environments/mdp/robot_configs.py` | `FRANKA_MIMIC_OSC_CFG` — custom Franka config for torque control |
| `isaaclab_arena_environments/mdp/env_callbacks.py` | `assembly_env_cfg_callback` — sets PhysX GPU collision stack size |
| `isaaclab_arena/tasks/rewards/gear_insertion_rewards.py` | 7 reward classes/functions (keypoint squashing, bonuses, penalties) |
| `isaaclab_arena/tasks/observations/gear_insertion_observations.py` | 5 observation functions (gear/peg pos, quat, delta) |
| `isaaclab_arena/tasks/terminations.py` | 3 termination functions (insertion success, gear dropped, orientation exceeded) |
| `isaaclab_arena/terms/events.py` | `place_gear_in_gripper` event |

---

## 7. Key Blocker: Isaac Lab Standard Training Path

The `--external_callback` mechanism exists in **RSL-RL's `train.py`** but **not in RL-Games' `train.py`**.

```
Isaac Lab rsl_rl/train.py:
  --external_callback  →  environment_registration_callback()  →  gym.register()  →  gym.make()
  ✅ Lift uses this path

Isaac Lab rl_games/train.py:
  No --external_callback  →  resolve_task_config(task, agent)  →  load_cfg_from_registry()
  ❌ NIST cannot use this path (no callback + YAML module path has no __init__.py)
```

### Options to resolve

| Option | What changes | Effort |
|--------|-------------|--------|
| **A. Switch NIST to RSL-RL** | Write `RslRlOnPolicyRunnerCfg` equivalent; retune hyperparams; lose LSTM (RSL-RL has GRU) | Medium |
| **B. Add `--external_callback` to RL-Games script** | ~8 lines added to `rl_games/train.py` (upstream PR to IsaacLab) + add `__init__.py` to `rl_policy/` | Small code, needs upstream approval |
| **C. Keep custom Arena scripts** | No changes needed, already working | Zero (but stays off standard path) |

---

## 8. Summary: What to Align

Sorted by priority for upstream integration:

1. **Training script path** — The biggest structural difference. Choose option A, B, or C above.
2. **Policy config format** — If staying on RL-Games, add `__init__.py` to `isaaclab_arena/policy/rl_policy/` so the YAML path is resolvable by `importlib`.
3. **Environment file complexity** — NIST's deep embodiment customization (overriding robot, actions, observations, rewards) is inherently more complex than lift. This is unavoidable given OSC control, but the pattern of null-then-replace could be formalized.
4. **Play/eval path** — `RlGamesActionPolicy` is now generic (no NIST defaults). Use `policy_runner.py --policy_type rl_games --agent_cfg_path <yaml> --checkpoint_path <pth>` as the primary eval path. RESOLVED.
5. **Task constructor** — 35+ params could be simplified with a config dataclass pattern (like lift's cleaner constructor), but this is cosmetic.
