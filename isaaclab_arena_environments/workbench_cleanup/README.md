# Workbench cleanup

## VLM goal experiments

Each atomic task has a separate experiment under `experiment_config/`:

| Experiment | Task |
| --- | --- |
| [bin_shelf_vlm_goal_experiment.yaml](experiment_config/bin_shelf_vlm_goal_experiment.yaml) | Pick up the white bin and place it on the overhead shelf. |
| [close_drawer_vlm_goal_experiment.yaml](experiment_config/close_drawer_vlm_goal_experiment.yaml) | Close the top-left drawer. |
| [close_toolcase_vlm_goal_experiment.yaml](experiment_config/close_toolcase_vlm_goal_experiment.yaml) | Close the socket case lid. |
| [turn_off_scope_vlm_goal_experiment.yaml](experiment_config/turn_off_scope_vlm_goal_experiment.yaml) | Press the oscilloscope power button. |

These use `VLMGoalPolicy` and the DROID adapter, following
`droid_gear_insertion_vlm_goal_object_poses_experiment.yaml`. Inputs include camera
images, robot state, and **privileged simulator object poses**. The drawer and
experimental bin tasks use the stock policy. The toolcase and oscilloscope use
`WorkbenchVLMGoalPolicy`, which adds the selected articulation's joint positions;
the toolcase also gets its live lid contact point, outward normal, and closing
tangent. Goal execution and previous-goal feedback use the standard VLM policy.
There is no workbench motion interpolation, extended goal history, or bin grasp
helper. The VLM chooses each goal, and task success conditions are unchanged.
The files under `prompts/` identify the object origins and describe task geometry; they do not
replace the policy's action-format prompt. They describe this environment's
registry assets, so different geometry requires updating them. The runtime scene
uses the `simready_central_*` asset registrations. Prepared workbench, toolcase,
and oscilloscope assets currently require access to Arena's staging storage;
the remaining assets use their registered SimReady Central sources. The socket
set's `spawn_cfg_addon.prim_physics` applies the passive lid drive settings to
`Physics/Joints/case_lid` after loading the remote USD and before cloning. The
authored angular drive values are damping `0.0005235988`, max force `0.2`,
stiffness `0`, and target position `0`. The task reset sets the initial opening;
the remote source asset is unchanged.

The shared scene defaults to `droid_abs_ik`: unscaled absolute XYZ/XYZW commands
in the robot root frame, followed by a gripper command (0 opens, 1 closes).
Observation terms remain separate dictionaries. Experiments render cameras at
the policy's 512-pixel image limit, preserving their aspect ratio, and enable cameras;
other scene users can still leave them disabled. The power-button reference is
observational and does not take ownership of the articulation's reset.

Start with the close-drawer experiment. Run from the repository
root inside the clone's Arena container with `NV_API_KEY` set:

```bash
isaaclab.sh -p isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/workbench_cleanup/experiment_config/close_drawer_vlm_goal_experiment.yaml \
  --viz none --device cuda:0 \
  --output_base_dir artifacts/workbench_vlm_results
```

Each experiment defaults to five episodes in one environment, using GPU simulation
and the internal endpoint's `openai/openai/gpt-6-astra` model. Credentials are read
from the environment. To evaluate five environments concurrently, append
`runs.close_drawer.environment_builder.num_envs=5` (replace the run name for another
experiment). Goal execution uses the standard completion and timeout behavior.
Skipping inference for completed environments is covered separately by
[PR #1648](https://github.com/isaac-sim/IsaacLab-Arena/pull/1648).
Override `runs.close_drawer.policy.endpoint`, `.model`, or `.base_url` when
using another service. Replace the run name for another experiment.

For a bounded integration check, append:

```text
runs.close_drawer.rollout_limit.num_episodes=null
runs.close_drawer.rollout_limit.num_steps=120
```

A step-limited check verifies execution, not task success. Read the experiment's
`arena_experiment_result.json` and its per-episode result files for the outcome.
Add `--record_camera_video` when video recording is wanted.

### Validation status

On 2026-11-11, five parallel environments used seed 42, placement seed 17,
GPU simulation, and the configured Astra model. The reduced policy completed:

| Task | Successes / episodes | Control steps |
| --- | --- | --- |
| Close drawer, stock goal policy | 5 / 5 | 291, 294, 335, 348, 374 |
| Close toolcase, contact observations only | 5 / 5 | 92, 98, 109, 204, 216 |
| Press oscilloscope button, joint observations only | 5 / 5 | 77, 79, 92, 96, 350 |

These runs removed gradual pose interpolation, extended goal history, general
object-bound observations, and the 15-step replacement cadence. The standard
previous-goal outcome and policy camera sizing remain. The recorded scope and
toolcase runs included completed-environment action holding; that duplicate
implementation was subsequently removed in favor of PR #1648. The drawer
comparison disabled it and made 11 unnecessary requests after individual
episodes succeeded.

The stock-toolcase comparison had no completions and all five hinges still at
the open limit after step 216, when the policy with contact feedback had completed
all five. It was stopped around step 222. The scope baseline without joint feedback
was interrupted around step 99 with no completed episodes. These bounded
comparisons support retaining contact feedback; unfinished episodes are not
counted as failures. Retained prompt details and tolerances have not each been
ablated independently. Five episodes demonstrate
capability on these sampled placements, not a general success-rate guarantee.
The scope result verifies the PressButtonTask joint predicate; screen-material
switching was not separately validated.

The bin task remains experimental. Earlier experiments with the removed grasp
and seating helpers returned 0 / 5 on GPU and 0 / 5 on CPU. The simplified bin
configuration has been checked for loading but has not been reevaluated for
success. None of these results changes the task success conditions, and the
privileged observations mean they are not an image-only benchmark.
