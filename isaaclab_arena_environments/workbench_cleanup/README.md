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
images, robot state, and **privileged simulator object poses**. The toolcase, oscilloscope, and bin
experiments extend this with `WorkbenchVLMGoalPolicy`: gradual pose commands,
recent-goal history, and explicitly labeled simulator joint positions and local
object bounds. For the bin, it also reports grasp-motion measurements, live
side-wall contact estimates, tabletop and overhead-edge clearances, and the
same bounds-center check used by placement. Shelf seating estimates assume
the measured grasp stays fixed; a front-bay alternative reduces reach.
The VLM still chooses every goal; the extension does not supply
a scripted task sequence or alter success conditions. The files under
`prompts/` identify the object origins and describe task geometry; they do not
replace the policy's action-format prompt. They describe this environment's
registry assets, so different geometry requires updating them. The runtime scene
uses the `simready_central_*` asset registrations. Prepared workbench, toolcase,
and oscilloscope assets currently require access to Arena's staging storage;
the remaining assets use their registered SimReady Central sources. This VLM
extension does not reauthor or override those assets. In particular, the old
local toolcase hinge fix is not applied to the runtime asset.

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
experiment). Goal replacement requests are grouped every 15 control steps;
finished episode slots hold their last action without further inference.
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

The following results are historical, from the direct-assets branch on
2026-11-10, before retargeting to the runtime registry assets. Five parallel
environments used seed 42, placement seed 17, and the configured Astra model:

| Task | Successes / episodes |
| --- | --- |
| Close drawer | 5 / 5 |
| Close toolcase | 3 / 5 |
| Turn off oscilloscope | 5 / 5 |
| Bin to shelf | 0 / 5 |

The bin experiment remains unfinished: it grasps and lifts the bin but loses
the grasp during reorientation or transfer. A separate five-episode CPU trial,
using a 6 mm motion-goal tolerance and a 60-step goal limit, also returned 0 / 5.
Neither trial changed the placement success criteria. These results use the
privileged observations described above; they are not an image-only benchmark.
Interrupted development episodes are excluded from the counts.
