# Balance scale measurement

An apple rests on the left pan of an equal-arm balance. The robot (DROID) moves calibrated weights
(100 g ×2, 50 g, 20 g ×2, 10 g) onto the right pan, lets go, and waits until the beam is level and still.
The apple's mass is set per run with Arena's mass variation; the task reads it from the physics engine, so any
mass works.

| File | What it is |
| --- | --- |
| `balance_scale_measurement.yaml` | Environment graph: DROID, RoboLab table, scale, apple, six weights, placement relations, task |
| `object_control/weigh_170g.jsonl`, `weigh_240g.jsonl` | Object Control Runner recipes that balance a 170 g / 240 g apple |
| `object_control/wrong_200g.jsonl` | Negative recipe: 200 g against a 170 g apple must not succeed |
| `check_reachability.py` | cuRobo IK check of DROID grasps at the weights' start poses and drops over the right pan |
| `videos/*.mp4` | Recordings of the three recipes |

Code: `isaaclab_arena/assets/balance_scale_usd.py` (procedural USD articulation),
`BalanceScale` and `CalibratedWeight` in `isaaclab_arena/assets/object_library.py`,
`isaaclab_arena/tasks/balance_scale_task.py`.

## Run

```bash
env PYTHONPATH=$PWD OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \
  isaaclab_arena/agentic_tools/object_control_runner.py \
  --env_spec isaaclab_arena_environments/balance_scale/balance_scale_measurement.yaml \
  --commands isaaclab_arena_environments/balance_scale/object_control/weigh_170g.jsonl \
  --output runs/oc_170 --seed 42 --placement_seed 42 --visualizer kit \
  apple.mass.enabled=true 'apple.mass.sampler_cfg.low=[0.17]' 'apple.mass.sampler_cfg.high=[0.17]'
```

Use `weigh_240g.jsonl` with `[0.24]` (seed 43), or `wrong_200g.jsonl` with `[0.17]` and `--allow_incomplete`.

## Results

| Recipe | Apple | Weights placed | Result | Final beam angle |
| --- | --- | --- | --- | --- |
| `weigh_170g` | 170 g | 100 + 50 + 20 g | success at 33.8 s | −0.13° |
| `weigh_240g` | 240 g | 100 + 100 + 20 + 20 g | success at 41.2 s | −0.10° |
| `wrong_200g` | 170 g | 100 + 100 g | not successful (as intended) | +10° (on its stop) |

`check_reachability.py` on these three layouts: 18/18 weight grasps (with pre-grasp) and 48/48 right-pan drops
(4 slots × 4 weight sizes × beam at −10°, 0°, +10°) are reachable collision-free with the finger pads at most
3.5 cm above the pan rim.

## What can be changed

- Apple mass: `apple.mass.sampler_cfg.low/high` (uniform per environment and reset).
- Measured object: any rigid library object that fits the 13 cm pan, via `registry_name` and its anchored pose.
- Weights: `mass_g` (10, 20, 50 or 100) per weight, the `counterweights` list, their start box and spacing.
- Scale: `initial_pose`, `initial_beam_angle_deg`, `beam_damping`, `pan_damping`; geometry in `BalanceScaleGeometry`.
- Task tolerances: `mass_tolerance_kg`, `level_tolerance_deg`, `beam_speed_threshold_rad_s`, `balance_hold_steps`,
  `gripper_clearance_m`.

## Known gaps

- The robot cannot report the measured grams: Arena has no answer channel from policy to task.
- The scale and apple have fixed poses; placement cannot put an object inside a part of an articulated object.
- The recipes move objects directly (Object Control Runner); no robot policy has run this task yet.
