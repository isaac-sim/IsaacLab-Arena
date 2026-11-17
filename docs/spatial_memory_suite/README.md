# Spatial-memory task suite

Ten Isaac Lab-Arena environments in which the geometry that matters at action time is out of the egocentric camera's
view, so a policy must have remembered or mapped it. Every task has four observation variants, a difficulty knob, a
privileged scripted oracle, a "blind" scripted controller, collision and visibility logging, and tests that check that
the suite tests what it claims. Design notes and the decisions taken along the way are in [PLAN.md](PLAN.md).

The code lives in `isaaclab_arena_environments/spatial_memory/` (shared infrastructure and one module per task under
`tasks/`), with one registered environment per task in `isaaclab_arena_environments/spatial_memory_<task>_environment.py`
and the tests in `isaaclab_arena/tests/test_spatial_memory_*.py`.

## Common conventions

- **Embodiment and camera.** Tasks 1, 2 and 4 to 10 use a Franka on a counter (`franka_joint_pos` with the high-gain,
  gravity-compensated arm) and its wrist camera only. Task 3 uses the G1 humanoid with Arena's whole-body controller
  (`g1_wbc_joint`) and its head camera only. There is no external camera.
- **Scenes.** Fixtures and items are primitive boxes and cylinders in distinct colours, because the per-episode
  geometry (door angle, slot width, shelf height, ...) must change per environment and per reset, and because the
  oracle map and the visibility model need their exact geometry. Real assets can replace the items later.
- **Hidden phase.** Each task defines when its relevant geometry is claimed to be out of view (its "action phase").
  Visibility is computed every step by sampling points on the claimed surfaces and testing them against the camera
  frustum and against every box in the scene (and a box model of the Franka hand). It is checked against rendered
  instance segmentation for tasks 1, 4 and 7 in the camera tests; a one-off check of task 3's head camera agreed on
  99.3% of samples, with no analytically hidden part ever rendered.
- **Memory probe.** In the "turn away and return" tasks (2, 6, 8, 9) the instruction or signal is revealed only after
  the memorised geometry has been out of view for 3.5 s, longer than the default 3 s frame history.
- **Collisions.** Any contact of the robot or of a carried item with scene geometry the task does not need ends the
  episode as a failure; the first contact (bodies, time and position) is logged.

## The tasks

| # | Environment | Difficulty knob (default) |
| - | --- | --- |
| 1 | `spatial_memory_mug_past_door` | door opening angle in degrees (90) |
| 2 | `spatial_memory_fetch_behind` | number of mugs, 2 to 6 (5) |
| 3 | `spatial_memory_tray_carry` | box height in metres (0.25) |
| 4 | `spatial_memory_dishwasher_rack` | slot clearance in metres (0.03) |
| 5 | `spatial_memory_back_of_shelf` | gap width between the front items in metres (0.30) |
| 6 | `spatial_memory_high_shelf` | number of items already on the shelf, 1 to 4 (4) |
| 7 | `spatial_memory_machine_tending` | opening width in metres (0.44) |
| 8 | `spatial_memory_kitting` | number of bins, 2 to 6 (5) |
| 9 | `spatial_memory_restock_slot` | number of empty bins, 1 to 3 (1) |
| 10 | `spatial_memory_moved_cart` | re-observation window in seconds (1.0) |

**1. Put the mug away past the open door.** The Franka picks a mug from the counter and puts it on the indicated shelf
of a wall cabinet whose door stands open beside the opening. Once the tool is 5 cm inside the cabinet, the door and the
compartment's front edges are out of view while the hand, wrist and forearm pass right by them. Knob: the door angle
(smaller is harder).

**2. Fetch the mug from the shelf behind you.** N coloured mugs stand on an open two-tier shelf beside the robot. The
robot looks at it, turns to the counter, and only then is told which colour to fetch. It must reach back and grasp the
right mug on the first reach; every mug is out of view at the start of the reach. Re-observing the shelf for more
than 4 s before the grasp counts as "searched again" (a failure). Knob: the number of mugs.

**3. Carry the box past the chair and the counter corner.** A G1 humanoid squeezes a box off a table between its
hands, walks back, turns and carries it about 3.5 m to a destination table, past a chair and a counter corner that
stand on the straight line, then sets the box down and steps back. While the robot passes an obstacle, the obstacle is
out of the head camera's view. Knob: the box height.

**4. Load the dishwasher lower rack.** The stand-in for a dishwasher is a low rack behind a front lip, with the door
lying open in front and dividers forming three slots. The Franka lowers a cup or a plate into the target slot; once
the item is below the lip, the lip and the door are behind and below the hand, out of view. Knob: the slot clearance
(smaller is harder).

**5. Take the item from the back of the shelf.** A covered shelf holds a front row of items with one gap and a short
target behind a front item next to the gap. The Franka reaches in through the gap, moves sideways behind the front
row, grasps the target and brings it out the same way. Once the hand is past the front row, the front items are beside
and behind the wrist camera. Knob: the gap width (smaller is harder).

**6. Put the item on the high shelf.** A shelf at about the Franka's shoulder height has rails along its front and ends
and five slots, N of them holding items lower than the rails. The robot looks down at the shelf once, picks up a tall
item from a stand with the hand level (the wrist camera under the wrist), and once the shelf has been out of view for
3.5 s must lift the item over the rail into a free slot. From below the rails nothing on the shelf is visible. Knob:
the number of items already on the shelf.

**7. Machine tending through an opening.** The Franka carries a part through the framed opening of a box enclosure and
seats it in a cradle between clamp blocks. While the hand is inside, the opening's frame and the clamps are behind the
camera, around the forearm and wrist. Knob: the opening width (smaller is harder).

**8. Kitting from bins.** A rack of N bins beside the robot holds parts of distinct colours; a kit tray lies in front.
After one look at the rack the robot receives two requests, one at a time, each revealed only once the rack has been
out of view for 3.5 s, and must pick each requested part on the first reach and place it in the tray. Knob: the number
of bins.

**9. Restock into the empty slot.** Six bins beside the robot, all but K holding stock. After one look at the rack the
robot picks up a new item from the counter; the restock signal comes once the rack has been out of view for 3.5 s, and
the robot must put the item into a bin that was empty, a free region rather than an object. Knob: the number of empty
bins (fewer is harder).

**10. The cart someone moved.** The scene of task 1 (lower shelf) plus a tall cart standing either in the lane between
the mug and the cabinet or aside. While the robot grasps the mug, the cart is moved to the other spot with probability
0.8. After an initial look the wrist image is withheld (blank frames), except for a re-observation window that opens
when the cart is first back in view. The robot must avoid the cart at its new pose without a detour around the old
one: the carry path must stay within 10% (plus 4 cm) of the oracle route for the new pose. Knob: the length of the
re-observation window (shorter is harder).

## Launching a task

Every task is a registered environment, so the usual Arena runners work. Use the checkout's runtime (see the
`arena-runtime` skill); for example, with the Docker route, prefix the commands below with
`docker exec --user "$(id --user --name)" <container>` and run `isaaclab.sh -p` instead of `python`.

The environment flags select the observation variant and the difficulty:

| Flag | Meaning |
| --- | --- |
| `--observation_variant current` | the egocentric image and proprioception only (default) |
| `--observation_variant history_k` | as `current`, with the last `k` frames of every observation group stacked (`--history_k`, default 3 s of control steps) |
| `--observation_variant oracle_map` | adds an `oracle_map` group: a labelled point cloud of every fixture and object surface, hidden ones included, in the robot base frame |
| `--observation_variant mapped` | adds a `mapped` group filled by an online map provider; the default provider returns an empty map (the hook for an nvblox-next reconstruction, see `observations.OnlineMapProvider`) |
| `--difficulty <value>` | the task's difficulty knob (see the table above) |

Tasks that give an instruction or signal (2, 6, 8 and 9) also add a `task_obs` group with it in every variant
(for example the one-hot colour in tasks 2 and 8).
Add `--enable_cameras` to render the egocentric camera.

```bash
# Run a policy on task 1 with the oracle map, at a door angle of 80 degrees.
python isaaclab_arena/evaluation/policy_runner.py \
    --policy_type <dotted.path.to.YourPolicy> --num_episodes 10 --num_envs 10 --enable_cameras \
    spatial_memory_mug_past_door --observation_variant oracle_map --difficulty 80

# Task 6 with a 3 s frame history.
python isaaclab_arena/evaluation/policy_runner.py \
    --policy_type <dotted.path.to.YourPolicy> --num_episodes 10 --enable_cameras \
    spatial_memory_high_shelf --observation_variant history_k

# Task 3 (G1) with the online-map hook.
python isaaclab_arena/evaluation/policy_runner.py \
    --policy_type <dotted.path.to.YourPolicy> --num_episodes 4 --num_envs 4 --enable_cameras \
    spatial_memory_tray_carry --observation_variant mapped
```

## Running the oracle and the blind controller

`isaaclab_arena_environments/spatial_memory/policies.py` provides two privileged scripted controllers that work on
every task:

- `SpatialMemoryOraclePolicy` reads the true layout, hidden geometry included, and follows the task's waypoint plan
  (Franka: its own damped inverse kinematics with a null-space posture term; G1: velocity commands to the walking
  controller and arm postures). It shows that a policy with a perfect map can solve the task.
- `SpatialMemoryBlindPolicy` uses the same skills but ignores the hidden geometry: a straight-line reach or walk, a
  nominal-layout guess, or (task 10) the pose seen at the start. It shows that the task is not solved by luck.

```bash
python isaaclab_arena/evaluation/policy_runner.py \
    --policy_type isaaclab_arena_environments.spatial_memory.policies.SpatialMemoryOraclePolicy \
    --num_episodes 10 --num_envs 10 spatial_memory_mug_past_door

python isaaclab_arena/evaluation/policy_runner.py \
    --policy_type isaaclab_arena_environments.spatial_memory.policies.SpatialMemoryBlindPolicy \
    --num_episodes 10 --num_envs 10 spatial_memory_mug_past_door
```

The scripted controllers read privileged state; they ignore the observation variant.

## Metrics

| Metric | Definition |
| --- | --- |
| `success_rate` | fraction of episodes that ended with the task's success condition (which already excludes collisions and, where it applies, searching again) |
| `collision_rate` | fraction of episodes with at least one contact between the robot or a carried item and scene geometry the task does not need |
| `time_to_success_s` | mean episode time, in seconds, of the successful episodes |
| `searched_again_rate` | fraction of episodes in which the memorised scene was re-observed, after the instruction or signal and before the commitment (grasp or release), for longer than the task's budget (tasks 2, 6, 8 and 9) |

Each episode also writes a `spatial_memory` record to the episode JSONL: the sampled layout, the first collision
(bodies, time and position), whether the robot searched again and for how long, when the instruction came and when the
action phase started, which hidden targets were seen during the hidden phase and for how long, which targets were seen
before it, the largest displacement of static geometry while hidden, the tool (Franka) or pelvis (G1) path length,
and the success terms at the end.

## Tests and results

For every task, `isaaclab_arena/tests/test_spatial_memory_<task>.py` checks over 50 seeds (seed 1234):

1. the oracle succeeds on at least 90% of seeds at the default difficulty;
2. in every episode the oracle solved, no claimed hidden target was visible during the hidden phase, and the claimed
   targets were seen before it (so the hidden geometry is something to remember, not something never shown);
3. nothing static moved while hidden (except task 10, where the cart moves by design);
4. the blind controller fails on at least 70% of seeds;
5. resets are deterministic under a fixed seed;

plus a check that all four observation variants build and step. `test_spatial_memory_cameras.py` (with cameras)
checks the image observation variants, that the visibility model agrees with rendered instance segmentation (tasks 1,
4 and 7), and that task 10's withheld frames are blank exactly when withheld.

Success rates over 50 seeds (seed 1234) at the default difficulty:

| # | Task | Oracle | Blind |
| - | --- | --- | --- |
| 1 | Put the mug away past the open door | 0.96 | 0.02 |
| 2 | Fetch the mug from the shelf behind you | 0.98 | 0.24 |
| 3 | Carry the box past the chair and the counter corner (G1) | 0.96 | 0.00 |
| 4 | Load the dishwasher lower rack | 1.00 | 0.00 |
| 5 | Take the item from the back of the shelf | 0.98 | 0.00 |
| 6 | Put the item on the high shelf | 0.96 | 0.16 |
| 7 | Machine tending through an opening | 1.00 | 0.00 |
| 8 | Kitting from bins | 0.92 | 0.00 |
| 9 | Restock into the empty slot | 1.00 | 0.22 |
| 10 | The cart someone moved | 0.98 | 0.12 |

## Deviations from the brief

- **Embodiment.** The Franka runs joint-position control with Arena's high-gain arm instead of `franka_ik`, because
  the scripted controllers need to choose the elbow posture. A forward-level hand is only reachable well above the
  counter, so most insertions use a hand pitched 45 to 55 degrees down.
- **Assets.** Primitive stand-ins replace kitchen cabinets, dishwashers, mugs and chairs (see "Common conventions").
- **Task 2.** An open two-tier shelf replaces the cabinet (mugs are grasped from above), N ranges from 2 to 6 (the
  slots the Franka reaches reliably), and the re-observation budget is 4 s because the wrist camera necessarily sees
  the shelf during the final approach.
- **Task 3.** The squeezed box is at most about 0.3 m wide, and the G1 head camera's view is wide, so the box does not
  hide an obstacle the robot walks around while that obstacle is still ahead. The hidden claim therefore covers the
  pass (the obstacle level with or just behind the robot, where knees, arms and box could clip it), and the box height
  knob has little effect on visibility. Start pose randomisation is replaced by randomising the obstacles and the
  destination relative to the robot, because the pick posture is fixed relative to the pick table.
- **Task 4.** The dividers next to the item stay visible to a wrist camera, so only the lip and the door are claimed
  hidden.
- **Task 6.** The item is picked from a stand rather than the counter, so that it can be carried with the hand level
  (the Franka cannot hold a level hand at counter height). The existing items are lower than the shelf's rails: from
  below the rails nothing on the shelf is visible.
- **Task 7.** A seated part may tilt up to 25 degrees in the cradle.
- **Task 8.** The tray's visibility while the hand is over the bins is logged but not claimed, and the re-observation
  budget is 5 s.
- **Task 9.** The re-observation budget is 5 s.
- **Task 10.** The movable element is a cart (the brief allows a door, a chair or a cart): a door that swings between
  open and half-open barely changes the oracle's route, so a path-length tolerance would not test anything. Only the
  cart moves; there is no choice of which element moves.
