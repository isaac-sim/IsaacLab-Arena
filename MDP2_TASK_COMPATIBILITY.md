# Can MDP2 run Arena’s task rules?

**Discussion with Mustafa · 24 November 2026 · Branch `peterd/mdp2_task_exp`**

**Yes for the rigid pick-and-place cases we ran.** We took the actual Arena task definition, ran its rules on Mustafa’s MDP2 draft in Newton, then replayed the measured poses, velocities and contact forces through Arena’s original predicates and `ProgressTracker`. Across **20 episodes and 1,229 active world-steps**, both sides agreed on success, drop, timeout, termination step and partial score. We reproduced **two differences** in what a caller can learn about progress. Both have short examples below.

## Watch one episode

![Newton camera recording of the two-cube parallel task with live milestone and score captions](isaaclab_arena_examples/mdp2_tasks/evidence/parallel.gif)

The red and blue cubes start resting on a plate. A script lifts each cube by writing its pose. Newton simulates the fall, support force and settling. Starting on the destination is intentional: lying on the plate at step 1 must earn **zero placement credit** until that cube has been lifted. Configured placement requires four consecutive steps with sufficient upward contact force, the center over the plate, and low speed. The camera recording plays at one-sixth simulation speed and shows world 0.

| Step | What to watch | What both trackers earned |
|---:|---|---|
| 1 | Both cubes are at rest. Each acquires its own reference height. | 2 of 6 milestones |
| 15 | Red rises above its remembered height. Blue stays on the plate. | 3 of 6 |
| 35 | Red has landed and satisfied its four-step hold. | 4 of 6 |
| 42 | Blue rises. | 5 of 6 |
| 62 | Blue finishes its hold. Both branches are complete. | 6 of 6, success |

**Where did Arena’s state machine go?** Arena supplied the task definition and the thresholds. The prototype bound the cube poses, velocities and filtered contact forces to MDP2 quantities. `sequence(settle, lift, place)` gave Lab the order and the hold. `compose(red=…, blue=…)` let the two sequences advance independently in the same world. Lab advanced those milestones and a success termination read `progress.done`. The Arena tracker ran afterwards as an independent check on the saved physics measurements. [Arena task definition](isaaclab_arena/tasks/pick_and_place_task.py) · [Prototype task translation and MDP2 environment](isaaclab_arena_examples/mdp2_tasks/run_mdp2.py) · [Arena replay and assertions](isaaclab_arena_examples/mdp2_tasks/compare_arena.py)

## What the five experiments established

Each case ran four worlds. An active world-step is one world observed for one control step up to and including its terminal step.

| Case | Question | Observed result |
|---|---|---|
| **Single cube** · 209 world-steps | Do real predicates, a consecutive hold, drop and timeout agree? | Success on steps 35 and 62, drop on 22, timeout on 90. In the timeout episode Arena reported **2 of 4** placement steps held. Both trackers gave ⅔ credit. |
| **Red AND blue, either order** · 236 | Can two required subtasks progress independently? | World 0 placed red first, world 1 placed blue first. Both succeeded on step 62. World 3 finished only blue and timed out with ⅔ credit, including red’s settle milestone. Every milestone timestamp matched. |
| **(Red OR blue) AND yellow** · 284 | Does choosing one alternative preserve the same progress history? | Red won the alternative on step 35 in world 0. Lab also credited blue’s later lift on 42 and placement on 62. Arena froze that criterion after red won. Both waited for yellow and succeeded on 86. World 1 reproduced the difference with the colors reversed. |
| **Any two of three, weighted** · 236 | Do both systems give partial credit before the second placement? | Each cube earns 0.1 for settling, 0.3 for lifting and 0.6 for placement. The timeout world finished only blue, so both scored its best two branches as (1.0 + 0.1) / 2 = **55%**. The drop earned **25%**. |
| **Red THEN blue, both still placed at success** · 264 | Does order affect the verdict? What if an earlier condition later becomes false? | World 0 finished red on 35, started counting blue on 36, and succeeded on 62. World 1 lifted blue too early and timed out. On 70 we raised red again in the already-finished world. **Arena live success became false. Lab root `done` and `holding` both remained true.** The original success termination agreed. |

The ordered case uses one episode per world. Lab keeps finished worlds stepping while the other worlds finish. Step 70 probes the *current* success API after the success record, separately from the episode verdict. The five compressed traces contain every measured input, per-step score, milestone timestamp and episode record used by the comparison. [Evidence files](isaaclab_arena_examples/mdp2_tasks/evidence)

## What to take to Mustafa

1. **Success and history have different consumers.** The parallel OR case shows a reporting difference even when the termination and score agree. Can Arena request that unfinished alternatives stop advancing once a criterion is satisfied?
2. **We need a live composite success result.** The sequence case demonstrates the distinction between “has succeeded” and “meets all final conditions now.” Which public expression should Arena use? A sequence’s root `holding` follows its last child. [Lab progress update](https://github.com/StafaH/IsaacLab/blob/c4f22fdfc9983a23f1fe092d0d887c3ad05ffc11/source/isaaclab_experimental/isaaclab_experimental/mdp2/torch_backend.py#L287)
3. **Expose partial holds for failure analysis.** At timeout Arena can tell a user “2 of 4 steps.” Lab maintains a streak internally. Could the public progress API or episode record expose it? [Lab progress fields](https://github.com/StafaH/IsaacLab/blob/c4f22fdfc9983a23f1fe092d0d887c3ad05ffc11/source/isaaclab_experimental/isaaclab_experimental/mdp2/config.py#L347)

**Arena’s migration work.** An adapter can reuse criteria, thresholds and weights. It also has to bind each predicate to scene data and preserve its meaning. For example, “all objects are on their destinations **now**” is one combined predicate. “Each object has reached its destination **at some point**” can use parallel milestones. Unequal weights need translation too: with three equal criteria in A and one in B, completing only B earns 25%. Equal weights for the two Lab parent tasks would yield 50%. [Real sorting criterion](isaaclab_arena/tasks/sorting_task.py) · [Arena scoring](isaaclab_arena/progress_tracking/progress_tracker.py) · [Lab scoring](https://github.com/StafaH/IsaacLab/blob/c4f22fdfc9983a23f1fe092d0d887c3ad05ffc11/source/isaaclab_experimental/isaaclab_experimental/mdp2/torch_backend.py#L331)

## Reproduce it

**Fastest way to learn the difference.** From the Arena repo root, replay the saved choice episode. This runs Arena’s real task predicates against the recorded inputs, checks the outputs against Lab’s recorded results, and prints the four extra milestone timestamps. It uses the checkout’s native Arena environment.

```bash
demo=isaaclab_arena_examples/mdp2_tasks
PYTHONPATH="$PWD:$PWD/isaaclab_arena_embodiments" .venv/bin/python \
    "$demo/compare_arena.py" compare "$demo/evidence/choice-trace.json.gz"
```

**Run fresh physics on this workstation.** The launcher takes the pinned Lab source root and a Python environment with Lab’s Newton dependencies. It writes a fresh results directory under `/tmp`. Append `single`, `parallel`, `choice`, `sequence`, or `two_of_three` to select cases. The parallel run also records the camera.

```bash
bash isaaclab_arena_examples/mdp2_tasks/run_demo.sh \
    /tmp/arena-mdp2-lab.st5lotvq /home/peterd/workspaces/IsaacLab/.venv/bin/python
```

The first path is the prepared runtime for this run. If it expires, the same command accepts any source directory containing `source/` and `apps/` from Lab commit `c4f22fdfc9983a23f1fe092d0d887c3ad05ffc11`. Arena and Lab run in separate processes because this Arena revision’s eager G1 import uses an actuator constructor argument the draft Lab revision no longer accepts.

**Evidence boundary.** Arena baseline `51754b08c4`, Lab draft `c4f22fd`, Newton MJWarp on an RTX PRO 6000 Blackwell, four worlds at 60 control steps/s. The physical fixture uses centered rigid cubes and a fixed horizontal plate. Motion is scripted and the original Arena implementation scores a replay of the same measurements. Production USD assets, robot grasping, deformables, arbitrary predicate adapters, unequal-weight parent groups, and recycling individual worlds into new episodes still need validation.

**Verification.** The five trace replays check verdicts, terminal reasons, scores, and timestamps. The choice and ordered cases assert the specific differences above. Two existing Arena tests for live final conditions and ANY/K gating also passed.

**Version history.** Lab merged the experimental engine in [#9016](https://github.com/isaac-sim/IsaacLab/pull/9016). Xinjie’s [Arena gear prototype](https://github.com/isaac-sim/IsaacLab-Arena/pull/1706) used `40d2114` and implemented its own hold. The milestone API exercised here is the later commit on Mustafa’s [draft #9091](https://github.com/isaac-sim/IsaacLab/pull/9091).
