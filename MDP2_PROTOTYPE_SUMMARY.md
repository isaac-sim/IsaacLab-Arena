# Arena on MDP2: prototype results

**Mustafa’s MDP2 draft ran Arena’s rigid pick-and-place rules in sequence, in parallel, and with any two of three objects required.** Across five cases, 20 episodes, and 1,229 active world-steps, Arena and MDP2 agreed on partial score, outcome (success, drop, or timeout), and the exact step each episode ended. Two differences remain in the progress information Arena exposes to users.

## Where Arena’s state machine went

Arena supplied the task rules and thresholds. The prototype translated them into MDP2 quantities that read cube poses, velocities, and contact forces. Lab’s `sequence` tracked each cube through settling, lifting, and placement. Its `compose` let multiple cubes advance independently. A success termination read the overall `progress.done` flag. [Prototype implementation](isaaclab_arena_examples/mdp2_tasks/run_mdp2.py)

![Parallel pick-and-place in Newton with milestone and score captions](isaaclab_arena_examples/mdp2_tasks/evidence/parallel.gif)

Both cubes start on the destination plate. At **step 1**, each earns credit for settling. Red is lifted and finishes its four-step placement hold on **step 35**. Blue has been sitting on the plate the whole time and still has only its settle milestone. On **step 62**, blue finishes its own lift and placement, and the task succeeds. Reversing the order in another world produces the same verdict.

## What to take to Mustafa

| Question | Observed answer |
|---|---|
| Can MDP2 preserve weighted partial credit? | In the two-of-three case, only blue finished before timeout. Its score was 1.0 and the next-best cube had earned 0.1 for settling. Both trackers averaged the best two and awarded **55%**. |
| Do both trackers credit the same milestones? | For **(red OR blue) AND yellow**, red satisfied the alternative on step 35. Arena froze that criterion. MDP2 also credited blue’s later lift on 42 and placement on 62. Both still waited for yellow and succeeded on 86. **Can Arena freeze unfinished alternatives when the group succeeds?** |
| Can Arena ask whether the entire task still holds? | For **red THEN blue**, both recorded success on step 62. On step 70 we lifted red again while blue stayed placed. Arena’s live success became false. MDP2’s root `done` and `holding` both stayed true. **Which public expression gives Arena the current result across all required final conditions?** |

**Migration verdict.** These runs show that Lab can own stage progression and termination for the tested tasks. Preserving Arena’s reporting also requires agreement on which milestones get credit and how to read live composite success.

The experiment used scripted lifts, rigid cubes, and a fixed plate. Newton simulated the falls, contacts, and settling. Arena’s original predicates and tracker then scored the recorded physics measurements. Production robot grasping and arbitrary Arena predicates remain unverified. All five saved trace replays and two relevant Arena regression tests passed. [Full walkthrough and rerun commands](MDP2_TASK_COMPATIBILITY.md) · [Saved traces](isaaclab_arena_examples/mdp2_tasks/evidence)
