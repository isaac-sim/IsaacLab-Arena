"""Build Isaac Lab terminations from a GoalSpec.

Previously tried to translate every LLM success predicate into a custom
termination function. That doesn't scale — the LLM vocabulary grows.

Now we rely on `predicate_to_xyz.resolve_goal_xyz()` to convert every
`goal_relations` entry into a concrete target xyz, and use one generic
termination (`all_objects_at_goal`) to check "every target at its xyz AND
at rest". Time limit is Isaac Lab's built-in `mdp.time_out`. Sibling
module `predicate_to_relation.py` (complementary, not duplicative) handles
scene-setup placement.
"""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
import isaaclab.envs.mdp as mdp

from isaaclab_arena.tasks.terminations import all_objects_at_goal


def build_terminations_from_goal_xyz(
    goal_xyz: dict[str, tuple[float, float, float]],
    pos_tolerance: float = 0.05,
    velocity_threshold: float = 0.1,
) -> dict[str, TerminationTermCfg]:
    """Produce terminations for a GoalSpec given its resolved goal xyz.

    Returns a dict of {term_name: TerminationTermCfg} with:
    - `time_out` — Isaac Lab's MDP time_out (episode length).
    - `success`  — all goal-constrained objects at xyz AND at rest. Absent
                   if goal_xyz is empty (no goal_relations → no success check).
    """
    terms: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    }
    if not goal_xyz:
        return terms

    names = list(goal_xyz.keys())
    terms["success"] = TerminationTermCfg(
        func=all_objects_at_goal,
        params={
            "object_cfgs": [SceneEntityCfg(n) for n in names],
            "goal_xyzs": [goal_xyz[n] for n in names],
            "pos_tolerance": pos_tolerance,
            "velocity_threshold": velocity_threshold,
        },
    )
    return terms
