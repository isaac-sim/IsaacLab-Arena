# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Export Arena's task rules or check its original predicates against a Newton trace."""

import argparse
import gzip
import inspect
import json
import torch
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder, objects_settled
from isaaclab_arena.tasks.predicates.spatial import object_is_above_height, object_on_destination
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


def arena_definition(case):
    # The task only uses asset names, types and the drop plane to build its termination config.
    plate = SimpleNamespace(name="plate", object_type=ObjectType.RIGID)
    background = SimpleNamespace(object_min_z=-0.1)
    names = {
        "single": ["red"],
        "parallel": ["red", "blue"],
        "choice": ["red", "blue", "yellow"],
        "two_of_three": ["red", "blue", "yellow"],
        "sequence": ["red", "blue"],
    }[case]
    tasks = [
        PickAndPlaceTask(
            pick_up_object=SimpleNamespace(name=name, object_type=ObjectType.RIGID),
            destination_location=plate,
            background_scene=background,
            placement_consecutive_steps=4,
            episode_length_s=1.5,
        )
        for name in names
    ]
    if len(tasks) == 1:
        return tasks[0].get_termination_cfg()
    cfg = CompositeTaskBase(
        tasks,
        episode_length_s=1.5,
        subtasks_are_sequential=case == "sequence",
        desired_subtask_success_state=[True, True] if case == "sequence" else None,
    ).get_termination_cfg()
    if case == "choice":
        alternatives = CompletionCriteria(
            name="either",
            predicate_sequences={"red": cfg.success[0].predicate_sequence, "blue": cfg.success[1].predicate_sequence},
            logical="any",
        )
        cfg = replace(cfg, success=[alternatives, replace(cfg.success[2], parent_subtask_idx=None)])
    if case == "two_of_three":
        sequences = {}
        for name, criterion in zip(names, cfg.success, strict=True):
            sequences[name] = list(zip(criterion.predicate_sequence, (0.1, 0.3, 0.6), strict=True))
        cfg = replace(
            cfg, success=[CompletionCriteria(name="best_two", predicate_sequences=sequences, logical="choose", K=2)]
        )
    return cfg


def object_rules(chain, failure):
    """Translate the supported rigid pick-and-place definition, including its actual weights and thresholds."""
    (rest, _), (lift, _), (hold, _) = chain
    assert (rest.func, lift.func, hold.predicate.func) == (
        objects_settled,
        object_is_above_height,
        object_on_destination,
    )
    name = lift.keywords["object_name"]
    assert rest.keywords["object_names"] == [name] and lift.keywords["use_settled_state"]
    rest_args = inspect.signature(rest.func).bind_partial(**rest.keywords)
    rest_args.apply_defaults()
    lift_args = inspect.signature(lift.func).bind_partial(**lift.keywords)
    lift_args.apply_defaults()
    params = hold.predicate.keywords
    assert params["object_cfg"].name == name and params["destination_cfg"].name == "plate"
    return {
        "weights": [weight for _, weight in chain],
        "hold": hold.required_steps,
        "rest": {
            "linear_limit": rest_args.arguments["lin_vel_threshold"],
            "angular_limit": rest_args.arguments["ang_vel_threshold"],
        },
        "lift": {"distance": lift_args.arguments["distance"]},
        "place": {key: params[key] for key in ("force_threshold", "velocity_threshold", "support_cone_half_angle_rad")},
        "drop_height": failure.params["minimum_height"],
    }


def export_rules(cfg, case):
    objects, groups = {}, []
    failures = {term.params["asset_cfg"].name: term for term in cfg.failures.values()}
    for criteria in cfg.success:
        assert criteria.score == 1.0, "This fixture expects equally weighted criteria"
        children = []
        for sequence in criteria.sequence_names:
            chain = criteria.get_sequence(sequence)
            name = chain[0][0].keywords["object_names"][0]
            objects[name] = object_rules(chain, failures[name])
            children.append(name)
        required = {"all": len(children), "any": 1, "choose": criteria.K}[criteria.logical.value]
        groups.append({"children": children, "require": required})
    return {
        "case": case,
        "objects": objects,
        "groups": groups,
        "timeout": cfg.timeout_s,
        "sequential": cfg.subtasks_are_sequential,
        "final": cfg.desired_subtask_success_state,
    }


class RecordedScene(dict):
    """The fixture contains only centered rigid cuboids."""

    deformable_objects = {}


class RecordedWorld:
    """Supply Arena world queries from recorded Newton poses, twists and the known plate geometry."""

    def __init__(self, scene, plate_size):
        self.scene = scene
        extents = torch.tensor([plate_size]) / 2
        self.bounds = AxisAlignedBoundingBox(min_point=-extents, max_point=extents)

    def load(self, step):
        self.poses = {"plate": torch.tensor(step["plate_pose"])}
        self.velocities = {}
        for name, row in step["objects"].items():
            self.poses[name] = torch.tensor(row["pose"])
            self.velocities[name] = torch.tensor(row["velocity"])
            data = SimpleNamespace(root_pos_w=SimpleNamespace(torch=self.poses[name][:, :3]))
            self.scene[name] = SimpleNamespace(data=data)
            forces = torch.tensor(row["force"])[:, None, None, :]
            sensor = SimpleNamespace(force_matrix_w=SimpleNamespace(torch=forces))
            self.scene[f"contact_sensor_{name}"] = SimpleNamespace(data=sensor)

    def get_pose_w(self, name):
        return self.poses[name]

    def get_position_w(self, name):
        return self.poses[name][:, :3]

    get_centroid_w = get_position_w

    def get_root_linear_velocity_w(self, name):
        return self.velocities[name][:, :3]

    get_mean_linear_velocity_w = get_root_linear_velocity_w

    def get_root_angular_velocity_w(self, name):
        return self.velocities[name][:, 3:]

    def get_aabb_in_local_frame(self, name):
        assert name == "plate"
        return self.bounds


def compare(trace):
    names = tuple(trace["rules"]["objects"])
    case = trace["rules"]["case"]
    cfg = arena_definition(case)
    assert trace["rules"] == export_rules(cfg, case)
    worlds = len(trace["steps"][0]["score"])
    scene = RecordedScene()
    world = RecordedWorld(scene, trace["plate_size"])
    env = SimpleNamespace(
        scene=scene,
        arena_world=world,
        num_envs=worlds,
        device="cpu",
        object_initial_rest_pose_recorder=ObjectInitialRestPoseRecorder(worlds, "cpu"),
    )
    tracker = ProgressTracker(
        cfg.success,
        worlds,
        "cpu",
        env=env,
        subtasks_are_sequential=cfg.subtasks_are_sequential,
        desired_subtask_success_state=cfg.desired_subtask_success_state,
    )
    ended = {}
    stage_names = ("settle", "lift", "place")
    milestones = [{} for _ in range(worlds)]
    comparisons = 0
    differences = {}
    live_difference = None
    object_by_sequence = {
        (criteria.name, sequence): criteria.get_sequence(sequence)[0][0].keywords["object_names"][0]
        for criteria in cfg.success
        for sequence in criteria.sequence_names
    }
    for row in trace["steps"]:
        step = row["step"]
        world.load(row)
        tracker.step(env, step_index=torch.full((worlds,), step))
        states = tracker.get_state()
        events = tracker.get_events()
        failures = {key: term.func(env, **term.params) for key, term in cfg.failures.items()}
        for idx in range(worlds):
            if idx in ended:
                if case == "sequence" and idx == 0 and live_difference is None and not states[idx].all_complete:
                    live_difference = step
                    assert row["success"][idx] and row["root_holding"][idx], (step, idx)
                    print(
                        f"LIVE SUCCESS DIFFERENCE: step {step} world 0, Arena false, MDP2 done true, root holding true"
                    )
                continue
            comparisons += 1
            ctx = f"step {step}, world {idx}"
            assert states[idx].all_complete == row["success"][idx], ctx
            assert abs(states[idx].overall_score - row["score"][idx]) < 1e-6, ctx
            for event in events[idx]:
                name = object_by_sequence[event.criteria_name, event.sequence_name]
                milestones[idx][(name, stage_names[event.predicate_index])] = event.step
            for name in names:
                for stage in stage_names:
                    actual = row["objects"][name]["done_at"][stage][idx]
                    expected = milestones[idx].get((name, stage), -1)
                    if actual != expected:
                        # Previously completed ANY group: record the losing branch's first extra milestone.
                        assert case == "choice" and expected == -1 and actual > 0, (ctx, name, stage, actual)
                        differences.setdefault((idx, name, stage), actual)
            drop = any(mask[idx].item() for mask in failures.values())
            success = states[idx].all_complete
            timed_out = step >= round(cfg.timeout_s / trace["dt"]) and not (success or drop)
            assert (success or drop, timed_out) == (row["terminated"][idx], row["timeout"][idx]), ctx
            if success or drop or timed_out:
                outcome = "success" if success else "drop" if drop else "timeout"
                ended[idx] = outcome
                record = trace["episodes"]
                i = record["world"].index(idx)
                assert record["length"][i] == step and bool(record["termination/success"][i]) == success
                for name, mask in zip(names, failures.values(), strict=True):
                    assert bool(record[f"termination/{name}_dropped"][i]) == mask[idx].item(), ctx
                history = " ".join(f"{name}.{stage}={t}" for (name, stage), t in milestones[idx].items())
                print(f"world {idx}  {outcome:7}  step {step:2}  score {states[idx].overall_score:.3f}  {history}")
                if case == "single" and timed_out:
                    (hold,) = tracker.get_consecutive_step_summary(idx)
                    count, required = hold["consecutive_steps"], hold["required_steps"]
                    assert hold["status"] == "active" and 0 < count < required
                    print(f"Arena timeout diagnostic: placement held for {count}/{required} steps.")
    assert ended == {0: "success", 1: "timeout" if case == "sequence" else "success", 2: "drop", 3: "timeout"}, ended
    if case == "sequence":
        assert live_difference == 70, live_difference
    if case == "parallel":
        assert milestones[0][("red", "place")] < milestones[0][("blue", "place")]
        assert milestones[1][("blue", "place")] < milestones[1][("red", "place")]
    print(f"MATCH: verdict and score on {comparisons} active world-steps.")
    if case == "choice":
        assert set(differences) == {(0, "blue", "lift"), (0, "blue", "place"), (1, "red", "lift"), (1, "red", "place")}
        for (idx, name, stage), step in differences.items():
            print(f"HISTORY DIFFERENCE: world {idx} {name}.{stage}: Arena unreached, MDP2 step {step}")
    else:
        print("MATCH: every milestone timestamp.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("case", choices=("single", "parallel", "choice", "sequence", "two_of_three"))
    export.add_argument("path", type=Path)
    check = commands.add_parser("compare")
    check.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        with args.path.open("x") as stream:
            json.dump(export_rules(arena_definition(args.case), args.case), stream, indent=2)
    else:
        opener = gzip.open if args.path.suffix == ".gz" else open
        with opener(args.path, "rt") as stream:
            compare(json.load(stream))


if __name__ == "__main__":
    main()
