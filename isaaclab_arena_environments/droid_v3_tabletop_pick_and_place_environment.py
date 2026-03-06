# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DROID v3 tabletop pick-and-place environment using the droid_mimic_fixed embodiment.

Uses SortMultiObjectTask so that success termination only fires when ALL
pickable objects are placed in the bin, preventing premature env resets
during multi-object scripted pick-and-place.
"""

import argparse
import math
import random

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class DroidV3TabletopPickAndPlaceEnvironment(ExampleEnvironmentBase):
    """DROID v3 environment with flattened USD and mimic joint constraints for the Robotiq 2F-85 gripper."""

    name: str = 'droid_v3_tabletop_pick_and_place'

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        """Build and return the IsaacLab Arena environment."""
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.assets.object_set import RigidObjectSet
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.sorting_task import SortMultiObjectTask
        import isaaclab.sim as sim_utils
        from isaaclab_arena.relations.relations import IsAnchor
        from isaaclab_arena.utils.pose import Pose, PoseRange

        office_table = self.asset_registry.get_asset_by_name('office_table_background')()
        ground_plane = self.asset_registry.get_asset_by_name('ground_plane')()
        obj_1 = self.asset_registry.get_asset_by_name('tomato_soup_can')(scale=(0.7, 0.7, 0.6))
        obj_2 = self.asset_registry.get_asset_by_name('ketchup_bottle_hope_robolab')(scale=(0.7, 0.7, 0.6))
        obj_3 = self.asset_registry.get_asset_by_name('alphabet_soup_can_hope_robolab')(scale=(0.7, 0.7, 0.8))

        blue_sorting_bin = self.asset_registry.get_asset_by_name('blue_sorting_bin')(scale=(1.5, 0.8, 1.0))
        light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
        light = self.asset_registry.get_asset_by_name('light')(spawner_cfg=light_spawner_cfg)
        embodiment = self.asset_registry.get_asset_by_name('droid_differential_ik')(enable_cameras=args_cli.enable_cameras)

        office_table.set_initial_pose(Pose(position_xyz=(0.7, 0.5, 0.0), rotation_wxyz=(0.707, 0, 0, 0.707)))
        ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0)))
        embodiment.set_initial_pose(Pose(position_xyz=(0.1, 0.18, 0.75), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        blue_sorting_bin.set_initial_pose(
            Pose(
                position_xyz=(0.67, 0.4, 0.8),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        office_table.add_relation(IsAnchor())
        blue_sorting_bin.add_relation(IsAnchor())
        self._generate_object_layout(
            objects=[obj_1, obj_2, obj_3],
            table=office_table,
            bin_asset=blue_sorting_bin,
        )

        # Store for reuse by generate_target_positions()
        self._table = office_table
        self._bin = blue_sorting_bin
        self._pick_objects = [obj_1, obj_2, obj_3]

        # Shared destination for all objects
        destination_location = ObjectReference(
            name='destination_location',
            prim_path='{ENV_REGEX_NS}/blue_sorting_bin/Geometry/sm_bin_20x25x05cm_a01_01',
            parent_asset=blue_sorting_bin,
            object_type=ObjectType.RIGID,
        )

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        assets = [office_table, ground_plane, obj_1, obj_2, obj_3, blue_sorting_bin, light]


        scene = Scene(assets=assets)

        # All pickable objects share the same destination (the bin).
        # SortMultiObjectTask creates a contact sensor per object and only
        # fires the success termination when ALL objects are on the destination.
        pick_up_objects = [obj_1, obj_2, obj_3]
        destinations = [destination_location] * len(pick_up_objects)

        task = SortMultiObjectTask(
            pick_up_object_list=pick_up_objects,
            destination_location_list=destinations,
            background_scene=office_table,
            episode_length_s=600.0,
        )

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    def generate_target_positions(self) -> dict[str, tuple[float, float, float]]:
        """Generate a new random layout and return resolved world-frame positions.

        Reuses the stored objects from get_env(), clears their relations,
        calls _generate_object_layout with fresh randoms, resolves via
        ObjectPlacer, then restores original relations.

        Also captures init and target relations for symbolic planning
        (see plan_pick_order).
        """
        from isaaclab_arena.relations.object_placer import ObjectPlacer
        from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
        from isaaclab_arena.relations.relations import IsAnchor

        all_objects = [self._table, self._bin] + self._pick_objects

        # Save original (init) relations, then clear
        saved = {id(obj): obj.relations[:] for obj in all_objects}
        self._init_relations = {obj.name: saved[id(obj)][:] for obj in self._pick_objects}

        for obj in all_objects:
            obj.relations = []

        # Re-add anchors, generate fresh random layout
        self._table.add_relation(IsAnchor())
        self._bin.add_relation(IsAnchor())
        self._generate_object_layout(
            objects=self._pick_objects, table=self._table, bin_asset=self._bin,
        )

        # Capture target relations before restoring
        self._target_relations = {obj.name: obj.relations[:] for obj in self._pick_objects}

        # Resolve without mutating initial_pose
        params = ObjectPlacerParams(apply_positions_to_objects=False)
        result = ObjectPlacer(params=params).place(objects=all_objects)

        # Restore original relations
        for obj in all_objects:
            obj.relations = saved[id(obj)]

        positions = {obj.name: result.positions[obj] for obj in self._pick_objects}
        self._target_positions = positions
        return positions

    def plan_pick_order(self, verbose: bool = False) -> list[str]:
        """A* symbolic planning: find minimum-cost pick order respecting relation constraints.

        Builds a dependency graph from ``On`` and ``NextTo`` relations between
        pickable objects:

        - **Target relations** (placement correctness):
          - ``On(A)``: A must be placed before current object (bottom first).
          - ``NextTo(A)``: A must be placed before current object (reference first).
        - **Init relations** (pick safety):
          - ``On(A)``: current object (on top) must be picked before A.
          - ``NextTo(A)``: current object must be picked before A (avoid disturbance).
        - Relations referencing static objects (table, bin) create no constraints.
        - Target constraints take priority over init constraints when they conflict.

        Uses A* search over possible orderings to minimize total arm travel
        distance (init_pos → target_pos transitions) while respecting the
        dependency constraints. Init positions come from each object's resolved
        ``initial_pose``; target positions from ``generate_target_positions()``.

        Must be called after ``generate_target_positions()``.
        """
        import heapq
        from isaaclab_arena.relations.relations import NextTo, On

        _log = print if verbose else (lambda *a, **k: None)

        pick_obj_ids = {id(obj) for obj in self._pick_objects}
        pick_names = [obj.name for obj in self._pick_objects]

        _log(f"\n{'=' * 70}")
        _log("  SYMBOLIC A* PLANNER")
        _log(f"{'=' * 70}")
        _log(f"  Objects: {pick_names}")

        # must_precede[A] = set of names that must be placed before A
        must_precede: dict[str, set[str]] = {name: set() for name in pick_names}

        # Target: On(A) or NextTo(A) among pick objects → A before current
        _log(f"\n  --- Target relation constraints (placement correctness) ---")
        for obj in self._pick_objects:
            for rel in self._target_relations[obj.name]:
                if isinstance(rel, (NextTo, On)) and id(rel.parent) in pick_obj_ids:
                    rel_type = type(rel).__name__
                    _log(f"    {obj.name} {rel_type}({rel.parent.name})"
                         f" -> place {rel.parent.name} BEFORE {obj.name}")
                    must_precede[obj.name].add(rel.parent.name)

        # Init: On(A) or NextTo(A) among pick objects → current before A
        # (top/adjacent object picked first to avoid disturbance)
        # Skip if it conflicts with a target constraint
        _log(f"\n  --- Init relation constraints (pick safety) ---")
        for obj in self._pick_objects:
            for rel in self._init_relations[obj.name]:
                if isinstance(rel, (NextTo, On)) and id(rel.parent) in pick_obj_ids:
                    parent_name = rel.parent.name
                    rel_type = type(rel).__name__
                    # obj is on/next-to parent → pick obj before parent
                    # But skip if target already says obj must come after parent
                    # (i.e. parent_name is in must_precede[obj.name])
                    if parent_name not in must_precede[obj.name]:
                        _log(f"    {obj.name} {rel_type}({parent_name})"
                             f" -> pick {obj.name} BEFORE {parent_name}")
                        must_precede[parent_name].add(obj.name)
                    else:
                        _log(f"    {obj.name} {rel_type}({parent_name})"
                             f" -> SKIPPED (conflicts with target constraint)")

        _log(f"\n  --- Dependency graph ---")
        for name in pick_names:
            deps = must_precede[name]
            if deps:
                _log(f"    {name} must come after: {deps}")
            else:
                _log(f"    {name} (no dependencies, can go first)")

        # Get positions for A* cost computation
        init_pos: dict[str, tuple[float, float, float]] = {}
        for obj in self._pick_objects:
            pose = obj.get_initial_pose()
            if pose is not None:
                init_pos[obj.name] = pose.position_xyz

        target_pos = self._target_positions

        _log(f"\n  --- Object positions ---")
        for name in pick_names:
            ip = init_pos.get(name)
            tp = target_pos.get(name)
            ip_str = f"({ip[0]:.3f}, {ip[1]:.3f}, {ip[2]:.3f})" if ip else "N/A"
            tp_str = f"({tp[0]:.3f}, {tp[1]:.3f}, {tp[2]:.3f})" if tp else "N/A"
            _log(f"    {name}: init={ip_str}  target={tp_str}")

        def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
            return sum((ai - bi) ** 2 for ai, bi in zip(a, b)) ** 0.5

        # Heuristic: sum of pick-to-place distances for remaining objects (admissible)
        def heuristic(placed: frozenset[str]) -> float:
            return sum(
                _dist(init_pos[n], target_pos[n])
                for n in pick_names
                if n not in placed and n in init_pos and n in target_pos
            )

        def is_valid(name: str, placed: frozenset[str]) -> bool:
            return must_precede[name].issubset(placed)

        # A* search: state = frozenset of placed objects
        # (f_cost, g_cost, counter, placed, order)
        counter = 0
        nodes_expanded = 0
        start: frozenset[str] = frozenset()
        goal = frozenset(pick_names)
        heap = [(heuristic(start), 0.0, counter, start, [])]
        visited: set[frozenset[str]] = set()

        _log(f"\n  --- A* search ---")

        while heap:
            f, g, _, placed, order = heapq.heappop(heap)

            if placed == goal:
                _log(f"\n  --- Result ---")
                _log(f"    Nodes expanded : {nodes_expanded}")
                _log(f"    Total travel   : {g:.3f} m")
                _log(f"    Pick order     : {' -> '.join(order)}")
                _log(f"{'=' * 70}\n")
                print(f"[SYMBOLIC A*] Pick order: {order} (total cost: {g:.3f}m)")
                return order

            if placed in visited:
                continue
            visited.add(placed)
            nodes_expanded += 1

            valid_next = [n for n in pick_names if n not in placed and is_valid(n, placed)]
            _log(f"    [expand #{nodes_expanded}] placed={list(order)}  candidates={valid_next}")

            # Reference position: where arm is after placing the last object
            ref = target_pos.get(order[-1]) if order else None

            for name in valid_next:
                # Cost: travel ref → pick(name) → place(name)
                cost = 0.0
                if ref is not None and name in init_pos:
                    cost += _dist(ref, init_pos[name])
                if name in init_pos and name in target_pos:
                    cost += _dist(init_pos[name], target_pos[name])

                new_placed = placed | {name}
                new_g = g + cost
                new_h = heuristic(new_placed)
                counter += 1
                _log(f"      -> {name}: g={new_g:.3f} h={new_h:.3f} f={new_g + new_h:.3f}")
                heapq.heappush(
                    heap,
                    (new_g + new_h, new_g, counter, new_placed, order + [name]),
                )

        # Fallback if no valid ordering found (e.g. cyclic deps)
        _log(f"\n  --- Result ---")
        _log(f"    WARNING: no valid ordering found (cyclic dependencies?)")
        _log(f"    Falling back to alphabetical order")
        _log(f"{'=' * 70}\n")
        print("[SYMBOLIC A*] Warning: no valid ordering found, returning alphabetical")
        return sorted(pick_names)

    @staticmethod
    def _generate_object_layout(objects, table, bin_asset):
        """Randomly initialize object positions on the table relative to the bin.

        Topology (kept from v2):
          - obj[0]: On table, NextTo bin (-Y side), lying on side (roll=90deg)
          - obj[1]: On table, NextTo obj[0] (-X side), random yaw
          - obj[2]: On table, NextTo bin (+Y side), upright
        Distances are randomized within safe IK ranges.
        """
        from isaaclab_arena.relations.relations import (
            AtPosition,
            IsAnchor,
            NextTo,
            On,
            RandomAroundSolution,
            RotateAroundSolution,
            Side,
        )
        all_sides = [Side.POSITIVE_X, Side.NEGATIVE_X, Side.POSITIVE_Y, Side.NEGATIVE_Y]
        obj_1, obj_2, obj_3 = objects[0], objects[1], objects[2]

        # obj_1: next to bin on -Y side, lying on its side
        obj_1.add_relation(On(table))
        obj_1.add_relation(NextTo(
            bin_asset, side=Side.NEGATIVE_Y,
            distance_m=random.uniform(0.15, 0.25),  # Do not go over 0.25 as IK may fail
        ))
        obj_1.add_relation(RotateAroundSolution(roll_rad=math.pi / 2, yaw_rad=0))

        # obj_2: next to obj_1 on -X side, random yaw
        obj_2.add_relation(On(table))
        obj_2.add_relation(NextTo(
            obj_1, side=Side.NEGATIVE_X,
            distance_m=random.uniform(0.05, 0.15),
        ))
        obj_2.add_relation(RotateAroundSolution(yaw_rad=math.radians(random.randint(0, 360))))

        # obj_3: next to bin on +Y side, upright
        obj_3.add_relation(On(table))
        obj_3.add_relation(NextTo(
            bin_asset, side=random.choice(all_sides),
            distance_m=random.uniform(0.05, 0.15),
        ))

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Add CLI arguments specific to this environment."""
        parser.add_argument('--object', type=str, default='tomato_soup_can')
        parser.add_argument('--object_set', nargs='+', type=str, default=None)
        parser.add_argument('--teleop_device', type=str, default=None)
