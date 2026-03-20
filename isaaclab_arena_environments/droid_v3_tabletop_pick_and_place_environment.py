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
        obj_4 = self.asset_registry.get_asset_by_name('bowl_ycb_robolab')()
        obj_5 = self.asset_registry.get_asset_by_name('red_container')(scale=(0.5, 0.5, 0.5))

        blue_sorting_bin = self.asset_registry.get_asset_by_name('blue_sorting_bin')(scale=(1.5, 0.8, 1.0))
        light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
        light = self.asset_registry.get_asset_by_name('light')(spawner_cfg=light_spawner_cfg)
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        office_table.set_initial_pose(Pose(position_xyz=(0.7, 0.5, 0.0), rotation_wxyz=(0.707, 0, 0, 0.707)))
        ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0)))
        embodiment.set_initial_pose(Pose(position_xyz=(0.1, 0.18, 0.75), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        blue_sorting_bin.set_initial_pose(
            Pose(
                position_xyz=(0.67, 0.4, 0.8),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        # Static objects need initial poses since they are anchors
        obj_4.set_initial_pose(Pose(position_xyz=(0.67, 0.6, 0.8), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        obj_5.set_initial_pose(Pose(position_xyz=(0.67, -0.3, 0.8), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

        office_table.add_relation(IsAnchor())
        blue_sorting_bin.add_relation(IsAnchor())
        self._place_static_objects(
            static_objects=[obj_4, obj_5],
            table=office_table,
            bin_asset=blue_sorting_bin,
        )
        self._generate_object_layout(
            objects=[obj_1, obj_2, obj_3],
            table=office_table,
            bin_asset=blue_sorting_bin,
        )

        # Store for reuse by generate_target_positions()
        self._table = office_table
        self._bin = blue_sorting_bin
        self._pick_objects = [obj_1, obj_2, obj_3]
        self._static_objects = [obj_4, obj_5]

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

        assets = [office_table, ground_plane, obj_1, obj_2, obj_3, obj_4, obj_5, blue_sorting_bin, light]


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

    def randomize_initial_positions(self, env=None) -> None:
        """Re-solve the init layout and update pick objects' positions.

        Args:
            env: If provided, directly writes new poses to the sim via
                 write_root_pose_to_sim. Call this after env.reset().
        """
        from isaaclab_arena.relations.object_placer import ObjectPlacer
        from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
        from isaaclab_arena.relations.relations import IsAnchor

        all_objects = [self._table, self._bin] + self._static_objects + self._pick_objects

        saved = {id(obj): obj.relations[:] for obj in all_objects}
        for obj in all_objects:
            obj.relations = []

        self._table.add_relation(IsAnchor())
        self._bin.add_relation(IsAnchor())
        for obj in self._static_objects:
            obj.add_relation(IsAnchor())
        self._generate_object_layout(
            objects=self._pick_objects, table=self._table, bin_asset=self._bin,
        )

        # Solve without applying (we'll write to sim directly)
        params = ObjectPlacerParams(apply_positions_to_objects=False)
        result = ObjectPlacer(params=params).place(objects=all_objects)

        # Write new poses directly to the sim via PhysX view
        if env is not None:
            import torch as _torch
            from isaaclab_arena.utils.pose import Pose
            for obj in self._pick_objects:
                pos = result.positions[obj]
                pose = Pose(position_xyz=pos, rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
                asset = env.scene[obj.name]
                # Build 7-dim tensor: [x, y, z, qx, qy, qz, qw] for PhysX
                pose7 = pose.to_tensor(device=env.device)  # [x,y,z, qw,qx,qy,qz]
                # Convert wxyz -> xyzw for PhysX
                pos_xyz = pose7[:3] + env.scene.env_origins[0]
                quat_wxyz = pose7[3:]
                quat_xyzw = _torch.tensor(
                    [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
                    device=env.device,
                )
                physx_pose = _torch.cat([pos_xyz, quat_xyzw]).unsqueeze(0)
                physx_vel = _torch.zeros(1, 6, device=env.device)
                indices = _torch.tensor([0], dtype=_torch.int32, device=env.device)
                asset.root_physx_view.set_transforms(physx_pose, indices=indices)
                asset.root_physx_view.set_velocities(physx_vel, indices=indices)
                print(f"[INIT] {obj.name} -> {pos}")
            # Step sim so poses take effect (must be in inference_mode for articulation tensors)
            with _torch.inference_mode():
                env.sim.step(render=False)
                env.scene.update(dt=env.step_dt)

        # Restore original relations
        for obj in all_objects:
            obj.relations = saved[id(obj)]

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

        all_objects = [self._table, self._bin] + self._static_objects + self._pick_objects

        # Save original (init) relations, then clear
        saved = {id(obj): obj.relations[:] for obj in all_objects}
        self._init_relations = {obj.name: saved[id(obj)][:] for obj in self._pick_objects}

        for obj in all_objects:
            obj.relations = []

        # Re-add anchors, generate fresh random layout
        self._table.add_relation(IsAnchor())
        self._bin.add_relation(IsAnchor())
        for obj in self._static_objects:
            obj.add_relation(IsAnchor())
        self._generate_object_layout(
            objects=self._pick_objects, table=self._table, bin_asset=self._bin,
            static_objects=self._static_objects,
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

    def plan_pick_order(
        self, verbose: bool = False, ik_cost_fn=None,
    ) -> tuple[list[str], float]:
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
        distance (init_pos -> target_pos transitions) while respecting the
        dependency constraints. Must be called after ``generate_target_positions()``.

        Args:
            verbose: Print detailed search trace.
            ik_cost_fn: Optional sequential IK callback with signature
                ``(name, init_pos, target_pos, prev_joint_config) ->
                (penalty, new_joint_config)``.

        Returns:
            ``(pick_order, total_ik_penalty)`` where ``total_ik_penalty`` is
            the cumulative IK infeasibility cost in the chosen ordering
            (0.0 means the full sequence is IK-feasible).
        """
        import heapq
        from isaaclab_arena.relations.relations import NextTo, On

        _log = print if verbose else (lambda *a, **k: None)

        pick_obj_ids = {id(obj) for obj in self._pick_objects}
        pick_names = [obj.name for obj in self._pick_objects]

        sep = '=' * 70
        _log(f"\n{sep}")
        _log('SYMBOLIC A* PLANNER')
        _log(sep)
        _log(f"Objects: {pick_names}")

        # must_precede[A] = set of names that must be placed before A
        must_precede: dict[str, set[str]] = {name: set() for name in pick_names}

        _log(f"\n--- Target relation constraints (placement correctness) ---")
        for obj in self._pick_objects:
            for rel in self._target_relations[obj.name]:
                if isinstance(rel, (NextTo, On)) and id(rel.parent) in pick_obj_ids:
                    rel_type = type(rel).__name__
                    _log(f"  {obj.name} {rel_type}({rel.parent.name})"
                         f" -> place {rel.parent.name} BEFORE {obj.name}")
                    must_precede[obj.name].add(rel.parent.name)

        # Init: On(A) or NextTo(A) among pick objects -> current before A
        # Skip if it conflicts with a target constraint
        _log(f"\n--- Init relation constraints (pick safety) ---")
        for obj in self._pick_objects:
            for rel in self._init_relations[obj.name]:
                if isinstance(rel, (NextTo, On)) and id(rel.parent) in pick_obj_ids:
                    parent_name = rel.parent.name
                    rel_type = type(rel).__name__
                    if parent_name not in must_precede[obj.name]:
                        _log(f"  {obj.name} {rel_type}({parent_name})"
                             f" -> pick {obj.name} BEFORE {parent_name}")
                        must_precede[parent_name].add(obj.name)
                    else:
                        _log(f"  {obj.name} {rel_type}({parent_name})"
                             f" -> SKIPPED (conflicts with target constraint)")

        _log(f"\n--- Dependency graph ---")
        for name in pick_names:
            deps = must_precede[name]
            if deps:
                _log(f"  {name} must come after: {deps}")
            else:
                _log(f"  {name} (no dependencies, can go first)")

        # Get positions for A* cost computation
        init_pos: dict[str, tuple[float, float, float]] = {}
        for obj in self._pick_objects:
            pose = obj.get_initial_pose()
            if pose is not None:
                init_pos[obj.name] = pose.position_xyz

        target_pos = self._target_positions

        def _fmt_pos(p):
            return f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})" if p else "N/A"

        _log(f"\n--- Object positions ---")
        for name in pick_names:
            _log(f"  {name}: init={_fmt_pos(init_pos.get(name))} target={_fmt_pos(target_pos.get(name))}")

        def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
            return sum((ai - bi) ** 2 for ai, bi in zip(a, b)) ** 0.5

        def heuristic(placed: frozenset[str]) -> float:
            """Sum of pick-to-place distances for remaining objects (admissible)."""
            return sum(
                _dist(init_pos[n], target_pos[n])
                for n in pick_names
                if n not in placed and n in init_pos and n in target_pos
            )

        def is_valid(name: str, placed: frozenset[str]) -> bool:
            return must_precede[name].issubset(placed)

        # A* search. When ik_cost_fn is provided, IK feasibility is evaluated
        # sequentially (joint config chains), making cost order-dependent.
        # Joint configs are stored in a side dict keyed by node counter.
        counter = 0
        nodes_expanded = 0
        start: frozenset[str] = frozenset()
        goal = frozenset(pick_names)
        joint_configs: dict[int, object] = {counter: None}
        ik_totals: dict[int, float] = {counter: 0.0}
        heap = [(heuristic(start), 0.0, counter, start, [])]
        visited_order: set[tuple[str, ...]] = set()
        visited_set: set[frozenset[str]] = set()

        mode_label = '(sequential IK)' if ik_cost_fn else ''
        _log(f"\n--- A* search {mode_label} ---")

        while heap:
            f, g, node_id, placed, order = heapq.heappop(heap)
            prev_jc = joint_configs.pop(node_id, None)
            prev_ik_total = ik_totals.pop(node_id, 0.0)

            if placed == goal:
                _log(f"\n--- Result ---")
                _log(f"  Nodes expanded: {nodes_expanded}")
                _log(f"  Total cost:     {g:.3f} (IK penalty: {prev_ik_total:.1f})")
                _log(f"  Pick order:     {' -> '.join(order)}")
                _log(sep)
                print(f"[SYMBOLIC A*] Pick order: {order} "
                      f"(cost: {g:.3f}m, ik_penalty: {prev_ik_total:.1f})")
                joint_configs.clear()
                ik_totals.clear()
                return order, prev_ik_total

            # Duplicate detection: order-based when IK is active (cost is
            # sequence-dependent), set-based otherwise.
            if ik_cost_fn is not None:
                order_key = tuple(order)
                if order_key in visited_order:
                    continue
                visited_order.add(order_key)
            else:
                if placed in visited_set:
                    continue
                visited_set.add(placed)

            nodes_expanded += 1

            valid_next = [n for n in pick_names if n not in placed and is_valid(n, placed)]
            _log(f"  [expand #{nodes_expanded}] placed={list(order)} candidates={valid_next}")

            ref = target_pos.get(order[-1]) if order else None

            for name in valid_next:
                dist_cost = 0.0
                if ref is not None and name in init_pos:
                    dist_cost += _dist(ref, init_pos[name])
                if name in init_pos and name in target_pos:
                    dist_cost += _dist(init_pos[name], target_pos[name])

                ik_penalty = 0.0
                next_jc = prev_jc
                if ik_cost_fn is not None and name in init_pos and name in target_pos:
                    ik_penalty, next_jc = ik_cost_fn(name, init_pos[name], target_pos[name], prev_jc)

                new_placed = placed | {name}
                new_g = g + dist_cost + ik_penalty
                new_h = heuristic(new_placed)
                counter += 1
                joint_configs[counter] = next_jc
                ik_totals[counter] = prev_ik_total + ik_penalty
                ik_tag = f" IK_PENALTY={ik_penalty:.1f}" if ik_penalty > 0 else ''
                _log(f"    -> {name}: g={new_g:.3f} h={new_h:.3f} f={new_g + new_h:.3f}{ik_tag}")
                heapq.heappush(
                    heap,
                    (new_g + new_h, new_g, counter, new_placed, order + [name]),
                )

        joint_configs.clear()
        ik_totals.clear()

        _log(f"\n--- Result ---")
        _log(f"  WARNING: no valid ordering found (cyclic dependencies?)")
        _log(f"  Falling back to alphabetical order")
        _log(sep)
        print("[SYMBOLIC A*] Warning: no valid ordering found, returning alphabetical")
        return sorted(pick_names), float('inf')

    @staticmethod
    def _generate_object_layout(objects, table, bin_asset, static_objects=None):
        """Randomly initialize object positions on the table relative to the bin and static objects.

        Topology:
          - obj[0]: On table, NextTo bin (-Y side), lying on side (roll=90deg)
          - obj[1]: On table, NextTo obj[0] (-X side), random yaw
          - obj[2]: On table, NextTo static_1 (bowl) on a random side, upright
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
            Inside
        )
        all_sides = [Side.POSITIVE_X, Side.NEGATIVE_X, Side.POSITIVE_Y, Side.NEGATIVE_Y]
        obj_1, obj_2, obj_3 = objects[0], objects[1], objects[2]
        static_1 = static_objects[0] if static_objects else bin_asset
        static_2 = static_objects[1] if static_objects and len(static_objects) > 1 else bin_asset

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

        # obj_3: inside static_1 (bowl) if available, otherwise next to bin
        if static_objects:
            obj_3.add_relation(Inside(static_1, clearance_m=0.02))
        else:
            obj_3.add_relation(On(table))
            obj_3.add_relation(NextTo(
                bin_asset, side=Side.NEGATIVE_Y,
                distance_m=random.uniform(0.05, 0.15),
            ))

    @staticmethod
    def _place_static_objects(static_objects, table, bin_asset):
        """Place non-pickable static objects on the table.

        Returns:
            List of static objects with relations applied (for goal config generation).
        """
        from isaaclab_arena.relations.relations import (
            IsAnchor,
            NextTo,
            On,
            Side,
            AtPosition,
        )
        static_1, static_2 = static_objects[0], static_objects[1]

        # static_1 (bowl): on table, next to bin on +X side
        static_1.add_relation(IsAnchor())
        static_1.add_relation(On(table))
        static_1.add_relation(NextTo(
            bin_asset, side=Side.POSITIVE_X,
            distance_m=random.uniform(0.10, 0.20),
        ))

        # static_2 (red container): on table, next to bin on -X side
        bowl_pose = static_1.get_initial_pose()
        static_2.add_relation(IsAnchor())
        static_2.add_relation(On(table))
        static_2.add_relation(AtPosition(x=bowl_pose.position_xyz[0]+1, y=bowl_pose.position_xyz[1]))


    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Add CLI arguments specific to this environment."""
        parser.add_argument('--object', type=str, default='tomato_soup_can')
        parser.add_argument('--object_set', nargs='+', type=str, default=None)
        parser.add_argument('--embodiment', type=str, default='droid_differential_ik')
        parser.add_argument('--teleop_device', type=str, default=None)
