# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render a ``Placement`` to a registered ArenaEnvironment Python module.

The pipeline is split into two stages:

  1. :func:`isaaclab_arena.llm_env_gen.placement_proposer.propose_placement`
     decides *what* should happen — which task to wire, how to anchor the
     tabletop, what relations each item gets. It produces a pure-data
     ``Placement`` bundle with no source strings.
  2. :func:`write_env` (this module) consumes a ``Placement`` and emits
     a ``@register_environment`` module. All template / indentation /
     import concerns live here; no placement decisions are made.

Separating the two lets feasibility gates (IK reachability, motion-plan
validity) run against a Placement before any file hits disk.
"""

from __future__ import annotations

from pathlib import Path

from .placement_proposer import (
    Placement,
    PlacementItem,
    RelationSpec,
    RobotPlacement,
    TabletopAnchorPlan,
    block_initial_goal_satisfaction,
    propose_placement,
)
from .resolver import ResolvedScene
from .schema import SceneSpec


def write_env(
    resolved: ResolvedScene,
    spec: SceneSpec,
    out_path: str | Path,
    attempt: int = 0,
    env_suffix: str = "",
) -> Path:
    """Render the env module and return the final path.

    ``out_path`` may be either an explicit ``*.py`` file or a directory —
    in the latter case the filename is derived from the env name so the
    generated module on disk matches the registered env name.

    ``attempt`` is forwarded to :func:`propose_placement` to vary the
    sampled robot placement across retries (see auto_generate_env).

    ``env_suffix`` is appended to the env / class names so the auto
    driver can register a unique env per attempt (e.g. ``"_t0"``,
    ``"_t1"``). This sidesteps Isaac Sim's gym / scene caching, which
    causes init_state from a re-registered same-name env to be ignored
    on the third attempt onward.
    """
    placement = propose_placement(resolved, spec, attempt=attempt)
    placement = block_initial_goal_satisfaction(placement, resolved)
    if env_suffix:
        # Camelize "_t3" -> "T3" so the class name stays PEP-8-friendly.
        suffix_camel = "".join(part.capitalize() for part in env_suffix.lstrip("_").split("_"))
        placement.env_name = f"{placement.env_name}{env_suffix}"
        placement.class_name = f"{placement.class_name}{suffix_camel}"

    out_path = Path(out_path)
    if out_path.suffix != ".py":
        out_path = out_path / f"{placement.env_name}.py"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    body = _render_module(placement)
    out_path.write_text(body)
    return out_path


# ---------------------------------------------------------------------------
# Rendering helpers — all purely string manipulation.
# ---------------------------------------------------------------------------


def _render_module(placement: Placement) -> str:
    item_decls = _render_item_decls(placement.items)
    anchor_setup = _render_anchor_setup(placement.tabletop_anchor_plan)
    bbox_setup = _render_bbox_setup(placement.tabletop_anchor_plan)
    robot_pose_setup = _render_robot_pose(placement.robot_placement)
    relations_src = _render_relations(placement.items)
    goal_comments = "\n".join(placement.task_plan.goal_comments)
    asset_list = ", ".join([
        "background",
        "ground_plane",
        "light",
        *placement.extra_scene_assets,
        *(i.var_name for i in placement.items),
    ])

    tabletop_header_note = placement.tabletop_anchor_plan.header_note(placement.background_name)

    # Only import symbols that are actually used in the generated module.
    # Walk inner specs so the kind wrapped by Not(...) is counted too —
    # e.g. Not(In(bowl)) needs both ``Not`` and ``In`` imported.
    def _collect_kinds(specs):
        seen: set[str] = set()
        for spec in specs:
            seen.add(spec.kind)
            if spec.inner is not None:
                seen.update(_collect_kinds([spec.inner]))
        return seen

    used_relation_kinds = _collect_kinds([r for i in placement.items for r in i.relations])
    relation_imports = sorted(
        {"IsAnchor"}  # always needed for tabletop anchor
        | ({"In"} if "in" in used_relation_kinds else set())
        | ({"Not"} if "not" in used_relation_kinds else set())
        | ({"On"} if "on" in used_relation_kinds else set())
        | ({"PositionLimits"} if "position_limits" in used_relation_kinds else set())
        | ({"RotateAroundSolution"} if "rotate_around_solution" in used_relation_kinds else set())
    )
    relations_import_line = f"        from isaaclab_arena.relations.relations import {', '.join(relation_imports)}"
    # ObjectType is only needed when the anchor ObjectReference uses ObjectType.RIGID explicitly.
    needs_object_type = (
        placement.tabletop_anchor_plan.kind == "reference"
        and placement.tabletop_anchor_plan.anchor_object_type != "BASE"
    )
    object_type_import_line = (
        "        from isaaclab_arena.assets.object_base import ObjectType\n" if needs_object_type else ""
    )
    # ObjectReference is only needed when the tabletop anchor is a sub-prim reference.
    needs_object_reference = placement.tabletop_anchor_plan.kind == "reference"
    object_reference_import_line = (
        "        from isaaclab_arena.assets.object_reference import ObjectReference\n" if needs_object_reference else ""
    )

    return f'''# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Auto-generated by isaaclab_arena.llm_env_gen.

Task: {placement.task_description}

Task wiring: {placement.task_plan.header_note}
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@register_environment
class {placement.class_name}(ExampleEnvironmentBase):
    """{placement.task_description}"""

    name: str = "{placement.env_name}"

    def get_env(self, args_cli: argparse.Namespace) -> "IsaacLabArenaEnvironment":
{object_type_import_line}{object_reference_import_line}        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
{relations_import_line}
        from isaaclab_arena.scene.scene import Scene
        {placement.task_plan.task_import}
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("{placement.background_name}")()
        background.set_initial_pose(Pose(position_xyz=(0.5, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.7071068, 0.7071068)))
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -1.05)))
        light = self.asset_registry.get_asset_by_name("light")()

        # Tabletop anchor — {tabletop_header_note}.
{anchor_setup}

{bbox_setup}

        # No kwargs — works for both no_embodiment and robot embodiments whose
        # optional flags (enable_cameras, etc.) default to safe values.
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)()
{robot_pose_setup}

{item_decls}

{relations_src}

        scene = Scene(assets=[{asset_list}])

{goal_comments}

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task={placement.task_plan.task_expr},
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="{placement.embodiment_default}")
'''


def _render_item_decls(items: list[PlacementItem]) -> str:
    # TODO: Support multiple instances of the same library asset (e.g. two
    # bananas in one scene). Right now each item gets the asset's registered
    # name by default, so two bananas would collide on the scene-level name.
    # When we need this, re-introduce instance_name="..." on the emitted
    # line and derive a unique suffix from the LLM's instance_name / query.
    return "\n".join(f'        {i.var_name} = self.asset_registry.get_asset_by_name("{i.asset_name}")()' for i in items)


def _render_anchor_setup(plan: TabletopAnchorPlan) -> str:
    if plan.kind == "reference":
        if plan.anchor_object_type == "BASE":
            # Tabletop sub-prim has no RigidBodyAPI — omit object_type to avoid RuntimeError.
            return (
                "        tabletop_anchor = ObjectReference(\n"
                '            name="table",\n'
                f'            prim_path="{plan.prim_path}",\n'
                "            parent_asset=background,\n"
                "        )\n"
                "        tabletop_anchor.add_relation(IsAnchor())"
            )
        return (
            "        tabletop_anchor = ObjectReference(\n"
            '            name="table",\n'
            f'            prim_path="{plan.prim_path}",\n'
            "            parent_asset=background,\n"
            "            object_type=ObjectType.RIGID,\n"
            "        )\n"
            "        tabletop_anchor.add_relation(IsAnchor())"
        )
    # "none" and "background" both anchor on the background asset.
    return "        background.add_relation(IsAnchor())"


def _render_bbox_setup(plan: TabletopAnchorPlan) -> str:
    if not plan.emit_position_limits:
        return "        # (no tabletop bbox — background not in _BACKGROUND_TABLETOP_ANCHOR)"
    return (
        "        # Runtime-derived XY footprint from the tabletop bbox — no\n"
        "        # hardcoded bounds, works across tables of any size. The\n"
        "        # bbox stores (N, 3) tensors; we squeeze to plain floats\n"
        "        # for the first env since PositionLimits takes scalars and\n"
        "        # the tabletop is static across envs.\n"
        f"        _tbl_bbox = {plan.anchor_var}.get_world_bounding_box()\n"
        "        _tbl_min_xyz = [float(_tbl_bbox.min_point[0, i]) for i in range(3)]\n"
        "        _tbl_max_xyz = [float(_tbl_bbox.max_point[0, i]) for i in range(3)]\n"
        f"        _tbl_margin = {plan.margin_m}"
    )


def _render_robot_pose(rp: RobotPlacement | None) -> str:
    """Emit the lines that place the embodiment along the sampled table edge.

    Relies on ``_tbl_min_xyz`` / ``_tbl_max_xyz`` / ``_tbl_margin`` having
    been materialized by :func:`_render_bbox_setup`, so this must be
    rendered *after* the bbox setup block in the generated module.
    """
    if rp is None:
        return ""

    # Compact axis encoding so we can share one template across edges.
    # The sampled edge picks which XY axis is "fixed at the edge" vs
    # "interpolated between the two corners", and which direction the
    # outward offset points.
    frac = round(rp.fraction, 4)
    if rp.edge == "x_min":
        x_expr = f"_tbl_min_xyz[0] - {rp.offset_m}"
        y_expr = f"(1 - {frac}) * _tbl_min_xyz[1] + {frac} * _tbl_max_xyz[1]"
    elif rp.edge == "x_max":
        x_expr = f"_tbl_max_xyz[0] + {rp.offset_m}"
        y_expr = f"(1 - {frac}) * _tbl_min_xyz[1] + {frac} * _tbl_max_xyz[1]"
    elif rp.edge == "y_min":
        x_expr = f"(1 - {frac}) * _tbl_min_xyz[0] + {frac} * _tbl_max_xyz[0]"
        y_expr = f"_tbl_min_xyz[1] - {rp.offset_m}"
    elif rp.edge == "y_max":
        x_expr = f"(1 - {frac}) * _tbl_min_xyz[0] + {frac} * _tbl_max_xyz[0]"
        y_expr = f"_tbl_max_xyz[1] + {rp.offset_m}"
    else:
        return f"        # TODO(robot_placement): unsupported edge {rp.edge!r}"

    return (
        "\n"
        f"        # Robot placement sampled on tabletop edge '{rp.edge}' at fraction "
        f"{rp.fraction:.3f};\n"
        f"        # base sits {rp.offset_m} m outside the edge, yaw faces the table center.\n"
        f"        _robot_x = {x_expr}\n"
        f"        _robot_y = {y_expr}\n"
        f"        embodiment.set_initial_pose(Pose(\n"
        f"            position_xyz=(_robot_x, _robot_y, {rp.z_m}),\n"
        f"            rotation_xyzw={rp.rotation_xyzw},\n"
        f"        ))"
    )


def _render_relations(items: list[PlacementItem]) -> str:
    lines: list[str] = []
    for item in items:
        if not item.relations:
            continue
        for rel in item.relations:
            lines.extend(_render_one_relation(item.var_name, rel))
    if not lines:
        return "        # (no initial relations)"
    return "\n".join(lines)


def _relation_call_expr(rel: RelationSpec) -> str | None:
    """Return the Python call expression for a RelationSpec (no leading var).

    Used as the argument to ``Not(...)`` when wrapping, and inlined into
    ``var.add_relation(...)`` for top-level relations.
    """
    if rel.kind == "on":
        return f"On({rel.on_target_var}, clearance_m={rel.on_clearance_m})"
    if rel.kind == "in":
        return f"In({rel.in_target_var})"
    return None


def _render_one_relation(var: str, rel: RelationSpec) -> list[str]:
    if rel.kind == "on":
        return [f"        {var}.add_relation({_relation_call_expr(rel)})"]
    if rel.kind == "in":
        return [f"        {var}.add_relation({_relation_call_expr(rel)})"]
    if rel.kind == "not":
        if rel.inner is None:
            return ["        # TODO(not): missing inner spec"]
        inner_expr = _relation_call_expr(rel.inner)
        if inner_expr is None:
            return [f"        # TODO(not): inner kind {rel.inner.kind!r} has no call expr"]
        return [f"        {var}.add_relation(Not({inner_expr}))"]
    if rel.kind == "position_limits":
        if rel.pl_source != "bbox":
            return [f"        # TODO(position_limits): unsupported source {rel.pl_source!r}"]
        return [
            f"        {var}.add_relation(PositionLimits(",
            "            x_min=_tbl_min_xyz[0] + _tbl_margin,",
            "            x_max=_tbl_max_xyz[0] - _tbl_margin,",
            "            y_min=_tbl_min_xyz[1] + _tbl_margin,",
            "            y_max=_tbl_max_xyz[1] - _tbl_margin,",
            "        ))",
        ]
    if rel.kind == "rotate_around_solution":
        return [f"        {var}.add_relation(RotateAroundSolution(yaw_rad={rel.rotate_yaw_rad}))"]
    if rel.kind == "unsupported":
        return [f"        # TODO({rel.reason}): {rel.raw_relation}"]
    return [f"        # TODO(unknown relation kind {rel.kind!r}): {rel.raw_relation}"]
