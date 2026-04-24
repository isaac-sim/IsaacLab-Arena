"""Scene generator orchestrator — ties all pieces together.

Pipeline:
    scene_themes → asset_manager → llm_agent → predicate_translator → adaptive_placer → Arena Scene

Usage:
    # Single scene
    gen = SceneGenerator()
    scene = gen.generate_scene("kitchen counter with fruits", max_objects=8)

    # Batch generation (automatic prompts)
    scenes = gen.generate_batch(num_scenes=100)
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Optional

# Path to the base scene template (table + ground + physics)
BASE_SCENE_PATH = os.path.join(os.path.dirname(__file__), "scenes", "base_empty.usda")

from isaaclab_arena.relations.relations import IsAnchor
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.scene_gen.arena_asset_manager import ArenaAssetManager, SCENE_GEN_TABLES
from isaaclab_arena.scene_gen.adaptive_placer import place_objects_adaptive
from isaaclab_arena.scene_gen.feedback_system import FeedbackSystem
from isaaclab_arena.scene_gen.llm_agent import LLMAgent
from isaaclab_arena.scene_gen.predicate_translator import translate_predicates
from isaaclab_arena.scene_gen.scene_themes import (
    generate_scene_prompts,
    ARTICULATED_THEMES,
)
from isaaclab_arena.utils.pose import Pose


class SceneGenerator:
    """Orchestrates LLM-driven scene generation for Arena.

    Combines:
    - ArenaAssetManager: 719 objects with dims
    - LLMAgent: Claude Opus 4.6 for predicate generation
    - RoboLab spatial solver: collision-free placement
    - Arena Scene: native output for env gen
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "aws/anthropic/bedrock-claude-opus-4-6",
        base_url: str = "https://inference-api.nvidia.com",
        output_dir: Optional[str] = None,
        max_retries: int = 3,
        table_top_z: float = 0.0,  # Table surface at Z=0 (table at Z=-0.35, height ~0.35m)
    ):
        """Initialize the scene generator.

        Args:
            api_key: API key for LLM. Reads NV_API_KEY env var if None.
            model: LLM model identifier.
            base_url: LLM API base URL.
            output_dir: Directory to save generated scene metadata.
            max_retries: Max LLM retries per scene on failure.
            table_top_z: Z height of the table surface (meters).
        """
        self.asset_manager = ArenaAssetManager()
        self.llm_agent = LLMAgent(api_key=api_key, model=model, base_url=base_url)
        self.feedback_system = FeedbackSystem()
        self.max_retries = max_retries
        self.table_top_z = table_top_z

        # Default output under scene_gen/generated/
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "generated")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_scene(
        self,
        prompt: str,
        max_objects: int = 10,
        table_name: Optional[str] = None,
        scene_name: Optional[str] = None,
        has_articulated: bool = False,
    ) -> tuple[Optional[Scene], Optional[str]]:
        """Generate a single scene from a natural language prompt.

        Args:
            prompt: Natural language scene description.
            max_objects: Maximum number of objects to place.
            table_name: Which table to use. Random if None.
            scene_name: Optional name for metadata saving.
            has_articulated: If True, ensures articulated objects are available.

        Returns:
            (Arena Scene, USD file path) tuple. (None, None) if generation failed.
        """
        from isaaclab_arena.assets.asset_registry import AssetRegistry
        registry = AssetRegistry()

        # 1. Select table
        if table_name is None:
            table_name = self.asset_manager.get_random_table()
        table_info = self.asset_manager.get_table_info(table_name)
        table_bounds = self.asset_manager.get_table_bounds(table_name)

        print(f"[SceneGen] Table: {table_name}")
        print(f"[SceneGen] Prompt: {prompt}")
        print(f"[SceneGen] Max objects: {max_objects}")

        # 2. Create table asset as anchor with table-specific pose
        table_pose = table_info.get("pose", {})
        table_pos = table_pose.get("position", (0.547, 0.0, -0.35))
        table_rot = table_pose.get("rotation", (1.0, 0.0, 0.0, 0.0))

        table = registry.get_asset_by_name(table_info["registry_name"])()
        table.set_initial_pose(Pose(
            position_xyz=table_pos,
            rotation_wxyz=table_rot,
        ))
        table.add_relation(IsAnchor())

        # 3. Select candidate objects
        candidates = self.asset_manager.get_objects_for_scene(max_objects)
        llm_catalog = self.asset_manager.get_catalog_for_llm(candidates)

        # 4. Build articulated objects info for LLM
        artic_objects = {}
        for name in candidates:
            if self.asset_manager.needs_fixed_orientation(name):
                artic_objects[name] = self.asset_manager.get_affordances(name)

        # 5. LLM generation with two-level retry:
        #    Outer: reduce object count if solver keeps failing (like RoboLab)
        #    Inner: retry LLM with feedback at current object count
        objects_to_try = [max_objects]
        if max_objects >= 14:
            objects_to_try.extend([max(10, max_objects - 4), max(8, max_objects - 6)])
        elif max_objects >= 10:
            objects_to_try.extend([max(7, max_objects - 3), max(5, max_objects - 5)])
        elif max_objects >= 6:
            objects_to_try.append(max(3, max_objects - 2))

        for current_max in objects_to_try:
            if current_max < max_objects:
                print(f"[SceneGen] Reducing to {current_max} objects")
                candidates = self.asset_manager.get_objects_for_scene(current_max)
                llm_catalog = self.asset_manager.get_catalog_for_llm(candidates)

            feedback = None
            spatial_failures = 0

            for attempt in range(self.max_retries):
                if attempt > 0:
                    print(f"[SceneGen] Retry {attempt}/{self.max_retries - 1}")

                try:
                    # Call LLM
                    self.llm_agent.reset_conversation()
                    llm_result = self.llm_agent.generate_predicates(
                        prompt=prompt,
                        object_catalog=llm_catalog,
                        max_objects=current_max,
                        feedback=feedback,
                        preselected_objects=candidates,
                        articulated_objects=artic_objects if artic_objects else None,
                    )

                    num_objects = len(llm_result.get("objects", []))
                    num_predicates = len(llm_result.get("predicates", []))
                    print(f"[SceneGen] LLM returned {num_objects} objects, {num_predicates} predicates")

                    # Validate minimum objects
                    if num_objects < max(2, current_max * 0.5):
                        feedback = self.feedback_system.generate_solver_feedback(
                            False, f"Too few objects: got {num_objects}, need at least {max(2, int(current_max * 0.6))}")
                        continue

                    # 6. Translate predicates → Arena Objects with Relations
                    objects = translate_predicates(llm_result, table, self.asset_manager)
                    print(f"[SceneGen] Translated {len(objects)} objects")

                    # 7. Place objects (spatial solver + stacking + containment)
                    result = place_objects_adaptive(
                        objects, table, self.asset_manager,
                        table_bounds=table_bounds,
                        table_top_z=self.table_top_z,
                        verbose=True,
                    )

                    if not result.success:
                        spatial_failures += 1
                        feedback = self.feedback_system.generate_solver_feedback(
                            False, "Collision resolution failed — objects overlap")
                        # If too many spatial failures at this count, try fewer objects
                        if spatial_failures >= 2:
                            print(f"[SceneGen] {spatial_failures} spatial failures — trying fewer objects")
                            break  # → outer loop reduces count
                        continue

                    # 8. Build Arena Scene
                    scene = Scene()
                    scene.add_asset(table)
                    for obj in objects:
                        scene.add_asset(obj)

                    # Add ground plane and light
                    try:
                        ground = registry.get_asset_by_name("ground_plane")()
                        ground.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -0.35)))
                        scene.add_asset(ground)
                    except Exception:
                        pass

                    try:
                        import isaaclab.sim as sim_utils
                        light_cfg = sim_utils.DomeLightCfg(
                            color=(0.75, 0.75, 0.75), intensity=1500.0)
                        light = registry.get_asset_by_name("light")(spawner_cfg=light_cfg)
                        scene.add_asset(light)
                    except Exception:
                        pass

                    print(f"[SceneGen] Scene built with {len(scene.assets)} assets")

                    # 9. Export to USD
                    if scene_name is None:
                        from datetime import datetime
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        prompt_slug = "".join(c if c.isalnum() else "_" for c in prompt[:30].lower())
                        scene_name = f"{prompt_slug}_{ts}"
                    usd_path = self._export_to_usd(
                        scene_name, objects, table_name, table_info,
                    )

                    # 10. Save metadata
                    if self.output_dir:
                        self._save_metadata(scene_name, prompt, table_name,
                                           llm_result, result.positions, usd_path)

                    return scene, usd_path

                except Exception as e:
                    print(f"[SceneGen] Error: {e}")
                    feedback = f"Generation error: {e}. Try again with simpler placement."
                    continue

        print(f"[SceneGen] Failed after all attempts")
        return None, None

    def generate_batch(
        self,
        num_easy: int = 15,
        num_medium: int = 70,
        num_hard: int = 15,
        appliance_ratio: float = 0.15,
    ) -> list[tuple[Optional[Scene], Optional[str]]]:
        """Generate a batch of scenes using automated prompts.

        Args:
            num_easy: Number of easy scenes (2-5 objects).
            num_medium: Number of medium scenes (7-10 objects).
            num_hard: Number of hard scenes (12-20 objects).
            appliance_ratio: Fraction of scenes with articulated objects.

        Returns:
            List of (Arena Scene, USD path) tuples.
        """
        scene_configs = generate_scene_prompts(
            num_easy=num_easy, num_medium=num_medium, num_hard=num_hard,
            appliance_ratio=appliance_ratio,
        )

        total = len(scene_configs)
        results = []
        success_count = 0

        for i, config in enumerate(scene_configs, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{total}] {config['name']} ({config['difficulty']})")
            print(f"{'='*60}")

            scene, usd_path = self.generate_scene(
                prompt=config["prompt"],
                max_objects=config["max_objects"],
                scene_name=config["name"],
                has_articulated=config.get("has_articulated", False),
            )

            results.append((scene, usd_path))
            if scene is not None:
                success_count += 1

            if i % 10 == 0:
                print(f"\n--- Progress: {i}/{total} ({success_count} success, "
                      f"{i - success_count} failed) ---\n")

        print(f"\nBatch complete: {success_count}/{total} scenes generated")
        self.asset_manager.print_coverage_report()

        # Print generated USD paths
        print(f"\nGenerated USD files:")
        for scene, path in results:
            if path:
                print(f"  {path}")

        return results

    def _export_to_usd(
        self,
        scene_name: str,
        objects: list,
        table_name: str,
        table_info: dict,
    ) -> str:
        """Export scene to .usda file — same approach as RoboLab.

        Opens base_empty.usda (which has table, ground plane, physics),
        swaps table payload if different table selected, adds objects
        as payloads with transforms, exports to output directory.

        Args:
            scene_name: Name for the output file.
            objects: List of placed Arena Object instances.
            table_name: Which table was used.
            table_info: Table config dict from SCENE_GEN_TABLES.

        Returns:
            Path to the exported .usda file.
        """
        from pxr import Gf, Usd, UsdGeom

        # Open base scene fresh (same as RoboLab)
        stage = Usd.Stage.Open(BASE_SCENE_PATH)

        # Swap table payload + transform if not oak (approach C)
        if table_name != "oak_table_robolab":
            table_prim = stage.GetPrimAtPath("/world/table")
            if table_prim.IsValid():
                # Swap payload
                table_prim.GetPayloads().ClearPayloads()
                from isaaclab_arena.assets.asset_registry import AssetRegistry
                registry = AssetRegistry()
                table_cls = registry.get_asset_by_name(table_info["registry_name"])
                table_usd = getattr(table_cls, "usd_path", None)
                if table_usd:
                    table_prim.GetPayloads().AddPayload(table_usd)

                # Update transform to match table-specific pose
                table_pose = table_info.get("pose", {})
                pos = table_pose.get("position", (0.547, 0.0, -0.35))
                rot = table_pose.get("rotation", (1.0, 0.0, 0.0, 0.0))

                xform = UsdGeom.Xformable(table_prim)
                # Update existing translate op
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
                    elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                        op.Set(Gf.Quatf(rot[0], rot[1], rot[2], rot[3]))

                # Apply scale if needed
                table_scale = table_info.get("scale", (1.0, 1.0, 1.0))
                if table_scale != (1.0, 1.0, 1.0):
                    for op in xform.GetOrderedXformOps():
                        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                            op.Set(Gf.Vec3f(table_scale[0], table_scale[1], table_scale[2]))

                print(f"[USD Export] Swapped table to {table_name} at pos={pos} rot={rot}")

        # Add each object (same pattern as RoboLab's make_new_scene.py)
        for obj in objects:
            usd_path = getattr(obj, "usd_path", None)
            if not usd_path:
                continue

            pose = obj.get_initial_pose()
            if pose is None:
                continue

            # Ensure unique prim path — use /World (capital W) so vomp material
            # bindings targeting /World/Looks/... resolve correctly
            name = obj.name
            prim_path = f"/World/{name}"
            counter = 1
            while stage.GetPrimAtPath(prim_path).IsValid():
                prim_path = f"/World/{name}_{counter}"
                counter += 1

            xform = UsdGeom.Xform.Define(stage, prim_path)
            xform.GetPrim().GetReferences().AddReference(usd_path)

            # Set transform — handle existing ops from reference
            pos = pose.position_xyz
            rot = pose.rotation_wxyz

            # Quaternion wxyz → yaw degrees
            w, x, y, z = rot
            yaw_rad = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            yaw_deg = math.degrees(yaw_rad)

            # Debug: print rotation ops from reference vs our yaw
            ref_ops = [(op.GetOpType(), op.GetOpName()) for op in xform.GetOrderedXformOps()]
            print(f"  [USD Export] {obj.name}: yaw={yaw_deg:.1f}°, xform_ops={ref_ops}")

            # Translate — reuse existing op if present (same as RoboLab)
            existing_t = [op for op in xform.GetOrderedXformOps()
                          if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
            if existing_t:
                existing_t[0].Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
            else:
                xform.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], pos[2]))

            # Rotate (RotateXYZ in degrees, same as RoboLab)
            existing_r = [op for op in xform.GetOrderedXformOps()
                          if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ]
            if existing_r:
                existing_r[0].Set(Gf.Vec3f(0.0, 0.0, yaw_deg))
            else:
                xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, yaw_deg))

            # Scale — use pre-computed effective_scale from asset manager
            # (registered_scale × auto_factor, computed once during catalog build)
            final_scale = self.asset_manager.get_object_scale(obj.name)

            print(f"  [USD Export] {obj.name}: effective_scale={final_scale}, dims={self.asset_manager.get_object_dims(obj.name)}")

            if final_scale != (1.0, 1.0, 1.0):
                existing_s = [op for op in xform.GetOrderedXformOps()
                              if op.GetOpType() == UsdGeom.XformOp.TypeScale]
                if existing_s:
                    existing_s[0].Set(Gf.Vec3f(final_scale[0], final_scale[1], final_scale[2]))
                else:
                    xform.AddScaleOp().Set(Gf.Vec3f(final_scale[0], final_scale[1], final_scale[2]))

        # Export (same as RoboLab — stage.Export, not stage.Save)
        output_path = str(self.output_dir / f"{scene_name}.usda")
        stage.Export(output_path)

        # Remove duplicate GroundPlanes/franka_tables from object payloads
        # (same as RoboLab's _remove_duplicate_ground_planes)
        self._remove_duplicate_prims(output_path)

        # Object payloads land under /World (uppercase) while infrastructure
        # sits under /world (lowercase, defaultPrim). UsdFileCfg follows only
        # defaultPrim, so /World prims would be invisible to Isaac Lab.
        # Consolidate everything under /world.
        self._consolidate_world_roots(output_path)

        # office_table has RigidBodyAPI on a nested prim; under the non-uniform
        # parent scale (0.7, 1, 0.9195) PhysX mishandles it and the table
        # falls. Tables are static infrastructure — kinematic is correct.
        self._kinematize_table_rigid_bodies(output_path)

        # Scene USDs carry their own PhysicsScene prim. Isaac Lab creates one
        # too → 'Physics scenes stepping is not the same' error. Strip ours.
        self._strip_physics_scenes(output_path)

        print(f"[USD Export] Saved: {output_path}")
        return output_path

    def _consolidate_world_roots(self, scene_path: str) -> None:
        """Flatten every /World/* prim into /world/* and remove /World.

        scene_gen composes base_empty.usda (defaultPrim='world') with object
        payloads authored under /World. UsdFileCfg only follows defaultPrim,
        so the objects are orphaned on load. This merges them in place.
        """
        from pxr import Sdf, Usd

        stage = Usd.Stage.Open(scene_path)
        world_lower = stage.GetPrimAtPath("/world")
        world_upper = stage.GetPrimAtPath("/World")
        if not world_lower.IsValid() or not world_upper.IsValid():
            return

        root_layer = stage.GetRootLayer()
        moved = 0
        for child in list(world_upper.GetChildren()):
            src = child.GetPath()
            dst = Sdf.Path(f"/world/{child.GetName()}")
            if stage.GetPrimAtPath(dst).IsValid():
                continue  # name collision — leave alone
            if Sdf.CopySpec(root_layer, src, root_layer, dst):
                moved += 1

        stage.RemovePrim("/World")
        if moved:
            root_layer.Save()
            print(f"[USD Export] Consolidated {moved} prim(s) from /World into /world")

    def _kinematize_table_rigid_bodies(self, scene_path: str) -> None:
        """Set kinematicEnabled=True on any rigid body under /world/table."""
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(scene_path)
        flipped = 0
        for root_name in ("/world/table", "/World/table"):
            table = stage.GetPrimAtPath(root_name)
            if not table.IsValid():
                continue
            for prim in Usd.PrimRange(table):
                rb = UsdPhysics.RigidBodyAPI(prim)
                if not (rb and rb.GetRigidBodyEnabledAttr().Get()):
                    continue
                kin = prim.GetAttribute("physics:kinematicEnabled")
                if kin and kin.Get() is True:
                    continue
                if kin:
                    kin.Set(True)
                else:
                    prim.CreateAttribute(
                        "physics:kinematicEnabled", Usd.ValueTypeNames.Bool
                    ).Set(True)
                flipped += 1
        if flipped:
            stage.GetRootLayer().Save()
            print(f"[USD Export] Kinematized {flipped} rigid body(ies) under /world/table")

    def _strip_physics_scenes(self, scene_path: str) -> None:
        """Remove PhysicsScene prims. Isaac Lab creates its own at runtime."""
        from pxr import Usd

        stage = Usd.Stage.Open(scene_path)
        to_remove = [
            p.GetPath() for p in stage.Traverse() if p.GetTypeName() == "PhysicsScene"
        ]
        for path in to_remove:
            stage.RemovePrim(path)
        if to_remove:
            stage.GetRootLayer().Save()
            print(f"[USD Export] Stripped {len(to_remove)} PhysicsScene prim(s)")

    def _remove_duplicate_prims(self, scene_path: str):
        """Remove duplicate GroundPlane/franka_table prims injected by object payloads.

        Object USDs were sometimes authored as complete scenes containing
        GroundPlane, franka_table, etc. When loaded as payloads these become
        duplicates. Keep only the main ones under /world.
        """
        from pxr import Usd

        stage = Usd.Stage.Open(scene_path)
        main_prims = {"/world/GroundPlane", "/world/franka_table", "/World/GroundPlane", "/World/franka_table"}
        duplicates = []

        for prim in stage.Traverse():
            name = prim.GetName()
            path = prim.GetPath().pathString
            if name in ("GroundPlane", "franka_table") and path not in main_prims:
                duplicates.append(prim.GetPath())

        for prim_path in duplicates:
            stage.RemovePrim(prim_path)

        if duplicates:
            stage.Save()
            print(f"[USD Export] Removed {len(duplicates)} duplicate prim(s)")

    def _save_metadata(self, scene_name, prompt, table_name, llm_result, positions, usd_path=None):
        """Save scene generation metadata to JSON."""
        metadata = {
            "scene_name": scene_name,
            "prompt": prompt,
            "table": table_name,
            "usd_path": usd_path,
            "llm_result": llm_result,
            "positions": {
                name: list(pos) for name, pos in positions.items()
            } if positions else {},
        }
        path = self.output_dir / f"{scene_name}_metadata.json"
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[SceneGen] Metadata saved: {path}")
