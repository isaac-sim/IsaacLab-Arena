"""Test environment factory: GoalSpec JSON → gymnasium env.

Run inside Isaac Sim:
    python test_env_factory.py
"""

from isaaclab.app import AppLauncher
# enable_cameras=True is required because the Franka embodiment spawns
# wrist_cam + external_cam. Without it, sensor init fails with
# "A camera was spawned without the --enable_cameras flag".
app_launcher = AppLauncher(headless=False, enable_cameras=True)

# Preload libassimp so pinocchio can find Assimp symbols
import ctypes, sys, os
for sp in [p for p in sys.path if "site-packages" in p]:
    assimp_path = os.path.join(sp, "cmeel.prefix", "lib", "libassimp.so.5")
    if os.path.exists(assimp_path):
        ctypes.CDLL(assimp_path, mode=ctypes.RTLD_GLOBAL)
        break

import json
from pathlib import Path


def _print_task_summary(spec: dict) -> None:
    print(f"  Task: {spec['task_name']}")
    print(f"  Instruction: {spec['instruction']}")
    print(f"  Contact objects: {spec['contact_objects']}")
    print(f"  Goal relations ({len(spec['goal_relations'])}) "
          f"— symbolic, describe starting placement:")
    for rel in spec['goal_relations']:
        print(f"    - {rel}")
    print(f"  Success conditions ({len(spec['success_conditions'])}) "
          f"— runtime termination predicates:")
    for cond in spec['success_conditions']:
        print(f"    - {cond}")


def _table_pose_for_scene(scene_usd_name: str, scene_dir: Path):
    """Return (table_pose, table_dims) matching the scene's table asset.

    Reads the scene's metadata JSON (written by scene_gen next to the USD)
    and picks the pose/dims registered in arena_asset_manager.
    """
    meta_path = scene_dir / f"{scene_usd_name.replace('.usda', '')}_metadata.json"
    if not meta_path.exists():
        return None, None
    import json as _json
    table_name = _json.loads(meta_path.read_text()).get("table", "")
    if table_name == "office_table_background":
        # After 90° Z-rot + scale(0.7, 1, 0.9195): world dims (0.802, 1.26, 0.697)
        return (0.55, 0.0, -0.697), (0.802, 1.26, 0.697)
    # bamboo/black and any other standard-table asset
    return (0.547, 0.0, -0.35), (0.7, 1.0, 0.35)


def _print_resolved_goal_xyz(spec_json_path: str, scene_dir: Path) -> None:
    """Run predicate→xyz solver and print the goal target per object."""
    from isaaclab_arena.env_gen.predicate_to_xyz import resolve_goal_xyz
    from isaaclab_arena.task_gen.goal_spec import GoalSpec

    goal_spec = GoalSpec.from_json(spec_json_path)
    table_pose, table_dims = _table_pose_for_scene(goal_spec.scene, scene_dir)

    kwargs = {}
    if table_pose is not None:
        kwargs["table_pose"] = table_pose
        kwargs["table_dims"] = table_dims

    resolved = resolve_goal_xyz(goal_spec, **kwargs)

    print(f"\n  Resolved goal xyz ({len(resolved)} target object(s)):")
    if not resolved:
        print("    (no goal_relations resolvable — missing dims or unknown target)")
        return
    for name, (gx, gy, gz) in resolved.items():
        init = goal_spec.initial_state.get(name)
        if init is not None:
            ix, iy, iz = init.position
            dx, dy, dz = gx - ix, gy - iy, gz - iz
            print(f"    {name}: goal=({gx:+.3f}, {gy:+.3f}, {gz:+.3f})  "
                  f"init=({ix:+.3f}, {iy:+.3f}, {iz:+.3f})  "
                  f"Δ=({dx:+.3f}, {dy:+.3f}, {dz:+.3f})")
        else:
            print(f"    {name}: goal=({gx:+.3f}, {gy:+.3f}, {gz:+.3f})")


def test_register_single(factory, json_path: str):
    """Register a single GoalSpec JSON as a gym env."""
    print(f"\n{'='*60}")
    print(f"Test: Register {json_path}")
    print(f"{'='*60}")

    with open(json_path) as f:
        spec = json.load(f)
    _print_task_summary(spec)

    env_name = factory.register_from_json(json_path)
    print(f"\n  Registered as: {env_name}")
    return env_name


def test_register_batch():
    """Register all GoalSpec JSONs from a scene folder."""
    from isaaclab_arena.env_gen.env_factory import EnvFactory

    task_dir = Path("isaaclab_arena/task_gen/generated")
    if not task_dir.exists():
        print(f"No generated tasks found at {task_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Test: Register all tasks from {task_dir}")
    print(f"{'='*60}")

    factory = EnvFactory()
    registered = factory.register_all_from_folder(str(task_dir))

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  Total registered: {len(registered)}")
    for name in registered[:10]:
        print(f"    {name}")
    if len(registered) > 10:
        print(f"    ... and {len(registered) - 10} more")

    return registered


def test_make_env(factory, env_name: str, steps: int = 200, keep_alive: bool = False) -> bool:
    """Instantiate the registered env via Isaac Lab and step it.

    Confirms: scene loads, termination manager parses, observation/action
    managers accept the empty configs, physics runs, and termination terms
    produce a reasonable verdict. Prints whether the env timed out or
    signaled success.

    Args:
        keep_alive: After the stepping loop, keep stepping the env with
            zero actions so the viewer stays open and the robot stays
            visible for inspection. Ctrl+C to exit.
    """
    print(f"\n{'='*60}")
    print(f"Test: make_env({env_name}) + {steps} steps")
    print(f"{'='*60}")

    try:
        env = factory.make_env(env_name)
    except Exception as e:
        print(f"  [FAIL] could not create env: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"  [OK] env created")
    print(f"  observation_space: {env.observation_space}")
    print(f"  action_space:      {env.action_space}")
    print(f"  num_envs:          {env.num_envs}")
    print(f"  device:            {env.device}")
    print(f"  active terminations: {list(env.termination_manager.active_terms)}")

    # Reset → step with zero actions. No robot, so this just advances physics.
    obs, _ = env.reset()
    import torch
    zero_action = torch.zeros((env.num_envs, env.action_space.shape[-1] if env.action_space.shape else 0),
                              device=env.device)
    success_step = None
    timeout_step = None
    for i in range(steps):
        obs, rew, terminated, truncated, info = env.step(zero_action)
        if terminated.any() and success_step is None:
            success_step = i
        if truncated.any() and timeout_step is None:
            timeout_step = i
        if terminated.any() or truncated.any():
            break

    print(f"\n  after {min(i+1, steps)} step(s):")
    print(f"    terminated (success): {success_step}")
    print(f"    truncated  (time_out): {timeout_step}")
    print(f"    done flags: term={bool(terminated.any())} trunc={bool(truncated.any())}")
    if keep_alive:
        print("  [keep_alive] Stepping with zero actions indefinitely. Ctrl+C to exit.")
        try:
            while True:
                env.step(zero_action)
        except KeyboardInterrupt:
            pass
    env.close()
    return True


if __name__ == "__main__":
    import argparse

    # parse_known_args so Isaac Lab's own sys.argv flags (passed through
    # AppLauncher earlier) don't trip this parser.
    parser = argparse.ArgumentParser(description="Test env_factory on one task.")
    parser.add_argument(
        "--scene-index", "-i", type=int, default=0,
        help="Index into the per-scene task list (0..N-1). Default 0.",
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Explicit task JSON path. Overrides --scene-index.",
    )
    parser.add_argument(
        "--mode", choices=("view", "make", "both"), default="view",
        help="view: open USD in viewer; make: run Isaac Lab env and step physics; both: run make then view.",
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Number of env.step() calls in 'make' mode.",
    )
    parser.add_argument(
        "--keep-alive", action="store_true",
        help="After stepping, keep the env running so the robot stays visible in the viewer.",
    )
    args, _ = parser.parse_known_args()

    # One factory for the whole run — otherwise view_scene() can't find
    # env_name in its local _registered dict (gym's registry is global,
    # EnvFactory's _registered is per-instance).
    from isaaclab_arena.env_gen.env_factory import EnvFactory

    task_root = Path("isaaclab_arena/task_gen/generated")
    # One task per scene folder so the index maps to distinct scenes.
    scene_folders = sorted(p for p in task_root.iterdir() if p.is_dir())
    tasks_by_scene = [sorted(folder.glob("*.json"))[0] for folder in scene_folders
                      if sorted(folder.glob("*.json"))]
    if not tasks_by_scene and args.json is None:
        print(f"No task JSONs found under {task_root}. Run test_task_generator.py first.")
        raise SystemExit(0)

    print(f"Available scenes ({len(tasks_by_scene)}):")
    for i, p in enumerate(tasks_by_scene):
        marker = " <-" if (args.json is None and i == args.scene_index) else ""
        print(f"  [{i}] {p.parent.name}/{p.name}{marker}")

    if args.json is not None:
        json_path = args.json
    else:
        if not 0 <= args.scene_index < len(tasks_by_scene):
            print(f"--scene-index {args.scene_index} out of range [0, {len(tasks_by_scene) - 1}]")
            raise SystemExit(2)
        json_path = str(tasks_by_scene[args.scene_index])

    factory = EnvFactory(scene_usd_dir="isaaclab_arena/scene_gen/tmp", background= "empty_warehouse")
    env_name = test_register_single(factory, json_path)

    if not env_name:
        raise SystemExit(1)

    if args.mode in ("make", "both"):
        test_make_env(factory, env_name, steps=args.steps, keep_alive=args.keep_alive)

    if args.mode in ("view", "both"):
        factory.view_scene(env_name)
