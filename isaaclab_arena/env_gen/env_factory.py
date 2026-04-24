"""Environment factory: GoalSpec JSON → registered gymnasium environment.

Scene loading: RoboLab's import_scene pattern (spawn=None + full USD as sublayer).
Positions read directly from scene USD — exact poses from scene gen.

Usage:
    factory = EnvFactory(scene_usd_dir="isaaclab_arena/scene_gen/tmp")
    env_name = factory.register_from_json("task.json")
    env = factory.make_env(env_name)
"""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.env_gen.embodiment import (
    get_franka_actions_cfg,
    get_franka_events_cfg,
    get_franka_observations_cfg,
    get_franka_scene_entities,
)
from isaaclab_arena.env_gen.predicate_map import build_terminations_from_goal_xyz
from isaaclab_arena.env_gen.predicate_to_xyz import resolve_goal_xyz
from isaaclab_arena.env_gen.scene_loader import _find_scene_usd, load_scene_from_goalspec
from isaaclab_arena.task_gen.goal_spec import GoalSpec


def _table_pose_for_scene(
    scene_usd_name: str, scene_dir: str,
) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    """Return (table_pose, table_dims) for a scene, from its metadata JSON.

    Scene metadata (written by scene_gen) records which table asset was used.
    Office_table vs bamboo/black have different world poses and dims, so the
    goal-xyz solver needs the right anchor. If metadata is missing, returns
    (None, None) → caller uses resolve_goal_xyz's defaults.
    """
    import json as _json
    stem = scene_usd_name.replace(".usda", "")
    meta = Path(scene_dir) / f"{stem}_metadata.json"
    if not meta.exists():
        return None, None
    table_name = _json.loads(meta.read_text()).get("table", "")
    if table_name == "office_table_background":
        # After 90° Z-rot + scale(0.7, 1, 0.9195): world dims (0.802, 1.26, 0.697)
        return (0.55, 0.0, -0.697), (0.802, 1.26, 0.697)
    return (0.547, 0.0, -0.35), (0.7, 1.0, 0.35)


class EnvFactory:
    """Register GoalSpec JSONs as gymnasium environments."""

    # Arena-hosted HDR maps (see `isaaclab_arena/assets/hdr_image_library.py`).
    # Used as dome-light texture + skybox so the viewport shows a realistic
    # indoor backdrop instead of an unlit white void.
    _HDR_BACKGROUNDS = {
        "home_office":       "srl_robolab_assets/backgrounds/default/home_office.exr",
        "empty_warehouse":   "srl_robolab_assets/backgrounds/default/empty_warehouse.hdr",
        "brown_photostudio": "srl_robolab_assets/backgrounds/default/brown_photostudio.hdr",
    }

    def __init__(
        self,
        scene_usd_dir: str | None = None,
        env_prefix: str = "Arena-",
        randomize_poses: bool = False,
        pose_noise: tuple = (0.02, 0.02, 0.0),
        add_embodiment: bool = True,
        enable_cameras: bool = True,
        background: str | None = "home_office",
        action_type: str = "joint_pos",
    ):
        """
        Args:
            scene_usd_dir: Where scene USD files live.
            env_prefix: Prefix for gym env names.
            randomize_poses: If True, add events to randomize object poses on reset
                             (RoboLab-style randomization on top of USD positions).
            pose_noise: (x, y, z) half-range in meters for pose randomization.
            add_embodiment: Add a Franka + EE frame + joint-position action + proprio
                            observations so make_env() returns a playable env. When
                            False, observations/actions are empty (scene-only).
            enable_cameras: Spawn wrist + external cameras on the robot. Irrelevant
                            if add_embodiment=False.
            background: Name of an Arena HDR backdrop to wrap the scene in
                        (see `_HDR_BACKGROUNDS` for available options, or pass
                        `None` to keep the plain gray dome light from
                        `scene_loader`).
            action_type: "joint_pos" (8-dim, default) or "ik_delta" (7-dim
                        delta-pose; required by the CuRobo runner).
        """
        self.scene_usd_dir = scene_usd_dir
        self.env_prefix = env_prefix
        self.randomize_poses = randomize_poses
        self.pose_noise = pose_noise
        self.add_embodiment = add_embodiment
        self.enable_cameras = enable_cameras
        self.background = background
        self.action_type = action_type
        self._registered: dict[str, GoalSpec] = {}

    def register_from_json(self, json_path: str, env_name: str | None = None) -> str:
        """Register a GoalSpec JSON as a gymnasium environment."""
        import gymnasium as gym
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab_arena.environments.isaaclab_arena_manager_based_env import (
            IsaacLabArenaManagerBasedRLEnvCfg,
        )
        from isaaclab_arena.utils.configclass import make_configclass

        goal_spec = GoalSpec.from_json(json_path)
        if env_name is None:
            env_name = f"{self.env_prefix}{goal_spec.task_name}-v0"

        # 1. Scene: RoboLab import_scene pattern (spawn=None + full USD sublayer)
        scene_cfgs = load_scene_from_goalspec(
            goal_spec,
            scene_usd_dir=self.scene_usd_dir or os.path.dirname(json_path),
        )
        # Replace the plain gray dome light with an HDR backdrop so the
        # viewport shows a real indoor environment (RoboLab pattern —
        # see `robolab/variations/backgrounds.py`).
        if self.background is not None:
            self._apply_hdr_background(scene_cfgs, self.background)
        # Embodiment: robot + ee_frame (+ cameras) as additional scene fields.
        # Names here must match the asset_name used by the action / observation
        # cfgs (`robot`, `ee_frame`). Appended after the scene objects so the
        # scene_positions extraction below still yields only USD object poses.
        if self.add_embodiment:
            scene_cfgs.update(get_franka_scene_entities(include_cameras=self.enable_cameras))
        scene_fields = [(name, type(cfg), cfg) for name, cfg in scene_cfgs.items()]
        SceneCfg = make_configclass("SceneCfg", scene_fields, bases=(InteractiveSceneCfg,))
        scene_cfg = SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)

        # 2. Resolve goal_relations → xyz, then build one generic termination
        # (all objects at goal AND at rest) + time_out. Avoids per-predicate
        # termination code — any LLM predicate supported by resolve_goal_xyz
        # works without new wiring here.
        table_pose, table_dims = _table_pose_for_scene(
            goal_spec.scene, self.scene_usd_dir or "",
        )
        # Pull every scene object's init position from scene_cfgs so the
        # resolver can anchor targets that task_gen filtered out of
        # goal_spec.initial_state (e.g. desk_caddy_001 used only as target).
        scene_positions: dict[str, tuple[float, float, float]] = {}
        for name, cfg in scene_cfgs.items():
            init_state = getattr(cfg, "init_state", None)
            pos = getattr(init_state, "pos", None) if init_state else None
            if pos is not None:
                scene_positions[name] = tuple(pos)

        resolve_kwargs = {"scene_positions": scene_positions}
        if table_pose is not None:
            resolve_kwargs["table_pose"] = table_pose
            resolve_kwargs["table_dims"] = table_dims
        goal_xyz = resolve_goal_xyz(goal_spec, **resolve_kwargs)
        term_map = build_terminations_from_goal_xyz(goal_xyz)
        term_fields = [(name, TerminationTermCfg, term) for name, term in term_map.items()]
        TerminationCfg = make_configclass("TerminationCfg", term_fields)

        # 3. Events: optional pose randomization on reset (RoboLab-style)
        # + Franka reset-to-home event when the embodiment is attached.
        event_fields = []
        if self.randomize_poses:
            dx, dy, dz = self.pose_noise
            from isaaclab.assets import RigidObjectCfg
            for name, cfg in scene_cfgs.items():
                if not isinstance(cfg, RigidObjectCfg):
                    continue
                event_fields.append((
                    f"randomize_{name}",
                    EventTermCfg,
                    EventTermCfg(
                        func=mdp.reset_root_state_uniform,
                        mode="reset",
                        params={
                            "pose_range": {"x": (-dx, dx), "y": (-dy, dy), "z": (-dz, dz)},
                            "velocity_range": {},
                            "asset_cfg": SceneEntityCfg(name),
                        },
                    ),
                ))
        if self.add_embodiment:
            import dataclasses as _dc
            franka_events = get_franka_events_cfg()
            for field in _dc.fields(franka_events):
                event_fields.append((
                    field.name,
                    EventTermCfg,
                    getattr(franka_events, field.name),
                ))
        EventsCfg = make_configclass("EventsCfg", event_fields)

        # 4. Observations / actions. With the Franka embodiment attached we get
        # a proprio ObsGroup + joint-position arm action + binary gripper. With
        # `add_embodiment=False` we fall back to empty managers (scene-only).
        if self.add_embodiment:
            observations_cfg = get_franka_observations_cfg()
            actions_cfg = get_franka_actions_cfg(self.action_type)
        else:
            ObsCfg = make_configclass("ObservationCfg", [])
            ActionsCfg = make_configclass("ActionsCfg", [])
            observations_cfg = ObsCfg()
            actions_cfg = ActionsCfg()

        env_cfg = IsaacLabArenaManagerBasedRLEnvCfg(
            scene=scene_cfg,
            observations=observations_cfg,
            actions=actions_cfg,
            events=EventsCfg(),
            terminations=TerminationCfg(),
            episode_length_s=goal_spec.episode_length_s,
        )
        # Pulled-back tabletop view: far enough to frame robot + table + scene
        # objects, close enough to actually see them. Isaac Lab's default is
        # (7.5, 7.5, 7.5) which is ~13 m out — way too distant.
        env_cfg.viewer.eye = (2.2, 1.6, 1.2)
        env_cfg.viewer.lookat = (0.3, 0.0, 0.2)

        gym.register(
            id=env_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": env_cfg},
            disable_env_checker=True,
        )

        self._registered[env_name] = goal_spec
        print(f"[EnvFactory] Registered: {env_name} ({goal_spec.instruction})")
        return env_name

    def _apply_hdr_background(self, scene_cfgs: dict, name: str) -> None:
        """Swap the default plain dome light for an HDR-textured one.

        The scene_loader registers a gray `DomeLightCfg(color=(0.75,...))` under
        the key `light`. We replace it in-place with an HDR-textured
        DomeLightCfg (Arena staging Nucleus path from
        `hdr_image_library.py`). `visible_in_primary_ray=True` makes the HDR
        visible as a skybox in the viewport, not just as illumination.
        """
        from isaaclab.assets import AssetBaseCfg
        import isaaclab.sim as sim_utils

        from isaaclab_arena.assets.object_library import ISAACLAB_STAGING_NUCLEUS_DIR

        suffix = self._HDR_BACKGROUNDS.get(name)
        if suffix is None:
            print(f"[EnvFactory] Unknown background '{name}'; keeping plain dome light.")
            return
        texture_url = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/{suffix}"
        scene_cfgs["light"] = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(
                texture_file=texture_url,
                texture_format="latlong",
                intensity=500.0,
                visible_in_primary_ray=True,
            ),
        )

    def make_env(self, env_name: str, device: str = "cuda:0", num_envs: int = 1):
        """Instantiate a registered environment."""
        import gymnasium as gym
        from isaaclab_tasks.utils import parse_env_cfg

        cfg = parse_env_cfg(env_name, device=device, num_envs=num_envs)
        return gym.make(env_name, cfg=cfg).unwrapped

    def view_scene(self, env_name: str):
        """Open the scene USD directly for visual inspection.

        Bypasses Isaac Lab env management — just loads the exact scene
        USD that was generated by scene gen. Use this to verify scenes
        look correct before running full env.
        """
        import omni.usd
        import omni.kit.app

        goal_spec = self._registered.get(env_name)
        if not goal_spec:
            print(f"[EnvFactory] Environment {env_name} not registered")
            return

        from isaaclab_arena.env_gen.scene_loader import _find_scene_usd
        scene_usd = _find_scene_usd(goal_spec, self.scene_usd_dir or "")
        if not scene_usd:
            print(f"[EnvFactory] Scene USD not found for {env_name}")
            return

        print(f"[EnvFactory] Opening scene: {scene_usd}")
        print(f"[EnvFactory] Task: {goal_spec.instruction}")
        omni.usd.get_context().open_stage(scene_usd)
        for _ in range(60):
            omni.kit.app.get_app().update()

        # Start physics so objects settle onto the table
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        print(f"[EnvFactory] Physics running. Objects settling...")
        for _ in range(300):  # ~5 seconds at 60Hz
            omni.kit.app.get_app().update()

        print(f"[EnvFactory] Scene loaded. Close window to exit.")
        while True:
            omni.kit.app.get_app().update()

    def register_all_from_folder(self, task_folder: str) -> list[str]:
        """Register all GoalSpec JSONs in a folder (recursive)."""
        task_folder = Path(task_folder)
        json_files = sorted(task_folder.rglob("*.json"))

        print(f"[EnvFactory] Found {len(json_files)} task JSONs in {task_folder}")

        registered = []
        for json_file in json_files:
            try:
                registered.append(self.register_from_json(str(json_file)))
            except Exception as e:
                print(f"[EnvFactory] Failed to register {json_file.name}: {e}")

        print(f"[EnvFactory] Registered {len(registered)}/{len(json_files)} environments")
        return registered

    def get_registered(self) -> dict[str, GoalSpec]:
        """Get all registered environment names and their GoalSpecs."""
        return dict(self._registered)
