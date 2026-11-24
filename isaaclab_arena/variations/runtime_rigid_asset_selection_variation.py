# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Episode-reset selection of preloaded Newton rigid-body variants."""

from __future__ import annotations

import hashlib
import json
import torch
from functools import partial
from typing import TYPE_CHECKING

import warp as wp
from isaaclab.managers import EventTermCfg
from isaaclab.sim import MultiAssetSpawnerCfg, UsdFileCfg
from isaaclab.sim.utils import standardize_xform_ops

from isaaclab_arena.assets.spawn_cfg_geometry import SpawnCfgGeometry
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.usd.rigid_body_path_alignment import align_rigid_body_paths
from isaaclab_arena.variations.rigid_asset_selection_variation import (
    RigidAssetSelectionVariationCfg,
    _assert_registered_rigid_object,
    _build_rigid_object,
)
from isaaclab_arena.variations.sequential_sampler import SequentialSampler
from isaaclab_arena.variations.variation_base import RunTimeVariationBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg

    from isaaclab_arena.assets.object import Object


class RuntimeRigidAssetSelectionVariation(RunTimeVariationBase):
    """Keep a scene role fixed while selecting its physical asset independently in each environment."""

    def __init__(self, rigid_object: Object, cfg: RigidAssetSelectionVariationCfg, num_envs: int) -> None:
        super().__init__(cfg, "asset_selection")
        self.object = rigid_object
        self.num_envs = num_envs
        self.candidates: dict[str, Object] = {}
        self.fingerprints: dict[str, str] = {}
        self.selected_names: list[str] = []
        self._pending_initial: set[int] = set()
        self._next_index = 0
        self._body_bounds: dict[str, AxisAlignedBoundingBox] = {}
        self._device_bounds = None
        self._selected_ids = None
        self._replay_initial_names = None
        self._validated_runtime = False

    def set_initial_replay_names(self, names: list[str]) -> None:
        """Reserve the first replay rows without advancing the episode scheduler."""
        self._replay_initial_names = names

    def _prepare_at_build_time(self) -> None:
        assert self.cfg.registry_names, "Reset asset selection requires registry_names"
        assert self.object.bounding_box is None, "Reset asset selection cannot use a shared bounding_box override"
        names = list(dict.fromkeys(self.cfg.registry_names))
        for name in names:
            _assert_registered_rigid_object(name)
            candidate = _build_rigid_object(name)
            self.candidates[name] = candidate
            self.fingerprints[name] = spawn_fingerprint(candidate.object_cfg.spawn)
        configs = align_rigid_body_paths([candidate.object_cfg.spawn for candidate in self.candidates.values()])
        for candidate, config in zip(self.candidates.values(), configs):
            candidate.object_cfg.spawn = config
        for name, candidate in self.candidates.items():
            self._body_bounds[name] = _body_bounds(candidate.object_cfg.spawn)
        self.object.object_cfg.spawn = MultiAssetSpawnerCfg(
            assets_cfg=configs, static=False, random_choice=False, activate_contact_sensors=None
        )
        self.object.runtime_asset_selection = self
        # Draw once for construction; notify recorders only when that episode actually starts.
        if self._replay_initial_names is None:
            preview_sampler = self.cfg.sampler_cfg.build()
            preview_sampler.set_generator(self._generator)
            self.selected_names = self._draw(preview_sampler, self.num_envs, None)
        else:
            self.selected_names = list(self._replay_initial_names)
            assert all(name in self.candidates for name in self.selected_names), "Unknown replay asset"
        self._pending_initial = set(range(self.num_envs))

    def _draw(self, sampler, count: int, env_ids) -> list[str]:
        kwargs = {"start_index": self._next_index} if isinstance(sampler, SequentialSampler) else {}
        names = sampler.sample(count, self.cfg.registry_names, env_ids=env_ids, **kwargs)
        self._next_index += count
        return names

    def select(self, env_ids: torch.Tensor) -> list[str]:
        """Select names for absolute, possibly reordered environment IDs; leave other slots unchanged."""
        ids = env_ids.tolist()
        if self._replay_sampler is not None:
            names = self._draw(self.sampler, len(ids), env_ids)
        else:
            # Preserve construction selections even when the first reset is partial or reordered.
            fresh_ids = [env_id for env_id in ids if env_id not in self._pending_initial]
            fresh_sampler = self.cfg.sampler_cfg.build()
            fresh_sampler.set_generator(self._generator)
            fresh_names = self._draw(fresh_sampler, len(fresh_ids), None)
            fresh = dict(zip(fresh_ids, fresh_names))
            names = [
                self.selected_names[env_id] if env_id in self._pending_initial else fresh[env_id] for env_id in ids
            ]
            self.sampler._notify(names, env_ids)
        for env_id, name in zip(ids, names):
            assert name in self.candidates, f"Unknown selected asset {name!r}"
            self.selected_names[env_id] = name
            self._pending_initial.discard(env_id)
        return names

    def get_selected_registry_names(self, env_ids=None) -> list[str]:
        """Return selected identities in the requested environment order."""
        ids = range(self.num_envs) if env_ids is None else env_ids
        return [self.selected_names[int(env_id)] for env_id in ids]

    def selected_metadata(self, env_id: int) -> dict[str, str]:
        """Return stable identity and spawn fingerprint for placement recording."""
        name = self.selected_names[env_id]
        return {"registry_name": name, "fingerprint": self.fingerprints[name]}

    def get_body_bounds(self, device) -> AxisAlignedBoundingBox:
        """Gather the currently selected bounds in each rigid body's link frame."""
        if self._device_bounds is None:
            bounds = list(self._body_bounds.values())
            self._device_bounds = (
                torch.cat([box.min_point for box in bounds]).to(device),
                torch.cat([box.max_point for box in bounds]).to(device),
            )
        ids = self._selected_ids
        assert ids is not None, "Asset selection reset must run before querying live bounds"
        lower, upper = self._device_bounds
        return AxisAlignedBoundingBox(lower[ids], upper[ids])

    def body_bounds_for_candidate(self, candidate: Object) -> AxisAlignedBoundingBox:
        """Return body-frame placement bounds for one catalogue member."""
        name = next(name for name, asset in self.candidates.items() if asset is candidate)
        return self._body_bounds[name]

    def apply(self, env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
        """Switch physical variants before pose, placement, and property-reset events."""
        if not self._validated_runtime:
            _validate_rendering(env)
            self._validated_runtime = True
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
        names = self.select(env_ids)
        index_by_name = {name: index for index, name in enumerate(self.candidates)}
        variant_ids = torch.tensor([index_by_name[name] for name in names], device=env.device, dtype=torch.int32)
        ids = env_ids.to(device=env.device, dtype=torch.int32)
        # Keep tensors alive through the Warp call, and order writes on Torch's CUDA stream.
        physics_stream = wp.get_stream(env.device)
        selection_stream = wp.stream_from_torch(torch.cuda.current_stream(env.device))
        selection_stream.wait_stream(physics_stream)
        with wp.ScopedStream(selection_stream):
            env.scene[self.object.name].write_mesh_variant_to_sim(variant_ids, ids)
        physics_stream.wait_stream(selection_stream)
        if self._selected_ids is None:
            self._selected_ids = torch.tensor(
                [index_by_name[name] for name in self.selected_names], device=env.device, dtype=torch.long
            )
        self._selected_ids[env_ids] = variant_ids.long()
        env.arena_world.register_runtime_geometry(self.object.name, self)

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        return f"{self.object.name}_asset_selection", EventTermCfg(
            func=_select_assets, mode="reset", params={"selection": _SelectionHandle(self)}
        )


class _SelectionHandle:
    """Keep the live catalogue shared across Lab's event-config copies."""

    __slots__ = ("variation",)

    def __init__(self, variation):
        self.variation = variation

    def __deepcopy__(self, memo):
        return self


def _select_assets(env, env_ids, selection):
    selection.variation.apply(env, env_ids)


def _validate_rendering(env):
    """Reject renderers that would keep showing the old variant after a physical switch."""
    for sensor in env.scene.sensors.values():
        renderer = getattr(sensor.cfg, "renderer_cfg", None)
        if renderer is not None:
            assert renderer.renderer_type == "newton_warp", (
                "Reset asset selection camera observations require NewtonWarpRendererCfg "
                f"(renderer_type=newton_warp); got {renderer.renderer_type}"
            )
    configs = env.sim.cfg.visualizer_cfgs
    if not isinstance(configs, (list, tuple)):
        configs = [configs]
    for config in configs:
        backends = getattr(config, "backend", [])
        if not isinstance(backends, (list, tuple)):
            backends = [backends]
        for backend in backends:
            kind = getattr(backend, "visualizer_type", None)
            assert kind in (
                None,
                "newton",
                "viser",
            ), "Reset asset selection requires --viz none, newton, or viser; Kit does not refresh switched meshes"
            assert getattr(backend, "initial_renderer", "opengl") != "rtx" and not getattr(
                backend, "rtx_camera", False
            ), "Reset asset selection requires the Newton GL viewer without an RTX camera"


def spawn_fingerprint(spawn_cfg: SpawnerCfg) -> str:
    """Hash spawn settings including source, scale, USD variants, and physics properties."""

    def json_value(value):
        if isinstance(value, dict):
            return {key: json_value(item) for key, item in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [json_value(item) for item in value]
        if isinstance(value, partial):
            return {
                "function": json_value(value.func),
                "args": json_value(value.args),
                "keywords": json_value(value.keywords),
            }
        if callable(value):
            return f"{value.__module__}.{value.__qualname__}"
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    payload = json.dumps(json_value(spawn_cfg.to_dict()), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _body_bounds(spawn_cfg: SpawnerCfg) -> AxisAlignedBoundingBox:
    geometry = SpawnCfgGeometry(spawn_cfg)
    if not isinstance(spawn_cfg, UsdFileCfg):
        return geometry.get_bounding_box()
    from isaaclab_arena.environments.arena_world_scene_access import _compute_geometry_bounds_in_prim_frame

    stage = geometry.open_usd_stage()
    root = stage.GetDefaultPrim()
    standardize_xform_ops(root, translation=(0, 0, 0), orientation=(0, 0, 0, 1), scale=spawn_cfg.scale)
    body = stage.GetPrimAtPath(str(root.GetPath()) + geometry.get_contact_body_path())
    return _compute_geometry_bounds_in_prim_frame(body)
