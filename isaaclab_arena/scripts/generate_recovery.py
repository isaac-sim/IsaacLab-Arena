# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate recovery demonstrations by resuming failed episodes from where they went wrong.

For each reset point chosen by ``tools/reset_points.py``, restore the simulator to the recorded state
and hand control to a scripted expert. The result is a *correction*: the policy's own trajectory up
to the moment it still had the task, followed by a demonstration of what to do next.

This is the simulation form of RaC (arXiv:2509.07953). On hardware, recovery data requires human
teleoperated interventions, which is what caps its scale; in simulation a restorable state plus an
expert makes it free.

Usage (from the Arena repo root, with the venv active)::

    python isaaclab_arena/scripts/generate_recovery.py \
        --reset_points <reset_points.json> \
        --source_hdf5 <corpus hdf5> \
        --output_base_dir <out> \
        --embodiment droid_differential_ik \
        [--max_recovery_steps 300]

Note the embodiment: the scripted expert emits delta-pose commands, so it runs on
``droid_differential_ik``. The evaluation embodiment is ``droid_abs_joint_pos``. That difference is
safe here because the scene, robot and cameras are identical — only the action term differs — and
imitation labels are taken from recorded joint positions rather than from the raw actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset_points", type=Path, required=True)
    parser.add_argument("--source_hdf5", type=Path, required=True)
    parser.add_argument("--output_base_dir", type=str, required=True)
    parser.add_argument("--embodiment", type=str, default="droid_differential_ik")
    parser.add_argument("--pick_up_object", type=str, default="sugar_box_ycb_robolab")
    parser.add_argument("--destination_location", type=str, default="bowl_ycb_robolab")
    parser.add_argument("--hdr", type=str, default="brown_photostudio_robolab")
    parser.add_argument("--max_recovery_steps", type=int, default=300)
    parser.add_argument("--record_camera_video", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None, help="only generate this many recoveries")
    return parser


def main() -> None:
    args = _cli().parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": True, "enable_cameras": True})
    simulation_app = app_launcher.app

    # Imports deferred until the SimulationApp exists — Isaac Sim requires it before isaaclab.
    import h5py
    import numpy as np
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder, ArenaEnvBuilderCfg
    from isaaclab_arena.recording.sim_state_terms import unflatten_state
    from isaaclab_arena.policy.scripted_pick_place_policy import (
        ScriptedPickPlacePolicy,
        ScriptedPickPlacePolicyCfg,
    )
    from isaaclab_arena_environments.etl_pnp_maple_table_environment import (
        EtlPnpMapleTableEnvironment,
        EtlPnpMapleTableEnvironmentCfg,
    )

    points = json.loads(Path(args.reset_points).read_text())
    if args.limit:
        points = points[: args.limit]
    layout_path = Path(args.source_hdf5).parent / "sim_state_layout.json"
    if not layout_path.exists():
        raise SystemExit(
            f"missing {layout_path}: the flat sim_state vector cannot be sliced back into the nested "
            "dict reset_to requires. Re-record with the sim_state term enabled."
        )
    layout: dict[str, int] = json.loads(layout_path.read_text())

    env_cfg = EtlPnpMapleTableEnvironmentCfg(
        embodiment=args.embodiment,
        enable_cameras=True,
        pick_up_object=args.pick_up_object,
        destination_location=args.destination_location,
        hdr=args.hdr,
    )
    arena_env = EtlPnpMapleTableEnvironment().build(env_cfg)
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg())
    env = builder.make_registered(render_mode="rgb_array" if args.record_camera_video else None)

    output_dir = Path(args.output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env.unwrapped.episode_recorder.set_job_name("recovery")
    env.unwrapped.episode_recorder.set_output_path(str(output_dir / "episode_results_rank0.jsonl"))

    policy = ScriptedPickPlacePolicy(
        ScriptedPickPlacePolicyCfg(
            object_name=args.pick_up_object,
            destination_name=args.destination_location,
        )
    )

    device = torch.device(env.unwrapped.device)
    keys = sorted(layout)
    widths = [layout[k] for k in keys]
    offsets = np.cumsum([0] + widths)

    def restore(flat_vector: np.ndarray) -> None:
        """Slice the flat state vector back into the nested dict and apply it."""
        nested = unflatten_state(
            {
                key: torch.as_tensor(flat_vector[offsets[i] : offsets[i + 1]], device=device).unsqueeze(0)
                for i, key in enumerate(keys)
            }
        )
        env.unwrapped.reset_to(nested, env_ids=None, is_relative=True)

    manifest = []
    with h5py.File(args.source_hdf5, "r") as h:
        for rp in points:
            grp = h["data"].get(rp["demo"])
            if grp is None or "sim_state" not in grp:
                print(f"  {rp['demo']}: no sim_state, skipping")
                continue
            states = np.asarray(grp["sim_state"])
            idx = min(rp["capture_index"], states.shape[0] - 1)

            env.reset()
            restore(states[idx])
            policy.reset()

            obs = env.unwrapped.obs_buf
            steps = 0
            succeeded = False
            while steps < args.max_recovery_steps:
                action = policy.get_action(env, obs)
                obs, _, terminated, truncated, _ = env.step(action)
                steps += 1
                done = bool(terminated[0].item() or truncated[0].item())
                if done:
                    succeeded = bool(terminated[0].item())
                    break

            manifest.append(
                {
                    "source_demo": rp["demo"],
                    "source_episode": rp["episode"],
                    "stalled_at": rp["furthest_stage"],
                    "reset_step": rp["reset_step"],
                    "recovery_steps": steps,
                    "recovered": succeeded,
                }
            )
            print(f"  {rp['demo']:10s} from step {rp['reset_step']:4d} ({rp['furthest_stage']:15s}) "
                  f"-> {steps:3d} steps, recovered={succeeded}")

    (output_dir / "recovery_manifest.json").write_text(json.dumps(manifest, indent=1))
    n_ok = sum(m["recovered"] for m in manifest)
    print(f"\n{n_ok}/{len(manifest)} recoveries succeeded -> {output_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
