# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import dataclasses
import os
import torch
import tqdm
from gymnasium.wrappers import RecordVideo
from importlib import import_module
from typing import TYPE_CHECKING, Any

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.camera_video import CameraObsVideoRecorder
from isaaclab_arena.evaluation.episode_outcome import classify_outcome
from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments
from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.utils.multiprocess import get_local_rank, get_world_size
from isaaclab_arena.utils.random import set_seed
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

if TYPE_CHECKING:
    from isaaclab_arena.policy.policy_base import PolicyBase


def get_policy_cls(policy_type: str) -> type["PolicyBase"]:
    """Get the policy class for the given policy type name.

    Note that this function:
    - first: checks for a registered policy type in the PolicyRegistry
    - if not found, it tries to dynamically import the policy class, treating
      the policy_type argument as a string representing the module path and class name.

    """
    from isaaclab_arena.assets.registries import PolicyRegistry

    policy_registry = PolicyRegistry()
    if policy_registry.is_registered(policy_type):
        return policy_registry.get_policy(policy_type)
    else:
        print(f"Policy {policy_type} is not registered. Dynamically importing from path: {policy_type}")
        assert "." in policy_type, (
            "policy_type must be a dotted Python import path of the form 'module.submodule.ClassName', got:"
            f" {policy_type}"
        )
        # Dynamically import the class from the string path
        module_path, class_name = policy_type.rsplit(".", 1)
        module = import_module(module_path)
        policy_cls = getattr(module, class_name)
        return policy_cls


def is_distributed(args_cli: argparse.Namespace) -> bool:
    return (
        "cuda" in args_cli.device and hasattr(args_cli, "distributed") and args_cli.distributed and get_world_size() > 1
    )


def prepare_env_cfg_for_datagen(env_cfg) -> list:
    """Prepare *env_cfg* for datagen collection. Call on the cfg *before* the env is built.

    Two changes, both so the dedicated datagen cameras capture clean frames:

    1. Remove the termination terms so the env never auto-resets inside ``step()``. Isaac
       Lab resets a done env *within* ``step()`` and re-renders before the new scene is
       flushed, so a dedicated camera reads back the previous episode's final frame as the
       first frame of the next episode. The rollout loop instead evaluates the returned
       terms manually and drives a clean, explicit ``env.reset()`` between episodes (which
       flushes to the renderer before re-rendering). Mirrors
       ``submodules/IsaacLab/scripts/tools/record_demos.py``.
    2. Drop the metrics and their recorder terms. The datagen collector writes its own
       per-episode HDF5, and the success-rate recorder asserts the (now removed) success
       termination is active, so it must not run.

    Returns:
        The stashed non-timeout terms as ``(name, term)`` pairs (e.g. success, object_dropped)
        for manual evaluation. Timeout terms are replaced by the env's ``max_episode_length``
        cap in the loop.
    """
    # Datagen has its own writer; drop the env's metrics + recorder terms (the success
    # recorder also depends on the success termination removed below). recorders=None makes
    # the RecorderManager a no-op.
    if hasattr(env_cfg, "metrics"):
        env_cfg.metrics = None
    if hasattr(env_cfg, "recorders"):
        env_cfg.recorders = None

    terminations = getattr(env_cfg, "terminations", None)
    if terminations is None:
        return []
    stashed = []
    for field in dataclasses.fields(terminations):
        term = getattr(terminations, field.name)
        # Skip unset/empty fields; real termination terms are TerminationTermCfg (have .func).
        if term is None or not hasattr(term, "func"):
            continue
        setattr(terminations, field.name, None)
        is_timeout = field.name == "time_out" or getattr(term, "time_out", False)
        if not is_timeout:
            stashed.append((field.name, term))
    return stashed


def _manual_episode_done(env, reset_terms: list) -> str | None:
    """Return the name of the first stashed termination term that fires, else ``None``."""
    base_env = env.unwrapped
    for name, term in reset_terms:
        result = term.func(base_env, **(term.params or {}))
        if bool(torch.as_tensor(result).any()):
            return name
    return None


def _run_datagen_rollout(
    env, policy, collector, pbar, num_steps, num_episodes, reset_terms, max_episode_length, obs
) -> None:
    """Rollout loop for datagen collection with the env's auto-reset disabled.

    Records every (settled) frame, decides episode end from the stashed termination terms
    plus a max-length cap, then flushes the episode and performs a clean explicit
    ``env.reset()`` before the next one so the first frame of each episode is correct.
    """
    assert max_episode_length is not None, "datagen rollout requires max_episode_length"
    num_episodes_completed = 0
    num_steps_completed = 0
    steps_in_episode = 0
    while True:
        with torch.inference_mode():
            actions = policy.get_action(env, obs)
            obs, _, _, _, _ = env.step(actions)
        steps_in_episode += 1
        collector.on_step(env, obs, actions, num_steps_completed)
        num_steps_completed += 1

        if num_steps is not None:
            pbar.update(1)
            if num_steps_completed >= num_steps:
                break

        ended_by = _manual_episode_done(env, reset_terms)
        hit_cap = steps_in_episode >= max_episode_length
        if ended_by is not None or hit_cap:
            collector.end_episode(env, outcome=classify_outcome(ended_by))
            num_episodes_completed += 1
            if num_episodes is not None:
                pbar.update(1)
                if num_episodes_completed >= num_episodes:
                    break
            obs, _ = env.reset()
            policy.reset()
            steps_in_episode = 0


def rollout_policy(
    env,
    policy: "PolicyBase",
    num_steps: int | None,
    num_episodes: int | None,
    language_instruction: str | None = None,
    collector: Any = None,
    datagen_reset_terms: list | None = None,
    max_episode_length: int | None = None,
) -> dict[str, Any]:
    """Roll out *policy* in *env*.

    Args:
        collector: Optional data collector. When provided, ``collector.on_step(env,
            obs, actions, step_idx)`` is called after every environment step and
            ``collector.finalize(env)`` once the rollout finishes. Duck-typed so
            core does not depend on the datagen package (see
            ``isaaclab_arena_datagen.collection.collector.DatagenCollector``).
        datagen_reset_terms: Stashed termination terms returned by
            :func:`prepare_env_cfg_for_datagen`. Required whenever a collector is
            given: with the env's auto-reset disabled, the loop evaluates these terms
            (plus ``max_episode_length``) to end episodes and resets explicitly, so no
            frame is captured mid-reset.
        max_episode_length: Per-episode step cap (replaces the removed timeout term).
            Required when ``datagen_reset_terms`` is given.
    """
    assert num_steps is not None or num_episodes is not None, "Either num_steps or num_episodes must be provided"
    assert num_steps is None or num_episodes is None, "Only one of num_steps or num_episodes must be provided"
    assert collector is None or datagen_reset_terms is not None, (
        "A datagen collector requires the env's auto-reset to be disabled; pass datagen_reset_terms"
        " from prepare_env_cfg_for_datagen() (and max_episode_length)."
    )

    pbar = None
    try:
        obs, _ = env.reset()
        policy.reset()
        # Determine language instruction: CLI/job-level override takes precedence over the task's own
        # description. Use unwrapped to reach the base env through any gym wrappers (e.g. OrderEnforcing).
        task_description = language_instruction or env.unwrapped.cfg.isaaclab_arena_env.task.get_task_description()
        policy.set_task_description(task_description)

        # Setup progress bar based on num_steps or num_episodes
        if num_steps is not None:
            pbar = tqdm.tqdm(total=num_steps, desc="Steps", unit="step")
        else:
            pbar = tqdm.tqdm(total=num_episodes, desc="Episodes", unit="episode")

        if collector is not None:
            # Datagen path: env auto-reset is disabled, so we drive episode boundaries
            # and resets explicitly (see _run_datagen_rollout / record_demos.py).
            _run_datagen_rollout(
                env, policy, collector, pbar, num_steps, num_episodes, datagen_reset_terms, max_episode_length, obs
            )
        else:
            num_episodes_completed = 0
            num_steps_completed = 0

            while True:
                with torch.inference_mode():
                    actions = policy.get_action(env, obs)
                    obs, _, terminated, truncated, _ = env.step(actions)

                    if terminated.any() or truncated.any():
                        # Only reset policy for those envs that are terminated or truncated
                        print(
                            f"Resetting policy for terminated env_ids: {terminated.nonzero().flatten()}"
                            f" and truncated env_ids: {truncated.nonzero().flatten()}"
                        )
                        env_ids = (terminated | truncated).nonzero().flatten()
                        policy.reset(env_ids=env_ids)
                        # Break if number of episodes is reached
                        completed_episodes = env_ids.shape[0]
                        num_episodes_completed += completed_episodes
                        if hasattr(env.unwrapped.cfg, "metrics") and env.unwrapped.cfg.metrics is not None:
                            metrics = env.unwrapped.compute_metrics()
                            tqdm.tqdm.write(
                                f"[Rank {get_local_rank()}/{get_world_size()}] Metrics:"
                                f" {metrics_to_plain_python_types(metrics)}"
                            )
                        if num_episodes is not None:
                            pbar.update(completed_episodes)
                            if num_episodes_completed >= num_episodes:
                                break
                    # Break if number of steps is reached
                    num_steps_completed += 1
                    if num_steps is not None:
                        pbar.update(1)
                        if num_steps_completed >= num_steps:
                            break

        pbar.close()

    except Exception as e:
        if pbar is not None:
            pbar.close()
        raise RuntimeError(f"Error rolling out policy: {e}")

    else:

        # Persist and close any datagen dataset before returning.
        if collector is not None:
            collector.finalize(env)

        # Only compute metrics if env has non-None metrics.
        # Use unwrapped to reach the base env through any gym wrappers (e.g. OrderEnforcing)
        if hasattr(env.unwrapped.cfg, "metrics") and env.unwrapped.cfg.metrics is not None:
            return env.unwrapped.compute_metrics()
        return None


def main():
    """Run an IsaacLab Arena environment with a policy.
    Use --distributed with torchrun command for one process per GPU on multi-GPU machines. AppLauncher uses LOCAL_RANK for device.
    """
    args_parser = get_isaaclab_arena_cli_parser()
    # We do this as the parser is shared between the example environment and policy runner
    args_cli, unknown = args_parser.parse_known_args()

    local_rank = get_local_rank()
    world_size = get_world_size()
    # Setting device to local rank before SimulationAppContext
    if is_distributed(args_cli):
        args_cli.device = f"cuda:{local_rank}"
        print(f"[Rank {local_rank}/{world_size}] One Isaac Lab instance per process on cuda:{local_rank}")

    with SimulationAppContext(args_cli):

        # Get the policy-type flag before proceeding to other arguments
        add_policy_runner_arguments(args_parser)
        args_cli, _ = args_parser.parse_known_args()

        # Get the policy class from the policy type
        policy_cls = get_policy_cls(args_cli.policy_type)
        print(
            f"[Rank {local_rank}/{world_size}] Requested policy type: {args_cli.policy_type} -> Policy class:"
            f" {policy_cls}"
        )

        # Add the example environment arguments + policy-related arguments to the parser
        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_parser = policy_cls.add_args_to_parser(args_parser)
        args_cli = args_parser.parse_args()
        # Re-apply per-rank device after parse preventing device got overwritten by the default value
        if is_distributed(args_cli):
            args_cli.distributed = True
            args_cli.device = f"cuda:{local_rank}"

        # Build scene. Use rgb_array render mode when recording so RecordVideo can grab frames.
        arena_builder = get_arena_builder_from_cli(args_cli)
        render_mode = "rgb_array" if args_cli.video else None
        collect_datagen = getattr(args_cli, "collect_datagen", False)
        # For datagen, disable the env's auto-reset before building so we drive episode
        # boundaries and resets explicitly (avoids capturing a frame mid-reset).
        name, cfg = arena_builder.build_registered()
        datagen_reset_terms = prepare_env_cfg_for_datagen(cfg) if collect_datagen else None
        env = arena_builder.make_registered(cfg, render_mode=render_mode)

        # Per-rank seed when distributed so each process has a different seed
        seed = args_cli.seed
        if seed is not None and is_distributed(args_cli):
            seed = seed + local_rank
        if seed is not None:
            set_seed(seed, env)

        # Create the policy from the arguments
        policy = policy_cls.from_args(args_cli)

        # Simulation length.
        if policy.has_length():
            num_steps = policy.length()
            num_episodes = None
        else:
            if args_cli.num_steps is not None:
                num_steps = args_cli.num_steps
                num_episodes = None
                print(f"[Rank {local_rank}/{world_size}] Simulation length: {num_steps} steps")
            elif args_cli.num_episodes is not None:
                num_steps = None
                num_episodes = args_cli.num_episodes
                print(f"[Rank {local_rank}/{world_size}] Simulation length: {num_episodes} episodes")
            else:
                raise ValueError(f"[Rank {local_rank}/{world_size}] Either num_steps or num_episodes must be provided")

        # Optionally wrap with RecordVideo and/or CameraObsVideoRecorder. The two flags
        # are independent: --video records the kit viewport (via env.render()),
        # --camera_video records the embodiment-mounted cameras (from obs["camera_obs"]).
        if args_cli.video or args_cli.camera_video:
            os.makedirs(args_cli.video_dir, exist_ok=True)
            if num_steps is not None:
                video_length = num_steps
            else:
                # When num_episodes is set, capture exactly one episode's worth of frames.
                # max_episode_length is in environment steps, which matches our rollout cadence.
                video_length = num_episodes * env.unwrapped.max_episode_length

        if args_cli.video:
            env = RecordVideo(
                env,
                video_folder=args_cli.video_dir,
                step_trigger=lambda step: step == 0,
                video_length=video_length,
                disable_logger=True,
            )
            print(
                f"[Rank {local_rank}/{world_size}] Recording {video_length}-step viewport video to:"
                f" {args_cli.video_dir}"
            )

        if args_cli.camera_video:
            # Record one mp4 per camera in obs["camera_obs"] (what the policy sees),
            # using the same encoder as RecordVideo.
            env = CameraObsVideoRecorder(
                env,
                video_folder=args_cli.video_dir,
                step_trigger=lambda step: step == 0,
                video_length=video_length,
            )
            print(
                f"[Rank {local_rank}/{world_size}] Recording {video_length}-step per-camera videos to:"
                f" {args_cli.video_dir}"
            )

        # Optionally set up datagen data collection (opt-in via --collect-datagen).
        # Lazy import keeps core decoupled from the isaaclab_arena_datagen package
        # unless collection is actually requested.
        collector = None
        if collect_datagen:
            cameras_enabled = args_cli.enable_cameras or os.environ.get("ENABLE_CAMERAS") == "1"
            assert cameras_enabled, "--collect-datagen requires --enable_cameras or ENABLE_CAMERAS=1."
            from isaaclab_arena_datagen.collection.collector import DatagenCollector, DatagenCollectorConfig

            # Optional explicit camera viewpoint (look-at). When
            # --datagen-camera-position is given it overrides the env's
            # get_default_cameras / the default fallback view; --datagen-camera-target
            # defaults to the world origin if not supplied.
            datagen_cameras = None
            if args_cli.datagen_camera_position is not None:
                from isaaclab_arena_datagen.camera_trajectory import CameraViewTrajectory

                target = (
                    tuple(args_cli.datagen_camera_target)
                    if args_cli.datagen_camera_target is not None
                    else (0.0, 0.0, 0.0)
                )
                datagen_cameras = [
                    CameraViewTrajectory(
                        position=tuple(args_cli.datagen_camera_position),
                        target=target,
                        focal_length_mm=args_cli.datagen_focal_length,
                    )
                ]

            datagen_cfg = DatagenCollectorConfig(
                output_dir=args_cli.datagen_output_dir,
                cameras=datagen_cameras,
                width=args_cli.datagen_width,
                height=args_cli.datagen_height,
                mesh_sample_spacing=args_cli.datagen_mesh_sample_spacing,
            )
            collector = DatagenCollector.from_env(env, datagen_cfg, env_name=args_cli.example_environment)
            print(
                f"[Rank {local_rank}/{world_size}] Collecting datagen data to:"
                f" {args_cli.datagen_output_dir}/episode_NNNN/dataset.h5"
            )

        steps_str = f"{num_steps} steps" if num_steps is not None else f"{num_episodes} episodes"
        print(f"[Rank {local_rank}/{world_size}] Starting rollout ({steps_str})")
        metrics = rollout_policy(
            env,
            policy,
            num_steps,
            num_episodes,
            args_cli.language_instruction,
            collector=collector,
            datagen_reset_terms=datagen_reset_terms,
            max_episode_length=int(env.unwrapped.max_episode_length) if collect_datagen else None,
        )

        if metrics is not None:
            print(f"[Rank {local_rank}/{world_size}] Metrics: {metrics_to_plain_python_types(metrics)}")

        # NOTE(huikang, 2025-12-30)Explicitly clean up the remote policy client / server.
        # Do NOT rely on a __del__ destructor in policy for this, since destructors are
        # triggered implicitly and their execution time (or even whether they run)
        # is not guaranteed, which makes resource cleanup unreliable.
        if policy.is_remote:
            policy.shutdown_remote(kill_server=args_cli.remote_kill_on_exit)

        # Close the environment.
        env.close()


if __name__ == "__main__":
    main()
