# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import dataclasses
import torch
import tqdm
from importlib import import_module
from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.registries import PolicyRegistry
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.policy_runner_cli import (
    add_policy_cli_args,
    add_policy_runner_arguments,
    build_policy_from_cli,
)
from isaaclab_arena.evaluation.episode_outcome import classify_outcome
from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments
from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.utils.multiprocess import get_local_rank, get_world_size
from isaaclab_arena.video.video_recording import VideoRecordingCfg, timestamped_run_dir, wrap_env_for_video
from isaaclab_arena.visualization.report import build_report, serve_until_ctrl_c
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

if TYPE_CHECKING:
    from isaaclab_arena.metrics.metric_data import MetricsDataCollection
    from isaaclab_arena.policy.policy_base import PolicyBase


def get_policy_cls(policy_type: str) -> type[PolicyBase]:
    """Get the policy class for the given policy type name.

    Note that this function:
    - first: checks for a registered policy type in the PolicyRegistry
    - if not found, it tries to dynamically import the policy class, treating
      the policy_type argument as a string representing the module path and class name.

    """
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


def prepare_env_cfg_for_datagen(env_cfg) -> list[tuple[str, Any]]:
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
            # Re-aim the datagen cameras before the reset so the reset's RTX rerenders
            # (num_rerenders_on_reset) flush the new poses; otherwise the next episode's
            # first frame is rendered from the previous layout.
            collector.resample_cameras()
            obs, _ = env.reset()
            policy.reset()
            steps_in_episode = 0


def rollout_policy(
    env,
    policy: PolicyBase,
    num_steps: int | None,
    num_episodes: int | None,
    language_instruction: str | None = None,
    collector: Any = None,
    datagen_reset_terms: list | None = None,
    max_episode_length: int | None = None,
) -> MetricsDataCollection | None:
    """Roll out *policy* in *env*.

    Args:
        collector: Optional data collector. When provided, ``collector.on_step(env,
            obs, actions, step_idx)`` is called after every environment step and
            ``collector.finalize(env)`` once the rollout finishes. Duck-typed so
            core does not depend on any specific datagen collection package.
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
        policy.set_task_description(env.unwrapped.get_language_instruction())

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

    # --record_camera_video requires cameras to be enabled at sim startup, before SimulationAppContext.
    if "--record_camera_video" in unknown:
        args_cli.enable_cameras = True

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

        # Add the example environment arguments and config-derived policy arguments.
        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_parser = add_policy_cli_args(args_parser, policy_cls)
        args_cli, hydra_overrides = args_parser.parse_known_args()
        assert_hydra_overrides(hydra_overrides, args_parser)
        # Re-apply per-rank device after parse preventing device got overwritten by the default value
        if is_distributed(args_cli):
            args_cli.distributed = True
            args_cli.device = f"cuda:{local_rank}"
            # Per-rank seed when distributed so each process has a different seed
            if args_cli.seed is not None:
                args_cli.seed += local_rank

        # Re-apply enable_cameras: the full parse resets it to default False.
        if args_cli.record_camera_video:
            args_cli.enable_cameras = True

        # Build scene. Use rgb_array render mode when recording so the recorders can grab frames.
        arena_builder = get_arena_builder_from_cli(args_cli, hydra_overrides=hydra_overrides)

        if args_cli.list_variations:
            print(arena_builder.get_variations_catalogue_as_string())
            return

        output_dir = timestamped_run_dir(args_cli.output_base_dir)
        video_cfg = VideoRecordingCfg(
            record_viewport_video=args_cli.record_viewport_video,
            record_camera_video=args_cli.record_camera_video,
            video_base_dir=output_dir,
        )
        env = arena_builder.make_registered(render_mode=video_cfg.render_mode)

        # Write per-episode results to disk.
        results_path = os.path.join(output_dir, f"episode_results_rank{local_rank}.jsonl")
        env.unwrapped.episode_recorder.set_job_name("policy_runner")
        env.unwrapped.episode_recorder.set_output_path(results_path)

        # Create the policy through the typed config compatibility adapter.
        policy = build_policy_from_cli(policy_cls, args_cli)

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

        # Optionally wrap with the viewport/camera video recorders (both independent).
        env = wrap_env_for_video(env, video_cfg, num_steps, num_episodes)

        steps_str = f"{num_steps} steps" if num_steps is not None else f"{num_episodes} episodes"
        print(f"[Rank {local_rank}/{world_size}] Starting rollout ({steps_str})")
        metrics = rollout_policy(
            env,
            policy,
            num_steps,
            num_episodes,
            args_cli.language_instruction,
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

        # Write and serve the evaluation report.
        # Only the local rank 0 writes/serves it, to avoid races on a shared output dir.
        if get_local_rank() == 0:
            report_path = build_report(output_dir)
            if args_cli.serve_evaluation_report:
                serve_until_ctrl_c(report_path.parent, args_cli.evaluation_report_port, report_path.name)


if __name__ == "__main__":
    main()
