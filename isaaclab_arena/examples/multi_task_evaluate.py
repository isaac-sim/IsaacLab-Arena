# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import torch
import tqdm

from enum import Enum
import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()


class Status(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# select args_cli given a state machine enum
class JobManager:
    def __init__(self):
        self.jobs = [
            {
                "name": "gr1_open_microwave",
                "args_cli": [
                    "gr1_open_microwave",
                    "--object",
                    "cracker_box",
                ],
            },
            {
                "name": "kitchen_pick_and_place",
                "args_cli": [
                    "gr1_open_microwave",
                    "--object",
                    "sugar_box",
                ],
            },
        ]
        self.jobs_status = [Status.PENDING for _ in self.jobs]

    def get_next_job(self):
        for job, status in zip(self.jobs, self.jobs_status):
            if status == Status.PENDING:
                print(f"Found pending job: {job['name']}")
                return job
        return None

    def mark_job_as_running(self, job_name):
        for job, status in zip(self.jobs, self.jobs_status):
            if job["name"] == job_name:
                status = Status.RUNNING
                break
        print(f"self.jobs_status: {self.jobs_status}")

def load_env(index = 0):
    from isaaclab_arena.utils.reload_modules import reload_arena_modules

    reload_arena_modules()
    from isaaclab_arena.examples.example_environments.cli import (
        get_arena_builder_from_cli,
        get_isaaclab_arena_example_environment_cli_parser,
    )
    args_parser = get_isaaclab_arena_example_environment_cli_parser()
    if index == 0:
        args_cli = args_parser.parse_args([
            "gr1_open_microwave",
            "--object",
            "cracker_box",
        ])
    elif index == 1:
        args_cli = args_parser.parse_args([
            "gr1_open_microwave",
            "--object",
            "sugar_box",
        ])
    else:
        raise ValueError(f"Invalid index: {index}")
    arena_builder = get_arena_builder_from_cli(args_cli)
    env = arena_builder.make_registered()
    env.reset()
    return env

def env_step(env):
    NUM_STEPS = 200
    for _ in tqdm.tqdm(range(NUM_STEPS)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

def env_stop(env):
    from isaaclab.sim import SimulationContext

    simulation_context = SimulationContext.instance()
    simulation_context._disable_app_control_on_stop_handle = True
    simulation_context.stop()
    simulation_context.clear_instance()
    env.close()

    import omni.timeline

    omni.timeline.get_timeline_interface().stop()
    omni.usd.get_context().new_stage()

def main():
    """Script to run an IsaacLab Arena environment with a zero-action agent."""

    env = load_env(0)
    env_step(env)
    env_stop(env)
    env = load_env(1)
    env_step(env)
    env_stop(env)


if __name__ == "__main__":
    main()
