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

class Job:
    def __init__(self, name, args_cli):
        self.name = name
        self.args_cli = args_cli
        self.status = Status.PENDING

    def to_dict(self):
        return {
            "name": self.name,
            "args_cli": self.args_cli,
        }

    def from_dict(self, dict):
        self.name = dict["name"]
        self.args_cli = dict["args_cli"]
        self.status = dict["status"]

    def __str__(self):
        return f"Job(name={self.name}, args_cli={self.args_cli}, status={self.status})"

    def __repr__(self):
        return self.__str__()

# select args_cli given a state machine enum
class JobManager:
    def __init__(self, jobs: list[Job]):
        self.jobs = jobs
        self.jobs_status = [job.status for job in self.jobs]

    def get_next_job(self):
        for job in self.jobs:
            if job.status == Status.PENDING:
                print(f"Found pending job: {job.name}")
                return job
        return None

def load_env(job):
    from isaaclab_arena.utils.reload_modules import reload_arena_modules

    reload_arena_modules()
    from isaaclab_arena_environments.cli import (
        get_arena_builder_from_cli,
        get_isaaclab_arena_environments_cli_parser,
    )
    args_parser = get_isaaclab_arena_environments_cli_parser()

    args_cli = args_parser.parse_args(job.args_cli)
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
    job_1 = Job("gr1_open_microwave", [
        "gr1_open_microwave",
        "--object",
        "cracker_box",
    ])
    job_2 = Job("kitchen_pick_and_place", [
        "gr1_open_microwave",
        "--object",
        "sugar_box",
    ])
    job_manager = JobManager([job_1, job_2])
    for _ in range(2):
        job = job_manager.get_next_job()
        if job is not None:
            env = load_env(job)
            env_step(env)
            env_stop(env)
            job.status = Status.COMPLETED

if __name__ == "__main__":
    main()
