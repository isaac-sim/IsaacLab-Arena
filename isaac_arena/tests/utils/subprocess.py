# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import subprocess
import sys
from collections.abc import Callable
from typing import Any

from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
from isaac_arena.isaaclab_utils.simulation_app import SimulationAppContext


def run_subprocess(cmd, env=None):
    print(f"Running command: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            env=env,
            # Don't capture output, let it flow through in real-time
            capture_output=False,
            text=True,
            # Explicitly set stdout and stderr to None to use parent process's pipes
            stdout=None,
            stderr=None,
        )
        print(f"Command completed with return code: {result.returncode}")
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"Command failed with return code {e.returncode}: {e}\n")
        raise


def runner(q: multiprocessing.Queue, function: Callable[[SimulationAppContext, Any], bool], headless: bool, **kwargs):
    # The runner runs a function in a way that a result is returned to the main process, before
    # simulation_app.close() can ruin everything.
    # Simulation app args. For now, we just make these default + headless.
    # parser = argparse.ArgumentParser(description="Isaac Arena CLI parser.")
    # AppLauncher.add_app_launcher_args(parser)
    # simulation_app_args = parser.parse_args([])
    # simulation_app_args.headless = headless
    # TODO(alexmillane, 2025.09.01): For now we're taking default args (other than headless)
    # We're eventually going to want a way to override the args.
    parser = get_isaac_arena_cli_parser()
    simulation_app_args = parser.parse_args([])
    simulation_app_args.headless = headless
    # Launch the simulator
    with SimulationAppContext(simulation_app_args) as simulation_app:
        # Run the function
        try:
            test_passed = function(simulation_app, **kwargs)
        except Exception as e:
            print(f"Exception occurred while running the function: {e}")
            test_passed = False
        finally:
            # NOTE(alexmillane, 2025.04.09): Put the test result in the queue, so that the main process
            # can get it after the simulation app is closed.
            print("Communicating test result to main process...")
            q.put_nowait(test_passed)


def run_simulation_app_function_in_separate_process(
    function: Callable[..., bool], headless: bool = True, **kwargs
) -> bool:
    """Run a simulation app in a separate process.

    This is sometimes required to prevent simulation app shutdown interrupting pytest.

    Args:
        function: The function to run in the simulation app.
            - The function should take a SimulationAppContext instance as its first argument,
            and then a variable number of additional arguments.
            - The function should return a boolean indicating whether the test passed.
        *args: The arguments to pass to the function (after the SimulationAppContext instance).

    Returns:
        The boolean result of the function.
    """
    # NOTE(alexmillane, 2025.04.10): I got CUDA issues without this.
    multiprocessing.set_start_method("spawn", force=True)
    # Queue to communicate the test result to the main process.
    q = multiprocessing.Queue()
    # Start the test
    # NOTE(alexmillane, 2025.04.10): We need to start the test in a separate process
    # because the simulation app cannot be closed in the main process, because it
    # kills the entire pytest process.
    p = multiprocessing.Process(target=runner, args=(q, function, headless), kwargs=kwargs)
    p.start()
    p.join()

    # NOTE(alexmillane, 2025.04.10): This is sort of a useless check, because the calls to
    # close the simulation app in the child process appear to eat exceptions, so the exitcode
    # is always 0.
    assert p.exitcode == 0, "The closed loop dummy policy failed to run."

    # Get the test result from the child process.
    test_result = q.get()

    return test_result
