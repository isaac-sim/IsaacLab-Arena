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

import asyncio
import copy
import torch

from isaaclab.envs import SubTaskConstraintType
from isaaclab.managers import TerminationTermCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab_mimic.datagen.waypoint import MultiWaypoint

from isaaclab_arena.arena_g1.g1_whole_body_controller.wbc_policy.policy.action_constants import (
    NAVIGATE_CMD_END_IDX,
    NAVIGATE_CMD_START_IDX,
)


def patch_recorders():
    from isaaclab.envs.mdp.recorders import PreStepActionsRecorder
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

    # Record p-controller generated navigation commands in the action buffer
    def record_pre_step(self):
        actions = self._env.action_manager.action
        for term_name in self._env.action_manager.active_terms:
            if hasattr(self._env.action_manager.get_term(term_name), "navigate_cmd"):
                actions[:, NAVIGATE_CMD_START_IDX:NAVIGATE_CMD_END_IDX] = self._env.action_manager.get_term(
                    term_name
                ).navigate_cmd
        return "actions", actions

    # Record post step action observation group
    class PostStepFlatPolicyObservationsRecorder(RecorderTerm):
        def record_post_step(self):
            return "action", self._env.obs_buf["action"]

    @configclass
    class PostStepFlatPolicyObservationsRecorderCfg(RecorderTermCfg):
        class_type: type[RecorderTerm] = PostStepFlatPolicyObservationsRecorder

    ActionStateRecorderManagerCfg.record_post_step_flat_policy_observations = (
        PostStepFlatPolicyObservationsRecorderCfg()
    )
    PreStepActionsRecorder.record_pre_step = record_pre_step

    print("\nPatched recorders for G1 Locomanip Mimic\n")


def patch_generate():  # noqa: C901
    from isaaclab_mimic.datagen.data_generator import DataGenerator

    async def generate(
        self,
        env_id: int,
        success_term: TerminationTermCfg,
        env_reset_queue: asyncio.Queue | None = None,
        env_action_queue: asyncio.Queue | None = None,
        pause_subtask: bool = False,
        export_demo: bool = True,
    ) -> dict:
        """
        Attempt to generate a new demonstration.

        Args:
            env_id: environment index
            success_term: success function to check if the task is successful
            env_reset_queue: queue to store environment IDs for reset
            env_action_queue: queue to store actions for each environment
            pause_subtask: if True, pause after every subtask during generation, for debugging
            export_demo: if True, export the generated demonstration

        Returns:
            results: dictionary with the following items:
                initial_state (dict): initial simulator state for the executed trajectory
                states (list): simulator state at each timestep
                observations (list): observation dictionary at each timestep
                datagen_infos (list): datagen_info at each timestep
                actions (np.array): action executed at each timestep
                success (bool): whether the trajectory successfully solved the task or not
                src_demo_inds (list): list of selected source demonstration indices for each subtask
                src_demo_labels (np.array): same as @src_demo_inds, but repeated to have a label for each timestep of the trajectory
        """

        # reset the env to create a new task demo instance
        env_id_tensor = torch.tensor([env_id], dtype=torch.int64, device=self.env.device)
        self.env.recorder_manager.reset(env_ids=env_id_tensor)
        await env_reset_queue.put(env_id)
        await env_reset_queue.join()
        new_initial_state = self.env.scene.get_state(is_relative=True)

        # create runtime subtask constraint rules from subtask constraint configs
        runtime_subtask_constraints_dict = {}
        for subtask_constraint in self.env_cfg.task_constraint_configs:
            runtime_subtask_constraints_dict.update(subtask_constraint.generate_runtime_subtask_constraints())

        # save generated data in these variables
        generated_states = []
        generated_obs = []
        generated_actions = []
        generated_success = False

        # some eef-specific state variables used during generation
        current_eef_selected_src_demo_indices = {}
        current_eef_subtask_trajectories = {}
        current_eef_subtask_indices = {}
        current_eef_subtask_step_indices = {}
        eef_subtasks_done = {}
        for eef_name in self.env_cfg.subtask_configs.keys():
            current_eef_selected_src_demo_indices[eef_name] = None
            # prev_eef_executed_traj[eef_name] = None  # type of list of Waypoint
            current_eef_subtask_trajectories[eef_name] = None  # type of list of Waypoint
            current_eef_subtask_indices[eef_name] = 0
            current_eef_subtask_step_indices[eef_name] = None
            eef_subtasks_done[eef_name] = False

        prev_src_demo_datagen_info_pool_size = 0

        was_navigating = False
        while True:
            async with self.src_demo_datagen_info_pool.asyncio_lock:
                if len(self.src_demo_datagen_info_pool.datagen_infos) > prev_src_demo_datagen_info_pool_size:
                    # src_demo_datagen_info_pool at this point may be updated with new demos,
                    # so we need to updaet subtask boundaries again
                    randomized_subtask_boundaries = (
                        self.randomize_subtask_boundaries()
                    )  # shape [N, S, 2], last dim is start and end action lengths
                    prev_src_demo_datagen_info_pool_size = len(self.src_demo_datagen_info_pool.datagen_infos)

                # generate trajectory for a subtask for the eef that is currently at the beginning of a subtask
                for eef_name, eef_subtask_step_index in current_eef_subtask_step_indices.items():
                    if eef_subtask_step_index is None:
                        # current_eef_selected_src_demo_indices will be updated in generate_trajectory
                        subtask_trajectory = self.generate_trajectory(
                            env_id,
                            eef_name,
                            current_eef_subtask_indices[eef_name],
                            randomized_subtask_boundaries,
                            runtime_subtask_constraints_dict,
                            current_eef_selected_src_demo_indices,
                            current_eef_subtask_trajectories,
                        )
                        current_eef_subtask_trajectories[eef_name] = subtask_trajectory
                        current_eef_subtask_step_indices[eef_name] = 0
                        # current_eef_selected_src_demo_indices[eef_name] = selected_src_demo_inds
                        # two_arm_trajectories[task_spec_idx] = subtask_trajectory
                        # prev_executed_traj[task_spec_idx] = subtask_trajectory

            # determine the next waypoint for each eef based on the current subtask constraints
            eef_waypoint_dict = {}
            for eef_name in sorted(self.env_cfg.subtask_configs.keys()):
                # handle constraints
                step_ind = current_eef_subtask_step_indices[eef_name]
                subtask_ind = current_eef_subtask_indices[eef_name]
                if (eef_name, subtask_ind) in runtime_subtask_constraints_dict:
                    task_constraint = runtime_subtask_constraints_dict[(eef_name, subtask_ind)]
                    if task_constraint["type"] == SubTaskConstraintType._SEQUENTIAL_LATTER:
                        min_time_diff = task_constraint["min_time_diff"]
                        if not task_constraint["fulfilled"]:
                            if (
                                min_time_diff == -1
                                or step_ind >= len(current_eef_subtask_trajectories[eef_name]) - min_time_diff
                            ):
                                if step_ind > 0:
                                    # wait at the same step
                                    step_ind -= 1
                                    current_eef_subtask_step_indices[eef_name] = step_ind

                    elif task_constraint["type"] == SubTaskConstraintType.COORDINATION:
                        synchronous_steps = task_constraint["synchronous_steps"]
                        concurrent_task_spec_key = task_constraint["concurrent_task_spec_key"]
                        concurrent_subtask_ind = task_constraint["concurrent_subtask_ind"]
                        concurrent_task_fulfilled = runtime_subtask_constraints_dict[
                            (concurrent_task_spec_key, concurrent_subtask_ind)
                        ]["fulfilled"]

                        if (
                            task_constraint["coordination_synchronize_start"]
                            and current_eef_subtask_indices[concurrent_task_spec_key] < concurrent_subtask_ind
                        ):
                            # the concurrent eef is not yet at the concurrent subtask, so wait at the first action
                            # this also makes sure that the concurrent task starts at the same time as this task
                            step_ind = 0
                            current_eef_subtask_step_indices[eef_name] = 0
                        else:
                            if (
                                not concurrent_task_fulfilled
                                and step_ind >= len(current_eef_subtask_trajectories[eef_name]) - synchronous_steps
                            ):
                                # trigger concurrent task
                                runtime_subtask_constraints_dict[(concurrent_task_spec_key, concurrent_subtask_ind)][
                                    "fulfilled"
                                ] = True

                            if not task_constraint["fulfilled"]:
                                if step_ind >= len(current_eef_subtask_trajectories[eef_name]) - synchronous_steps:
                                    if step_ind > 0:
                                        step_ind -= 1
                                        current_eef_subtask_step_indices[eef_name] = step_ind  # wait here

                waypoint = current_eef_subtask_trajectories[eef_name][step_ind]
                eef_waypoint_dict[eef_name] = waypoint
            multi_waypoint = MultiWaypoint(eef_waypoint_dict)

            # execute the next waypoints for all eefs
            exec_results = await multi_waypoint.execute(
                env=self.env,
                success_term=success_term,
                env_id=env_id,
                env_action_queue=env_action_queue,
            )

            if "wbc" in exec_results["observations"][0]:
                obs = exec_results["observations"][0]["wbc"]
                if "is_navigating" in obs and "navigation_goal_reached" in obs:
                    is_navigating = obs["is_navigating"]
                    navigation_goal_reached = obs["navigation_goal_reached"]
            else:
                is_navigating = False
                navigation_goal_reached = False

            # update execution state buffers
            if len(exec_results["states"]) > 0:
                generated_states.extend(exec_results["states"])
                generated_obs.extend(exec_results["observations"])
                generated_actions.extend(exec_results["actions"])
                generated_success = exec_results["success"]

            processed_nav_subtask = False
            for eef_name in self.env_cfg.subtask_configs.keys():
                current_eef_subtask_step_indices[eef_name] += 1

                """
                Note: The following code sets the eef indices accordingly for the navigation subtask.
                The eef index buffers are updated twice due to the existing outer for loop.
                After the first eef is handled, both eefs index buffers will be updated.
                """

                # Execute locomanip navigation p-controller if it is enabled via the is_navigating flag
                if "body" in self.env_cfg.subtask_configs.keys():
                    # Repeat the last nav subtask action if the robot is navigating and hasn't reached the waypoint goal
                    if (
                        current_eef_subtask_step_indices["body"] == len(current_eef_subtask_trajectories["body"]) - 1
                        and not processed_nav_subtask
                    ):
                        if is_navigating and not navigation_goal_reached:
                            for name in self.env_cfg.subtask_configs.keys():
                                current_eef_subtask_step_indices[name] -= 1
                            processed_nav_subtask = True
                    # Skip to the end of the nav subtask if the robot has reached the waypoint goal before the end
                    # of the human recorded trajectory
                    elif was_navigating and not is_navigating and not processed_nav_subtask:
                        number_of_steps_to_skip = len(current_eef_subtask_trajectories["body"]) - (
                            current_eef_subtask_step_indices["body"] + 1
                        )
                        for name in self.env_cfg.subtask_configs.keys():
                            if current_eef_subtask_step_indices[name] + number_of_steps_to_skip < len(
                                current_eef_subtask_trajectories[name]
                            ):
                                current_eef_subtask_step_indices[name] = (
                                    current_eef_subtask_step_indices[name] + number_of_steps_to_skip
                                )
                            else:
                                current_eef_subtask_step_indices[name] = len(current_eef_subtask_trajectories[name]) - 1
                        processed_nav_subtask = True

                subtask_ind = current_eef_subtask_indices[eef_name]
                if current_eef_subtask_step_indices[eef_name] == len(
                    current_eef_subtask_trajectories[eef_name]
                ):  # subtask done
                    if (eef_name, subtask_ind) in runtime_subtask_constraints_dict:
                        task_constraint = runtime_subtask_constraints_dict[(eef_name, subtask_ind)]
                        if task_constraint["type"] == SubTaskConstraintType._SEQUENTIAL_FORMER:
                            constrained_task_spec_key = task_constraint["constrained_task_spec_key"]
                            constrained_subtask_ind = task_constraint["constrained_subtask_ind"]
                            runtime_subtask_constraints_dict[(constrained_task_spec_key, constrained_subtask_ind)][
                                "fulfilled"
                            ] = True
                        elif task_constraint["type"] == SubTaskConstraintType.COORDINATION:
                            concurrent_task_spec_key = task_constraint["concurrent_task_spec_key"]
                            concurrent_subtask_ind = task_constraint["concurrent_subtask_ind"]
                            # concurrent_task_spec_idx = task_spec_keys.index(concurrent_task_spec_key)
                            task_constraint["finished"] = True
                            # check if concurrent task has been finished
                            assert (
                                runtime_subtask_constraints_dict[(concurrent_task_spec_key, concurrent_subtask_ind)][
                                    "finished"
                                ]
                                or current_eef_subtask_step_indices[concurrent_task_spec_key]
                                >= len(current_eef_subtask_trajectories[concurrent_task_spec_key]) - 1
                            )

                    if pause_subtask:
                        input(
                            f"Pausing after subtask {current_eef_subtask_indices[eef_name]} of {eef_name} execution."
                            " Press any key to continue..."
                        )
                    # This is a check to see if this arm has completed all the subtasks
                    if current_eef_subtask_indices[eef_name] == len(self.env_cfg.subtask_configs[eef_name]) - 1:
                        eef_subtasks_done[eef_name] = True
                        # If all subtasks done for this arm, repeat last waypoint to make sure this arm does not move
                        current_eef_subtask_trajectories[eef_name].append(
                            current_eef_subtask_trajectories[eef_name][-1]
                        )
                    else:
                        current_eef_subtask_step_indices[eef_name] = None
                        current_eef_subtask_indices[eef_name] += 1

            was_navigating = copy.deepcopy(is_navigating)

            # Check if all eef_subtasks_done values are True
            if all(eef_subtasks_done.values()):
                break

        # merge numpy arrays
        if len(generated_actions) > 0:
            generated_actions = torch.cat(generated_actions, dim=0)

        # set success to the recorded episode data and export to file
        self.env.recorder_manager.set_success_to_episodes(
            env_id_tensor, torch.tensor([[generated_success]], dtype=torch.bool, device=self.env.device)
        )
        if export_demo:
            self.env.recorder_manager.export_episodes(env_id_tensor)

        results = dict(
            initial_state=new_initial_state,
            states=generated_states,
            observations=generated_obs,
            actions=generated_actions,
            success=generated_success,
        )
        return results

    DataGenerator.generate = generate

    print("\nPatched generate function for G1 Locomanip Mimic\n")


def patch_g1_locomanip_mimic():
    patch_recorders()
    patch_generate()
