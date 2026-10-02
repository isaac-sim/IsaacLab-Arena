# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compose servicing milestones with a current-state completion gate in Arena."""

from __future__ import annotations

import math
import torch
from functools import partial
from typing import TYPE_CHECKING

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg

from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.configclass import make_configclass

from .scenarios import SCENARIOS

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

    from .scene import ServiceScene


def service_condition(env, name: str, runtime_cfg) -> torch.Tensor:
    """Read one current condition after a single cached runtime update."""
    from .runtime import get_service_runtime

    runtime = get_service_runtime(env, runtime_cfg)
    runtime.update()
    return torch.tensor([getattr(status, name) for status in runtime.statuses], dtype=torch.bool, device=env.device)


def reset_service(env, env_ids, runtime_cfg) -> None:
    """Reset only the requested workcells after their ordinary physical scene reset."""
    from .runtime import get_service_runtime

    get_service_runtime(env, runtime_cfg).reset(env_ids)


def instrument_readings(env, runtime_cfg) -> torch.Tensor:
    """Expose the same public voltage and airflow readings shown by the physical instruments."""
    from .runtime import get_service_runtime

    runtime = get_service_runtime(env, runtime_cfg)
    values = []
    result_codes = {"idle": 0.0, "running": 1.0, "pass": 2.0, "fail": 3.0, "invalid": 4.0}
    for status in runtime.statuses:
        battery, airflow = status.battery_reading, status.airflow_reading
        values.append([
            battery.value if battery.value is not None else -1.0,
            result_codes[battery.result],
            airflow.value if airflow.value is not None else -1.0,
            result_codes[airflow.result],
        ])
    return torch.tensor(values, device=env.device, dtype=torch.float32)


class ReturnToServiceTask(TaskBase):
    """Require verified servicing and correct packing while allowing evidence-dependent rework."""

    def __init__(self, workcell: ServiceScene, scenario_names: list[str], episode_length_s: float = 600.0) -> None:
        assert scenario_names and all(name in SCENARIOS for name in scenario_names), "Select known return scenarios."
        assert math.isfinite(episode_length_s) and episode_length_s > 0
        from .runtime import ServiceRuntimeCfg

        self.workcell = workcell
        self.runtime_cfg = ServiceRuntimeCfg(workcell.layout, tuple(scenario_names))
        instructions = {SCENARIOS[name].work_order for name in scenario_names}
        assert len(instructions) == 1, "Parallel scenarios must share one public work order."
        super().__init__(episode_length_s, instructions.pop())

    def configure_for_embodiment(self, embodiment: EmbodimentBase) -> None:
        self.runtime_cfg.gripper = embodiment.gripper

    def get_scene_cfg(self):
        fields = []
        for socket_name in ("cradle", "battery", "cup", "filter", "battery_tester"):
            socket = self.workcell.layout.sockets[socket_name]
            parent = self.workcell.assets[socket.parent_name]
            for candidate in socket.candidate_names:
                sensor = self.workcell.assets[candidate].get_contact_sensor_cfg(contact_against_object=parent)
                fields.append((f"service_contact_{socket_name}_{candidate}", type(sensor), sensor))
        return make_configclass("ServiceContactsCfg", fields)()

    def get_termination_cfg(self) -> TaskTerminationCfg:
        criteria = []
        for name in ("battery_verified", "airway_serviced", "vacuum_verified", "kit_complete", "station_reset"):
            criteria.append(
                CompletionCriteria(
                    name=name,
                    predicate_sequence=[partial(service_condition, name=name, runtime_cfg=self.runtime_cfg)],
                )
            )
        # Milestones may have happened earlier. This gate rechecks the complete
        # serviced assembly, certificate, inventory, and isolation history now.
        criteria.append(
            CompletionCriteria(
                name="return_to_service",
                predicate_sequence=[
                    TrueForConsecutiveStepsCfg(
                        predicate=partial(service_condition, name="success", runtime_cfg=self.runtime_cfg),
                        required_steps=15,
                    )
                ],
            )
        )
        return TaskTerminationCfg(success=criteria, timeout_s=self.episode_length_s)

    def get_events_cfg(self):
        return make_configclass(
            "ServiceEventsCfg",
            [(
                "reset_service",
                EventTermCfg,
                EventTermCfg(func=reset_service, mode="reset", params={"runtime_cfg": self.runtime_cfg}),
            )],
        )()

    def get_observation_cfg(self):
        group = make_configclass(
            "ServiceInstrumentsCfg",
            [(
                "readings",
                ObservationTermCfg,
                ObservationTermCfg(func=instrument_readings, params={"runtime_cfg": self.runtime_cfg}),
            )],
            bases=(ObservationGroupCfg,),
        )()
        group.concatenate_terms = False
        group.enable_corruption = False
        return make_configclass("ServiceObservationsCfg", [("instruments", type(group), group)])()

    def get_mimic_env_cfg(self, arm_mode):
        return None

    def get_metrics(self):
        from .metrics import ServiceMetrics

        return [SuccessRateMetric(), ServiceMetrics()]

    def get_viewer_cfg(self) -> ViewerCfg:
        z = self.workcell.layout.table_height_m
        return ViewerCfg(eye=(1.4, 1.25, z + 1.25), lookat=(0.43, 0.0, z + 0.08))
