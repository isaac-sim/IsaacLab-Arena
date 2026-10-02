# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.managers import RecorderManagerBaseCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

# Import from the package root so this resolves whether MJWarpSolverCfg lives in
# newton_manager_cfg (older isaaclab_newton) or mjwarp_manager_cfg (Isaac Lab Beta 2).
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonMJWarpManager
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg
from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings
from isaaclab_tasks.utils import PresetCfg

from isaaclab_arena.environments.newton_buffer_monitor import NewtonBufferMonitor

if TYPE_CHECKING:
    from isaaclab_newton.assets.articulation.articulation import Articulation
    from isaaclab_newton.physics.mjwarp_tendon_control import MjWarpTendonControl
    from newton import Contacts, Control, Model, State


class NewtonArenaMJWarpManager(NewtonMJWarpManager):
    """Customize Isaac Lab's Newton/MuJoCo-Warp solver lifecycle for Arena.

    This manager delegates solver construction and stepping to Isaac Lab, adding checks that abort
    on buffer exhaustion and MuJoCo-Warp overflow flags. These failures may
    otherwise silently discard physics data while a run continues.

    The manager also handles models without MuJoCo actuators when creating fixed-tendon controls.

    Existing solver exceptions propagate normally.
    """

    _buffer_monitor: NewtonBufferMonitor | None = None
    """Buffer usage and failure status retained for the current solver's lifetime."""

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: MJWarpSolverCfg) -> None:
        """Construct the solver through Isaac Lab and monitor its allocated buffers.

        Args:
            model: Finalized Newton model used to construct the solver.
            solver_cfg: MuJoCo-Warp solver configuration, including buffer capacities.
        """
        super()._build_solver(model, solver_cfg)
        data = cls._solver.mjw_data
        assert data is not None, "Newton MuJoCo-Warp solver data is not initialized"
        cls._buffer_monitor = NewtonBufferMonitor(data)

    @classmethod
    def _step_solver(
        cls, state_0: State, state_1: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """Advance one solver substep while retaining buffer usage and overflow flags.

        Raw Newton contacts are recorded before transfer can silently clamp their
        count; constraint usage and solver flags are recorded after stepping. Both
        recordings run on the device and participate in CUDA graph capture.

        Args:
            state_0: Input Newton state.
            state_1: Output Newton state.
            control: Actuation inputs for this substep.
            contacts: Newton collision contacts, or None for internal MuJoCo contacts.
            substep_dt: Duration of the solver substep in seconds.
        """
        assert cls._buffer_monitor is not None, "Newton buffer monitor is not initialized"
        if cls._needs_collision_pipeline:
            assert contacts is not None, "Newton collision contacts are not initialized"
            cls._buffer_monitor.record_newton_contacts(contacts)
        super()._step_solver(state_0, state_1, control, contacts, substep_dt)
        cls._buffer_monitor.record_solver_status()

    @classmethod
    def _check_solver_status(cls) -> None:
        """Abort on recorded buffer failures after each physics-manager step.

        Isaac Lab invokes this hook outside CUDA graph capture, after stepping or
        replay. The monitor synchronizes a small status array to the host and
        asserts with capacity guidance if any monitored substep overflowed.
        """
        super()._check_solver_status()
        assert cls._buffer_monitor is not None, "Newton buffer monitor is not initialized"
        cls._buffer_monitor.check()

    @classmethod
    def _solver_specific_clear(cls) -> None:
        """Release the monitor during teardown so a rebuilt solver starts with fresh status."""
        cls._buffer_monitor = None
        super()._solver_specific_clear()

    @classmethod
    def create_fixed_tendon_control(cls, articulation: Articulation) -> MjWarpTendonControl | None:
        """Create a tendon adapter only when the model has MuJoCo actuator metadata.

        Newton-native actuators and passive tendons do not create the optional
        ``model.mujoco.actuator_world`` array assumed by Isaac Lab's tendon adapter.

        Args:
            articulation: Newton articulation whose fixed tendons should be controlled.

        Returns:
            Isaac Lab's tendon adapter, or None when no applicable MuJoCo actuators exist.
        """
        model = cls.get_model()
        if not hasattr(model.mujoco, "actuator_world"):
            return None
        return super().create_fixed_tendon_control(articulation)


@configclass
class ArenaPhysicsCfg(PresetCfg):
    """Physics backend presets available to all Arena environments.

    ``default`` / ``physx`` use the stock PhysX backend.
    ``newton`` uses MuJoCo-Warp via Newton with solver parameters tuned
    for dexterous manipulation, with larger contact and constraint buffers.
    """

    physx = PhysxCfg()
    newton = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            class_type=NewtonArenaMJWarpManager,
            solver="newton",
            integrator="implicitfast",
            njmax=8192,
            nconmax=8192,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
            iterations=100,
            ls_iterations=15,
            ls_parallel=False,
            use_mujoco_contacts=False,
            ccd_iterations=15000,
        ),
        num_substeps=2,
        debug_mode=False,
    )
    default = physx


@configclass
class IsaacLabArenaManagerBasedRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for an IsaacLab Arena environment."""

    # NOTE(alexmillane, 2025-07-29): The following definitions are taken from the base class.
    # scene: InteractiveSceneCfg
    # observations: object
    # actions: object
    # events: object
    # terminations: object
    # recorders: object

    # Kill the unused managers
    commands = None
    rewards = None
    curriculum = None

    metrics: object | None = None

    episode_recorders: object | None = None

    demo_recorder_config: RecorderManagerBaseCfg | None = None
    """Recorder configuration used by demonstration collection scripts."""

    # Task language description
    task_description: str | None = None

    # Control rate: sim.dt (1/120 s) x decimation (8) = 15 Hz
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=2,
    )
    decimation: int = 8
    wait_for_textures: bool = False


def apply_arena_global_settings() -> None:
    """Apply Arena's process-global RTX and physics settings before environment construction."""
    apply_isaac_rtx_global_settings(
        IsaacRtxRendererGlobalSettingsCfg(
            # Disable RTX's built-in scene ambient so USD light prims fully control illumination.
            ambient_light_intensity=0.0,
            carb_settings={
                # Work around IsaacLab #6424: do not recurse into leaf collision shapes when
                # resolving a contact filter that targets a multi-shape rigid body.
                "/physics/tensors/recursiveLeafPatternMatch": False,
            },
        )
    )


def set_control_rate_50hz(env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Set 50 Hz control (sim dt 1/200, decimation 4), Arena's pre-15 Hz default rate.

    Args:
        env_cfg: The environment configuration to modify in place.

    Returns:
        The same configuration, so this can be used directly as an ``env_cfg_callback``.
    """
    env_cfg.sim.dt = 1 / 200
    env_cfg.decimation = 4
    return env_cfg


@configclass
class IsaacArenaManagerBasedMimicEnvCfg(IsaacLabArenaManagerBasedRLEnvCfg, MimicEnvCfg):
    """Configuration for an IsaacLab Arena environment."""

    # NOTE(alexmillane, 2025-09-10): The following members are defined in the MimicEnvCfg class.
    # Restated here for clarity.
    # datagen_config: DataGenConfig = DataGenConfig()
    # subtask_configs: dict[str, list[SubTaskConfig]] = {}
    # task_constraint_configs: list[SubTaskConstraintConfig] = []

    # Data generation keeps the longer historical default so demos are not truncated; the task's
    # (shorter) episode length is only applied to non-mimic RL/eval envs by the env builder.
    episode_length_s: float = 50.0
