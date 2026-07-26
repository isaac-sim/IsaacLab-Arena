# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from isaaclab_arena.integrations.cap_barrier.franka_env import (
    FrankaSimulationAdapter,
    _configure_cap_droid_embodiment,
)
from isaaclab_arena.integrations.cap_barrier.grocery_close_guard import (
    CollisionOffsets,
    GroceryCloseAuthorizationError,
    GroceryCollisionOffsets,
    Pose,
)
from isaaclab_arena.integrations.cap_barrier.grocery_collision_runtime import (
    GroceryRuntimeGeometry,
)
from isaaclab_arena.integrations.cap_barrier.joint_mapping import (
    DROID_ABI_JOINTS,
    DROID_FINGER_JOINT,
    DROID_GRIPPER_CLOSED_POSITION_RAD,
    DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD,
    DROID_PHYSICAL_ARTICULATION_JOINTS,
    DROID_PHYSICAL_GRIPPER_JOINTS,
    DROID_SIMULATION_JOINTS,
    PANDA_ARM_JOINTS,
    JointOrderMapping,
    droid_binary_gripper_action,
    make_droid_joint_mapping,
    resolve_droid_physical_gripper_joint_indices,
)
from isaaclab_arena.integrations.cap_barrier.lockstep_manager import (
    ArenaLockstepManager,
)
from isaaclab_arena.integrations.cap_barrier.protocol import (
    ControllerTimingSpec,
    FrameKind,
    make_command_frame,
    validate_state_frame,
)


class _FakeSimulation:
    def __init__(self):
        self._joint_names = (
            "right_mimic_joint",
            DROID_FINGER_JOINT,
            *reversed(PANDA_ARM_JOINTS),
        )
        self.positions = [float(index) for index in range(9)]
        self.velocities = [0.1 * index for index in range(9)]
        self.efforts = [0.2 * index for index in range(9)]
        self.step_count = 0
        self.reset_count = 0
        self.targets: list[tuple[float, ...]] = []

    @property
    def joint_names(self) -> Sequence[str]:
        return self._joint_names

    def read_joint_state(self):
        return self.positions, self.velocities, self.efforts

    def step_position_targets(self, positions_in_abi_order):
        self.targets.append(tuple(positions_in_abi_order))
        self.step_count += 1

    def reset_without_physics_step(self):
        self.reset_count += 1
        self.positions = [float(index) for index in range(9)]


class _FakeBarrierClient:
    def __init__(self, simulation: _FakeSimulation):
        self.simulation = simulation
        self.generations = [1, 2]
        self.attached_generation = None
        self.expected_sequence = 0
        self.expected_physics_tick = 0
        self._pending_kind = None
        self._step_count_at_begin = 0
        self.completed: list[FrameKind] = []
        self.wait_deadline = None
        self.attach_deadline = None
        self.wait_is_startup = False
        self.attach_is_startup = False

    def wait_for_generation(
        self,
        *,
        after=None,
        timeout_s=10.0,
        deadline_monotonic_s=None,
        startup_rendezvous=False,
    ):
        del timeout_s
        self.wait_deadline = deadline_monotonic_s
        self.wait_is_startup = startup_rendezvous
        return self.generations[0] if after is None else self.generations[1]

    def attach_generation(
        self,
        generation,
        *,
        timeout_s=10.0,
        deadline_monotonic_s=None,
        startup_rendezvous=False,
    ):
        del timeout_s
        self.attach_deadline = deadline_monotonic_s
        self.attach_is_startup = startup_rendezvous
        self.attached_generation = generation
        self.expected_sequence = 0
        self.expected_physics_tick = 0

    def begin_exchange(self, state, *, timeout_s=1.0):
        del timeout_s
        validate_state_frame(state)
        assert state.header.generation == self.attached_generation
        assert state.header.sequence == self.expected_sequence
        assert state.header.physics_tick == self.expected_physics_tick
        self._pending_kind = FrameKind(state.header.frame_kind)
        self._step_count_at_begin = self.simulation.step_count
        due_mask = 0 if self._pending_kind == FrameKind.FENCE else 1
        command = make_command_frame(state, due_mask)
        for index in range(command.header.joint_count):
            command.joints[index].position = 0.25 * (index + 1)
        command.joints[7].position = DROID_GRIPPER_CLOSED_POSITION_RAD
        return command

    def complete_exchange(self):
        expected_steps = self._step_count_at_begin + (
            self._pending_kind == FrameKind.PHYSICS
        )
        assert self.simulation.step_count == expected_steps, (
            "barrier released before the Kit physics step completed"
        )
        self.completed.append(self._pending_kind)
        self.expected_sequence += 1
        if self._pending_kind == FrameKind.PHYSICS:
            self.expected_physics_tick += 1
        self._pending_kind = None

    def fail_pending_exchange(self, code):
        raise AssertionError(f"unexpected simulated exchange failure: {code}")


def _manager() -> tuple[ArenaLockstepManager, _FakeSimulation, _FakeBarrierClient]:
    simulation = _FakeSimulation()
    client = _FakeBarrierClient(simulation)
    mapping = make_droid_joint_mapping(simulation.joint_names)
    manager = ArenaLockstepManager(
        client=client,
        simulation=simulation,
        joint_mapping=mapping,
        controller_specs=[ControllerTimingSpec("hold_controller")],
    )
    return manager, simulation, client


def test_droid_mapping_is_explicit_and_order_preserving() -> None:
    simulation_names = (
        "right_mimic_joint",
        DROID_FINGER_JOINT,
        *reversed(PANDA_ARM_JOINTS),
    )
    mapping = make_droid_joint_mapping(simulation_names)
    assert mapping.abi_joint_names == DROID_ABI_JOINTS
    assert mapping.mapped_simulation_joint_names == DROID_SIMULATION_JOINTS
    assert mapping.to_abi_order(tuple(range(9))) == (8, 7, 6, 5, 4, 3, 2, 1)
    mapping.assert_action_order(DROID_SIMULATION_JOINTS)
    with pytest.raises(RuntimeError, match="action joint order"):
        mapping.assert_action_order((*PANDA_ARM_JOINTS, "right_mimic_joint"))


def test_mapping_rejects_missing_or_duplicate_joints() -> None:
    with pytest.raises(ValueError, match="missing"):
        make_droid_joint_mapping(PANDA_ARM_JOINTS)
    with pytest.raises(ValueError, match="duplicates"):
        JointOrderMapping.from_names(("a", "a"), {"a": "x"}, ("x",))


def test_physical_gripper_roster_pins_live_physx_order_not_stale_usd_order() -> None:
    assert DROID_PHYSICAL_GRIPPER_JOINTS == (
        "finger_joint",
        "right_outer_knuckle_joint",
        "left_inner_finger_joint",
        "right_inner_finger_joint",
        "left_inner_finger_knuckle_joint",
        "right_inner_finger_knuckle_joint",
    )
    assert DROID_PHYSICAL_ARTICULATION_JOINTS == (
        *PANDA_ARM_JOINTS,
        *DROID_PHYSICAL_GRIPPER_JOINTS,
    )
    assert resolve_droid_physical_gripper_joint_indices(
        DROID_PHYSICAL_ARTICULATION_JOINTS
    ) == tuple(range(7, 13))

    stale_usd_order = (
        *PANDA_ARM_JOINTS,
        "finger_joint",
        "right_outer_knuckle_joint",
        "right_inner_finger_joint",
        "right_inner_finger_knuckle_joint",
        "left_inner_finger_knuckle_joint",
        "left_inner_finger_joint",
    )
    assert resolve_droid_physical_gripper_joint_indices(stale_usd_order) == (
        7,
        8,
        12,
        9,
        11,
        10,
    )


def test_physical_gripper_roster_resolution_is_name_ordered_and_exact() -> None:
    reordered = tuple(reversed(DROID_PHYSICAL_ARTICULATION_JOINTS))
    indices = resolve_droid_physical_gripper_joint_indices(reordered)
    assert tuple(reordered[index] for index in indices) == DROID_PHYSICAL_GRIPPER_JOINTS

    with pytest.raises(ValueError, match="missing=.*finger_joint"):
        resolve_droid_physical_gripper_joint_indices(
            tuple(
                name
                for name in DROID_PHYSICAL_ARTICULATION_JOINTS
                if name != "finger_joint"
            )
        )
    with pytest.raises(ValueError, match="duplicates"):
        resolve_droid_physical_gripper_joint_indices(
            (*DROID_PHYSICAL_ARTICULATION_JOINTS, "finger_joint")
        )
    with pytest.raises(ValueError, match="extra=.*unexpected_joint"):
        resolve_droid_physical_gripper_joint_indices(
            (*DROID_PHYSICAL_ARTICULATION_JOINTS, "unexpected_joint")
        )


def test_fence_discards_command_and_physics_steps_exactly_once() -> None:
    manager, simulation, client = _manager()
    assert manager.attach_initial_generation() == 1
    assert simulation.reset_count == 1
    before = tuple(simulation.positions)

    fence = manager.fence()
    assert fence.frame_kind == FrameKind.FENCE
    assert simulation.step_count == 0
    assert tuple(simulation.positions) == before
    assert manager.sequence == 1
    assert manager.physics_tick == 0

    physics = manager.physics_step()
    assert physics.frame_kind == FrameKind.PHYSICS
    assert physics.sequence == 1
    assert physics.physics_tick == 0
    assert simulation.step_count == 1
    assert simulation.targets == [
        (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, DROID_GRIPPER_CLOSED_POSITION_RAD)
    ]
    assert manager.sequence == 2
    assert manager.physics_tick == 1
    assert client.completed == [FrameKind.FENCE, FrameKind.PHYSICS]


def test_initial_generation_uses_one_absolute_startup_deadline() -> None:
    manager, simulation, client = _manager()
    startup_deadline = time.monotonic() + 300.0

    assert (
        manager.attach_initial_generation(startup_deadline_monotonic_s=startup_deadline)
        == 1
    )

    assert simulation.reset_count == 1
    assert client.wait_deadline == startup_deadline
    assert client.attach_deadline == startup_deadline
    assert client.wait_is_startup
    assert client.attach_is_startup


def test_next_generation_resets_sim_without_stepping_and_restarts_counters() -> None:
    manager, simulation, _ = _manager()
    manager.attach_initial_generation()
    manager.fence()
    manager.physics_step()
    old_steps = simulation.step_count

    assert manager.attach_next_generation() == 2
    assert simulation.reset_count == 2
    assert simulation.step_count == old_steps
    assert manager.sequence == 0
    assert manager.physics_tick == 0
    first = manager.physics_step()
    assert first.sequence == 0
    assert first.physics_tick == 0


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        (0.0, 0.0),
        (-DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD, 0.0),
        (DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD, 0.0),
        (DROID_GRIPPER_CLOSED_POSITION_RAD, 1.0),
        (
            DROID_GRIPPER_CLOSED_POSITION_RAD - DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD,
            1.0,
        ),
        (
            DROID_GRIPPER_CLOSED_POSITION_RAD + DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD,
            1.0,
        ),
    ],
)
def test_droid_gripper_endpoint_bands_map_to_binary_polarity(
    position: float, expected: float
) -> None:
    assert DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD == 0.01
    assert droid_binary_gripper_action(position) == expected


@pytest.mark.parametrize(
    "position",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        math.nextafter(-DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD, -math.inf),
        math.nextafter(DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD, math.inf),
        0.5 * DROID_GRIPPER_CLOSED_POSITION_RAD,
        math.nextafter(
            DROID_GRIPPER_CLOSED_POSITION_RAD - DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD,
            -math.inf,
        ),
        math.nextafter(
            DROID_GRIPPER_CLOSED_POSITION_RAD + DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD,
            math.inf,
        ),
    ],
)
def test_droid_gripper_rejects_nonfinite_and_intermediate_positions(
    position: float,
) -> None:
    with pytest.raises(ValueError):
        droid_binary_gripper_action(position)


class _FakeActionManager:
    active_terms = ("arm_action", "gripper_action")
    action_term_dim = (7, 1)
    total_action_dim = 8

    @staticmethod
    def get_term(name: str):
        if name == "arm_action":
            return SimpleNamespace(_joint_names=PANDA_ARM_JOINTS)
        if name == "gripper_action":
            return SimpleNamespace(_joint_names=(DROID_FINGER_JOINT,))
        raise KeyError(name)


def test_cap_droid_profile_replaces_stand_and_disables_intermediate_joint_reset_randomization() -> (
    None
):
    arm_action = SimpleNamespace()
    joint_reset = SimpleNamespace(params={"mean": 1.0, "std": 0.02})
    original_stand_spawn = object()
    stand_spawn = object()
    embodiment = SimpleNamespace(
        action_config=SimpleNamespace(arm_action=arm_action),
        event_config=SimpleNamespace(randomize_franka_joint_state=joint_reset),
        scene_config=SimpleNamespace(stand=SimpleNamespace(spawn=original_stand_spawn)),
    )

    _configure_cap_droid_embodiment(embodiment, stand_spawn=stand_spawn)

    assert embodiment.event_config.randomize_franka_joint_state is joint_reset
    assert joint_reset.params == {"mean": 0.0, "std": 0.0}
    assert embodiment.scene_config.stand.spawn is stand_spawn
    assert embodiment.scene_config.stand.spawn is not original_stand_spawn
    assert arm_action.joint_names == list(PANDA_ARM_JOINTS)
    assert arm_action.preserve_order is True
    assert arm_action.scale == 1.0
    assert arm_action.offset == 0.0
    assert arm_action.use_default_offset is False


class _FakeDroidEnvironment:
    def __init__(self, torch):
        joint_names = ("right_mimic_joint", DROID_FINGER_JOINT, *PANDA_ARM_JOINTS)
        joint_position = torch.tensor(
            [[9.0, DROID_GRIPPER_CLOSED_POSITION_RAD / 3, *range(1, 8)]]
        )
        robot = SimpleNamespace(
            joint_names=joint_names,
            data=SimpleNamespace(
                joint_pos=joint_position,
                joint_vel=torch.zeros_like(joint_position),
                applied_torque=torch.zeros_like(joint_position),
            ),
        )
        self.unwrapped = SimpleNamespace(
            num_envs=1,
            scene={"robot": robot},
            action_manager=_FakeActionManager(),
            device="cpu",
        )
        self.actions = []

    def step(self, action) -> None:
        self.actions.append(action.clone())

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeGroceryCollisionRuntime:
    def require_dynamics_certificate(self) -> object:
        return object()

    def capture(self) -> GroceryRuntimeGeometry:
        proxy = CollisionOffsets(contact_m=0.00025, rest_m=0.0)
        return GroceryRuntimeGeometry(
            simulation_timestamp_s=1.0,
            gripper_base_pose=Pose(
                (0.0, 0.0, 0.3),
                (0.0, 0.0, 0.0, 1.0),
            ),
            left_inner_finger_pose=Pose(
                (0.0, 0.0, 0.3),
                (0.0, 0.0, 0.0, 1.0),
            ),
            right_inner_finger_pose=Pose(
                (0.0, 0.0, 0.3),
                (0.0, 0.0, 0.0, 1.0),
            ),
            can_pose=Pose(
                (0.13, 0.0, 0.3),
                (0.0, 0.0, 0.0, 1.0),
            ),
            bin_pose=Pose(
                (0.46, -0.15, 0.01),
                (0.0, 0.0, 0.0, 1.0),
            ),
            support_pose=Pose(
                (0.5485909, 0.0220630, -0.0169993),
                (0.0, 0.0, 0.0, 1.0),
            ),
            collision_offsets=GroceryCollisionOffsets(
                palm=proxy,
                left_finger4=proxy,
                left_fingertip=proxy,
                right_finger4=proxy,
                right_fingertip=proxy,
                can=proxy,
                bin=CollisionOffsets(contact_m=0.001, rest_m=0.0),
                support=CollisionOffsets(contact_m=0.005, rest_m=0.0),
                ground=CollisionOffsets(contact_m=0.005, rest_m=0.0),
            ),
        )


def test_droid_adapter_preserves_raw_finger_state_and_applies_only_supported_endpoints() -> (
    None
):
    torch = pytest.importorskip("torch")
    environment = _FakeDroidEnvironment(torch)
    adapter = FrankaSimulationAdapter(environment)
    assert adapter.abi_positions() == pytest.approx(
        (
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            DROID_GRIPPER_CLOSED_POSITION_RAD / 3,
        )
    )

    arm_target = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    adapter.step_position_targets((*arm_target, 0.0))
    adapter.step_position_targets((*arm_target, DROID_GRIPPER_CLOSED_POSITION_RAD))
    assert environment.actions[0][0].tolist() == pytest.approx([*arm_target, 0.0])
    assert environment.actions[1][0].tolist() == pytest.approx([*arm_target, 1.0])

    for invalid_target in (float("nan"), 0.5 * DROID_GRIPPER_CLOSED_POSITION_RAD):
        step_count = adapter.physics_step_count
        with pytest.raises(ValueError):
            adapter.step_position_targets((*arm_target, invalid_target))
        assert adapter.physics_step_count == step_count
        assert len(environment.actions) == step_count


def test_grocery_collision_null_adapter_rejects_close_before_step() -> None:
    torch = pytest.importorskip("torch")
    environment = _FakeDroidEnvironment(torch)
    adapter = FrankaSimulationAdapter(
        environment,
        grocery_close_disabled=True,
    )
    first_arm_target = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    second_arm_target = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

    adapter.step_position_targets((*first_arm_target, 0.0))
    adapter.step_position_targets((*second_arm_target, 0.0))
    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="disabled in diagnostic collision-null mode",
    ):
        adapter.step_position_targets(
            (*second_arm_target, DROID_GRIPPER_CLOSED_POSITION_RAD)
        )

    assert len(environment.actions) == 2
    assert environment.actions[0][0].tolist() == pytest.approx([*first_arm_target, 0.0])
    assert environment.actions[1][0].tolist() == pytest.approx(
        [*second_arm_target, 0.0]
    )
    assert adapter.physics_step_count == 2


def test_grocery_close_disable_flag_rejects_ambiguous_or_invalid_configuration() -> (
    None
):
    torch = pytest.importorskip("torch")
    environment = _FakeDroidEnvironment(torch)

    with pytest.raises(TypeError, match="must be boolean"):
        FrankaSimulationAdapter(
            environment,
            grocery_close_disabled=1,  # type: ignore[arg-type]
        )
    with pytest.raises(
        ValueError, match="both runtime-guarded and explicitly disabled"
    ):
        FrankaSimulationAdapter(
            environment,
            grocery_collision_runtime=_FakeGroceryCollisionRuntime(),
            grocery_close_disabled=True,
        )


def test_grocery_adapter_guards_initial_close_and_reset_reopens_proof() -> None:
    torch = pytest.importorskip("torch")
    environment = _FakeDroidEnvironment(torch)
    environment.unwrapped.scene["robot"].data.joint_pos[0, 1] = 0.0
    adapter = FrankaSimulationAdapter(
        environment,
        grocery_collision_runtime=_FakeGroceryCollisionRuntime(),
    )
    arm_target = tuple(float(value) for value in range(1, 8))

    adapter.step_position_targets((*arm_target, 0.0))
    adapter.step_position_targets((*arm_target, DROID_GRIPPER_CLOSED_POSITION_RAD))

    assert len(environment.actions) == 2
    assert adapter.last_grocery_close_evidence is not None
    assert adapter.last_grocery_close_evidence.newly_latched

    adapter.reset_without_physics_step()
    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="sequence-adjacent prior physics sample",
    ):
        adapter.step_position_targets((*arm_target, DROID_GRIPPER_CLOSED_POSITION_RAD))
    assert len(environment.actions) == 2


def test_grocery_adapter_clears_close_state_before_a_throwing_reset() -> None:
    torch = pytest.importorskip("torch")
    environment = _FakeDroidEnvironment(torch)
    environment.unwrapped.scene["robot"].data.joint_pos[0, 1] = 0.0
    adapter = FrankaSimulationAdapter(
        environment,
        grocery_collision_runtime=_FakeGroceryCollisionRuntime(),
    )
    arm_target = tuple(float(value) for value in range(1, 8))
    adapter.step_position_targets((*arm_target, 0.0))
    adapter.step_position_targets((*arm_target, DROID_GRIPPER_CLOSED_POSITION_RAD))
    assert adapter.last_grocery_close_evidence is not None

    def fail_reset() -> None:
        raise RuntimeError("reset failed")

    environment.reset = fail_reset
    with pytest.raises(RuntimeError, match="reset failed"):
        adapter.reset_without_physics_step()

    assert adapter.last_grocery_close_evidence is None
    assert not adapter._grocery_close_guard.close_authorized
    assert adapter._previous_guard_arm_positions is None
    assert adapter.reset_count == 0


def test_grocery_adapter_rolls_back_close_latch_when_step_throws() -> None:
    torch = pytest.importorskip("torch")
    environment = _FakeDroidEnvironment(torch)
    environment.unwrapped.scene["robot"].data.joint_pos[0, 1] = 0.0
    adapter = FrankaSimulationAdapter(
        environment,
        grocery_collision_runtime=_FakeGroceryCollisionRuntime(),
    )
    arm_target = tuple(float(value) for value in range(1, 8))
    adapter.step_position_targets((*arm_target, 0.0))

    def fail_step(action) -> None:
        del action
        raise RuntimeError("step failed")

    environment.step = fail_step
    with pytest.raises(RuntimeError, match="step failed"):
        adapter.step_position_targets((*arm_target, DROID_GRIPPER_CLOSED_POSITION_RAD))

    assert adapter.last_grocery_close_evidence is None
    assert not adapter._grocery_close_guard.close_authorized
    assert adapter._previous_guard_arm_positions is None
    assert adapter.physics_step_count == 1


def test_grocery_adapter_rejects_arm_motion_before_simulation_step() -> None:
    torch = pytest.importorskip("torch")
    environment = _FakeDroidEnvironment(torch)
    environment.unwrapped.scene["robot"].data.joint_pos[0, 1] = 0.0
    adapter = FrankaSimulationAdapter(
        environment,
        grocery_collision_runtime=_FakeGroceryCollisionRuntime(),
    )
    arm_target = tuple(float(value) for value in range(1, 8))
    adapter.step_position_targets((*arm_target, 0.0))
    environment.unwrapped.scene["robot"].data.joint_pos[0, 2] += 0.001

    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="derived rate",
    ):
        adapter.step_position_targets(
            (*adapter.arm_positions(), DROID_GRIPPER_CLOSED_POSITION_RAD)
        )

    assert len(environment.actions) == 1
