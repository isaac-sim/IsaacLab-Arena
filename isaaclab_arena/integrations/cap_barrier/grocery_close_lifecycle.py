# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure all-joint lifecycle for a guarded CAP grocery close command."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real

from .joint_mapping import (
    DROID_GRIPPER_CLOSED_POSITION_RAD,
    DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD,
    DROID_GRIPPER_OPEN_POSITION_RAD,
    DROID_PHYSICAL_GRIPPER_JOINTS,
)

PINNED_SIMULATION_DT_S = 0.005
PINNED_SIMULATION_DT_TOLERANCE_S = 1.0e-9
_PINNED_SIMULATION_DT_FLOAT_TOLERANCE_S = PINNED_SIMULATION_DT_TOLERANCE_S + math.ulp(
    PINNED_SIMULATION_DT_S
)
CLOSURE_SETTLED_RATE_BOUND_RAD_S = 1.0e-3
# DroidEmbodimentCfg pins the finger_joint actuator velocity limit to 1 rad/s.
DROID_DRIVER_RATE_BOUND_RAD_S = 1.0
_REQUIRED_SETTLED_PAIR_COUNT = 2
_DRIVER_JOINT_INDEX = DROID_PHYSICAL_GRIPPER_JOINTS.index("finger_joint")
_GRIPPER_JOINT_COUNT = len(DROID_PHYSICAL_GRIPPER_JOINTS)


class GroceryCloseLifecycleError(RuntimeError):
    """A close lifecycle transition could not be proven from its samples."""


class GroceryCloseLifecycleState(Enum):
    """The fail-closed states of the binary close lifecycle."""

    OPEN = "OPEN"
    CLOSE_TRANSITION = "CLOSE_TRANSITION"
    CLOSED_HOLD = "CLOSED_HOLD"


class GroceryGripperTarget(Enum):
    """The only targets accepted by the binary gripper lifecycle."""

    OPEN = "OPEN"
    CLOSE = "CLOSE"


@dataclass(frozen=True)
class GroceryCloseLifecycleSample:
    """One pre-step physical-joint sample and caller-supplied geometry proof.

    ``physical_sample_safe`` is supplied by the same-frame geometry layer. In
    ``CLOSE_TRANSITION`` it proves sample adjacency, timestamp agreement, and
    fixture clearance while allowing finger motion. This lifecycle derives all
    six gripper-joint rates itself; the geometry flag never substitutes for
    physical joint settling. Can/bin motion remains diagnostic and is not part
    of this safety predicate.
    """

    sequence: int
    simulation_timestamp_s: float
    gripper_joint_positions_rad: tuple[float, ...]
    requested_target: GroceryGripperTarget
    physical_sample_safe: bool

    @property
    def driver_position_rad(self) -> float:
        """Return the commanded finger joint from the physical-joint sample."""
        return self.gripper_joint_positions_rad[_DRIVER_JOINT_INDEX]


@dataclass(frozen=True)
class GroceryCloseLifecycleEvidence:
    """The physical gripper lifecycle fact proven by one accepted sample.

    This is not, by itself, a complete physical-terminal proof.
    """

    state: GroceryCloseLifecycleState
    sequence: int
    simulation_timestamp_s: float
    gripper_joint_positions_rad: tuple[float, ...]
    derived_gripper_joint_rates_rad_s: tuple[float, ...] | None
    max_abs_derived_gripper_joint_rate_rad_s: float | None
    closure_settled_pair_count: int
    newly_closure_settled: bool

    @property
    def driver_position_rad(self) -> float:
        """Return the commanded finger joint from the physical-joint evidence."""
        return self.gripper_joint_positions_rad[_DRIVER_JOINT_INDEX]

    @property
    def derived_driver_rate_rad_s(self) -> float | None:
        """Return the derived commanded-finger rate, when a pair was observed."""
        rates = self.derived_gripper_joint_rates_rad_s
        return None if rates is None else rates[_DRIVER_JOINT_INDEX]

    @property
    def closure_settled(self) -> bool:
        """Whether the lifecycle has proven the closed endpoint settled."""
        return self.state is GroceryCloseLifecycleState.CLOSED_HOLD

    @property
    def closure_settled_marker(self) -> str | None:
        """Return the transition marker only when closure first settles."""
        if not self.newly_closure_settled:
            return None
        driver_rate = self.derived_driver_rate_rad_s
        max_rate = self.max_abs_derived_gripper_joint_rate_rad_s
        if driver_rate is None or max_rate is None:
            raise GroceryCloseLifecycleError(
                "closure-settled evidence is missing its derived gripper rates"
            )
        return (
            "CAP_GROCERY_CLOSURE_SETTLED "
            f"sequence={self.sequence} "
            f"driver_position_rad={self.driver_position_rad:.9f} "
            f"derived_driver_rate_rad_s={driver_rate:.9f} "
            f"max_abs_gripper_rate_rad_s={max_rate:.9f} "
            f"closure_settled_pairs={self.closure_settled_pair_count}"
        )


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise GroceryCloseLifecycleError(
            f"{label} must be a real number, not {value!r}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise GroceryCloseLifecycleError(f"{label} must be finite, got {value!r}")
    return result


def _finite_tuple(
    value: object,
    *,
    length: int,
    label: str,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise GroceryCloseLifecycleError(
            f"{label} must contain exactly {length} numbers"
        )
    try:
        result = tuple(
            _finite_float(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)  # type: ignore[arg-type]
        )
    except TypeError as exc:
        raise GroceryCloseLifecycleError(
            f"{label} must contain exactly {length} numbers"
        ) from exc
    if len(result) != length:
        raise GroceryCloseLifecycleError(
            f"{label} must contain exactly {length} numbers, got {len(result)}"
        )
    return result


def _validated_sample(
    sample: GroceryCloseLifecycleSample,
) -> GroceryCloseLifecycleSample:
    if not isinstance(sample, GroceryCloseLifecycleSample):
        raise GroceryCloseLifecycleError("sample must be a GroceryCloseLifecycleSample")
    if isinstance(sample.sequence, bool) or not isinstance(sample.sequence, Integral):
        raise GroceryCloseLifecycleError(
            f"sequence must be an integer, not {sample.sequence!r}"
        )
    sequence = int(sample.sequence)
    if sequence < 0:
        raise GroceryCloseLifecycleError(
            f"sequence must be nonnegative, got {sequence}"
        )
    if not isinstance(sample.requested_target, GroceryGripperTarget):
        raise GroceryCloseLifecycleError(
            "requested_target must be GroceryGripperTarget.OPEN or "
            "GroceryGripperTarget.CLOSE"
        )
    if not isinstance(sample.physical_sample_safe, bool):
        raise GroceryCloseLifecycleError(
            f"physical_sample_safe must be boolean, not {sample.physical_sample_safe!r}"
        )
    return GroceryCloseLifecycleSample(
        sequence=sequence,
        simulation_timestamp_s=_finite_float(
            sample.simulation_timestamp_s,
            label="simulation_timestamp_s",
        ),
        gripper_joint_positions_rad=_finite_tuple(
            sample.gripper_joint_positions_rad,
            length=_GRIPPER_JOINT_COUNT,
            label="gripper_joint_positions_rad",
        ),
        requested_target=sample.requested_target,
        physical_sample_safe=sample.physical_sample_safe,
    )


def _is_in_endpoint_band(position_rad: float, endpoint_rad: float) -> bool:
    return (
        endpoint_rad - DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD
        <= position_rad
        <= endpoint_rad + DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD
    )


def _require_transition_position(position_rad: float) -> None:
    lower = (
        min(
            DROID_GRIPPER_OPEN_POSITION_RAD,
            DROID_GRIPPER_CLOSED_POSITION_RAD,
        )
        - DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD
    )
    upper = (
        max(
            DROID_GRIPPER_OPEN_POSITION_RAD,
            DROID_GRIPPER_CLOSED_POSITION_RAD,
        )
        + DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD
    )
    if not lower <= position_rad <= upper:
        raise GroceryCloseLifecycleError(
            "driver position is outside the physical endpoint envelope: "
            f"position_rad={position_rad!r}, envelope_rad=[{lower!r}, {upper!r}]"
        )


def _derive_adjacent_gripper_rates(
    previous: GroceryCloseLifecycleSample,
    current: GroceryCloseLifecycleSample,
) -> tuple[float, ...]:
    if current.sequence != previous.sequence + 1:
        raise GroceryCloseLifecycleError(
            "close lifecycle requires sequence-adjacent samples: "
            f"previous={previous.sequence}, current={current.sequence}"
        )
    delta_s = current.simulation_timestamp_s - previous.simulation_timestamp_s
    if delta_s <= 0.0:
        raise GroceryCloseLifecycleError(
            "close lifecycle requires a positive simulation timestamp delta: "
            f"delta_s={delta_s!r}"
        )
    if not math.isclose(
        delta_s,
        PINNED_SIMULATION_DT_S,
        rel_tol=0.0,
        abs_tol=_PINNED_SIMULATION_DT_FLOAT_TOLERANCE_S,
    ):
        raise GroceryCloseLifecycleError(
            "close lifecycle sample delta drifted from the pinned simulation "
            f"step: delta_s={delta_s!r}, expected_s={PINNED_SIMULATION_DT_S!r}, "
            f"tolerance_s={PINNED_SIMULATION_DT_TOLERANCE_S!r}"
        )
    rates = tuple(
        (current_position - previous_position) / delta_s
        for previous_position, current_position in zip(
            previous.gripper_joint_positions_rad,
            current.gripper_joint_positions_rad,
            strict=True,
        )
    )
    if any(not math.isfinite(rate) for rate in rates):
        raise GroceryCloseLifecycleError(
            f"derived gripper-joint rates must be finite, got {rates!r}"
        )
    return rates


class GroceryCloseLifecycle:
    """Track one binary close through transition and settled closure."""

    def __init__(self) -> None:
        self._state = GroceryCloseLifecycleState.OPEN
        self._previous_sample: GroceryCloseLifecycleSample | None = None
        self._closure_settled_pair_count = 0

    @property
    def state(self) -> GroceryCloseLifecycleState:
        """Return the current lifecycle state."""
        return self._state

    @property
    def closure_settled(self) -> bool:
        """Whether the closed endpoint has been proven settled."""
        return self._state is GroceryCloseLifecycleState.CLOSED_HOLD

    def reset(self) -> None:
        """Clear every fact used to authorize a close."""
        self._state = GroceryCloseLifecycleState.OPEN
        self._previous_sample = None
        self._closure_settled_pair_count = 0

    def observe_pre_step(
        self,
        sample: GroceryCloseLifecycleSample,
    ) -> GroceryCloseLifecycleEvidence:
        """Consume one pre-step sample or reset and reject it fail closed."""
        try:
            current = _validated_sample(sample)
            previous = self._previous_sample
            if previous is not None and current.sequence <= previous.sequence:
                raise GroceryCloseLifecycleError(
                    "sample sequence must strictly increase: "
                    f"previous={previous.sequence}, current={current.sequence}"
                )
            if current.requested_target is GroceryGripperTarget.OPEN:
                self.reset()
                self._previous_sample = current
                return GroceryCloseLifecycleEvidence(
                    state=self._state,
                    sequence=current.sequence,
                    simulation_timestamp_s=current.simulation_timestamp_s,
                    gripper_joint_positions_rad=current.gripper_joint_positions_rad,
                    derived_gripper_joint_rates_rad_s=None,
                    max_abs_derived_gripper_joint_rate_rad_s=None,
                    closure_settled_pair_count=0,
                    newly_closure_settled=False,
                )
            evidence = self._observe_close(current, previous)
            self._previous_sample = current
            return evidence
        except BaseException:
            self.reset()
            raise

    def _observe_close(
        self,
        current: GroceryCloseLifecycleSample,
        previous: GroceryCloseLifecycleSample | None,
    ) -> GroceryCloseLifecycleEvidence:
        if previous is None:
            raise GroceryCloseLifecycleError(
                "initial close requires a prior sequence-adjacent physically-open "
                "sample"
            )
        rates_rad_s = _derive_adjacent_gripper_rates(previous, current)
        driver_rate_rad_s = rates_rad_s[_DRIVER_JOINT_INDEX]
        max_abs_rate_rad_s = max(abs(rate) for rate in rates_rad_s)

        if self._state is GroceryCloseLifecycleState.CLOSED_HOLD:
            self._require_closed_hold(
                current,
                rates_rad_s,
                max_abs_rate_rad_s,
            )
            return self._evidence(
                current,
                rates_rad_s,
                max_abs_rate_rad_s,
                newly_closure_settled=False,
            )

        if self._state is GroceryCloseLifecycleState.OPEN:
            if not _is_in_endpoint_band(
                previous.driver_position_rad,
                DROID_GRIPPER_OPEN_POSITION_RAD,
            ):
                raise GroceryCloseLifecycleError(
                    "initial close requires the prior physical driver position "
                    "in its open endpoint band"
                )
            self._state = GroceryCloseLifecycleState.CLOSE_TRANSITION

        if not current.physical_sample_safe:
            raise GroceryCloseLifecycleError(
                "close transition lost its same-frame geometry safety proof"
            )
        _require_transition_position(current.driver_position_rad)
        if driver_rate_rad_s < 0.0:
            raise GroceryCloseLifecycleError(
                "close transition driver motion reversed away from the DROID "
                f"close endpoint: derived_rate_rad_s={driver_rate_rad_s!r}"
            )
        if driver_rate_rad_s > DROID_DRIVER_RATE_BOUND_RAD_S:
            raise GroceryCloseLifecycleError(
                "close transition driver rate exceeded its pinned actuator bound: "
                f"derived_rate_rad_s={driver_rate_rad_s!r}, "
                f"bound_rad_s={DROID_DRIVER_RATE_BOUND_RAD_S!r}"
            )

        pair_is_settled = (
            max_abs_rate_rad_s < CLOSURE_SETTLED_RATE_BOUND_RAD_S
            and _is_in_endpoint_band(
                previous.driver_position_rad,
                DROID_GRIPPER_CLOSED_POSITION_RAD,
            )
            and _is_in_endpoint_band(
                current.driver_position_rad,
                DROID_GRIPPER_CLOSED_POSITION_RAD,
            )
        )
        self._closure_settled_pair_count = (
            self._closure_settled_pair_count + 1 if pair_is_settled else 0
        )
        newly_closure_settled = (
            self._closure_settled_pair_count >= _REQUIRED_SETTLED_PAIR_COUNT
        )
        if newly_closure_settled:
            self._state = GroceryCloseLifecycleState.CLOSED_HOLD
        return self._evidence(
            current,
            rates_rad_s,
            max_abs_rate_rad_s,
            newly_closure_settled=newly_closure_settled,
        )

    def _require_closed_hold(
        self,
        current: GroceryCloseLifecycleSample,
        rates_rad_s: tuple[float, ...],
        max_abs_rate_rad_s: float,
    ) -> None:
        if not current.physical_sample_safe:
            raise GroceryCloseLifecycleError(
                "closed lifecycle lost its same-frame physical sample proof"
            )
        if not _is_in_endpoint_band(
            current.driver_position_rad,
            DROID_GRIPPER_CLOSED_POSITION_RAD,
        ):
            raise GroceryCloseLifecycleError(
                "closed lifecycle lost the physical full-close endpoint band"
            )
        if max_abs_rate_rad_s >= CLOSURE_SETTLED_RATE_BOUND_RAD_S:
            raise GroceryCloseLifecycleError(
                "closed lifecycle lost its strictly-settled all-joint rate proof: "
                f"derived_rates_rad_s={rates_rad_s!r}, "
                f"max_abs_rate_rad_s={max_abs_rate_rad_s!r}, "
                f"bound_rad_s={CLOSURE_SETTLED_RATE_BOUND_RAD_S!r}"
            )

    def _evidence(
        self,
        current: GroceryCloseLifecycleSample,
        rates_rad_s: tuple[float, ...],
        max_abs_rate_rad_s: float,
        *,
        newly_closure_settled: bool,
    ) -> GroceryCloseLifecycleEvidence:
        return GroceryCloseLifecycleEvidence(
            state=self._state,
            sequence=current.sequence,
            simulation_timestamp_s=current.simulation_timestamp_s,
            gripper_joint_positions_rad=current.gripper_joint_positions_rad,
            derived_gripper_joint_rates_rad_s=rates_rad_s,
            max_abs_derived_gripper_joint_rate_rad_s=max_abs_rate_rad_s,
            closure_settled_pair_count=self._closure_settled_pair_count,
            newly_closure_settled=newly_closure_settled,
        )
