# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Strict loader for the CAP grocery dynamics certificate.

The certificate is content-addressed without a recursive self-hash:
``canonical_sha256`` is the SHA-256 of canonical JSON after removing only that
top-level field. Canonical JSON uses sorted keys, no insignificant whitespace,
ASCII escaping, and rejects non-finite numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Mapping

GROCERY_DYNAMICS_CERTIFICATE_SCHEMA_VERSION = 1
GROCERY_DYNAMICS_BODY_ROSTER = (
    "base_link",
    "left_outer_knuckle",
    "left_outer_finger",
    "left_inner_knuckle",
    "left_inner_finger",
    "right_outer_knuckle",
    "right_outer_finger",
    "right_inner_knuckle",
    "right_inner_finger",
    "can",
    "bin",
)

_MAX_CERTIFICATE_BYTES = 256 * 1024
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_QUATERNION_NORM_TOLERANCE = 1.0e-6
_INERTIA_SYMMETRY_TOLERANCE = 1.0e-12


class GroceryDynamicsCertificateError(ValueError):
    """The dynamics certificate is malformed or does not match this runtime."""


@dataclass(frozen=True)
class GroceryDynamicsSourcePin:
    """One immutable source-asset identity."""

    source_id: str
    uri_suffix: str
    sha256: str


GROCERY_DYNAMICS_SOURCE_PINS = (
    GroceryDynamicsSourcePin(
        source_id="robot",
        uri_suffix=(
            "/Arena/assets/robot_library/droid/franka_robotiq_2f_85_flattened.usd"
        ),
        sha256=("c8d72259834e2e5290754f8580b37efbc0dec079ac6a98b27b167efe6461eb2c"),
    ),
    GroceryDynamicsSourcePin(
        source_id="can",
        uri_suffix=(
            "/Arena/assets/object_library/srl_robolab_assets/"
            "objects/hope/alphabet_soup_can.usd"
        ),
        sha256=("84b730aecba2efed94eeba61befd152d619f59810463bc6b23b3cc3e54836dfb"),
    ),
    GroceryDynamicsSourcePin(
        source_id="bin",
        uri_suffix=(
            "/Arena/assets/object_library/srl_robolab_assets/fixtures/grey_bin.usd"
        ),
        sha256=("e4c1c2d34dce8d642b3ffd8a1b468ecd40f6b8bd2b365f3d41592eb191ce99ea"),
    ),
)

_SOURCE_PINS_BY_ID = {
    source.source_id: source for source in GROCERY_DYNAMICS_SOURCE_PINS
}


@dataclass(frozen=True)
class GroceryDynamicsTolerances:
    """Absolute comparison tolerances for one body."""

    mass_kg: float
    center_of_mass_m: float
    principal_axes_angle_rad: float
    principal_moments_kg_m2: float
    inertia_kg_m2: float


@dataclass(frozen=True)
class GroceryBodyDynamics:
    """Certified inertial properties for one exact body."""

    name: str
    mass_kg: float
    center_of_mass_m: tuple[float, float, float]
    principal_axes_xyzw: tuple[float, float, float, float]
    principal_moments_kg_m2: tuple[float, float, float]
    inertia_kg_m2: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    tolerances: GroceryDynamicsTolerances


@dataclass(frozen=True)
class GroceryDynamicsCertificate:
    """Validated, immutable dynamics certificate."""

    certificate_id: str
    canonical_sha256: str
    engine_id: str
    engine_version: str
    stage_meters_per_unit: float
    num_envs: int
    sources: tuple[GroceryDynamicsSourcePin, ...]
    bodies: tuple[GroceryBodyDynamics, ...]

    def body(self, name: str) -> GroceryBodyDynamics:
        """Return one body by its exact roster name."""

        for body in self.bodies:
            if body.name == name:
                return body
        raise KeyError(name)


def _strict_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise GroceryDynamicsCertificateError(f"duplicate certificate key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise GroceryDynamicsCertificateError(
        f"non-finite JSON number is forbidden: {value}"
    )


def canonical_certificate_sha256(document: Mapping[str, object]) -> str:
    """Hash canonical JSON after excluding only ``canonical_sha256``."""

    if not isinstance(document, Mapping):
        raise GroceryDynamicsCertificateError(
            "certificate document must be a JSON object"
        )
    unsigned = dict(document)
    unsigned.pop("canonical_sha256", None)
    try:
        payload = json.dumps(
            unsigned,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise GroceryDynamicsCertificateError(
            "certificate cannot be represented as canonical JSON"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _require_exact_keys(
    value: object,
    expected: set[str],
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise GroceryDynamicsCertificateError(f"{label} must be a JSON object")
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise GroceryDynamicsCertificateError(
            f"{label} keys do not match schema: missing={missing!r}, extra={extra!r}"
        )
    return value


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise GroceryDynamicsCertificateError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    result = _require_string(value, label=label)
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise GroceryDynamicsCertificateError(
            f"{label} must be a lowercase 64-hex SHA-256"
        )
    return result


def _require_float(
    value: object,
    *,
    label: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise GroceryDynamicsCertificateError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise GroceryDynamicsCertificateError(f"{label} must be finite")
    if positive and result <= 0.0:
        raise GroceryDynamicsCertificateError(f"{label} must be strictly positive")
    if nonnegative and result < 0.0:
        raise GroceryDynamicsCertificateError(f"{label} must be nonnegative")
    return result


def _require_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GroceryDynamicsCertificateError(f"{label} must be an integer")
    return value


def _require_vector(
    value: object,
    *,
    length: int,
    label: str,
    positive: bool = False,
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise GroceryDynamicsCertificateError(
            f"{label} must be a JSON array of {length} numbers"
        )
    return tuple(
        _require_float(
            item,
            label=f"{label}[{index}]",
            positive=positive,
        )
        for index, item in enumerate(value)
    )


def _require_matrix3(
    value: object,
    *,
    label: str,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    if not isinstance(value, list) or len(value) != 3:
        raise GroceryDynamicsCertificateError(f"{label} must be a 3x3 JSON array")
    rows = tuple(
        _require_vector(row, length=3, label=f"{label}[{index}]")
        for index, row in enumerate(value)
    )
    return (rows[0], rows[1], rows[2])


def _require_normalized_quaternion(
    value: object,
    *,
    label: str,
) -> tuple[float, float, float, float]:
    quaternion = _require_vector(value, length=4, label=label)
    norm = math.sqrt(sum(component * component for component in quaternion))
    if not math.isclose(
        norm,
        1.0,
        rel_tol=0.0,
        abs_tol=_QUATERNION_NORM_TOLERANCE,
    ):
        raise GroceryDynamicsCertificateError(
            f"{label} must be normalized in xyzw order"
        )
    return (
        quaternion[0],
        quaternion[1],
        quaternion[2],
        quaternion[3],
    )


def _require_positive_definite_inertia(
    matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    *,
    label: str,
) -> None:
    xx, xy, xz = matrix[0]
    yx, yy, yz = matrix[1]
    zx, zy, zz = matrix[2]
    for left, right in ((xy, yx), (xz, zx), (yz, zy)):
        if not math.isclose(
            left,
            right,
            rel_tol=0.0,
            abs_tol=_INERTIA_SYMMETRY_TOLERANCE,
        ):
            raise GroceryDynamicsCertificateError(f"{label} must be symmetric")
    leading_two = xx * yy - xy * yx
    determinant = (
        xx * (yy * zz - yz * zy) - xy * (yx * zz - yz * zx) + xz * (yx * zy - yy * zx)
    )
    if xx <= 0.0 or leading_two <= 0.0 or determinant <= 0.0:
        raise GroceryDynamicsCertificateError(f"{label} must be positive definite")


def _load_tolerances(
    value: object,
    *,
    label: str,
) -> GroceryDynamicsTolerances:
    data = _require_exact_keys(
        value,
        {
            "mass_kg",
            "center_of_mass_m",
            "principal_axes_angle_rad",
            "principal_moments_kg_m2",
            "inertia_kg_m2",
        },
        label=label,
    )
    return GroceryDynamicsTolerances(
        mass_kg=_require_float(
            data["mass_kg"],
            label=f"{label}.mass_kg",
            nonnegative=True,
        ),
        center_of_mass_m=_require_float(
            data["center_of_mass_m"],
            label=f"{label}.center_of_mass_m",
            nonnegative=True,
        ),
        principal_axes_angle_rad=_require_float(
            data["principal_axes_angle_rad"],
            label=f"{label}.principal_axes_angle_rad",
            nonnegative=True,
        ),
        principal_moments_kg_m2=_require_float(
            data["principal_moments_kg_m2"],
            label=f"{label}.principal_moments_kg_m2",
            nonnegative=True,
        ),
        inertia_kg_m2=_require_float(
            data["inertia_kg_m2"],
            label=f"{label}.inertia_kg_m2",
            nonnegative=True,
        ),
    )


def _load_body(value: object, *, index: int) -> GroceryBodyDynamics:
    label = f"bodies[{index}]"
    data = _require_exact_keys(
        value,
        {
            "name",
            "mass_kg",
            "center_of_mass_m",
            "principal_axes_xyzw",
            "principal_moments_kg_m2",
            "inertia_kg_m2",
            "tolerances",
        },
        label=label,
    )
    name = _require_string(data["name"], label=f"{label}.name")
    center_of_mass = _require_vector(
        data["center_of_mass_m"],
        length=3,
        label=f"{label}.center_of_mass_m",
    )
    principal_moments = _require_vector(
        data["principal_moments_kg_m2"],
        length=3,
        label=f"{label}.principal_moments_kg_m2",
        positive=True,
    )
    inertia = _require_matrix3(
        data["inertia_kg_m2"],
        label=f"{label}.inertia_kg_m2",
    )
    _require_positive_definite_inertia(
        inertia,
        label=f"{label}.inertia_kg_m2",
    )
    return GroceryBodyDynamics(
        name=name,
        mass_kg=_require_float(
            data["mass_kg"],
            label=f"{label}.mass_kg",
            positive=True,
        ),
        center_of_mass_m=(
            center_of_mass[0],
            center_of_mass[1],
            center_of_mass[2],
        ),
        principal_axes_xyzw=_require_normalized_quaternion(
            data["principal_axes_xyzw"],
            label=f"{label}.principal_axes_xyzw",
        ),
        principal_moments_kg_m2=(
            principal_moments[0],
            principal_moments[1],
            principal_moments[2],
        ),
        inertia_kg_m2=inertia,
        tolerances=_load_tolerances(
            data["tolerances"],
            label=f"{label}.tolerances",
        ),
    )


def _load_sources(value: object) -> tuple[GroceryDynamicsSourcePin, ...]:
    data = _require_exact_keys(
        value,
        set(_SOURCE_PINS_BY_ID),
        label="identity.sources",
    )
    sources: list[GroceryDynamicsSourcePin] = []
    for expected in GROCERY_DYNAMICS_SOURCE_PINS:
        label = f"identity.sources.{expected.source_id}"
        source = _require_exact_keys(
            data[expected.source_id],
            {"uri_suffix", "sha256"},
            label=label,
        )
        uri_suffix = _require_string(
            source["uri_suffix"],
            label=f"{label}.uri_suffix",
        )
        sha256 = _require_sha256(
            source["sha256"],
            label=f"{label}.sha256",
        )
        if uri_suffix != expected.uri_suffix:
            raise GroceryDynamicsCertificateError(
                f"{label}.uri_suffix does not match the pinned source"
            )
        if sha256 != expected.sha256:
            raise GroceryDynamicsCertificateError(
                f"{label}.sha256 does not match the pinned source"
            )
        sources.append(expected)
    return tuple(sources)


def _load_document(
    document: object,
    *,
    expected_engine_id: str,
    expected_engine_version: str,
) -> GroceryDynamicsCertificate:
    expected_engine_id = _require_string(
        expected_engine_id,
        label="expected_engine_id",
    )
    expected_engine_version = _require_string(
        expected_engine_version,
        label="expected_engine_version",
    )
    data = _require_exact_keys(
        document,
        {
            "schema_version",
            "certificate_id",
            "canonical_sha256",
            "identity",
            "bodies",
        },
        label="certificate",
    )
    schema_version = _require_integer(
        data["schema_version"],
        label="schema_version",
    )
    if schema_version != GROCERY_DYNAMICS_CERTIFICATE_SCHEMA_VERSION:
        raise GroceryDynamicsCertificateError(
            f"unsupported schema_version {schema_version}"
        )

    claimed_digest = _require_sha256(
        data["canonical_sha256"],
        label="canonical_sha256",
    )
    actual_digest = canonical_certificate_sha256(data)
    if claimed_digest != actual_digest:
        raise GroceryDynamicsCertificateError(
            "canonical_sha256 does not match the certificate contents"
        )

    identity = _require_exact_keys(
        data["identity"],
        {
            "engine_id",
            "engine_version",
            "stage_meters_per_unit",
            "num_envs",
            "sources",
        },
        label="identity",
    )
    engine_id = _require_string(
        identity["engine_id"],
        label="identity.engine_id",
    )
    engine_version = _require_string(
        identity["engine_version"],
        label="identity.engine_version",
    )
    if engine_id != expected_engine_id:
        raise GroceryDynamicsCertificateError(
            "identity.engine_id does not match the expected runtime"
        )
    if engine_version != expected_engine_version:
        raise GroceryDynamicsCertificateError(
            "identity.engine_version does not match the expected runtime"
        )
    stage_meters_per_unit = _require_float(
        identity["stage_meters_per_unit"],
        label="identity.stage_meters_per_unit",
        positive=True,
    )
    if stage_meters_per_unit != 1.0:
        raise GroceryDynamicsCertificateError(
            "identity.stage_meters_per_unit must equal 1.0"
        )
    num_envs = _require_integer(
        identity["num_envs"],
        label="identity.num_envs",
    )
    if num_envs != 1:
        raise GroceryDynamicsCertificateError("identity.num_envs must equal 1")
    sources = _load_sources(identity["sources"])

    body_values = data["bodies"]
    if not isinstance(body_values, list):
        raise GroceryDynamicsCertificateError("bodies must be a JSON array")
    bodies = tuple(
        _load_body(body, index=index) for index, body in enumerate(body_values)
    )
    names = tuple(body.name for body in bodies)
    if names != GROCERY_DYNAMICS_BODY_ROSTER:
        raise GroceryDynamicsCertificateError(
            "body roster/order does not match the exact 11-body contract: "
            f"expected={GROCERY_DYNAMICS_BODY_ROSTER!r}, actual={names!r}"
        )

    return GroceryDynamicsCertificate(
        certificate_id=_require_string(
            data["certificate_id"],
            label="certificate_id",
        ),
        canonical_sha256=claimed_digest,
        engine_id=engine_id,
        engine_version=engine_version,
        stage_meters_per_unit=stage_meters_per_unit,
        num_envs=num_envs,
        sources=sources,
        bodies=bodies,
    )


def load_grocery_dynamics_certificate(
    path: str | Path,
    *,
    expected_engine_id: str,
    expected_engine_version: str,
) -> GroceryDynamicsCertificate:
    """Load and validate one exact CAP grocery dynamics certificate."""

    certificate_path = Path(path)
    try:
        size = certificate_path.stat().st_size
    except OSError as exc:
        raise GroceryDynamicsCertificateError(
            f"cannot stat dynamics certificate: {certificate_path}"
        ) from exc
    if size > _MAX_CERTIFICATE_BYTES:
        raise GroceryDynamicsCertificateError(
            f"dynamics certificate exceeds {_MAX_CERTIFICATE_BYTES} bytes"
        )
    try:
        payload = certificate_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise GroceryDynamicsCertificateError(
            f"cannot read dynamics certificate: {certificate_path}"
        ) from exc
    try:
        document = json.loads(
            payload,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonfinite_json,
        )
    except json.JSONDecodeError as exc:
        raise GroceryDynamicsCertificateError(
            f"dynamics certificate is not valid JSON: {exc.msg}"
        ) from exc
    return _load_document(
        document,
        expected_engine_id=expected_engine_id,
        expected_engine_version=expected_engine_version,
    )
