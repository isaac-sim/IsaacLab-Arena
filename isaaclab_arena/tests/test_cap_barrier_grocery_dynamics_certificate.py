# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import FrozenInstanceError

import pytest

from isaaclab_arena.integrations.cap_barrier.grocery_dynamics_certificate import (
    GROCERY_DYNAMICS_BODY_ROSTER,
    GROCERY_DYNAMICS_SOURCE_PINS,
    GroceryDynamicsCertificateError,
    canonical_certificate_sha256,
    load_grocery_dynamics_certificate,
)

_ENGINE_ID = "physx"
_ENGINE_VERSION = "isaac-sim-6.0.0.1"
_TOLERANCES = {
    "mass_kg": 1.0e-9,
    "center_of_mass_m": 1.0e-9,
    "principal_axes_angle_rad": 1.0e-8,
    "principal_moments_kg_m2": 1.0e-12,
    "inertia_kg_m2": 1.0e-12,
}


def _body(name: str, index: int) -> dict[str, object]:
    scale = float(index + 1)
    moments = [0.001 * scale, 0.002 * scale, 0.003 * scale]
    return {
        "name": name,
        "mass_kg": 0.1 * scale,
        "center_of_mass_m": [0.001 * index, 0.0, -0.001 * index],
        "principal_axes_xyzw": [0.0, 0.0, 0.0, 1.0],
        "principal_moments_kg_m2": moments,
        "inertia_kg_m2": [
            [moments[0], 0.0, 0.0],
            [0.0, moments[1], 0.0],
            [0.0, 0.0, moments[2]],
        ],
        "tolerances": dict(_TOLERANCES),
    }


def _certificate() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "certificate_id": "arena_droid_b1.grocery_dynamics.v1.test",
        "identity": {
            "engine_id": _ENGINE_ID,
            "engine_version": _ENGINE_VERSION,
            "stage_meters_per_unit": 1.0,
            "num_envs": 1,
            "sources": {
                pin.source_id: {
                    "uri_suffix": pin.uri_suffix,
                    "sha256": pin.sha256,
                }
                for pin in GROCERY_DYNAMICS_SOURCE_PINS
            },
        },
        "bodies": [
            _body(name, index)
            for index, name in enumerate(GROCERY_DYNAMICS_BODY_ROSTER)
        ],
    }
    value["canonical_sha256"] = canonical_certificate_sha256(value)
    return value


def _write(tmp_path, value: object, *, raw: str | None = None):
    path = tmp_path / "dynamics.json"
    path.write_text(
        raw if raw is not None else json.dumps(value, indent=2),
        encoding="utf-8",
    )
    return path


def _load(tmp_path, value: object):
    return load_grocery_dynamics_certificate(
        _write(tmp_path, value),
        expected_engine_id=_ENGINE_ID,
        expected_engine_version=_ENGINE_VERSION,
    )


def _resign(value: dict[str, object]) -> dict[str, object]:
    value["canonical_sha256"] = canonical_certificate_sha256(value)
    return value


def test_roster_and_raw_asset_identities_are_literal_pins() -> None:
    assert GROCERY_DYNAMICS_BODY_ROSTER == (
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
    assert {
        pin.source_id: (pin.uri_suffix, pin.sha256)
        for pin in GROCERY_DYNAMICS_SOURCE_PINS
    } == {
        "robot": (
            "/Arena/assets/robot_library/droid/franka_robotiq_2f_85_flattened.usd",
            "c8d72259834e2e5290754f8580b37efbc0dec079ac6a98b27b167efe6461eb2c",
        ),
        "can": (
            "/Arena/assets/object_library/srl_robolab_assets/"
            "objects/hope/alphabet_soup_can.usd",
            "84b730aecba2efed94eeba61befd152d619f59810463bc6b23b3cc3e54836dfb",
        ),
        "bin": (
            "/Arena/assets/object_library/srl_robolab_assets/fixtures/grey_bin.usd",
            "e4c1c2d34dce8d642b3ffd8a1b468ecd40f6b8bd2b365f3d41592eb191ce99ea",
        ),
    }


def test_loads_exact_immutable_eleven_body_certificate(tmp_path) -> None:
    certificate = _load(tmp_path, _certificate())

    assert tuple(body.name for body in certificate.bodies) == (
        GROCERY_DYNAMICS_BODY_ROSTER
    )
    assert certificate.sources == GROCERY_DYNAMICS_SOURCE_PINS
    assert certificate.body("can").mass_kg == pytest.approx(1.0)
    assert certificate.body("bin").principal_moments_kg_m2 == pytest.approx(
        (0.011, 0.022, 0.033)
    )
    with pytest.raises(FrozenInstanceError):
        certificate.bodies[0].mass_kg = 7.0
    with pytest.raises(KeyError):
        certificate.body("missing")


def test_canonical_hash_excludes_only_self_hash_and_ignores_formatting(
    tmp_path,
) -> None:
    value = _certificate()
    digest = value["canonical_sha256"]
    reordered = {key: value[key] for key in reversed(tuple(value))}

    assert canonical_certificate_sha256(reordered) == digest
    assert _load(tmp_path, reordered).canonical_sha256 == digest

    value["certificate_id"] = "mutated"
    with pytest.raises(
        GroceryDynamicsCertificateError,
        match="canonical_sha256 does not match",
    ):
        _load(tmp_path, value)


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("{", "not valid JSON"),
        (
            '{"schema_version":1,"schema_version":1}',
            "duplicate certificate key",
        ),
        ("[1,2,3]", "must be a JSON object"),
        ('{"value":NaN}', "non-finite JSON number"),
    ],
)
def test_rejects_malformed_duplicate_nonobject_and_nonfinite_json(
    tmp_path,
    raw: str,
    match: str,
) -> None:
    with pytest.raises(GroceryDynamicsCertificateError, match=match):
        load_grocery_dynamics_certificate(
            _write(tmp_path, None, raw=raw),
            expected_engine_id=_ENGINE_ID,
            expected_engine_version=_ENGINE_VERSION,
        )


@pytest.mark.parametrize(
    ("path", "operation"),
    [
        ("certificate", lambda value: value.update({"unknown": 1})),
        ("certificate", lambda value: value.pop("certificate_id")),
        (
            "identity",
            lambda value: value["identity"].update({"unknown": 1}),
        ),
        (
            "identity.sources.robot",
            lambda value: value["identity"]["sources"]["robot"].update({"unknown": 1}),
        ),
        (
            "bodies[0]",
            lambda value: value["bodies"][0].update({"unknown": 1}),
        ),
        (
            "bodies[0].tolerances",
            lambda value: value["bodies"][0]["tolerances"].pop("mass_kg"),
        ),
    ],
)
def test_rejects_unknown_or_missing_keys_at_every_schema_level(
    tmp_path,
    path: str,
    operation,
) -> None:
    value = _certificate()
    operation(value)
    _resign(value)

    with pytest.raises(
        GroceryDynamicsCertificateError,
        match=rf"{re.escape(path)} keys do not match schema",
    ):
        _load(tmp_path, value)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["identity"].update({"engine_id": "newton"}),
            "engine_id does not match",
        ),
        (
            lambda value: value["identity"].update({"engine_version": "different"}),
            "engine_version does not match",
        ),
        (
            lambda value: value["identity"].update({"stage_meters_per_unit": 0.01}),
            "must equal 1.0",
        ),
        (
            lambda value: value["identity"].update({"num_envs": 2}),
            "num_envs must equal 1",
        ),
        (
            lambda value: value["identity"].update({"num_envs": True}),
            "num_envs must be an integer",
        ),
        (
            lambda value: value.update({"schema_version": True}),
            "schema_version must be an integer",
        ),
    ],
)
def test_rejects_runtime_identity_and_unit_drift(
    tmp_path,
    mutation,
    match: str,
) -> None:
    value = _certificate()
    mutation(value)
    _resign(value)

    with pytest.raises(GroceryDynamicsCertificateError, match=match):
        _load(tmp_path, value)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["identity"]["sources"]["robot"].update(
                {"sha256": "0" * 64}
            ),
            "sha256 does not match the pinned source",
        ),
        (
            lambda value: value["identity"]["sources"]["can"].update(
                {"uri_suffix": "/wrong.usd"}
            ),
            "uri_suffix does not match the pinned source",
        ),
        (
            lambda value: value["identity"]["sources"]["bin"].update({"sha256": "ABC"}),
            "lowercase 64-hex",
        ),
        (
            lambda value: value["identity"]["sources"].pop("can"),
            "identity.sources keys do not match schema",
        ),
        (
            lambda value: value["identity"]["sources"].update(
                {"other": {"uri_suffix": "/other", "sha256": "0" * 64}}
            ),
            "identity.sources keys do not match schema",
        ),
    ],
)
def test_rejects_source_roster_uri_and_digest_drift(
    tmp_path,
    mutation,
    match: str,
) -> None:
    value = _certificate()
    mutation(value)
    _resign(value)

    with pytest.raises(GroceryDynamicsCertificateError, match=match):
        _load(tmp_path, value)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda bodies: bodies.pop(),
        lambda bodies: bodies.append(copy.deepcopy(bodies[-1])),
        lambda bodies: bodies.__setitem__(
            1,
            {**bodies[1], "name": "wrong"},
        ),
        lambda bodies: bodies.reverse(),
    ],
)
def test_rejects_missing_extra_wrong_or_reordered_body_roster(
    tmp_path,
    mutation,
) -> None:
    value = _certificate()
    mutation(value["bodies"])
    _resign(value)

    with pytest.raises(
        GroceryDynamicsCertificateError,
        match="body roster/order does not match",
    ):
        _load(tmp_path, value)


@pytest.mark.parametrize(
    ("field", "bad_value", "match"),
    [
        ("mass_kg", True, "must be a real number"),
        ("mass_kg", 0.0, "strictly positive"),
        ("center_of_mass_m", [0.0, 0.0], "array of 3"),
        (
            "principal_axes_xyzw",
            [0.0, 0.0, 0.0, 0.9],
            "must be normalized",
        ),
        (
            "principal_moments_kg_m2",
            [0.1, -0.2, 0.3],
            "strictly positive",
        ),
        (
            "inertia_kg_m2",
            [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "must be symmetric",
        ),
        (
            "inertia_kg_m2",
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            "must be positive definite",
        ),
    ],
)
def test_rejects_invalid_body_properties(
    tmp_path,
    field: str,
    bad_value: object,
    match: str,
) -> None:
    value = _certificate()
    value["bodies"][0][field] = bad_value
    _resign(value)

    with pytest.raises(GroceryDynamicsCertificateError, match=match):
        _load(tmp_path, value)


@pytest.mark.parametrize(
    ("field", "bad_value", "match"),
    [
        ("mass_kg", True, "must be a real number"),
        ("center_of_mass_m", -1.0e-9, "must be nonnegative"),
        (
            "principal_axes_angle_rad",
            float("inf"),
            "non-finite JSON number",
        ),
        ("principal_moments_kg_m2", -1.0, "must be nonnegative"),
        ("inertia_kg_m2", -1.0, "must be nonnegative"),
    ],
)
def test_rejects_invalid_tolerances(
    tmp_path,
    field: str,
    bad_value: object,
    match: str,
) -> None:
    value = _certificate()
    value["bodies"][0]["tolerances"][field] = bad_value
    if math_is_finite_json(bad_value):
        _resign(value)

    with pytest.raises(GroceryDynamicsCertificateError, match=match):
        if not math_is_finite_json(bad_value):
            _write(
                tmp_path,
                value,
                raw=json.dumps(value, allow_nan=True),
            )
            load_grocery_dynamics_certificate(
                tmp_path / "dynamics.json",
                expected_engine_id=_ENGINE_ID,
                expected_engine_version=_ENGINE_VERSION,
            )
        else:
            _load(tmp_path, value)


def math_is_finite_json(value: object) -> bool:
    return not isinstance(value, float) or math.isfinite(value)


def test_rejects_oversized_certificate_before_json_parse(tmp_path) -> None:
    path = _write(tmp_path, None, raw=" " * (256 * 1024 + 1))

    with pytest.raises(
        GroceryDynamicsCertificateError,
        match="exceeds 262144 bytes",
    ):
        load_grocery_dynamics_certificate(
            path,
            expected_engine_id=_ENGINE_ID,
            expected_engine_version=_ENGINE_VERSION,
        )
