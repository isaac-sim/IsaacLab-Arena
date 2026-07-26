# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

#
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import dataclasses
import datetime
import pathlib
import tomllib


def get_project_version() -> str:
    """Return the ``project.version`` string from the repository's pyproject.toml.

    Reading the version at build time keeps the docs header in sync with the
    checked-out source. sphinx-multiversion builds each ref from its own source
    tree, so this yields the correct version for every documented version.
    """
    pyproject_path = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as file:
        pyproject = tomllib.load(file)
    return pyproject["project"]["version"]


def to_datetime(date_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(date_str, "%d.%m.%Y")


def is_expired(start_date: datetime.datetime, days: int) -> bool:
    today = datetime.datetime.now()
    delta = datetime.timedelta(days=days)
    return today > (start_date + delta)


@dataclasses.dataclass
class TemporaryLinkcheckIgnore:
    url: str
    start_date: datetime.datetime
    days: int
