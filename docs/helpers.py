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
import re


def version_sort(versions: list) -> list:
    """Order sphinx-multiversion version objects for the header switcher.

    Non-numeric refs (e.g. ``main``) come first, then semantic versions descending, with
    a bare release ranked above its own prereleases (``0.3.0`` before ``0.3.0-prerelease``).
    """

    def key(version):
        label = version.name.removeprefix("release/")
        match = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?(.*)", label)
        if not match:
            return (0,)  # non-numeric refs (e.g. main) sort first
        major, minor, patch, suffix = match.groups()
        is_release = 1 if not suffix else 0  # a bare version outranks its prereleases
        return (1, -int(major), -int(minor), -int(patch or 0), -is_release)

    return sorted(versions, key=key)


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
