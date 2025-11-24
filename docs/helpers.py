# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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
