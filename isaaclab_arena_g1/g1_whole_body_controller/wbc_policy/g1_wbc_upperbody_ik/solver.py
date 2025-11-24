# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any


class Solver(ABC):
    def __init__(self):
        pass

    def register_robot(self, robot):
        pass

    def calibrate(self, data):
        pass

    @abstractmethod
    def __call__(self, target) -> Any:
        pass
