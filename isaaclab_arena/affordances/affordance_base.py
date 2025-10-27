# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod


class AffordanceBase(ABC):
    """Base class for affordances."""

    @property
    @abstractmethod
    def name(self) -> str:
        # NOTE(alexmillane, 2025.09.19) Affordances always have be combined with
        # an Asset which has a "name" property. By declaring this property
        # abstract here, we enforce this.
        pass
