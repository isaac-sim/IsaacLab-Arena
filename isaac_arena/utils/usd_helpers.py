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


from pxr import Usd, UsdPhysics


def get_all_prims(
    stage: Usd.Stage, prim: Usd.Prim | None = None, prims_list: list[Usd.Prim] | None = None
) -> list[Usd.Prim]:
    """Get all prims in the stage.

    Performs a Depth First Search (DFS) through the prims in a stage
    and returns all the prims.

    Args:
        stage: The stage to get the prims from.
        prim: The prim to start the search from. Defaults to the pseudo-root.
        prims_list: The list to store the prims in. Defaults to an empty list.

    Returns:
        A list of prims in the stage.
    """
    if prims_list is None:
        prims_list = []
    if prim is None:
        prim = stage.GetPseudoRoot()
    for child in prim.GetAllChildren():
        prims_list.append(child)
        get_all_prims(stage, child, prims_list)
    return prims_list


def is_articulation_root(prim: Usd.Prim) -> bool:
    """Check if prim is articulation root"""
    return prim.HasAPI(UsdPhysics.ArticulationRootAPI)


def is_rigid_body(prim: Usd.Prim) -> bool:
    """Check if prim is rigidbody"""
    return prim.HasAPI(UsdPhysics.RigidBodyAPI)


def get_prim_depth(prim: Usd.Prim) -> int:
    """Get the depth of a prim"""
    return len(str(prim.GetPath()).split("/")) - 2
