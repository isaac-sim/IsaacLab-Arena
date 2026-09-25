# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for decorator-based Isaac CAP component registration."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

pytestmark = pytest.mark.isaac_cap


def _test_isaac_cap_components_registered(_simulation_app) -> bool:
    from isaaclab_arena.assets.registries import AssetRegistry, EnvironmentRegistry, PolicyRegistry, TaskRegistry
    from isaaclab_arena_environments.isaac_cap import register_components
    from isaaclab_arena_environments.isaac_cap import registration as cap_registration

    register_components()

    from isaaclab_arena_environments.isaac_cap import cap_policy
    from isaaclab_arena_environments.isaac_cap.cable_routing import environment as cable_environment
    from isaaclab_arena_environments.isaac_cap.cable_routing import task as cable_task
    from isaaclab_arena_environments.isaac_cap.cable_routing_v2 import environment as cable_v2_environment
    from isaaclab_arena_environments.isaac_cap.cable_routing_v2 import task as cable_v2_task
    from isaaclab_arena_environments.isaac_cap.embodiments import cable_routing as cable_embodiment
    from isaaclab_arena_environments.isaac_cap.embodiments import insertion_task as insertion_embodiment
    from isaaclab_arena_environments.isaac_cap.gear_insertion import asset_factories as gear_assets
    from isaaclab_arena_environments.isaac_cap.gear_insertion import task as gear_task
    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2 import asset_factories as gear_v2_assets
    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2 import embodiment as gear_v2_embodiment
    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2 import gear_mesh_environment
    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2.task import task as gear_v2_task
    from isaaclab_arena_environments.isaac_cap.syringe_sort.environments import assets as syringe_assets
    from isaaclab_arena_environments.isaac_cap.syringe_sort.environments import environment as syringe_environment
    from isaaclab_arena_environments.isaac_cap.syringe_sort.tasks import task as syringe_task
    from isaaclab_arena_environments.isaac_cap.tool_sorting import assets as tool_sorting_assets
    from isaaclab_arena_environments.isaac_cap.tool_sorting import embodiment as tool_sorting_embodiment
    from isaaclab_arena_environments.isaac_cap.tool_sorting import task as tool_sorting_task
    from isaaclab_arena_environments.isaac_cap.usbc_insertion import assets as usbc_assets
    from isaaclab_arena_environments.isaac_cap.usbc_insertion import environment as usbc_environment
    from isaaclab_arena_environments.isaac_cap.usbc_insertion import task as usbc_task

    expected_assets = {
        "industrial_fr3_robotiq_2f85": insertion_embodiment.IndustrialFr3Robotiq2f85Embodiment,
        "industrial_fr3_robotiq_2f85_differential_ik": (
            insertion_embodiment.IndustrialFr3Robotiq2f85DifferentialIKEmbodiment
        ),
        "industrial_bimanual_yam": cable_embodiment.IndustrialBimanualYamEmbodiment,
        "factory_gear_base": gear_assets.make_factory_gear_base,
        "factory_gear_small": gear_assets.make_factory_gear_small,
        "factory_gear_medium": gear_assets.make_factory_gear_medium,
        "factory_gear_large": gear_assets.make_factory_gear_large,
        "fr3_workcell_table": gear_assets.IndustrialFr3WorkcellTable,
        "hdr_shadow_receiver": gear_assets.IndustrialHdrShadowReceiver,
        "empty_warehouse_dome_light": gear_assets.IndustrialEmptyWarehouseDomeLight,
        "syringe": syringe_assets.SyringeRedCap,
        "syringe_blank": syringe_assets.SyringeWhiteCap,
        "instrument_tray": syringe_assets.InstrumentTray,
        "sharps_container": syringe_assets.SharpsContainer,
        "usbc_insertion_easy_plug": usbc_assets.UsbcEasyPlug,
        "usbc_insertion_medium_plug": usbc_assets.UsbcMediumPlug,
        "usbc_insertion_easy_port": usbc_assets.UsbcEasyPort,
        "usbc_insertion_bulkhead": usbc_assets.UsbcBulkhead,
        "usbc_insertion_bench": usbc_assets.UsbcBench,
        "usbc_insertion_cradle_front": usbc_assets.UsbcCradleFront,
        "usbc_insertion_cradle_rear": usbc_assets.UsbcCradleRear,
        "usbc_insertion_yam_table": usbc_assets.UsbcYamWorkcellTable,
        "usbc_insertion_hdr_shadow_receiver": usbc_assets.UsbcHdrShadowReceiver,
        "usbc_insertion_dome_light": usbc_assets.UsbcDomeLight,
        "usbc_insertion_connector_cable": usbc_assets.UsbcConnectorCable,
        "industrial__gear_mesh_16t": gear_v2_assets.make_gear_mesh_16t,
        "industrial__gear_mesh_20t": gear_v2_assets.make_gear_mesh_20t,
        "industrial__gear_mesh_24t": gear_v2_assets.make_gear_mesh_24t,
        "industrial__gear_mesh_board_16": gear_v2_assets.make_gear_mesh_board_16,
        "industrial__gear_mesh_board_20": gear_v2_assets.make_gear_mesh_board_20,
        "industrial__gear_mesh_board_24": gear_v2_assets.make_gear_mesh_board_24,
        "industrial__gear_mesh_mat": gear_v2_assets.make_gear_mesh_mat,
        "industrial__fr3_workcell_table": gear_v2_assets.IndustrialFr3WorkcellTable,
        "industrial__hdr_shadow_receiver": gear_v2_assets.IndustrialHdrShadowReceiver,
        "industrial__empty_warehouse_dome_light": gear_v2_assets.IndustrialEmptyWarehouseDomeLight,
        "industrial_fr3_robotiq_2f85_v2": gear_v2_embodiment.IndustrialFr3Robotiq2f85Embodiment,
        "industrial_fr3_robotiq_2f85_differential_ik_v2": (
            gear_v2_embodiment.IndustrialFr3Robotiq2f85DifferentialIKEmbodiment
        ),
        "tool_sorting_fr3_robotiq_2f85": tool_sorting_embodiment.ToolSortingFr3Robotiq2f85Embodiment,
        "vabar_tool_sort__adjustable_wrench": tool_sorting_assets.IndustrialToolSortAdjustableWrench,
        "vabar_tool_sort__battery": tool_sorting_assets.IndustrialToolSortBattery,
        "vabar_tool_sort__breadboard": tool_sorting_assets.IndustrialToolSortBreadboard,
        "vabar_tool_sort__combination_pliers": tool_sorting_assets.IndustrialToolSortCombinationPliers,
        "vabar_tool_sort__cutting_pliers": tool_sorting_assets.IndustrialToolSortCuttingPliers,
        "vabar_tool_sort__flashlight": tool_sorting_assets.IndustrialToolSortFlashlight,
        "vabar_tool_sort__insulating_tape": tool_sorting_assets.IndustrialToolSortInsulatingTape,
        "vabar_tool_sort__multimeter": tool_sorting_assets.IndustrialToolSortMultimeter,
        "vabar_tool_sort__safety_glasses": tool_sorting_assets.IndustrialToolSortSafetyGlasses,
        "vabar_tool_sort__slotted_screwdriver": tool_sorting_assets.IndustrialToolSortSlottedScrewdriver,
        "vabar_tool_sort__tape_measure": tool_sorting_assets.IndustrialToolSortTapeMeasure,
        "vabar_tool_sort__wire_spool": tool_sorting_assets.IndustrialToolSortWireSpool,
        "industrial__tool_sort_bin": tool_sorting_assets.IndustrialToolSortBin,
    }
    asset_registry = AssetRegistry()
    for name, component in expected_assets.items():
        assert asset_registry.get_component_by_name(name) is component

    expected_tasks = {
        "GearInsertionTask": gear_task.GearInsertionTask,
        "CableRoutingTask": cable_task.CableRoutingTask,
        "SyringeSortTask": syringe_task.SyringeSortTask,
        "UsbcInsertionTask": usbc_task.UsbcInsertionTask,
        "GearMeshTaskV2": gear_v2_task.GearMeshTaskV2,
        "CableRoutingTaskV2": cable_v2_task.CableRoutingTaskV2,
        "ObjectsInRegionsTask": tool_sorting_task.ObjectsInRegionsTask,
    }
    task_registry = TaskRegistry()
    for name, component in expected_tasks.items():
        assert task_registry.get_component_by_name(name) is component

    expected_environments = {
        "cable_routing__medium": (
            cable_environment.CableRoutingMediumEnvironment,
            cable_environment.CableRoutingMediumEnvironmentCfg,
        ),
        "cable_routing__easy": (
            cable_environment.CableRoutingEasyEnvironment,
            cable_environment.CableRoutingEasyEnvironmentCfg,
        ),
        "syringe_single_newton": (
            syringe_environment.SyringeSingleEnvironment,
            syringe_environment.SyringeSortEnvironmentCfg,
        ),
        "syringe_both_newton": (
            syringe_environment.SyringeBothEnvironment,
            syringe_environment.SyringeBothEnvironmentCfg,
        ),
        "syringe_cluttered_newton": (
            syringe_environment.SyringeClutteredEnvironment,
            syringe_environment.SyringeClutteredEnvironmentCfg,
        ),
        "vabar_contact_rich_insertion__usbc_insertion_easy": (
            usbc_environment.UsbcInsertionEasyEnvironment,
            usbc_environment.UsbcInsertionEasyEnvironmentCfg,
        ),
        "vabar_contact_rich_insertion__usbc_insertion_medium": (
            usbc_environment.UsbcInsertionMediumEnvironment,
            usbc_environment.UsbcInsertionMediumEnvironmentCfg,
        ),
        "vabar_contact_rich_insertion_v2__gear_easy": (
            gear_mesh_environment.GearInsertionEasyNewtonEnvironment,
            gear_mesh_environment.GearInsertionEasyNewtonEnvironmentCfg,
        ),
        "vabar_contact_rich_insertion_v2__gear_easy_pair": (
            gear_mesh_environment.GearMeshPairNewtonEnvironment,
            gear_mesh_environment.GearMeshPairNewtonEnvironmentCfg,
        ),
        "vabar_contact_rich_insertion_v2__gear_medium_train": (
            gear_mesh_environment.GearMeshTrainNewtonEnvironment,
            gear_mesh_environment.GearMeshTrainNewtonEnvironmentCfg,
        ),
        "vabar_cable_routing_v2__medium": (
            cable_v2_environment.CableRoutingMediumEnvironment,
            cable_v2_environment.CableRoutingMediumEnvironmentCfg,
        ),
        "vabar_cable_routing_v2__easy": (
            cable_v2_environment.CableRoutingEasyEnvironment,
            cable_v2_environment.CableRoutingEasyEnvironmentCfg,
        ),
    }
    environment_registry = EnvironmentRegistry()
    for name, (factory_type, cfg_type) in expected_environments.items():
        assert environment_registry.get_component_by_name(name) is factory_type
        assert environment_registry.get_environment_cfg_type(factory_type) is cfg_type

    policy_registry = PolicyRegistry()
    assert policy_registry.get_component_by_name("cap_remote") is cap_policy.CapPolicy
    assert policy_registry.get_policy_cfg_type(cap_policy.CapPolicy) is cap_policy.CapPolicyCfg

    class ConflictingEnvironment:
        name = "vabar_cable_routing_v2__easy"

    with pytest.raises(AssertionError, match="Conflicting Isaac CAP environment registration"):
        cap_registration.register_environment(cfg_type=cable_v2_environment.CableRoutingEasyEnvironmentCfg)(
            ConflictingEnvironment
        )

    return True


def test_isaac_cap_components_registered() -> None:
    """Isaac CAP graph names resolve to their decorator-registered components."""
    assert run_function_with_persistent_simulation_app(_test_isaac_cap_components_registered)
