# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module


def import_class_from_path(class_path: str, separator: str = ".") -> type:
    """Import the class named by a ``module_path<separator>ClassName`` path.

    Args:
        class_path: Module path and class name joined by ``separator``, for example ``my_package.envs.MyEnvironment``.
        separator: Text between the module path and the class name.
    """
    assert (
        separator in class_path
    ), f"Expected a class path of the form 'module_path{separator}ClassName', got {class_path!r}"
    module_path, class_name = class_path.rsplit(separator, 1)
    imported_object = getattr(import_module(module_path), class_name, None)
    assert imported_object is not None, f"{class_path!r}: {module_path} has no {class_name}"
    assert isinstance(imported_object, type), f"{class_path!r} is not a class"
    return imported_object
