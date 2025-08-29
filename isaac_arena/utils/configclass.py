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

import dataclasses
import keyword
import types
from typing import Any

from isaaclab.utils import configclass


# NOTE(alexmillane, 2025-07-24): This is copied from dataclasses.py, but altered in the final line
# to produce a configclass instead of a dataclass
# NOTE(alexmillane, 2025-07-24): The file this was taken from has no license header.
def make_configclass(
    cls_name,
    fields,
    *,
    bases=(),
    namespace=None,
    init=True,
    repr=True,
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=False,
    match_args=True,
    kw_only=False,
    slots=False,
):
    """Return a new dynamically created dataclass.

    The dataclass name will be 'cls_name'.  'fields' is an iterable
    of either (name), (name, type) or (name, type, Field) objects. If type is
    omitted, use the string 'typing.Any'.  Field objects are created by
    the equivalent of calling 'field(name, type [, Field-info])'.

      C = make_dataclass('C', ['x', ('y', int), ('z', int, field(init=False))], bases=(Base,))

    is equivalent to:

      @dataclass
      class C(Base):
          x: 'typing.Any'
          y: int
          z: int = field(init=False)

    For the bases and namespace parameters, see the builtin type() function.

    The parameters init, repr, eq, order, unsafe_hash, and frozen are passed to
    dataclass().
    """

    if namespace is None:
        namespace = {}

    # While we're looking through the field names, validate that they
    # are identifiers, are not keywords, and not duplicates.
    seen = set()
    annotations = {}
    defaults = {}
    for item in fields:
        if isinstance(item, str):
            name = item
            tp = "typing.Any"
        elif len(item) == 2:
            (
                name,
                tp,
            ) = item
        elif len(item) == 3:
            name, tp, spec = item
            defaults[name] = spec
        else:
            raise TypeError(f"Invalid field: {item!r}")

        if not isinstance(name, str) or not name.isidentifier():
            raise TypeError(f"Field names must be valid identifiers: {name!r}")
        if keyword.iskeyword(name):
            raise TypeError(f"Field names must not be keywords: {name!r}")
        if name in seen:
            raise TypeError(f"Field name duplicated: {name!r}")

        seen.add(name)
        annotations[name] = tp

    # Update 'ns' with the user-supplied namespace plus our calculated values.
    def exec_body_callback(ns):
        ns.update(namespace)
        ns.update(defaults)
        ns["__annotations__"] = annotations

    # We use `types.new_class()` instead of simply `type()` to allow dynamic creation
    # of generic dataclasses.
    cls = types.new_class(cls_name, bases, {}, exec_body_callback)

    # Apply the normal decorator.
    return configclass(
        cls,
        init=init,
        repr=repr,
        eq=eq,
        order=order,
        unsafe_hash=unsafe_hash,
        frozen=frozen,
        match_args=match_args,
        kw_only=kw_only,
        slots=slots,
    )


def get_field_info(config_class: configclass) -> list[tuple[str, type, Any]]:
    """Get the field info of a configclass.

    Args:
        config_class: The configclass to get the field info of.

    Returns:
        A list of tuples, where each tuple contains:
            - the name
      - the type
      - and the (optional) default value or default factory of a field.
    """
    field_info_list = []
    for f in dataclasses.fields(config_class):
        field_info = (f.name, f.type)
        if f.default is not dataclasses.MISSING:
            field_info += (f.default,)
        elif f.default_factory is not dataclasses.MISSING:
            field_info += (f.default_factory,)
        field_info_list.append(field_info)
    return field_info_list


def combine_configclasses(name: str, *input_configclasses: configclass) -> configclass:
    """Combine a list of configclasses into a single configclass.

    Args:
        name: The name of the new configclass.
        input_configclasses: The configclasses to combine.

    Returns:
        A new configclass that is the combination of the input configclasses.
    """
    field_info_list = []
    for d in input_configclasses:
        field_info_list.extend(get_field_info(d))
    # Check for duplicate field names
    names = [f[0] for f in field_info_list]
    unique_names = list(set(names))
    assert len(unique_names) == len(names), "Duplicate field names in the input configclasses"
    return make_configclass(name, field_info_list)


def combine_configclass_instances(name: str, *input_configclass_instances: configclass) -> configclass:
    """Combine a list of configclass instances into a single configclass instance.

    Args:
        name: The name of the new configclass.
        input_configclass_instances: The configclass instances to combine.

    Returns:
        A new configclass instance that is the combination of the input configclass instances.
    """
    input_configclass_instances_not_none = [i for i in input_configclass_instances if i is not None]
    input_configclasses_not_none: list[type] = [type(i) for i in input_configclass_instances_not_none]
    combined_configclass = combine_configclasses(name, *input_configclasses_not_none)
    # Create an instance of the combined type
    combined_configclass_instance = combined_configclass()
    # Copy in the values from the input configclass instances
    for configclass_instance in input_configclass_instances_not_none:
        for field in dataclasses.fields(configclass_instance):
            setattr(combined_configclass_instance, field.name, getattr(configclass_instance, field.name))
    return combined_configclass_instance
