# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import keyword
import types
from collections import OrderedDict
from collections.abc import Callable
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


def get_field_info(field: dataclasses.Field) -> tuple[str, type, Any]:
    """Get the field info of a configclass.
    Args:
        config_class: The configclass to get the field info of.
    Returns:
        A list of tuples, where each tuple contains:
            - the name
      - the type
      - and the (optional) default value or default factory of a field.
    """
    field_info = (field.name, field.type)
    if field.default is not dataclasses.MISSING:
        field_info += (field.default,)
    elif field.default_factory is not dataclasses.MISSING:
        field_info += (field.default_factory,)
    return field_info


def combine_configclasses(name: str, *input_configclasses: type, bases: tuple[type, ...] = ()) -> configclass:
    field_map: "OrderedDict[str, tuple]" = OrderedDict()
    for cls in input_configclasses:
        for current_field in dataclasses.fields(cls):
            if current_field.name in field_map:
                previous_field = field_map[current_field.name]
                # same → skip; different types → error; else last wins
                # The 1 field index is the type
                if previous_field[1] == current_field.type:
                    continue
                raise ValueError(f"Field {current_field.name} has different types in the input configclasses")
            field_info = get_field_info(current_field)
            field_map[current_field.name] = field_info

    new_configclass = make_configclass(name, list(field_map.values()), bases=bases)
    new_configclass.__post_init__ = combine_post_inits(*input_configclasses)
    return new_configclass


def combine_configclass_instances(
    name: str, *input_configclass_instances: type, bases: tuple[type, ...] = ()
) -> configclass:
    """Combine a list of configclass instances into a single configclass instance.

    Args:
        name: The name of the new configclass.
        input_configclass_instances: The configclass instances to combine.

    Returns:
        A new configclass instance that is the combination of the input configclass instances.
    """
    input_configclass_instances_not_none = [i for i in input_configclass_instances if i is not None]
    input_configclasses_not_none: list[type] = [type(i) for i in input_configclass_instances_not_none]
    combined_configclass = combine_configclasses(name, *input_configclasses_not_none, bases=bases)
    # Create an instance of the combined type
    combined_configclass_instance = combined_configclass()
    # Copy in the values from the input configclass instances
    for configclass_instance in input_configclass_instances_not_none:
        for field in dataclasses.fields(configclass_instance):
            setattr(combined_configclass_instance, field.name, getattr(configclass_instance, field.name))
    return combined_configclass_instance


def combine_post_inits(*cls_list: type) -> Callable:
    """Takes a list of classes and returns a function that calls the
    __post_init__ method of each class.

    Args:
        cls_list: The list of classes to combine.

    Returns:
        A function that calls the __post_init__ method of each class.
    """
    post_init_list: list[Callable] = []
    seen: set[object] = set()
    for cls in cls_list:
        f = getattr(cls, "__post_init__", None)
        if f is None:
            continue
        key = getattr(f, "__func__", f)  # handle bound methods just in case
        if key in seen:
            continue
        seen.add(key)
        post_init_list.append(f)

    def new_post_init(self):
        for post_init in post_init_list:
            post_init(self)

    return new_post_init
