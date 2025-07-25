import inspect
import random
from typing import Any, List

import isaac_arena.scene.background.background as background
import isaac_arena.scene.pick_up_object.pick_up_object as pick_up_object


class SingletonMeta(type):
    """
    Metaclass that overrides __call__ so that only one instance
    of any class using it is ever created.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # first time: actually create the instance
            cls._instances[cls] = super().__call__(*args, **kwargs)
        # afterwards: always return the same object
        return cls._instances[cls]


class ObjectRegistry(metaclass=SingletonMeta):

    def __init__(self):
        self.registry = {}
        self.register_in_built_objects()

    def register_in_built_objects(self):
        for _, cls in inspect.getmembers(background, inspect.isclass):
            if issubclass(cls, background.Background) and cls is not background.Background:
                instance = cls()
                self.register_object(instance.get_name(), instance)
        for _, cls in inspect.getmembers(pick_up_object, inspect.isclass):
            if issubclass(cls, pick_up_object.PickUpObject) and cls is not pick_up_object.PickUpObject:
                instance = cls()
                self.register_object(instance.get_name(), instance)

    def register_object(self, name: str, object):
        assert name not in self.registry, f"Object {name} already registered"
        self.registry[name] = object

    def get_object_by_name(self, name) -> Any:
        return self.registry[name]

    def get_objects_by_tag(self, tag) -> list[Any]:
        return [object for object in self.registry.values() if tag in object.tags]

    def get_random_object_by_tag(self, tag) -> Any:
        return random.choice(self.get_objects_by_tag(tag))
