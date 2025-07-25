from typing import List


class Asset:
    """
    Base class for all assets.
    """

    def __init__(self, name: str, tags: list[str]):
        self.name = name
        self.tags = tags

    def get_name(self) -> str:
        return self.name

    def get_tags(self) -> list[str]:
        return self.tags
