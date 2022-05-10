from abc import abstractmethod
from argparse import Namespace
from typing import Any


class AttributeHolder(Namespace):
    def __init__(self, attributes: dict = None, **kwargs):
        super().__init__(**kwargs)
        if attributes is not None:
            self.__dict__.update(attributes)

    def update(self, attributes: dict):
        self.__dict__.update(attributes)

    def add(self, name: str, attribute: Any):
        setattr(self, name, attribute)

    def pop(self, name: str):
        delattr(self, name)


class Arguments(AttributeHolder):
    @abstractmethod
    def parse(self, *args, **kwargs):
        pass
