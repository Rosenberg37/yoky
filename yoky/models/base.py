from abc import abstractmethod

from yoky import nn


class Model(nn.Module):
    @abstractmethod
    def decode(self, *args, **kwargs):
        pass

    @abstractmethod
    def detail(self, *args, **kwargs):
        pass
