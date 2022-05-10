from typing import Optional

import torch
from torch import nn, Tensor

from yoky.nn import functional


class WeightSoftmax(nn.Module):
    def __init__(self, dim: Optional[int] = None) -> None:
        super(WeightSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor, weight: Tensor) -> Tensor:
        return functional.weight_softmax(x, weight, self.dim)

    def extra_repr(self) -> str:
        return 'dim={dim}'.format(dim=self.dim)


class Sinh(nn.Module):
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return torch.sinh(x)
