import torch
from torch import Tensor

from yoky import nn


class MaxPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.max(x, dim=self.dim)[0]
