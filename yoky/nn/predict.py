import torch
from torch import Tensor

from yoky import nn


class Locator(nn.Module):
    def __init__(self, hidden_size: int, coordinate_dim: int = 2):
        super().__init__()
        self.coordinate_dim = coordinate_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, coordinate_dim * (coordinate_dim + 1))
        )

    def forward(self, queries: Tensor) -> tuple[Tensor, Tensor]:
        """

        :param queries: [..., hidden_size]
        :return:
            'centers': [..., 2(start&end)]
            'factors': [..., 2, 2]
        """
        splits = [self.coordinate_dim, self.coordinate_dim * self.coordinate_dim]
        offsets, factors = torch.split(self.mlp(queries), splits, dim=-1)
        factors = factors.view(*factors.shape[:-1], self.coordinate_dim, self.coordinate_dim)
        factors = factors @ factors.transpose(-1, -2)
        return offsets, factors


class Classifier(nn.Module):
    def __init__(self, hidden_size: int, types_num: int, additional_none: bool = True):
        super().__init__()
        self.labels_num = types_num
        if additional_none:
            self.none_label = types_num
            self.labels_num += 1

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, self.labels_num)
        )

    def forward(self, queries: Tensor) -> Tensor:
        """

        :param queries: [..., features_num, hidden_size]
        :return: [..., features_num, labels_num]
        """
        return self.mlp(queries)
