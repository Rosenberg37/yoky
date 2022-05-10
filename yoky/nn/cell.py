import torch
from torch import nn, Tensor

from yoky import nn


class GateCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = input_dim if hidden_dim is None else hidden_dim

        self.matrix_xr = nn.Linear(input_dim, hidden_dim)
        self.matrix_hr = nn.Linear(input_dim, hidden_dim)

        self.matrix_xz = nn.Linear(input_dim, hidden_dim)
        self.matrix_hz = nn.Linear(input_dim, hidden_dim)

        self.matrix_xn = nn.Linear(input_dim, hidden_dim)
        self.matrix_hn = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tensor:
        """

        :param x: [*, input_dim]
        :param h: [*, input_dim]
        :return: [*, hidden_dim]
        """
        r = torch.sigmoid(self.matrix_xr(x) + self.matrix_hr(h))
        z = torch.sigmoid(self.matrix_xz(x) + self.matrix_hz(h))
        n = torch.tanh(self.matrix_xn(x) + r * self.matrix_hn(h))
        return (1 - z) * n + z * h
