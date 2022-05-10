from typing import Optional

import torch
from torch import Tensor, nn

from yoky import nn
from yoky.nn import functional


class Pyramid(nn.Module):
    def __init__(self, hidden_size: int, kernels_size: list, dropout: float, no_backward: bool = False):
        super().__init__()

        self.kernels_size = kernels_size
        self.visions_size = [1]
        for k in self.kernels_size:
            self.visions_size.append(self.visions_size[-1] + k - 1)

        self.forward_blocks = nn.ModuleList([ForwardBlock(hidden_size, dropout, k) for k in self.kernels_size + [None]])
        if not no_backward:
            self.backward_blocks = nn.ModuleList(
                [BackwardBlock(hidden_size, dropout, *args)
                 for args in zip(self.visions_size, [None] + self.kernels_size)]
            )

    def forward(self, batch_hiddens: Tensor, batch_masks: Tensor) -> tuple[Tensor, ...]:
        """

        :param batch_masks:  [batch_size, sentence_length]
        :param batch_hiddens: [batch_size, sentence_length, hidden_size]
        :return:
            features: [batch_size, num_layers, sentence_length, hidden_size]
            features_locs: [batch_size, num_layers, sentence_length, 2]
            features_masks: [batch_size, num_layers, sentence_length]
        """
        batch_size, sentence_length = batch_hiddens.shape[:2]
        indices = torch.arange(0, sentence_length, device=batch_hiddens.device)

        features = [batch_hiddens]
        features_locs, features_masks = list(), list()

        for i, (block, v_size) in enumerate(zip(self.forward_blocks, self.visions_size)):
            locs_s, locs_e = indices[:sentence_length - v_size + 1], indices[v_size - 1:]
            locs = torch.stack([locs_s, locs_e], dim=-1)
            features_locs.append(locs.unsqueeze(0).expand(batch_size, -1, -1))
            features_masks.append(batch_masks[:, locs_e])

            features[i], next_features = block(features[i], features_masks[-1])
            features.append(next_features)

            if features[i] is next_features:
                break

        features, next_features = features[:-1], features[-1]
        if hasattr(self, 'backward_blocks'):
            for i, mask in enumerate(reversed(features_masks)):
                block = self.backward_blocks[len(features_masks) - i - 1]
                features[-(i + 1)], next_features = block(features[-(i + 1)], next_features, mask, batch_hiddens)

        return tuple(map(functional.padding, [features, features_locs, features_masks]))


class ForwardBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, kernel_size: Optional[int] = None):
        super().__init__()
        self.kernel_size = kernel_size

        self.rnn_block = nn.RNNBlock(hidden_size, dropout)
        if kernel_size is not None:
            self.conv_block = nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, (kernel_size,)),
                nn.GELU(),
            )

    def forward(self, features: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
        """

        :param masks: [batch_size, sentence]
        :param features: [batch_size, features, hidden_size]
        :return: [batch_size, hidden_size, features - kernel_size + 1]
        """
        features = self.rnn_block(features, masks)
        if not hasattr(self, 'conv_block') or features.shape[1] < self.kernel_size:
            return features, features
        else:
            next_features = self.conv_block(features.transpose(-1, -2)).transpose(-1, -2)
            return features, next_features


class BackwardBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, vision_size: int, kernel_size: Optional[int] = None):
        super().__init__()
        self.rnn_block = nn.RNNBlock(hidden_size, dropout)
        if kernel_size is not None:
            self.conv_block = nn.Sequential(
                nn.ConvTranspose1d(hidden_size, hidden_size, (kernel_size,)),
                nn.GELU(),
            )
        self.linear_block = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.pool = nn.AvgPool1d(vision_size, stride=1)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, features: Tensor, next_features: Tensor, masks: Tensor, batch_hiddens: Tensor) -> tuple[Tensor, Tensor]:
        """

        :param batch_hiddens: [batch_size, sentence_length, hidden_size]
        :param features: [batch_size, features_size, hidden_size]
        :param next_features: [batch_size, features, hidden_size]
        :param masks: [batch_size, sentence]
        :return: [batch_size, hidden_size, features - kernel_size + 1]
        """
        features = torch.cat([features, self.rnn_block(next_features, masks)], dim=-1)
        cut_features = self.pool(batch_hiddens.transpose(-1, -2)).transpose(-1, -2)
        features = self.norm(cut_features + self.linear_block(features))
        if hasattr(self, 'conv_block'):
            next_features = self.conv_block(features.transpose(-1, -2)).transpose(-1, -2)
        return features, next_features
