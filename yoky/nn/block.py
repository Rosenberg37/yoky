import math

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from yoky import nn
from yoky.nn import functional


class SpatialAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, no_locations_iter: bool, no_spatial: bool, dropout: float):
        super(SpatialAttentionBlock, self).__init__()
        self.no_spatial = no_spatial

        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.factor = math.sqrt(head_dim)

        self.transform = nn.Sequential(
            nn.Linear(hidden_size, 3 * num_heads * head_dim),
            nn.Unflatten(-1, [3 * num_heads, head_dim])
        )

        if not no_locations_iter:
            self.anchor_attn = nn.Sequential(
                nn.Linear(head_dim, head_dim // 2),
                nn.GELU(),
                nn.Linear(head_dim // 2, 2)
            )

        self.locator = nn.Locator(head_dim)

        self.norm = nn.LayerNorm(hidden_size)
        self.trans_out = nn.Sequential(
            nn.Linear(num_heads * head_dim, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, queries: Tensor, locations: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
        """


        :param queries: [batch_size, queries_num, hidden_size]
        :param locations: [batch_size, queries_num, 2]
        :param masks: [batch_size, queries_num]
        :return:
            queries: [batch_size, queries_size, hidden_size]
            locations: [batch_size, queries_num, 2]
        """

        queries_h, keys_h, values_h = torch.chunk(self.transform(queries).transpose(1, 2), 3, dim=1)
        score = torch.einsum('bnqh,bnvh->bnqv', queries_h, keys_h) / self.factor
        score.masked_fill_(~masks.unsqueeze(1).unsqueeze(1), -1e8)

        offsets, factors = self.locator(queries_h)
        locs = locations.unsqueeze(1) + offsets

        if not self.no_spatial:
            spatial_weight = functional.distribute(locations.unsqueeze(1), masks.unsqueeze(1), locs, factors)
            attn_weight = functional.weight_softmax(score, spatial_weight)
        else:
            attn_weight = torch.softmax(score, dim=-1)

        outputs_h = torch.einsum('bnqv,bnvh->bqnh', attn_weight, values_h)

        if hasattr(self, 'anchor_attn'):
            heads_scores = self.anchor_attn(outputs_h)
            locs_weights = torch.softmax(heads_scores, dim=-2)
            locations = torch.einsum('bnqc,bqnc->bqc', locs, locs_weights)

        queries = self.norm(self.trans_out(outputs_h.flatten(-2)) + queries)

        return queries, locations


class LinearBlock(nn.Module):
    def __init__(self, hidden_size: int, dim_feedforward: int, dropout: float):
        super(LinearBlock, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(hidden_size, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, queries: Tensor) -> tuple[Tensor, Tensor]:
        """

        :param queries: [batch_size, queries_num, hidden_size]
        :return: [batch_size, queries_num, hidden_size]
        """
        return self.norm(queries + self.transform(queries))


class RNNBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.transform = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, features: Tensor, masks: Tensor) -> Tensor:
        """

        :param features: [batch_size, features_size, hidden_size]
        :param masks: [batch_size, sentence]
        :return: [batch_size, features, hidden_size]
        """
        features = self.dropout(self.norm(features))
        lengths = torch.clamp_min(torch.sum(masks, dim=-1), 1).cpu()
        packed_features = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
        features = pad_packed_sequence(self.rnn(packed_features)[0], batch_first=True)[0]
        return self.transform(features)


class GateBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super(GateBlock, self).__init__()
        self.cell = nn.GateCell(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tensor:
        """

        :param x: [*, input_dim]
        :param h: [*, input_dim]
        :return: [*, hidden_dim]
        """
        return self.norm(self.dropout(self.cell(x, h)) + h)
