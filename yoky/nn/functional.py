import torch
from torch import Tensor

# noinspection PyUnresolvedReferences
from torch.nn.functional import *
from torch.nn.utils.rnn import pad_sequence


def weight_softmax(x: Tensor, weight: Tensor, dim: int = -1):
    """

    :param dim: softmax on which dim
    :param x: [..., tags_num]
    :param weight: [..., tags_num]
    :return: pros: [..., tags_num]
    """
    x = x + weight
    x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    return x / torch.sum(x, dim=dim, keepdim=True)


def limit_sigmoid(x: Tensor, threshold: float = 0.5):
    """

    :param threshold: with values larger than 0.5, probability will be larger than 0.5
    :param x: [...]
    :return: [...] same shape as x.
    """
    a = threshold / (1 - threshold)
    return torch.sigmoid(1 / (1 - x) - a / x)


def padding(tensors: list[Tensor]) -> Tensor:
    """

    :param tensors: (num_layers) [batch_size, length, *]
    :return:  [batch_size, num_layers, sentence_length, *]
    """
    batch_tensors = pad_sequence(list(map(lambda a: a.transpose(0, 1), tensors)))
    return batch_tensors.transpose(0, 2)


def distribute(locations: Tensor, mask: Tensor, centers: Tensor, factors: Tensor):
    """
    discrete clamp normal distribution

    :param locations: [..., location_num, 2(start&end)]
    :param mask: [..., location_num], True means not masked.
    :param centers: [..., queries_num, 2(start&end)]
    :param factors: [..., queries_num, 4]
    :return: probabilities: [..., queries_num, location_num]
    """
    rel_locs = (locations.unsqueeze(-3) - centers.unsqueeze(-2)).unsqueeze(-1)
    dis = -rel_locs.transpose(-1, -2) @ factors.unsqueeze(-3) @ rel_locs
    return torch.masked_fill(dis.view(dis.shape[:-2]), ~mask.unsqueeze(-2), 0)
