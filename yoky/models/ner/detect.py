import argparse
import logging
import math
from functools import reduce
from typing import Optional, Union

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.types import Device

import yoky.nn
from yoky import nn
from yoky.models.base import Model
from yoky.nn import functional
from yoky.utils import Arguments
from yoky.utils.data import Sentence, Entity

logger = logging.getLogger('yoky')


class EntityDetection(Model):
    class ModelArguments(Arguments):
        def parse(self):
            parser = argparse.ArgumentParser(description='Process model hyper-parameter.')
            parser.add_argument('--pretrain_select', type=str, default="bert-base-chinese", help='Selection of pretrain model.')
            parser.add_argument('--pos_dim', type=int, default=25, help='Dimension of part-of-speech embedding.')
            parser.add_argument('--char_dim', type=int, default=50, help='Dimension of char embedding.')
            parser.add_argument('--word_dim', type=int, default=50, help='Dimension of word embedding.')
            parser.add_argument('--word2vec_select', type=str, default='chinese', help='Select pretrain word2vec embeddings.')
            parser.add_argument('--kernels_size', type=int, default=[2, 3], nargs='*', help='Convolution size of proposer.')
            parser.add_argument('--num_heads', type=int, default=8, help='Number of heads of attention.')
            parser.add_argument('--num_layers', type=int, default=3, help='Layers of regressor.')
            parser.add_argument('--dropout', type=float, default=0.1, help='The general model dropout.')
            parser.add_argument('--no_spatial', default=False, action='store_true',
                                help='Whether use the spatial guidance.')
            parser.add_argument('--no_gate', default=False, action='store_true',
                                help='Whether use the gate block.')
            parser.add_argument('--no_backward', default=False, action='store_true',
                                help='Whether use the backward block in proposer.')
            parser.add_argument('--no_category_embed', default=False, action='store_true',
                                help='Whether use the category embedding.')
            parser.add_argument('--no_logarithm_iter', default=False, action='store_true',
                                help='Whether use the iterative logits.')
            parser.add_argument('--no_locations_iter', default=False, action='store_true',
                                help='Whether use the iterative locations.')
            self.__dict__.update(vars(parser.parse_known_args()[0]))
            return self

        @property
        def encoder_kargs(self):
            return {
                'dropout': self.dropout,
                'embedding_kargs': {
                    'pos_dim': self.pos_dim,
                    'char_dim': self.char_dim,
                    'word_dim': self.word_dim,
                    'word2vec_select': self.word2vec_select,
                    'chars_list': self.metadata.chars_list,
                    'pos_list': self.metadata.pos_list,
                    'pretrain_select': self.pretrain_select,
                }
            }

        @property
        def detector_kargs(self):
            return {
                'hidden_size': self.hidden_size,
                'dropout': self.dropout,
                'types2idx': self.metadata.types2idx,
                'idx2types': self.metadata.idx2types,
                'proposer_kargs': {
                    'no_backward': self.no_backward,
                    'kernels_size': self.kernels_size
                },
                'regressor_kargs': {
                    'num_layers': self.num_layers,
                    'layer_kargs': {
                        'no_logarithm_iter': self.no_logarithm_iter,
                        'no_category_embed': self.no_category_embed,
                        'no_locations_iter': self.no_locations_iter,
                        'no_gate': self.no_gate,
                        'no_spatial': self.no_spatial,
                        'num_heads': self.num_heads,
                    }
                },
            }

    def __init__(self, model_args: ModelArguments):
        super(EntityDetection, self).__init__()
        self.encoder = Encoder(**model_args.encoder_kargs)
        model_args.add('hidden_size', self.encoder.hidden_size)
        self.detector = Detector(**model_args.detector_kargs)

    def forward(self, batch_sentences: list[Sentence]) -> Tensor:
        """

        :param batch_sentences: (batch_size, sentence_length)
        :return: loss
        """
        batch_hiddens, batch_masks = self.encoder(batch_sentences)
        return self.detector(batch_hiddens, batch_masks, batch_sentences)

    def decode(self, batch_sentences: list[Sentence]) -> list[list[Entity]]:
        """

        :param batch_sentences: (batch_size, sentence_length)
        :return: entities or more detailed results.
        """
        return self.detector.decode(*self.encoder(batch_sentences))

    def detail(self, batch_sentences: list[Sentence]):
        """

        :param batch_sentences: (batch_size, sentence_length)
        :return: detailed results.
        """
        return self.detector.detail(*self.encoder(batch_sentences))


class Detector(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            dropout: float,
            proposer_kargs: dict,
            regressor_kargs: dict,
            types2idx: dict,
            idx2types: dict
    ):
        super().__init__()
        types_num = len(types2idx)
        self.formatter = Formatter(types2idx, idx2types)

        general_args = {
            'types_num': types_num,
            'hidden_size': hidden_size,
            'dropout': dropout,
        }

        proposer_kargs.update(general_args)
        self.proposer = Proposer(**proposer_kargs)

        regressor_kargs['layer_kargs'].update(general_args)
        self.regressor = Regressor(**regressor_kargs)

        self.predictor = Predictor(hidden_size, types_num)

    def forward(self, batch_hiddens: Tensor, batch_masks: Tensor, batch_sentences: list[Sentence]) -> Tensor:
        """

        :param batch_hiddens: [batch_size, sentence_length, hidden_size]
        :param batch_masks: [batch_size, sentence_length]
        :param batch_sentences: (batch_size) list of sentence containing entities
        :return: loss
        """
        batch_targets = self.formatter(batch_sentences, batch_hiddens.device)
        (labels_pros, locs_pros), _ = self.predict(batch_hiddens, batch_masks)

        losses = list()
        for labels_p, locs_p, masks, targets in zip(labels_pros, locs_pros, batch_masks, batch_targets):
            entity_p = locs_p[masks][:, flatten(targets[:, :2])] * labels_p[masks][:, targets[:, 2]]
            none_p = labels_p[masks][:, -1:]
            cost = -torch.log(torch.clamp_min(torch.cat([entity_p, none_p], dim=-1), 1e-8))
            proposal_indices, entity_indices = assignment(cost)
            losses.append(torch.mean(cost[proposal_indices, entity_indices]))
        return sum(losses) / len(losses)

    def decode(self, batch_hiddens: Tensor, batch_masks: Tensor) -> Union[list[list[Entity]]]:
        """

        :param batch_hiddens: [batch_size, sentence_length, hidden_size]
        :param batch_masks: [batch_size, sentence_length]
        :return: (batch_size, entities_num, 3)
            0: start_idx
            1: end_idx
            2: type
        """
        length = torch.as_tensor([batch_hiddens.shape[1]]).to(batch_hiddens.device)
        (labels_pros, locs_pros), results = self.predict(batch_hiddens, batch_masks)

        batch_triples = list()
        for labels_p, locs_p, mask in zip(labels_pros, locs_pros, batch_masks):
            locs_p, labels_p = locs_p[mask], labels_p[mask]
            locs_p, loc = torch.max(locs_p, dim=-1)
            types_p, types = torch.max(labels_p[:, :-1], dim=-1)
            sel = (locs_p * types_p).gt(labels_p[:, -1])
            triples = torch.cat([relocate(loc, length), types.unsqueeze(-1)], dim=-1)
            batch_triples.append(triples[sel].tolist())

        return self.formatter.decode(batch_triples)

    def detail(self, batch_hiddens: Tensor, batch_masks: Tensor) -> tuple[list[list[Entity]], list[dict[str, Tensor]]]:
        """

        :param batch_hiddens: [batch_size, sentence_length, hidden_size]
        :param batch_masks: [batch_size, sentence_length]
        :return: (batch_size, entities_num, 3)
            0: start_idx
            1: end_idx
            2: type
        """
        length = torch.as_tensor([batch_hiddens.shape[1]]).to(batch_hiddens.device)
        (labels_pros, locs_pros), results = self.predict(batch_hiddens, batch_masks, True)

        batch_triples = list()
        for labels_p, locs_p, mask in zip(labels_pros, locs_pros, batch_masks):
            locs_p, labels_p = locs_p[mask], labels_p[mask]
            locs_p, loc = torch.max(locs_p, dim=-1)
            types_p, types = torch.max(labels_p[:, :-1], dim=-1)
            sel = (locs_p * types_p).gt(labels_p[:, -1])
            triples = torch.cat([relocate(loc, length), types.unsqueeze(-1)], dim=-1)
            batch_triples.append(triples[sel].tolist())

            for result in results:
                for key, value in result.items():
                    result[key] = value[0, sel]

        batch_entities = self.inverse_format(batch_triples)
        return batch_entities, results

    def predict(
            self,
            batch_hiddens: Tensor,
            batch_masks: Tensor,
            return_detailed_results: bool = False
    ) -> tuple[tuple[Tensor, Tensor], Optional[list[dict[str, Tensor]]]]:
        if not return_detailed_results:
            queries_kargs = self.proposer(batch_hiddens, batch_masks)
            queries_kargs = self.regressor(queries_kargs, batch_masks)
            return self.predictor(batch_hiddens, batch_masks, **queries_kargs), None
        else:
            iterated_kargs = [self.proposer(batch_hiddens, batch_masks)]
            iterated_kargs.extend(self.regressor(iterated_kargs[-1], batch_masks, True))
            proposals = list(map(lambda kargs: {
                'queries_locs': kargs['queries_locs'],
                'queries_logits': torch.softmax(kargs['queries_logits'], dim=-1),
            }, iterated_kargs))
            return self.predictor(batch_hiddens, batch_masks, **iterated_kargs[-1]), proposals


class Encoder(nn.Module):
    def __init__(self, embedding_kargs: dict, dropout: float):
        super(Encoder, self).__init__()
        embedding_kargs.update({
            'cache_dir': yoky.cache_root,
            'dropout': dropout
        })
        self.embedding = nn.StackEmbedding(**embedding_kargs)
        self.rnn = nn.GRU(self.embedding_length, self.embedding_length, bidirectional=True, batch_first=True)
        self.transforms = nn.Sequential(
            nn.Linear(2 * self.embedding_length, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, batch_sentences: list[Sentence]) -> tuple[Tensor, Tensor]:
        """

        :param batch_sentences: (batch_size, sentence_length)
        :return:
            context: [batch_size, sentence_length, hidden_size]
            types: [batch_size, sentence_length, hidden_size]
            mask: [batch_size, sentence_length]
        """
        lengths = torch.as_tensor(list(map(len, batch_sentences)))

        batch_embeds, batch_masks = self.embedding(batch_sentences)
        batch_embeds = pack_padded_sequence(batch_embeds, lengths, batch_first=True, enforce_sorted=False)
        batch_embeds = pad_packed_sequence(self.rnn(batch_embeds)[0], batch_first=True)[0]

        batch_hiddens = self.transforms(batch_embeds)
        return batch_hiddens, batch_masks

    @property
    def hidden_size(self) -> int:
        return self.embedding.token2vec.pretrain.config.hidden_size

    @property
    def embedding_length(self) -> int:
        return self.embedding.embedding_length


class Formatter:
    def __init__(self, types2idx: dict, idx2types: dict):
        super().__init__()
        self.types2idx, self.idx2types = types2idx, idx2types

    def __call__(self, batch_sentences: list[Sentence], device: Device) -> list[Tensor]:
        """

        :param device: device tensor created on
        :param batch_sentences: (batch_size, entities_num) of dicts
             'start_idx': start index of entity
             'end_idx': end index of entity
             'type': type of entity
        :return: (batch_size) list of [entities_num, 3]
        """
        batch_targets = list()
        for sentence in batch_sentences:
            entities = sentence.entities
            if len(entities) == 0:
                targets = torch.zeros([0, 3], device=device, dtype=torch.long)
            else:
                targets = [[entity.start, entity.end - 1, self.types2idx[entity.type]] for entity in entities]
                targets = torch.as_tensor(targets, device=device, dtype=torch.long)
            batch_targets.append(targets)
        return batch_targets

    def decode(self, batch_triples: list[list]) -> list[list[Entity]]:
        """

        :param batch_triples: (batch_size, entities_num, 3)
        :return: batch_entities: (batch_size, entities_num) of dicts
        """
        batch_entities = list()
        for i, triples in enumerate(batch_triples):
            batch_entities.append([Entity({
                'start': triple[0],
                'end': triple[1] + 1,
                'type': self.idx2types[triple[2]]
            }) for triple in triples])
        return list(map(lambda a: reduce(lambda x, y: x if y in x else x + [y], [[], *a]), batch_entities))


class Proposer(nn.Module):
    def __init__(self, types_num: int, hidden_size: int, kernels_size: list[int], no_backward: bool, dropout: float):
        super().__init__()
        self.pyramid = nn.Pyramid(hidden_size, kernels_size, dropout, no_backward)
        self.anchor_attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 2)
        )
        self.classifier = nn.Classifier(hidden_size, types_num)

    def forward(self, batch_hiddens: Tensor, batch_masks: Tensor) -> dict[str, Tensor]:
        """

        :param batch_masks:  [batch_size, sentence_length]
        :param batch_hiddens: [batch_size, sentence_length, hidden_size]
        :return: queries_kargs
            'queries': [batch_size, sentence_length, hidden_size]
            'queries_locs': [batch_size, sentence_length, 2]
            'queries_logits': [batch_size, sentence_length, types_num + 1]
        """
        batch_size, sentence_length = batch_hiddens.shape[:2]
        features, features_locs, features_masks = self.pyramid(batch_hiddens, batch_masks)

        queries = features[:, 0]
        queries_logits = self.classifier(queries)

        scores = self.anchor_attn(features).masked_fill_(~features_masks.unsqueeze(-1), -1e8)

        features_locs = features_locs.flatten(1, 2)
        indices = torch.arange(sentence_length, device=batch_hiddens.device).unsqueeze(-1)
        sel = indices.ge(features_locs[0, :, 0]) & indices.le(features_locs[0, :, 1])
        sel = torch.split(torch.nonzero(sel)[:, 1], torch.sum(sel, dim=-1).tolist())
        sel = pad_sequence(sel, padding_value=-1, batch_first=True)

        features_locs = features_locs[:, sel].float()
        scores = scores.flatten(1, 2)[:, sel]
        scores.masked_fill_(sel.eq(-1).view(1, *sel.shape, 1), -1e8)

        locs_weights = torch.softmax(scores, dim=-2)
        queries_locs = torch.einsum('blsc,blsc->blc', locs_weights, features_locs)

        return {
            'queries': queries,
            'queries_locs': queries_locs,
            'queries_logits': queries_logits,
        }


class Regressor(nn.Module):
    def __init__(self, num_layers: int, layer_kargs: dict):
        super(Regressor, self).__init__()
        self.layers = nn.ModuleList([IterativeLayer(**layer_kargs) for _ in range(num_layers)])

    def forward(
            self,
            queries_kargs: dict[str, Tensor],
            batch_masks: Tensor,
            return_layers: bool = False
    ) -> Union[list[dict[str, Tensor]], dict[str, Tensor]]:
        """

        :param return_layers: whether return each layers' results.
        :param batch_masks: [batch_size, sentence_length]
        :param queries_kargs:
            'queries': [batch_size, sentence_length, hidden_size]
            'locations': [batch_size, sentence_length, 2]
            'logits': [batch_size, sentence_length, types_num + 1]
        :return: queries_kargs:
            'queries': [batch_size, sentence_length, hidden_size]
            'locations': [batch_size, sentence_length, 2]
            'logits': [batch_size, sentence_length, types_num + 1]
        """
        if not return_layers:
            for layer in self.layers:
                queries_kargs.update(layer(batch_masks=batch_masks, **queries_kargs))
            return queries_kargs
        else:
            layers_kargs = [queries_kargs]
            for layer in self.layers:
                layers_kargs.append(layer(batch_masks=batch_masks, **layers_kargs[-1]))
            return layers_kargs


class IterativeLayer(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            types_num: int,
            no_category_embed: bool,
            no_locations_iter: bool,
            no_logarithm_iter: bool,
            no_spatial: bool,
            no_gate: bool,
            dropout: float
    ):
        super(IterativeLayer, self).__init__()
        if not no_category_embed:
            self.types_embeds = nn.Parameter(torch.empty(types_num + 1, hidden_size))
            torch.nn.init.normal_(self.types_embeds)

        self.attention_block = nn.SpatialAttentionBlock(hidden_size, num_heads, no_locations_iter, no_spatial, dropout)
        if not no_gate:
            self.gate_block = nn.GateBlock(hidden_size, dropout)

        if not no_logarithm_iter:
            self.classifier = nn.Classifier(hidden_size, types_num)

    def forward(self, queries: Tensor, queries_locs: Tensor, queries_logits: Tensor, batch_masks: Tensor) -> dict[str, Tensor]:
        """


        :param queries: [batch_size, queries_num, hidden_size]
        :param queries_locs: [batch_size, queries_num, 2]
        :param queries_logits: [batch_size, queries_num, types_num + 1]
        :param batch_masks: [batch_size, queries_num]
        :return: queries_kargs:
            'queries': [batch_size, sentence_length, hidden_size]
            'locations': [batch_size, sentence_length, 2]
            'logits': [batch_size, sentence_length, types_num + 1]
        """
        if hasattr(self, 'types_embeds'):
            types_weights = torch.softmax(queries_logits, dim=-1)
            queries = queries + torch.matmul(types_weights, self.types_embeds)
        outputs, queries_locs = self.attention_block(queries, queries_locs, batch_masks)
        if hasattr(self, 'gate_block'):
            queries = self.gate_block(outputs, queries)
        else:
            queries = outputs
        if hasattr(self, 'classifier'):
            queries_logits = queries_logits + self.classifier(queries)
        return {
            'queries': queries,
            'queries_locs': queries_locs,
            'queries_logits': queries_logits,
        }


class Predictor(nn.Module):
    def __init__(self, hidden_size: int, types_num: int):
        super().__init__()
        self.factor = math.sqrt(hidden_size)

        self.classifier = nn.Classifier(hidden_size, types_num)
        self.trans_queries = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.Unflatten(-1, [2, hidden_size])
        )
        self.trans_context = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.Unflatten(-1, [2, hidden_size])
        )
        self.locator = nn.Locator(hidden_size)

    def forward(self, batch_hiddens: Tensor, batch_masks: Tensor, queries: Tensor, queries_locs: Tensor, queries_logits: Tensor):
        """

        :param queries_logits: [batch_size, queries_size, types_num + 1]
        :param queries: [batch_size, queries_num, hidden_size]
        :param queries_locs: [batch_size, queries_num, 2]
        :param batch_hiddens: [batch_size, sentence_length, hidden_size]
        :param batch_masks: [batch_size, sentence_length]
        :return:
        """
        labels_pros = torch.softmax(queries_logits + self.classifier(queries), dim=-1)

        queries_h, context_h = self.trans_queries(queries), self.trans_context(batch_hiddens)

        score = torch.einsum('bqch,bvch->bcqv', queries_h, context_h) / self.factor
        score.masked_fill_(~batch_masks.unsqueeze(1).unsqueeze(1), -1e8)

        pairs = get_table(torch.as_tensor([batch_hiddens.shape[1]], device=batch_hiddens.device))
        score = score[:, 0, :, pairs[:, 0]] + score[:, 1, :, pairs[:, 1]]

        mask, pairs = batch_masks[:, pairs[:, -1]], pairs.unsqueeze(0)
        offset, factors = self.locator(queries)
        weight = functional.distribute(pairs, mask, queries_locs + offset, factors)
        locs_pros = functional.weight_softmax(score, weight)

        return labels_pros, locs_pros


def assignment(cost: Tensor, one_to_many: bool = True) -> tuple[list, list]:
    """

    :param one_to_many: whether use one to many assignment.
    :param cost: [proposals_num, entities_num + 1]
    :return: [proposals_num], [proposals_num]
    """
    row_indices, col_indices = linear_sum_assignment(cost[:, :-1].cpu().detach().numpy())
    row_indices, col_indices = map(list, [row_indices, col_indices])
    other_row_indices = list(set(range(cost.shape[0])) - set(row_indices))
    if one_to_many:
        other_col_indices = torch.argmin(cost[other_row_indices], dim=-1).tolist()
    else:
        other_col_indices = [-1] * len(other_row_indices)
    return row_indices + other_row_indices, col_indices + other_col_indices


def flatten(loc: Tensor) -> Tensor:
    """
    flatten the start&end format coordinate into linear format
    :param loc: [..., 2(start&end)]
    :return: [...]
    """

    return torch.div((1 + loc[..., 1]) * loc[..., 1], 2, rounding_mode='floor') + loc[..., 0]


def relocate(loc: Tensor, length: Tensor) -> Tensor:
    """
    relocate linear coordinate into the start&end form
    :param loc: [proposals_num]
    :param length: sentence length
    :return: [proposals_num, 2(start&end)]
    """

    return get_table(length)[loc]


def get_table(length: Tensor) -> Tensor:
    """
    0,0 0,1 0,2 0,3
        1,1 1,2 1,3
            2,2 2,3
                3,3
    :param length: [1], tensor of single element for search of the s&e table
    :return: [(length + 1) * length / 2, 2]
    """

    start = reduce(lambda a, b: a + b, [list(range(i + 1)) for i in range(length)])
    end = reduce(lambda a, b: a + b, [[i] * (i + 1) for i in range(length)])
    return torch.as_tensor(list(map(lambda args: [*args], zip(start, end))), device=length.device)
