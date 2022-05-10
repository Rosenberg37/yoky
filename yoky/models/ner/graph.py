import argparse
import logging
import math
from enum import IntEnum
from typing import Union, Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence, PackedSequence

import yoky
import yoky.nn.cell
from yoky import nn
from yoky.models.base import Model
from yoky.utils import Sentence, Entity, Arguments

logger = logging.getLogger('yoky')


class GraphNERArguments(Arguments):
    def parse(self):
        parser = argparse.ArgumentParser(description='Process model hyper-parameter.')
        parser.add_argument('--pretrain_select', type=str, default="dmis-lab/biobert-base-cased-v1.2",
                            help='Selection of pretrain model.')
        parser.add_argument('--pos_dim', type=int, default=25, help='Dimension of part-of-speech embedding.')
        parser.add_argument('--char_dim', type=int, default=50, help='Dimension of char embedding.')
        parser.add_argument('--word_dim', type=int, default=50, help='Dimension of word embedding.')
        parser.add_argument('--word2vec_select', type=str, default='bio', help='Select pretrain word2vec embeddings.')
        parser.add_argument('--num_heads', type=int, default=8, help='Number of heads of attention.')
        parser.add_argument('--num_layers', type=int, default=3, help='Times of graph iteration.')
        parser.add_argument('--window_size', type=int, default=1, help='Size of context window.')
        parser.add_argument('--updates', type=str, action='append', default=['context', 'types'],
                            help='Control the update of node')
        parser.add_argument('--no_gate', default=False, action='store_true',
                            help='Whether use gate mechanism.')
        parser.add_argument('--no_hybrid', default=False, action='store_true',
                            help='Whether use hybrid attention.')
        self.__dict__.update(vars(parser.parse_known_args()[0]))
        return self

    @property
    def encoder_kargs(self):
        return {
            'types_num': len(self.metadata.types2idx),
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
    def graph_kargs(self):
        return {
            'num_layers': self.num_layers,
            'layer_kargs': {
                'hidden_size': self.hidden_size,
                'num_heads': self.num_heads,
                'window_size': self.window_size,
                'no_gate': self.no_gate,
                'no_hybrid': self.no_hybrid,
                'updates': self.updates,
            }
        }

    @property
    def decoder_kargs(self):
        return {
            'hidden_size': self.hidden_size,
            'types_num': len(self.metadata.types2idx),
            'types2idx': self.metadata.types2idx,
            'idx2types': self.metadata.idx2types,
        }


class GraphNER(Model):
    def __init__(self, model_args: GraphNERArguments):
        super(GraphNER, self).__init__()
        self.output_attentions = model_args.output_attentions

        self.encoder = Encoder(**model_args.encoder_kargs)
        model_args.add('hidden_size', self.encoder.hidden_size)

        self.graph = Graph(**model_args.graph_kargs)
        self.decoder = GraphDecoder(**model_args.decoder_kargs)

    def forward(self, batch_sentences: list[Sentence]) -> Tensor:
        """

        :param batch_sentences: (batch_size)
        :return: loss
        """
        context, types, mask = self.encoder(batch_sentences)
        context, types = self.graph(context, types, mask)
        return self.decoder(context, types, mask, batch_sentences)

    def decode(self, batch_sentences: list[Sentence]) -> list[list[Entity]]:
        """

        :param batch_sentences: (batch_size, sentence_length)
        :return:  (batch_size, entities_num, 3)
        """
        context, types, mask = self.encoder(batch_sentences)
        context, types = self.graph(context, types, mask)
        return self.decoder.decode(context, types, mask)

    def detail(self, batch_sentences: list[Sentence]):
        return self.graph.get_attentions(**self.encoder(batch_sentences))


class Encoder(nn.Module):
    def __init__(self, types_num: int, embedding_kargs: dict):
        super(Encoder, self).__init__()
        embedding_kargs.update({'cache_dir': yoky.cache_root})
        self.embedding = nn.StackEmbedding(**embedding_kargs)
        self.hidden_size = self.embedding.token2vec.pretrain.config.hidden_size
        embedding_length = self.embedding.embedding_length

        self.out_rnn = nn.GRU(embedding_length, embedding_length, bidirectional=True, batch_first=True)

        self.transforms = nn.ModuleDict({
            'types': nn.Sequential(
                nn.Linear(2 * embedding_length, types_num * self.hidden_size),
                nn.MaxPool(dim=1),
                nn.Unflatten(-1, [types_num, self.hidden_size]),
                nn.LayerNorm(self.hidden_size),
            ),
            'context': nn.Sequential(
                nn.Linear(2 * embedding_length, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
            )
        })

    def forward(self, batch_sentences: list[Sentence]) -> tuple[Tensor, Tensor, Tensor]:
        """

        :param batch_sentences: (batch_size, sentence_length)
        :return:
            context: [batch_size, sentence_length, hidden_size]
            types: [batch_size, sentence_length, hidden_size]
            mask: [batch_size, sentence_length]
        """
        lengths = torch.as_tensor(list(map(len, batch_sentences)))

        batch_embeds, mask = self.embedding(batch_sentences)
        batch_embeds = pack_padded_sequence(batch_embeds, lengths, batch_first=True, enforce_sorted=False)
        batch_embeds = pad_packed_sequence(self.out_rnn(batch_embeds)[0], batch_first=True)[0]

        context = self.transforms['context'](batch_embeds)
        types = self.transforms['types'](batch_embeds)
        return context, types, mask


class Graph(nn.Module):
    def __init__(self, num_layers: int, layer_kargs: dict):
        super().__init__()
        self.layers = nn.ModuleList([UpdateLayer(**layer_kargs) for _ in range(num_layers)])

    def forward(self, context: Tensor, types: Tensor, context_mask: Tensor) -> tuple[Tensor, Tensor]:
        """

        :param context: [batch_size, sentence_length, hidden_size]
        :param types: [batch_size, types_num, hidden_size]
        :param context_mask: [batch_size, sentence_length]
        :return:
            context: [batch_size, sentence_length, hidden_size]
        """
        for layer in self.layers:
            types, context = layer(types, context, context_mask)
        return types, context

    def get_attentions(self, context: Tensor, types: Tensor, context_mask: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        context_attentions, types_attentions = list(), list()
        for layer in self.layers:
            context, types, context_attention, types_attention = layer.attn_forward(types, context, context_mask)
            context_attentions.append(context_attention)
            types_attentions.append(types_attention)
        return context_attentions, types_attentions


class UpdateLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, window_size: int, no_gate: bool, no_hybrid: bool, updates: list[str]):
        super(UpdateLayer, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.updates = updates

        edge_block = EdgeBlock(hidden_size, num_heads, no_hybrid)

        self.norms = nn.ModuleDict()
        if 'types' in updates:
            self.types_block = TypesBlock(hidden_size, num_heads, edge_block, no_gate)
            self.norms['types'] = nn.LayerNorm(hidden_size)

        if 'context' in updates:
            self.context_block = ContextBlock(hidden_size, num_heads, window_size, edge_block, no_gate)
            self.norms['context'] = nn.LayerNorm(hidden_size)

    def forward(self, types: Tensor, context: Tensor, context_mask: Tensor):
        """

        :param types: [batch_size, types_num, hidden_size]
        :param context: [batch_size, sentence_length, hidden_size]
        :param context_mask: [batch_size, sentence_length]
        :return:
            types: [batch_size, types_num, hidden_size]
            context: [batch_size, sentence_length, hidden_size]
        """
        if 'context' in self.updates:
            context = self.norms['context'](context + self.context_block(context, context_mask, types)[0])
        if 'types' in self.updates:
            types = self.norms['types'](types + self.types_block(types, context, context_mask)[0])
        return types, context

    def attn_forward(self, types: Tensor, context: Tensor, context_mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """

        :param types: [batch_size, types_num, hidden_size]
        :param context: [batch_size, sentence_length, hidden_size]
        :param context_mask: [batch_size, sentence_length]
        :return:
            context_attention: [batch_size, num_heads, types_num + sentence_length, types_num + sentence_length]
            types_attention: [batch_size, num_heads, types_num + sentence_length, types_num + sentence_length]
        """
        device = types.device
        batch_size, types_num = types.shape[:2]
        batch_size, sentence_length = context.shape[:2]

        context_update, types_attention, window_attention = self.context_block(context, context_mask, types)
        attentions_list = list()
        padding = torch.zeros([batch_size, self.num_heads, sentence_length], device=device)
        for i, attn in enumerate(window_attention.permute(2, 0, 1, 3)):  # [batch_size, num_heads, value_size]
            attn_left = max(self.window_size - i, 0)
            attn_right = 2 * self.window_size + 2 - max(i + self.window_size - sentence_length + 2, 0)
            attn = attn[..., attn_left:attn_right]

            padding_before = padding[..., :max(i - self.window_size, 0)]
            padding_last = padding[..., min(i + self.window_size + 1, sentence_length):]
            attentions_list.append(torch.cat([padding_before, attn, padding_last], dim=-1))
        context_attention = torch.stack(attentions_list, dim=-2)
        context_attention = torch.cat([types_attention, context_attention], dim=-1)
        padding = torch.zeros([batch_size, self.num_heads, types_num, types_num + sentence_length], device=device)
        context_attention = torch.cat([padding, context_attention], dim=-2)

        types_update, types_attention = self.types_block(types, context, context_mask)
        padding = torch.zeros([batch_size, self.num_heads, types_num, types_num], device=device)
        types_attention = torch.cat([padding, types_attention], dim=-1)
        padding = torch.zeros([batch_size, self.num_heads, sentence_length, types_num + sentence_length], device=device)
        types_attention = torch.cat([types_attention, padding], dim=-2)

        context = self.norms['context'](context + context_update)
        types = self.norms['types'](types + types_update)

        return context, types, context_attention, types_attention


class EdgeBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, no_hybrid: bool, negative_slope: float = 5):
        super(EdgeBlock, self).__init__()
        self.no_hybrid = no_hybrid
        head_dim = hidden_size // num_heads
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        gain = nn.init.calculate_gain('leaky_relu', negative_slope)

        self.upon = nn.Parameter(torch.empty(num_heads, head_dim))
        nn.init.xavier_uniform_(self.upon, gain)
        self.down = nn.Parameter(torch.empty(num_heads, head_dim))
        nn.init.xavier_uniform_(self.down, gain)
        if not no_hybrid:
            self.cross = nn.Parameter(torch.empty(num_heads, head_dim, head_dim))
            nn.init.xavier_uniform_(self.cross, gain)

    def cross_attn(self, query_h: Tensor, value_h: Tensor) -> Tensor:
        """

        :param query_h: [batch_size, query_size, num_heads, head_dim]
        :param value_h: [batch_size, value_size, num_heads, head_dim]
        :return: [batch_size, query_size, num_heads, head_dim]
        """
        upon_s = torch.einsum('bqnd,nd->bnq', query_h, self.upon)
        down_s = torch.einsum('bvnd,nd->bnv', value_h, self.down)
        concat_score = upon_s.unsqueeze(-1) + down_s.unsqueeze(-2)
        if self.no_hybrid:
            return self.leaky_relu(concat_score)
        else:
            product_score = torch.einsum('bqnd,ndh,bvnh->bnqv', query_h, self.cross, value_h)
            return self.leaky_relu(concat_score + product_score)

    def self_attn(self, hidden: Tensor, mask: Tensor, window_size: int):
        """

        :param window_size: the size of self connect window
        :param hidden: [batch_size, sentence_length, num_heads, head_dim]
        :param mask: [batch_size, sentence_length]
        :return: [batch_size, sentence_length, num_heads, head_dim]
        """
        length = mask.shape[1]

        indices_u = torch.cat([torch.zeros(window_size + 1, dtype=torch.long), torch.arange(1, window_size + 1)])
        indices_u = (torch.arange(length).unsqueeze(1) + indices_u).to(hidden.device)

        indices_d = torch.cat([torch.arange(-window_size, 0), torch.zeros(window_size + 1, dtype=torch.long)])
        indices_d = (torch.arange(length).unsqueeze(1) + indices_d).to(hidden.device)

        indices_mask = indices_d.ge(0) & indices_u.lt(length)
        indices_u, indices_d = indices_u % length, indices_d % length

        upon_s = torch.einsum('blnd,nd->bnl', hidden, self.upon)
        down_s = torch.einsum('blnd,nd->bnl', hidden, self.down)
        concat_score = upon_s[:, :, indices_u] + down_s[:, :, indices_d]

        window_indices = torch.cat([indices_d[:, :0], indices_u[:, 0:]], dim=-1)
        mask = mask[:, window_indices % length] & indices_mask

        value_h = hidden[:, window_indices]

        if self.no_hybrid:
            window_score = self.leaky_relu(concat_score).masked_fill(~mask.unsqueeze(1), -1e12)
        else:
            product_score = torch.einsum('blnd,ndh,blvnh->bnlv', hidden, self.cross, value_h)
            window_score = self.leaky_relu(concat_score + product_score).masked_fill(~mask.unsqueeze(1), -1e12)

        return window_score, value_h


class TypesBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, edge_block: EdgeBlock, no_gate: bool):
        super(TypesBlock, self).__init__()
        head_dim = hidden_size // num_heads
        self.no_gate = no_gate

        self.transforms = nn.ModuleDict({
            'types': nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Unflatten(-1, [num_heads, head_dim])
            ),
            'context': nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Unflatten(-1, [num_heads, head_dim])
            )
        })
        self.edge_block = edge_block

        if no_gate:
            self.out = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh()
            )
        else:
            self.cell = yoky.nn.cell.GateCell(hidden_size)

    def forward(self, types: Tensor, context: Tensor, context_mask: Tensor) -> tuple[Tensor, Tensor]:
        """

        :param types: [batch_size, types_num, hidden_size]
        :param context: [batch_size, sentence_length, hidden_size]
        :param context_mask: [batch_size, sentence_length]
        :return:
            types: [batch_size, types_num, hidden_size]
            weights: [batch_size, types_num, sentence_length]
        """
        types_h = self.transforms['types'](types)
        context_h = self.transforms['context'](context)
        score = self.edge_block.cross_attn(types_h, context_h)
        score = score.masked_fill(~context_mask.unsqueeze(1).unsqueeze(1), -1e12)
        weights = torch.softmax(score, dim=-1)
        update = torch.einsum('bntl,blnd->btnd', weights, context_h).flatten(-2)

        update = self.out(update + types) if self.no_gate else self.cell(update + types, types)
        return update, weights


class ContextBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, window_size: int, edge_block: EdgeBlock, no_gate: bool):
        super(ContextBlock, self).__init__()
        head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.no_gate = no_gate

        self.transforms = nn.ModuleDict({
            'types': nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Unflatten(-1, [num_heads, head_dim])
            ),
            'context': nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Unflatten(-1, [num_heads, head_dim])
            )
        })
        self.edge_block = edge_block

        if no_gate:
            self.out = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh()
            )
        else:
            self.cell = yoky.nn.cell.GateCell(hidden_size)

    def forward(self, context: Tensor, context_mask: Tensor, types: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """

        :param types: [batch_size, hidden_size]
        :param context: [batch_size, sentence_length, hidden_size]
        :param context_mask: [batch_size, sentence_length]
        :return:
            context: [batch_size, sentence_length, hidden_size]
            weights: [batch_size, sentence_length, types_num + window_size * 2 + 1]
        """
        types_num = types.shape[1]

        types_h = self.transforms['types'](types)
        context_h = self.transforms['context'](context)

        types_s = self.edge_block.cross_attn(context_h, types_h)
        context_s, value_h = self.edge_block.self_attn(context_h, context_mask, self.window_size)

        weights = torch.softmax(torch.cat([types_s, context_s], dim=-1), dim=-1)
        types_weights, context_weights = weights.split([types_num, self.window_size * 2 + 1], -1)

        update_types = torch.einsum('bnlt,btnd->blnd', types_weights, types_h)
        update_context = torch.einsum('bnlv,blvnd->blnd', context_weights, value_h)
        update = (update_types + update_context).flatten(-2)

        update = self.out(update + types) if self.no_gate else self.cell(update + types, types)
        return update, types_weights, context_weights


class Tags(IntEnum):
    begin = 0
    inside = 1
    out = 2
    end = 3
    single = 4


class GraphDecoder(nn.Module):
    def __init__(self, hidden_size: int, types_num: int, types2idx: dict, idx2types: dict):
        super(GraphDecoder, self).__init__()
        self.tagger = Tagger(types2idx, idx2types)

        self.rnns = nn.ModuleList([nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True) for _ in range(types_num)])
        self.reduces = nn.Parameter(torch.empty(types_num, hidden_size * 2, hidden_size))
        nn.init.kaiming_uniform_(self.reduces, a=math.sqrt(5))
        self.scores = nn.Parameter(torch.empty(types_num, hidden_size, len(Tags)))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        self.crf = NestCRF()

    def emission(self, types: Tensor, context: Tensor, lengths: Tensor):
        """

        :param lengths: [batch_size]
        :param types: [batch_size, types_num, hidden_size]
        :param context: [batch_size, sentence_length, hidden_size]
        :return: [batch_size, types_num, sentence_length, num_tags]
        """
        pack_context = pack_padded_sequence(context, lengths, enforce_sorted=False, batch_first=True)
        rnn_hiddens = torch.repeat_interleave(types.unsqueeze(0), 2, dim=0)  # [2, batch_size, types_num, hidden_size]

        batch_hiddens = list()
        for rnn, hidden in zip(self.rnns, rnn_hiddens.permute(2, 0, 1, 3)):
            packed_hiddens = rnn(pack_context, hidden)[0]
            batch_hiddens.append(pad_packed_sequence(packed_hiddens, batch_first=True)[0])
        batch_hiddens = torch.stack(batch_hiddens, dim=1)
        batch_hiddens = torch.einsum('btlh,thd->btld', batch_hiddens, self.reduces)
        batch_hiddens = torch.maximum(batch_hiddens + context.unsqueeze(1), types.unsqueeze(2))
        return torch.einsum('btld,tdg->btlg', batch_hiddens, self.scores)

    def forward(self, types: Tensor, context: Tensor, context_mask: Tensor, batch_sentences: list[Sentence]):
        """

        :param batch_sentences: (batch_size, sentence_length)
        :param types: [batch_size, types_num, hidden_size]
        :param context: [batch_size, sentence_length, hidden_size]
        :param context_mask: [batch_size, sentence_length]
        :return: loss
        """
        batch_scores = self.emission(types, context, torch.sum(context_mask, dim=-1).cpu())
        batch_tags = self.tagger(batch_sentences, batch_scores.shape[:3], batch_scores.device)
        mask = context_mask.unsqueeze(1).expand_as(batch_tags)
        return self.crf(*map(lambda a: a.flatten(0, 1), [batch_scores, batch_tags, mask]))

    def decode(self, types: Tensor, context: Tensor, context_mask: Tensor):
        """

        :param types: [batch_size, types_num, hidden_size]
        :param context: [batch_size, types_num, sentence_length, hidden_size]
        :param context_mask: [batch_size, sentence_length]
        :return: (batch_size, entities_num, 3)
        """
        batch_scores = self.emission(types, context, torch.sum(context_mask, dim=-1).cpu())
        mask = context_mask.unsqueeze(1).expand(batch_scores.shape[:-1])
        batch_tags = self.crf.decode(*map(lambda a: a.flatten(0, 1), [batch_scores, mask]))
        batch_tags = [torch.as_tensor(tags) for tags in batch_tags]
        batch_tags = pad_sequence(batch_tags, batch_first=True, padding_value=float(Tags.out))
        batch_tags = batch_tags.view(batch_scores.shape[:-1]).long()
        return self.tagger.decode(batch_tags)


class ScoreBlock(nn.Module):
    def __init__(self, hidden_size: int, num_tags: int):
        super().__init__()
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.reduce = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, num_tags)

    def forward(self, type_embed: Tensor, pack_context: PackedSequence, context: Tensor):
        """

        :param type_embed: [batch_size, hidden_size]
        :param context: [batch_size, sentence_length, hidden_size]
        :param pack_context: [batch_size, sentence_length, hidden_size] as packed sequence
        :return: [batch_size, types_num, sentence_length, num_tags]
        """
        batch_hiddens = self.rnn(pack_context, torch.repeat_interleave(type_embed.unsqueeze(0), 2, dim=0))[0]
        batch_hiddens = pad_packed_sequence(batch_hiddens, batch_first=True)[0]
        batch_hiddens = type_embed.unsqueeze(1) + context + self.reduce(batch_hiddens)
        return self.score(batch_hiddens)


class Tagger:
    def __init__(self, types2idx: dict, idx2types: dict):
        super(Tagger, self).__init__()
        self.types2idx, self.idx2types = types2idx, idx2types

    def __call__(self, batch_sentences: list[Sentence], shape: Union[torch.Size, list[int]], device: str = 'cpu') -> Tensor:
        """

        :param device: device sentence created on.
        :param shape: batch_size, types_num, sentence_length
        :param batch_sentences: (batch_size, entities_num)
        :return:  (batch_size, entities_num, 3)
        """
        batch_tags = torch.full(shape, Tags.out, device=device, dtype=torch.long)
        for sentence, types_tags in zip(batch_sentences, batch_tags):
            entities = list(filter(lambda a: a.end <= shape[-1], sentence.entities))
            triples = list(map(lambda a: (a.start, a.end, self.types2idx[a.type]), entities))
            for s, e, t in triples:
                types_tags[t, s:e] = Tags.inside
            for s, e, t in filter(lambda a: a[0] != a[1] - 1, triples):
                types_tags[t, s], types_tags[t, e - 1] = Tags.begin, Tags.end
            for s, e, t in filter(lambda a: a[0] == a[1] - 1, triples):
                types_tags[t, s] = Tags.single

        return batch_tags

    def decode(self, batch_tags: Tensor) -> list[list[Entity]]:
        """

        :param batch_tags: [batch_size, types_num, sentence_length]
        :return: (batch_size, entities_num, 1 + 1 + 1(start, end, type))
        """

        batch_entities = list()
        for types_tags in batch_tags:
            entities = list()
            for t, tags in enumerate(types_tags):
                starts = [i for i, tag in enumerate(tags) if tag in [Tags.begin, Tags.single]]
                ends = [i for i, tag in enumerate(tags) if tag in [Tags.end, Tags.single]]
                tags = torch.cat([tags, torch.as_tensor([Tags.out], device=tags.device)])

                while len(starts) != 0 and len(ends) != 0:
                    for s in starts:
                        for e in ends:
                            if s == e:
                                entities.append(Entity({'start': s, 'end': e + 1, 'type': self.idx2types[t]}))
                                if tags[s + 1] == Tags.out and tags[s - 1] in [Tags.begin, Tags.inside]:
                                    tags[s] = Tags.end
                                    starts.remove(s)
                                elif tags[s + 1] in [Tags.end, Tags.inside] and tags[s - 1] == Tags.out:
                                    tags[s] = Tags.begin
                                    ends.remove(e)
                                else:
                                    starts.remove(s), ends.remove(e)
                                    if tags[s + 1] in [Tags.out, Tags.single] and tags[s - 1] in [Tags.out, Tags.single]:
                                        tags[s] = Tags.out
                                    else:
                                        tags[s] = Tags.inside
                                break
                            elif tags[s] == Tags.begin and tags[e] == Tags.end and \
                                    sum(map(lambda a: a == Tags.inside, tags[s + 1:e])) == e - s - 1:
                                entities.append(Entity({'start': s, 'end': e + 1, 'type': self.idx2types[t]}))
                                if tags[s - 1] in [Tags.end, Tags.out, Tags.single]:
                                    ends.remove(e)
                                    tags[e] = Tags.inside if tags[e + 1] in [Tags.inside, Tags.end] else Tags.out
                                if tags[e + 1] in [Tags.begin, Tags.out, Tags.single]:
                                    starts.remove(s)
                                    tags[s] = Tags.inside if tags[s - 1] in [Tags.inside, Tags.begin] else Tags.out
                                if tags[s - 1] in [Tags.inside, Tags.begin] and tags[e + 1] in [Tags.inside, Tags.end]:
                                    starts.remove(s), ends.remove(e)
                                    tags[s] = tags[e] = Tags.inside
                                break
                        else:
                            continue
                        break
            batch_entities.append(entities)
        return batch_entities


class NestCRF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_tags = len(Tags)
        self.register_buffer('start_transitions', torch.zeros(self.num_tags))
        self.register_buffer('end_transitions', torch.zeros(self.num_tags))
        self.register_buffer('transitions', torch.zeros(self.num_tags, self.num_tags))

        for tag in ['inside', 'end']:
            self.start_transitions[Tags[tag]] = -1e12

        for tag in ['begin', 'inside']:
            self.end_transitions[Tags[tag]] = -1e12

        pairs = {
            'begin': ['out'],
            'inside': ['out'],
            'out': ['inside', 'end']
        }
        for pre in pairs.keys():
            for post in pairs[pre]:
                self.transitions[Tags[pre], Tags[post]] = -1e12

    def forward(self, emissions: Tensor, tags: Tensor, mask: Optional[Tensor] = None) -> torch.Tensor:
        """
        Compute the conditional log likelihood of a sequence of tags given emission scores.
        :param emissions: (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
        :param tags: (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
        :param mask: (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
        :return: `~torch.Tensor`: The negative log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        if mask is None:  # 0 means padding, 1 means not padding.
            mask = torch.ones_like(tags)

        # transform batch_size dimension to the dimension 1.
        emissions = emissions.transpose(0, 1)
        tags = tags.transpose(0, 1)
        mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = denominator - numerator

        return torch.mean(llh / mask.sum(0))

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.ByteTensor] = None) -> list[list[int]]:
        """
        Find the most likely tag sequence using Viterbi algorithm.
        :param emissions: (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
        :param mask: (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
        :return: List of list containing the best tag sequence for each batch.
        """
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.long)

        emissions = emissions.transpose(0, 1)
        mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _compute_score(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor) -> torch.Tensor:
        """

        :param emissions: (seq_length, batch_size, num_tags)
        :param tags: (seq_length, batch_size)
        :param mask: (seq_length, batch_size)
        :return:
        """

        seq_length, batch_size = tags.shape

        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        seq_ends = mask.sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        """

        :param emissions: (seq_length, batch_size, num_tags)
        :param mask: (seq_length, batch_size)
        :return:
        """

        seq_length = emissions.shape[0]

        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        score += self.end_transitions

        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor, mask: torch.ByteTensor) -> list[list[int]]:
        """

        :param emissions: (seq_length, batch_size, num_tags)
        :param mask: (seq_length, batch_size)
        :return:
        """

        seq_length, batch_size = mask.shape

        score = self.start_transitions + emissions[0]
        history = []

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        score += self.end_transitions
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list
