import json
import os.path
from pathlib import Path

import numpy as np
import torch
import transformers
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchtext import vocab

from yoky import nn
from yoky.utils import Sentence


class StackEmbedding(nn.Module):
    def __init__(
            self,
            cache_dir: Path,
            pretrain_select: str,
            pos_dim: int,
            char_dim: int,
            word_dim: int,
            word2vec_select: str,
            chars_list: list[str],
            pos_list: list[str],
            dropout: float,
    ):
        super(StackEmbedding, self).__init__()
        self.token2vec = Token2Vec(cache_dir, pretrain_select)
        self._embedding_length = self.token2vec.hidden_size

        if word_dim is not None:
            self.word2vec = Word2Vec(cache_dir, word_dim, word2vec_select, dropout)
            self._embedding_length += self.word2vec.word_dim

        if char_dim is not None:
            self.char2vec = Char2Vec(chars_list, char_dim, dropout)
            self._embedding_length += char_dim * 2

        if pos_dim is not None:
            self.pos2vec = Pos2Vec(pos_list, pos_dim, dropout)
            self._embedding_length += pos_dim

    def forward(self, batch_sentences: list[Sentence]) -> tuple[Tensor, Tensor]:
        """

        :param batch_sentences: (batch_size, sentence_length)
        :return:
            batch_embeds: [batch_size, sentence_length, hidden_size]
            mask: [batch_size, sentence_length]
        """
        vectors = list()
        token2vec, batch_masks = self.token2vec(batch_sentences)
        vectors.append(token2vec)

        if hasattr(self, 'word2vec'):
            word2vec = self.word2vec(batch_sentences, token2vec.device)
            vectors.append(word2vec)

        if hasattr(self, 'char2vec'):
            char2vec = self.char2vec(batch_sentences)
            vectors.append(char2vec)

        if hasattr(self, 'pos2vec'):
            pos2vec = self.pos2vec(batch_sentences)
            vectors.append(pos2vec)

        return torch.cat(vectors, dim=-1), batch_masks

    @property
    def embedding_length(self):
        return self._embedding_length


class Token2Vec(nn.Module):
    def __init__(self, cache_dir: Path, pretrain_select: str):
        super(Token2Vec, self).__init__()
        cache_dir = os.path.join(cache_dir, pretrain_select)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrain_select,
            cache_dir=cache_dir,
            padding_side='right'
        )
        self.pretrain = transformers.AutoModel.from_pretrained(pretrain_select, cache_dir=cache_dir)

    def forward(self, batch_sentences: list[Sentence]) -> tuple[Tensor, Tensor]:
        """

        :param batch_sentences: (batch_size, sentence_length)
        :return: [batch_size, sentence_length, hidden_size]
        """
        batch_context = list(map(lambda a: a.context, batch_sentences))
        lengths = list(map(len, batch_context))

        batch_encoding = self.tokenizer(
            batch_context,
            is_split_into_words=True,
            return_tensors='pt',
            padding=True
        ).to(next(self.parameters()).device)

        output = self.pretrain(output_hidden_states=True, **batch_encoding)
        hidden_states = torch.stack(output.hidden_states[-4:], dim=-1)
        hidden_state = torch.mean(hidden_states, dim=-1)

        token_embeds, sub_lengths = list(), list()
        for i, length in enumerate(lengths):
            for j in range(length):
                s, e = batch_encoding.word_to_tokens(i, j)
                token_embeds.append(hidden_state[i, s:e])
                sub_lengths.append(e - s)

        sub_lengths = torch.as_tensor(sub_lengths, device=hidden_state.device)
        token_embeds = pad_sequence(token_embeds, padding_value=0)
        token_embeds = torch.sum(token_embeds, dim=0) / sub_lengths.unsqueeze(-1)

        token_embeds = token_embeds.split(lengths, dim=0)
        token_embeds = pad_sequence(token_embeds, batch_first=True)
        return self.span_select(token_embeds, batch_sentences)

    @staticmethod
    def span_select(batch_embeds: Tensor, batch_sentences: list[Sentence]) -> tuple[Tensor, Tensor]:
        """

        :param batch_embeds: [batch_size, context_length, hidden_size]
        :param batch_sentences: (batch_size, sentence_length)
        :return:
            context: [batch_size, sentence_length, hidden_size]
            mask: [batch_size, sentence_length]
        """
        hiddens, batch_masks = list(), list()
        for sentence, embeds in zip(batch_sentences, batch_embeds):
            s, e = sentence.start_pos, sentence.end_pos
            hiddens.append(embeds[s:e])
            batch_masks.append(torch.ones(e - s, device=embeds.device, dtype=torch.bool))
        hiddens = pad_sequence(hiddens, batch_first=True)
        batch_masks = pad_sequence(batch_masks, padding_value=False, batch_first=True)
        return hiddens, batch_masks

    @property
    def hidden_size(self):
        return self.pretrain.config.hidden_size


class Word2Vec(nn.Module):
    def __init__(self, cache_dir: Path, dim: int, word2vec_select: str, dropout: float):
        super().__init__()
        self.word2vec_select = word2vec_select
        if word2vec_select == 'glove':
            self._word_dim = dim
            self.vectors = vocab.GloVe(name='6B', dim=dim, cache=os.path.join(cache_dir, "glove"))
        elif word2vec_select == 'bio':
            self.word2idx = json.load(open(os.path.join(cache_dir, "bio-word2vec/vocab_bio.json")))
            vectors = torch.from_numpy(np.load(os.path.join(cache_dir, "bio-word2vec/vocab_embed_bio.npy")))
            self.embeds = nn.Embedding.from_pretrained(vectors.float())
            self._word_dim = self.embeds.embedding_dim
        elif word2vec_select == 'chinese':
            self._word_dim = 300
            self.vectors = vocab.Vectors(name='chinese-word2vec.char', cache=os.path.join(cache_dir, "chinese-word2vec"))
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_sentences: list[Sentence], device: str) -> Tensor:
        """

        :param device: create word embedding on which device
        :param batch_sentences: (batch_size, sentence_length)
        :return: [batch_size, sentence_length, word_dim]
        """
        embeds = list(map(self.get_vectors_by_tokens, batch_sentences))
        return self.dropout(pad_sequence(embeds, batch_first=True).to(device))

    def get_vectors_by_tokens(self, sentence: Sentence):
        if self.word2vec_select in ['glove', 'chinese']:
            return self.vectors.get_vecs_by_tokens(sentence.sentence_tokens)
        elif self.word2vec_select == 'bio':
            indices = [self.word2idx.get(token, 0) for token in sentence.sentence_tokens]
            return self.embeds(torch.as_tensor(indices, device=self.embeds.weight.device))

    @property
    def word_dim(self):
        return self._word_dim


class Char2Vec(nn.Module):
    def __init__(self, chars_list: list[str], char_dim: int, dropout: float):
        super().__init__()
        self.char2idx = dict(zip(chars_list, range(len(chars_list))))
        self.char2vec = nn.Embedding(len(self.char2idx), char_dim)
        self.char_rnn = nn.GRU(char_dim, char_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_sentences: list[Sentence]) -> Tensor:
        """

        :param batch_sentences: (batch_size, sentence_length)
        :return: [batch_size, sentence_length, char_size]
        """
        device = self.char2vec.weight.device
        lengths = list(map(len, batch_sentences))

        indices, char_lengths = list(), list()
        for sentence in batch_sentences:
            for token in sentence:
                char_text = token.text
                indices.append(torch.as_tensor([self.char2idx[c] for c in char_text], device=device))
                char_lengths.append(len(char_text))

        char_embeds = self.char2vec(pad_sequence(indices))
        char_lengths = torch.as_tensor(char_lengths, device=char_embeds.device)

        char_embeds = pack_padded_sequence(char_embeds, char_lengths.cpu(), enforce_sorted=False)
        char_embeds = pad_packed_sequence(self.char_rnn(char_embeds)[0], padding_value=0)[0]
        char_embeds = torch.sum(char_embeds, dim=0) / char_lengths.unsqueeze(-1)

        char_embeds = torch.split(char_embeds, lengths, dim=0)
        return self.dropout(pad_sequence(char_embeds, batch_first=True))


class Pos2Vec(nn.Module):
    def __init__(self, pos_list: list[str], pos_dim: int, dropout: float):
        super().__init__()
        self.pos2idx = dict(zip(pos_list, range(len(pos_list))))
        self.pos2vec = nn.Embedding(len(self.pos2idx), pos_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_sentences: list[Sentence]) -> tuple[Tensor, Tensor]:
        """

        :param batch_sentences: (batch_size, sentence_length)
        :return: [batch_size, sentence_length, pos_dim]
        """
        device = self.pos2vec.weight.device
        indices = list()
        for sentence in batch_sentences:
            pos_indices = list(map(lambda a: self.pos2idx[a], sentence.pos_tags))
            indices.append(torch.as_tensor(pos_indices, dtype=torch.long, device=device))
        return self.dropout(self.pos2vec(pad_sequence(indices).T))
