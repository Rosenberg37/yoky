import argparse
import json
import random
from typing import Union, TextIO, List

import langdetect
from flair.data import Token, iob2, iob_iobes
from flair.tokenization import SegtokTokenizer
from torch.utils.data import Subset
from tqdm import tqdm

from yoky.utils.base import AttributeHolder, Arguments
from yoky.utils.data import DataPoint, logger, YokyDataset


class Entity:
    def __init__(self, data: dict, sentence=None):
        self.start = data['start']
        self.end = data['end']
        self.type = data['type']
        self.sentence = sentence

        self._nest = None
        self._nest_pair = None

    @staticmethod
    def spawn(data: dict, sentence=None):
        return Entity(data, sentence)

    @property
    def nest(self) -> list[str]:
        if self._nest is None:
            if self._nest_pair is not None:
                raise RuntimeError("Unexpected situation.")
            self.init_nest()
        return self._nest

    @property
    def nest_pair(self):
        if self._nest_pair is None:
            if self._nest is not None:
                raise RuntimeError("Unexpected situation.")
            self.init_nest()
        return self._nest_pair

    def init_nest(self):
        self._nest = list()
        self._nest_pair = list()
        for entity in self.sentence.entities:
            if entity is not self:
                s_1, e_1, t_1 = entity.start, entity.end - 1, entity.type
                s_2, e_2, t_2 = self.start, self.end - 1, self.type

                self._nest_pair.append(entity)
                if s_1 == s_2 and e_1 == e_2 and t_1 != t_2:
                    self._nest.append("ME")
                elif e_2 >= e_1 >= s_1 >= s_2 and t_1 == t_2:
                    self._nest.append("NST")
                elif e_2 >= e_1 >= s_1 >= s_2 and t_1 != t_2:
                    self._nest.append("NDT")
                elif e_1 > e_2 >= s_1 > s_2 and t_1 == t_2:
                    self._nest.append("OST")
                elif e_1 > e_2 >= s_1 > s_2 and t_1 != t_2:
                    self._nest.append("ODT")
                else:
                    self._nest_pair.pop(-1)

    def __eq__(self, other) -> bool:
        return self.start == other.start and self.end == other.end and self.type == other.type

    def __repr__(self):
        return f"(start_index:{self.start}, end_index:{self.end}, type:{self.type})"


class Sentence(DataPoint):
    def __init__(
            self,
            text: Union[str, List[str]],
            max_length: int = 512,
            entities: list[Entity] = None,
            language_code: str = None,
            start_position: int = None
    ):
        """
        Class to hold all meta related to a text (tokens, predictions, language code, ...)
        :param text: original string (sentence), or a list of string tokens (words)
        :param language_code: Language of the sentence
        :param start_position: Start char offset of the sentence in the superordinate document
        """
        super().__init__()

        self.entities = entities
        self.max_length = max_length

        self._context = None
        self._next_sentence = None
        self._previous_sentence = None
        self._embeddings = None

        self.tokens: List[Token] = []

        self.language_code: str = language_code

        self.start_pos = start_position
        self.end_pos = start_position + len(text) if start_position is not None else None

        tokenizer = SegtokTokenizer()

        # if text is passed, instantiate sentence with tokens (words)
        if text is not None:
            if isinstance(text, (list, tuple)):
                for token in text:
                    self.add_token(token)
            else:
                for token in tokenizer.tokenize(text):
                    self.add_token(token)

                    # log a warning if the dataset is empty
        if text == "":
            logger.warning(
                "Warning: An empty Sentence was created! Are there empty strings in your dataset?"
            )

        self.tokenized = None

        # some sentences represent a document boundary (but most do not)
        self.is_document_boundary: bool = False

    @property
    def memory_cost(self):
        return self.__len__()

    def get_token(self, token_id: int) -> Token:
        for token in self.tokens:
            if token.idx == token_id:
                return token

    def add_token(self, token: Union[Token, str]):

        if type(token) is str:
            token = Token(token)

        token.text = token.text.replace('\u200c', '')
        token.text = token.text.replace('\u200b', '')
        token.text = token.text.replace('\ufe0f', '')
        token.text = token.text.replace('\ufeff', '')

        # data with zero-width characters cannot be handled
        if token.text == '':
            return

        self.tokens.append(token)

        # set token idx if not set
        token.sentence = self
        if token.idx is None:
            token.idx = len(self.tokens)

    def get_label_names(self):
        label_names = []
        for label in self.labels:
            label_names.append(label.value)
        return label_names

    def to_tagged_string(self, main_tag=None) -> str:
        token_lists = []
        for token in self.tokens:
            token_lists.append(token.text)

            tags: List[str] = []
            for label_type in token.annotation_layers.keys():

                if main_tag is not None and main_tag != label_type:
                    continue

                if token.get_labels(label_type)[0].value == "O":
                    continue
                if token.get_labels(label_type)[0].value == "_":
                    continue

                tags.append(token.get_labels(label_type)[0].value)
            all_tags = "<" + "/".join(tags) + ">"
            if all_tags != "<>":
                token_lists.append(all_tags)
        return " ".join(token_lists)

    def to_tokenized_string(self) -> str:

        if self.tokenized is None:
            self.tokenized = " ".join([t.text for t in self.tokens])

        return self.tokenized

    def to_plain_string(self):
        plain = ""
        for token in self.tokens:
            plain += token.text
            if token.whitespace_after:
                plain += " "
        return plain.rstrip()

    def convert_tag_scheme(self, tag_type: str = "ner", target_scheme: str = "iob"):

        tags = list()
        for token in self.tokens:
            tags.append(token.get_tag(tag_type))

        if target_scheme == "iob":
            iob2(tags)

        if target_scheme == "iobes":
            iob2(tags)
            tags = iob_iobes(tags)

        for index, tag in enumerate(tags):
            self.tokens[index].set_label(tag_type, tag)

    def to_original_text(self) -> str:
        if len(self.tokens) > 0 and (self.tokens[0].start_pos is None):
            return " ".join([t.text for t in self.tokens])
        string = ""
        pos = 0
        for t in self.tokens:
            while t.start_pos > pos:
                string += " "
                pos += 1

            string += t.text
            pos += len(t.text)

        return string

    def __getitem__(self, idx: int) -> Token:
        return self.tokens[idx]

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self) -> int:
        return min(len(self.tokens), self.max_length)

    def __repr__(self):
        tagged_string = self.to_tagged_string()
        tokenized_string = self.to_tokenized_string()

        # add Sentence labels to output if they exist
        sentence_labels = f"  − Sentence-Labels: {self.annotations}" if self.annotations != {} else ""

        # add Token labels to output if they exist
        token_labels = f'  − Token-Labels: "{tagged_string}"' if tokenized_string != tagged_string else ""

        return f'Sentence: "{tokenized_string}"   [− Tokens: {len(self)}{token_labels}{sentence_labels}]'

    def __str__(self) -> str:

        tagged_string = self.to_tagged_string()
        tokenized_string = self.to_tokenized_string()

        # add Sentence labels to output if they exist
        sentence_labels = f"  − Sentence-Labels: {self.annotations}" if self.annotations != {} else ""

        # add Token labels to output if they exist
        token_labels = f'  − Token-Labels: "{tagged_string}"' if tokenized_string != tagged_string else ""

        return f'Sentence: "{tokenized_string}"   [− Tokens: {len(self)}{token_labels}{sentence_labels}]'

    def get_language_code(self) -> str:
        if self.language_code is None:
            self.language_code = langdetect.detect(self.to_plain_string())
        return self.language_code

    @property
    def next_sentence(self):
        """
        Get the next sentence in the document (works only if context is set through dataloader or elsewhere)
        :return: next Sentence in document if set, otherwise None
        """
        return self._next_sentence

    @property
    def previous_sentence(self):
        """
        Get the previous sentence in the document (works only if context is set through dataloader or elsewhere)
        :return: previous Sentence in document if set, otherwise None
        """

        return self._previous_sentence

    def get_labels(self, label_type: str = None):
        return self.annotations[label_type] if label_type in self.annotations else []

    @property
    def sentence_tokens(self):
        return list(map(lambda a: a.text, self.tokens))[:self.max_length]

    @property
    def previous_tokens(self):
        return self.previous_sentence.sentence_tokens

    @property
    def next_tokens(self):
        return self.next_sentence.sentence_tokens

    @property
    def nested_entities(self) -> list[Entity]:
        return list(filter(lambda a: len(a.nest) > 0, self.entities))

    @property
    def flat_entities(self) -> list[Entity]:
        return list(filter(lambda a: len(a.nest) == 0, self.entities))

    @property
    def context(self):
        if self._context is None:
            self._context = self.sentence_tokens
            if len(self._context) < self.max_length:
                self._context = self._context + self.next_tokens
                if len(self._context) < self.max_length:
                    self._context = self.previous_tokens + self._context
                    if len(self._context) < self.max_length:
                        offset = len(self.previous_sentence)
                        self.start_pos, self.end_pos = offset, offset + len(self.tokens)
                    else:
                        self._context = self._context[-self.max_length:]
                        offset = self.max_length - len(self._context)
                        self.start_pos, self.end_pos = offset, offset + len(self.tokens)
                else:
                    self._context = self._context[:self.max_length]
                    self.start_pos, self.end_pos = 0, self.max_length
            else:
                self._context = self._context[:self.max_length]
                self.start_pos, self.end_pos = 0, self.max_length
        return self._context

    @property
    def pos_tags(self):
        return self.get_labels('pos')[0].value[:self.max_length]

    def get_nested_entities(self, kind: str) -> list[Entity]:
        return list(filter(lambda a: kind in a.nest, self.entities))


class SentenceDataset(YokyDataset):
    def __init__(self, file: TextIO, name: str, max_length: int = None, mini_size: int = None, length_cut: int = 4):
        self.length_cut = length_cut
        self.name = name
        data_list = json.load(file)
        if mini_size is not None:
            data_list = random.sample(data_list, mini_size)

        self._sentences = list()
        for data in tqdm(data_list, desc=f"Formatting {self.name}"):
            sentence = Sentence(data['tokens'], max_length)
            sentence._previous_sentence = Sentence(data['ltokens'], max_length)
            sentence._next_sentence = Sentence(data['rtokens'], max_length)
            sentence.add_label('pos', data['pos'])
            if 'tags' in data.keys():
                sentence.add_label('tags', data['tags'])
            sentence.entities = list(map(lambda a: Entity.spawn(a, sentence), data['entities']))
            self._sentences.append(sentence)

    def add_data(self, sentences: list[Sentence]):
        self._sentences += sentences

    def get_nested_entities_num(self, kind: str):
        return sum(map(lambda a: len(a.get_nested_entities(kind)), self._sentences))

    @property
    def sentences(self):
        return self._sentences

    @property
    def entities_num(self):
        return sum(map(lambda a: len(a.entities), self._sentences))

    @property
    def max_entities_num(self):
        return max(map(lambda a: len(a.entities), self._sentences))

    @property
    def entities_num_by_length(self):
        nums = dict()
        for sentence in self.sentences:
            for entity in sentence.entities:
                entity_length = entity.end - entity.start
                if entity_length > self.length_cut:
                    entity_length = self.length_cut + 1
                nums[entity_length] = nums.get(entity_length, 0) + 1
        return nums

    @property
    def nested_entities_num(self):
        return sum(map(lambda a: len(a.nested_entities), self._sentences))

    @property
    def flat_entities_num(self):
        return sum(map(lambda a: len(a.flat_entities), self._sentences))

    @property
    def average_sentences_length(self):
        return sum(map(len, self.sentences)) / len(self.sentences)

    @property
    def average_entities_length(self):
        return sum(map(lambda a: sum(map(lambda a: a.end - a.start, a.entities)), self.sentences)) / self.entities_num

    def __getitem__(self, index: int) -> Sentence:
        return self._sentences[index]

    def __len__(self) -> int:
        return len(self._sentences)

    @staticmethod
    def is_in_memory() -> bool:
        return True

    @staticmethod
    def create(dataset_path: str, mode: str, *args, **kargs):
        with open(f"{dataset_path}/processed/{mode}.json", encoding="utf-8") as file:
            dataset = SentenceDataset(file, mode, *args, **kargs)
        return dataset


class CorpusArguments(Arguments):
    def parse(self):
        parser = argparse.ArgumentParser(description='Process dataset parameter.')
        parser.add_argument('--corpus_name', type=str, default='weiboNER', help='Selection of dataset.')
        parser.add_argument('--mini_size', type=int, default=None, help='How large the mini dataset will be.')
        parser.add_argument('--filter_length', type=int, default=128, help='Sentences with max context length to be filtered.')
        parser.add_argument('--max_length', type=int, default=512, help='Maximum context length.')
        parser.add_argument('--concat_train_dev', default=False, action='store_true',
                            help='Whether concatenate the train and development dataset.')
        self.__dict__.update(vars(parser.parse_known_args()[0]))
        return self


class Corpus:
    def __init__(self, dataset_path: str, corpus_args: CorpusArguments):
        self.corpus_name = corpus_args.corpus_name
        self.concat_train_dev = corpus_args.concat_train_dev
        self.filter_length = corpus_args.filter_length
        self.mini_size = corpus_args.mini_size
        self.max_length = corpus_args.max_length

        self._metadata = None
        self._dataset_path = dataset_path
        self._train, self._dev, self._test = None, None, None

    def filter(self, condition: callable, dataset_names: list[str] = None):
        if dataset_names is None:
            dataset_names = ['train']

        for name in dataset_names:
            dataset = self.__dict__[f'_{name}']
            self.__dict__[f'_{name}'] = self._filter(dataset, condition)

    @staticmethod
    def _filter(dataset: SentenceDataset, condition: callable) -> SentenceDataset:

        # find out empty sentence indices
        empty_sentence_indices = []
        non_empty_sentence_indices = []
        index = 0

        for sentence in dataset:
            if condition(sentence):
                non_empty_sentence_indices.append(index)
            else:
                empty_sentence_indices.append(index)
            index += 1

        # create subset of non-empty sentence indices
        subset = Subset(dataset, non_empty_sentence_indices)
        logger.info(f'{dataset.name}: filtered sentences num:{len(empty_sentence_indices)}')

        return subset

    def get_nested_entities_nums(self, kind: str):
        return {
            'train': self.train.get_nested_entities_num(kind),
            'dev': self.dev.get_nested_entities_num(kind),
            'test': self.test.get_nested_entities_num(kind),
        }

    @property
    def dataset_path(self):
        return self._dataset_path

    @property
    def metadata(self) -> AttributeHolder:
        if self._metadata is None:
            metadata = {
                'pos_list': set(),
                'chars_list': set(),
                'types': set(),
            }

            for divide in ['train', 'dev', 'test']:
                with open(f"{self._dataset_path}/processed/{divide}.json", 'r', encoding='utf-8') as file:
                    for data in tqdm(json.load(file), desc=f"Generating Metadata: {divide}"):
                        metadata['pos_list'].update(set(data['pos']))
                        for token in data['tokens']:
                            metadata['chars_list'].update(set(token))
                        for token in data['ltokens']:
                            metadata['chars_list'].update(set(token))
                        for token in data['rtokens']:
                            metadata['chars_list'].update(set(token))
                        for entity in data['entities']:
                            metadata['types'].add(entity['type'])

            for key, value in metadata.items():
                metadata[key] = list(value)

            types = metadata.pop('types')
            types_idx = range(len(types))
            metadata.update({
                'types2idx': dict(zip(types, types_idx)),
                'idx2types': dict(zip(types_idx, types)),
            })

            self._metadata = AttributeHolder(metadata)
        return self._metadata

    @property
    def datasets(self) -> list[SentenceDataset]:
        datasets = map(lambda a: self.__dict__[f'_{a}'], ['train', 'dev', 'test'])
        return list(filter(lambda a: a is not None, datasets))

    @property
    def train(self) -> SentenceDataset:
        if self._train is None:
            self._train = SentenceDataset.create(self.dataset_path, 'train', self.max_length, self.mini_size)
            if self.concat_train_dev:
                self._train.add_data(self.dev.sentences)
                self._dev = None

            if self.filter_length is not None:
                logger.info("Filtering sentences with long context and empty.")
                self.filter(lambda a: 0 < len(a.context) < self.filter_length, ['train'])
        return self._train

    @property
    def dev(self) -> SentenceDataset:
        if self._dev is None:
            self._dev = SentenceDataset.create(self.dataset_path, 'dev', self.max_length)
        return self._dev

    @property
    def test(self) -> SentenceDataset:
        if self._test is None:
            self._test = SentenceDataset.create(self.dataset_path, 'test', self.max_length)
        return self._test

    @property
    def sentences_nums(self):
        return {
            'train': len(self.train),
            'dev': len(self.dev),
            'test': len(self.test),
        }

    @property
    def entities_nums(self):
        return {
            'train': self.train.entities_num,
            'dev': self.dev.entities_num,
            'test': self.test.entities_num,
        }

    @property
    def max_entities_nums(self):
        return {
            'train': self.train.max_entities_num,
            'dev': self.dev.max_entities_num,
            'test': self.test.max_entities_num,
        }

    @property
    def entities_nums_by_length(self):
        return {
            'train': self.train.entities_num_by_length,
            'dev': self.dev.entities_num_by_length,
            'test': self.test.entities_num_by_length,
        }

    @property
    def nested_entities_nums(self):
        return {
            'train': self.train.nested_entities_num,
            'dev': self.dev.nested_entities_num,
            'test': self.test.nested_entities_num,
        }

    @property
    def flat_entities_nums(self):
        return {
            'train': self.train.flat_entities_num,
            'dev': self.dev.flat_entities_num,
            'test': self.test.flat_entities_num,
        }

    @property
    def average_sentence_lengths(self):
        return {
            'train': self.train.average_sentences_length,
            'dev': self.dev.average_sentences_length,
            'test': self.test.average_sentences_length,
        }

    @property
    def average_entities_lengths(self):
        return {
            'train': self.train.average_entities_length,
            'dev': self.dev.average_entities_length,
            'test': self.test.average_entities_length,
        }

    def __len__(self):
        return len(self.train)

    def get_statistic(self) -> dict:
        return {
            "Name": self.corpus_name,
            "sentences_nums": self.sentences_nums,
            "entities_nums": self.entities_nums,
            "max_entities_nums": self.max_entities_nums,
            "entities_nums_by_length": self.entities_nums_by_length,
            "nested_entities_nums": self.nested_entities_nums,
            "flat_entities_nums": self.flat_entities_nums,
            "ME": self.get_nested_entities_nums('ME'),
            "NST": self.get_nested_entities_nums('NST'),
            "NDT": self.get_nested_entities_nums('NDT'),
            "OST": self.get_nested_entities_nums('OST'),
            "ODT": self.get_nested_entities_nums('ODT'),
            "average_sentence_lengths": self.average_sentence_lengths,
            "average_entities_lengths": self.average_entities_lengths,
        }

    def __str__(self) -> str:
        return "Corpus: %d train + %d dev + %d test sentences" % (
            len(self.train) if self.train else 0,
            len(self.dev) if self.dev else 0,
            len(self.test) if self.test else 0,
        )
