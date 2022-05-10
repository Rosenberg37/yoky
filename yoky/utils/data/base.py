import logging
from abc import abstractmethod
from typing import List, Iterator, Union

from torch.utils.data import Sampler, SequentialSampler, Dataset, DataLoader, RandomSampler
from torch.utils.data.dataset import T_co

logger = logging.getLogger('yoky')


class Label:
    """
    This class represents a label.
    """

    def __init__(self, value: str):
        self._value = value
        super().__init__()

    def set_value(self, value: str):
        self.value = value

    @staticmethod
    def spawn(value: str):
        return Label(value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not value and value != "":
            raise ValueError("Incorrect label value provided. Label value needs to be set.")
        else:
            self._value = value

    def __str__(self):
        return f"{self._value}"

    def __repr__(self):
        return f"{self._value}"

    def __eq__(self, other):
        return self.value == other.value


class DataPoint:
    """
    This is the parent class of all data points.
    """

    def __init__(self):
        self.annotations = {}

    def add_label(self, name: str, value: str):

        if name not in self.annotations:
            self.annotations[name] = [Label(value)]
        else:
            self.annotations[name].append(Label(value))

        return self

    def set_label(self, typename: str, value: str):
        self.annotations[typename] = [Label(value)]
        return self

    def remove_labels(self, typename: str):
        if typename in self.annotations.keys():
            self.annotations.pop(typename)

    def get_labels(self, typename: str = None):
        if typename is None:
            return self.labels

        return self.annotations[typename] if typename in self.annotations else []

    @property
    @abstractmethod
    def memory_cost(self):
        pass

    @property
    def labels(self) -> List[Label]:
        all_labels = []
        for key in self.annotations.keys():
            all_labels.extend(self.annotations[key])
        return all_labels


class YokyDataset(Dataset):
    @abstractmethod
    def __getitem__(self, index) -> T_co:
        pass

    @abstractmethod
    def __len__(self) -> T_co:
        pass


class YokyDataLoader(DataLoader):
    def __init__(
            self,
            dataset: YokyDataset,
            batch_size: int = None,
            shuffle: bool = True,
            drop_last: bool = False,
            use_dynamic_batch: bool = False,
            memory_limit: float = None,
            num_workers=0,
    ):
        """

        :param dataset (Dataset): dataset from which to load the data.
        :param batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        :param use_dynamic_batch: whether we use dynamic batch size.
        :param memory_limit: memory limitation if we use dynamic batch_size
        :param shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        :param num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        :param drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        """
        if hasattr(dataset, 'is_in_memory') and dataset.is_in_memory():
            num_workers = 0

        batch_sampler = None
        if use_dynamic_batch:
            batch_sampler = DynamicSampler(dataset, batch_size, shuffle, drop_last, memory_limit)
            shuffle, batch_size, drop_last = None, 1, None

        super(YokyDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=list,
            drop_last=drop_last,
        )


class DynamicSampler(Sampler):
    r"""Generate dynamic mini-batch of indices."""

    def __init__(
            self,
            data_source: YokyDataset,
            batch_size: int,
            shuffle: bool,
            drop_last: bool,
            memory_limit: float,
    ):
        super().__init__(data_source)
        if memory_limit is None:
            raise ValueError("Use dynamic batch_size without specified memory limitation.")

        self.sampler: Sampler = RandomSampler(data_source) if shuffle else SequentialSampler(data_source)
        self.data_source: YokyDataset = data_source
        self.batch_size: int = batch_size
        self.drop_last: bool = drop_last
        self.memory_limit: float = memory_limit

        self._batch_indices: Union[list[list[int]], None] = None

    @property
    def batch_indices(self) -> list[list[int]]:
        batch_size = -1 if self.batch_size is None else self.batch_size
        if self._batch_indices is None:
            self._batch_indices = list()
            batch, memory_cost = list(), 0
            for idx in self.sampler:
                cost = self.data_source[idx].memory_cost
                if memory_cost + cost > self.memory_limit or len(batch) == batch_size:
                    if len(batch) == 0:
                        self._batch_indices.append([idx])
                        batch, memory_cost = list(), 0
                    else:
                        self._batch_indices.append(batch)
                        batch, memory_cost = [idx], cost
                else:
                    memory_cost += cost
                    batch.append(idx)
            if not self.drop_last and len(batch) > 0:
                self._batch_indices.append(batch)
        return self._batch_indices

    def __iter__(self) -> Iterator[List[int]]:
        for indices in self.batch_indices:
            yield indices
        self._batch_indices = None

    def __len__(self) -> int:
        return len(self.batch_indices)
