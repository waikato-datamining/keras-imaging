from abc import ABC, abstractmethod
from collections import OrderedDict
from random import Random

from . import number_of_subsets, subset_number_to_subset
from ._types import Dataset, Split
from ._util import per_label


class Splitter(ABC):
    @abstractmethod
    def __call__(self, dataset: Dataset) -> Split:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class RandomSplit(Splitter):
    """
    TODO
    """
    def __init__(self, count: int, random: Random = Random()):
        self._count = count
        self._random = random

    def __str__(self) -> str:
        return f"rand-{self._count}"

    def __call__(self, dataset: Dataset) -> Split:
        dataset_size = len(dataset)

        # TODO: Change to order_matters = False when available
        choice_set = subset_number_to_subset(
            dataset_size,
            self._count,
            self._random.randrange(number_of_subsets(dataset_size, self._count, True)),
            True
        )
        choice_set = set(choice_set)

        result = OrderedDict(), OrderedDict()

        for index, filename in enumerate(dataset.keys()):
            result_index = 0 if index in choice_set else 1

            result[result_index][filename] = dataset[filename]

        return result


class StratifiedSplit(Splitter):
    """
    TODO
    """
    def __init__(self, count_per_label: int, random: Random = Random()):
        self._count_per_label = count_per_label
        self._random = random

    def __str__(self) -> str:
        return f"strat-{self._count_per_label}"

    def __call__(self, dataset: Dataset) -> Split:
        subsets_per_label = per_label(dataset)
        sub_splitter = RandomSplit(self._count_per_label, self._random)
        sub_splits = {
            label: sub_splitter(label_dataset)
            for label, label_dataset in subsets_per_label.items()
        }

        result = OrderedDict(), OrderedDict()

        for filename, label in dataset.items():
            result_index = 0 if filename in sub_splits[label][0] else 1

            result[result_index][filename] = label

        return result
