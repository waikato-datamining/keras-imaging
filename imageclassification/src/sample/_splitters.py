from abc import ABC, abstractmethod
from collections import OrderedDict
from random import Random
from typing import Dict, Set

import numpy as np

from ._math import number_of_subsets, subset_number_to_subset
from ._keras import dataset_predictions_ResNet50
from ._kernel import RBFKernel
from ._types import Dataset, Split
from ._util import compare_ignore_index, per_label


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

        if self._count > dataset_size:
            raise ValueError(f"Can't select sub-set of size {self._count} from data-set of size {dataset_size}")

        choice_set = subset_number_to_subset(
            dataset_size,
            self._count,
            self._random.randrange(number_of_subsets(dataset_size, self._count))
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
    def __init__(self, count_per_label: int, labels: Set[str], random: Random = Random()):
        self._count_per_label = count_per_label
        self._labels = labels
        self._random = random

    def __str__(self) -> str:
        return f"strat-{self._count_per_label}"

    def __call__(self, dataset: Dataset) -> Split:
        subsets_per_label = per_label(dataset)
        sub_splitter = RandomSplit(self._count_per_label, self._random)
        sub_splits = {
            label: sub_splitter(subsets_per_label[label])
            for label in self._labels
        }

        result = OrderedDict(), OrderedDict()

        for filename, label in dataset.items():
            result_index = 0 if filename in sub_splits[label][0] else 1

            result[result_index][filename] = label

        return result


class KernelHerdingSplit(Splitter):
    """
    TODO
    """
    _predictions_cache: Dict[str, np.ndarray] = {}

    def __init__(self, root_path: str, count: int):
        self._root_path = root_path
        self._count = count
        self._kernel = RBFKernel()

    def __call__(self, dataset: Dataset) -> Split:
        self._predictions_cache.update(
            dataset_predictions_ResNet50(
                self._root_path,
                OrderedDict({
                    filename: label
                    for filename, label in dataset.items()
                    if filename not in self._predictions_cache
                })
            )
        )
        print("UPDATED PREDICTIONS CACHE")

        kernel_list = [self._predictions_cache[filename] for filename in dataset.keys()]
        dataset_size = len(dataset)

        self._kernel.build_kernel(kernel_list)
        print("BUILT KERNEL")

        estimated_expected_similarity = [
            sum(self._kernel.eval(i, j, kernel_list[i]) for j in range(dataset_size)) / dataset_size
            for i in range(dataset_size)
        ]
        print("ESTIMATED EXPECTED SIMILARITY")

        index, max_score = max(enumerate(estimated_expected_similarity), key=compare_ignore_index)

        num_sampled = 0
        accumulated_similarity_to_sample = [0.0] * dataset_size
        selected = [False] * dataset_size
        while True:
            selected[index] = True
            num_sampled += 1
            print(f"SELECTED ITEM {index} OF {dataset_size}")
            if num_sampled >= self._count:
                break

            for i in range(dataset_size):
                if not selected[i]:
                    accumulated_similarity_to_sample[i] += self._kernel.eval(i, index, kernel_list[i])
            print("UPDATED ACCUMULATED SIMILARITIES")

            index, max_score = max(
                (
                    (i, estimated_expected_similarity[i] - accumulated_similarity_to_sample[i] / (num_sampled + 1))
                    for i in range(dataset_size)
                    if not selected[i]
                ),
                key=compare_ignore_index
            )

        result = OrderedDict(), OrderedDict()

        for index, filename in enumerate(dataset.keys()):
            result_index = 0 if selected[index] else 1

            result[result_index][filename] = dataset[filename]

        return result

    def __str__(self) -> str:
        return f"kh-{self._count}"
