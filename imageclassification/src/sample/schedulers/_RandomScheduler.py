from random import Random
from typing import List

from ._Scheduler import Scheduler
from .._math import subset_number_to_subset, number_of_subsets
from .._types import Dataset


class RandomScheduler(Scheduler):
    """
    TODO
    """
    def __init__(self, random: Random = Random()):
        self._random = random

    def __call__(self, dataset: Dataset) -> List[str]:
        file_list = list(dataset.keys())
        N = len(dataset)
        subset_number = self._random.randrange(number_of_subsets(N, N, True))
        subset = subset_number_to_subset(N, N, subset_number, True)

        return [
            file_list[index]
            for index in subset
        ]

    def __str__(self) -> str:
        return "rand"
