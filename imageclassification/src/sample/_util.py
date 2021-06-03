from collections import OrderedDict
import os
from typing import Dict, Iterable, Tuple

from ._types import Dataset


def per_label(dataset: Dataset) -> Dict[str, Dataset]:
    """
    TODO
    """
    result = {}

    for filename, label in dataset.items():
        if label in result:
            subset = result[label]
        else:
            subset = OrderedDict()
            result[label] = subset

        subset[filename] = label

    return result


def num_labels(dataset: Dataset) -> int:
    """
    TODO
    """
    return len(set(dataset.values()))


def merge(d1: Dataset, d2: Dataset) -> Dataset:
    """
    TODO
    """
    return OrderedDict(**d1, **d2)


def split_arg(arg: str) -> Tuple[str, str, str]:
    """
    TODO
    """
    path_split = os.path.split(arg)
    return (path_split[0], *os.path.splitext(path_split[1]))


def first(iterable: Iterable):
    for el in iterable:
        return el


def compare_ignore_index(item: Tuple[int, float]) -> float:
    return item[1]
