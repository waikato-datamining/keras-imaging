from collections import OrderedDict
import os
from random import Random
from typing import Iterable, Tuple, OrderedDict as ODict

from ._math import random_permutation
from ._types import Dataset, LabelIndices


def per_label(dataset: Dataset) -> ODict[str, Dataset]:
    """
    TODO
    """
    result = OrderedDict()

    for filename, label in dataset.items():
        if label in result:
            subset = result[label]
        else:
            subset = OrderedDict()
            result[label] = subset

        subset[filename] = label

    return result


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


def label_indices(dataset: Dataset) -> LabelIndices:
    """
    TODO
    """
    result = OrderedDict()
    for label in dataset.values():
        if label not in result:
            result[label] = len(result)
    return result


def predictions_file_header(indices: LabelIndices) -> str:
    """
    TODO
    """
    return f"filename,{','.join(f'{label}_prob' for label in indices.keys())}\n"


def top_n(dataset: Dataset, n: int) -> Dataset:
    """
    TODO
    """
    result = OrderedDict()
    for i, filename in enumerate(dataset.keys()):
        if i >= n:
            break
        result[filename] = dataset[filename]
    return result


def change_path(
        dataset: Dataset,
        path: str
) -> Dataset:
    """
    TODO
    """
    result = OrderedDict()
    for filename, label in dataset.items():
        _, file = os.path.split(filename)
        file, ext = os.path.splitext(file)
        changed_filename = os.path.join(path, f"{file}.xml")
        result[changed_filename] = dataset[filename]
    return result


def shuffle_dataset(dataset: Dataset, random: Random = Random()) -> Dataset:
    """
    TODO
    """
    order = random_permutation(list(dataset.keys()), random)
    result = OrderedDict()
    for filename in order:
        result[filename] = dataset[filename]
    return result


def rm_dir(path: str):
    """
    TODO
    """
    for dirpath, dirnames, filenames in os.walk(path, False, followlinks=False):
        for filename in filenames:
            os.remove(os.path.join(dirpath, filename))
        for dirname in dirnames:
            os.rmdir(os.path.join(dirpath, dirname))
    os.rmdir(path)
