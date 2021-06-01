import os.path
from collections import OrderedDict

from ._types import Dataset


def load_dataset(filename: str) -> Dataset:
    """
    TODO
    """
    result = OrderedDict()

    with open(filename, "r") as file:
        while True:
            item = file.readline().rstrip()
            if item == "":
                break
            label = os.path.split(os.path.split(item)[0])[1]
            result[item] = label

    return result
