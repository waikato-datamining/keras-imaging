import os.path
from collections import OrderedDict

import numpy as np

from ._types import Dataset, Predictions


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


def load_predictions(filename: str) -> Predictions:
    """
    TODO
    """
    result = OrderedDict()

    with open(filename, "r") as file:
        file.readline()
        while True:
            item = file.readline().rstrip()
            if item == "":
                break
            split = item.split(",")
            filename = split[0]
            probs = [float(prob) for prob in split[1:]]
            result[filename] = np.array(probs)

    return result
