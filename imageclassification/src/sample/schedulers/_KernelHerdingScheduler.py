from typing import List

from ._Scheduler import Scheduler
from .._types import Dataset
from ..splitters import KernelHerdingSplitter


class KernelHerdingScheduler(Scheduler):
    """
    TODO
    """
    def __init__(self, predictions_path: str):
        self._predictions_path = predictions_path

    def __call__(self, dataset: Dataset) -> List[str]:
        return list(
            KernelHerdingSplitter(self._predictions_path, len(dataset))(dataset)[0].keys()
        )

    def __str__(self) -> str:
        return "kh"
