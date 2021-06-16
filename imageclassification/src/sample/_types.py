from typing import OrderedDict, Tuple

import numpy as np

Dataset = OrderedDict[str, str]

DatasetWithData = OrderedDict[str, Tuple[str, np.ndarray]]

Split = Tuple[Dataset, Dataset]
