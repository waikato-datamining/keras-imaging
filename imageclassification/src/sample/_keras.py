import os
from typing import Dict

import numpy as np
from tensorflow import keras

from ._types import Dataset


def dataset_predictions_ResNet50(path: str, dataset: Dataset) -> Dict[str, np.ndarray]:
    model = keras.applications.ResNet50()
    model = keras.models.Model(model.input, model.layers[-2].output)
    dataset_size = len(dataset)

    result = {}

    for index, filename in enumerate(dataset.keys(), 1):
        result[filename] = model(
            np.array([
                keras.preprocessing.image.img_to_array(
                    keras.preprocessing.image.load_img(
                        os.path.join(path, filename),
                        target_size=(224, 224)
                    )
                )
            ])
        ).numpy()
        print(f"GENERATED PREDICTION {index} OF {dataset_size}")

    return result
