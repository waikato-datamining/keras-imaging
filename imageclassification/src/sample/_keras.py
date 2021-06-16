import os
from collections import OrderedDict
from typing import Dict, Iterator, Tuple

import numpy as np
from tensorflow import keras

from ._types import Dataset, DatasetWithData


def data_flow(
        dataset: DatasetWithData,
        label_indices: Dict[str, int],
        shuffle: bool,
        batch_size: int
):
    gen = keras.preprocessing.image.ImageDataGenerator()
    return gen.flow(
        np.array([item[1][0] for item in dataset.values()]),
        [label_indices[item[0]] for item in dataset.values()],
        batch_size=batch_size,
        shuffle=shuffle
    )


def ResNet50_for_fine_tuning(num_labels: int) -> keras.models.Model:
    base_model = keras.applications.ResNet50(include_top=False)
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    fine_tuning_model = base_model(inputs, training=False)
    fine_tuning_model = keras.layers.AveragePooling2D(pool_size=(7, 7))(fine_tuning_model)
    fine_tuning_model = keras.layers.Flatten(name="flatten")(fine_tuning_model)
    fine_tuning_model = keras.layers.Dense(256, activation="relu")(fine_tuning_model)
    fine_tuning_model = keras.layers.Dropout(0.5)(fine_tuning_model)
    fine_tuning_model = keras.layers.Dense(num_labels, activation="softmax")(fine_tuning_model)

    return keras.models.Model(inputs=inputs, outputs=fine_tuning_model)


def dataset_predictions_ResNet50(path: str, dataset: Dataset) -> Dict[str, np.ndarray]:
    model = keras.applications.ResNet50()
    model = keras.models.Model(model.input, model.layers[-2].output)
    dataset_size = len(dataset)

    result = {}

    for index, item in enumerate(dataset_iter_data(path, dataset), 1):
        filename, _, data = item
        result[filename] = model(data).numpy()
        print(f"GENERATED PREDICTION {index} OF {dataset_size}")

    return result


def dataset_with_data(path: str, dataset: Dataset) -> DatasetWithData:
    return OrderedDict((
        (filename, (label, data))
        for filename, label, data in dataset_iter_data(path, dataset)
    ))


def dataset_iter_data(path: str, dataset: Dataset) -> Iterator[Tuple[str, str, np.ndarray]]:
    for filename, label in dataset.items():
        yield filename, label, load_image(path, filename)


def load_image(path: str, filename: str) -> np.ndarray:
    return np.array([
        keras.preprocessing.image.img_to_array(
            keras.preprocessing.image.load_img(
                os.path.join(path, filename),
                target_size=(224, 224)
            )
        )
    ])
