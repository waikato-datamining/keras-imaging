from collections import OrderedDict

from pandas import DataFrame
from tensorflow import keras

from ._types import Dataset, Predictions, LabelIndices
from ._util import label_indices


def data_flow_from_disk(
        path: str,
        dataset: Dataset,
        label_indices: LabelIndices,
        shuffle: bool,
        batch_size: int,
        seed: int
):
    gen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=keras.applications.resnet.preprocess_input
    )

    dataframe = DataFrame(
        data={
            "filename": list(dataset.keys()),
            "class": list(label_indices[label] for label in dataset.values())
        },
        columns=["filename", "class"]
    )

    return gen.flow_from_dataframe(
        dataframe,
        path,
        target_size=(224, 224),
        class_mode="raw",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed
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


class MyLogger(keras.callbacks.Callback):
    def __init__(self, batch_size: int, num_items: int):
        self._batch_size = batch_size
        self._num_items = num_items

    def on_predict_batch_end(self, batch, logs=None):
        print(f"predicted {(batch + 1) * self._batch_size} of {self._num_items}")


def dataset_predictions_ResNet50(path: str, dataset: Dataset) -> Predictions:
    model = keras.applications.ResNet50()
    model = keras.models.Model(model.input, model.layers[-2].output)
    dataset_size = len(dataset)

    predictions = model.predict(
        data_flow_from_disk(
            path,
            dataset,
            label_indices(dataset),  # Doesn't actually matter
            False,
            5,
            0
        ),
        callbacks=[MyLogger(5, dataset_size)]
    )

    result = OrderedDict()
    for i, filename in enumerate(dataset.keys()):
        result[filename] = predictions[i]

    return result
