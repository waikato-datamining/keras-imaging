import os
import sys
from collections import OrderedDict

from tensorflow import keras

from sample import *

INIT_LR = 1e-4
BS = 5
NUM_EPOCHS = 10
SEED = 42

SOURCE_PATH, SOURCE_DATASET, SOURCE_EXT = split_arg(sys.argv[1])

source_dataset_without_data = load_dataset(os.path.join(SOURCE_PATH, SOURCE_DATASET + SOURCE_EXT))

label_indices = OrderedDict()
for label in source_dataset_without_data.values():
    if label not in label_indices:
        label_indices[label] = len(label_indices)

PREDICTIONS_FILE_HEADER = "filename,label"
for label in label_indices.keys():
    PREDICTIONS_FILE_HEADER += f",{label}_prob"
PREDICTIONS_FILE_HEADER += "\n"

holdout_dataset = load_dataset("holdout.txt")

holdout_gen = data_flow_from_disk(SOURCE_PATH, holdout_dataset, label_indices, False, BS, SEED)

iteration = 0
while True:
    if not os.path.exists(f"train.{iteration}.txt") or not os.path.exists(f"validation.{iteration}.txt"):
        break

    print(f"ITERATION {iteration}")

    model = ResNet50_for_fine_tuning(len(label_indices))
    opt = keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    train_dataset = load_dataset(f"train.{iteration}.txt")
    validation_dataset = load_dataset(f"validation.{iteration}.txt")

    train_gen = data_flow_from_disk(SOURCE_PATH, train_dataset, label_indices, True, BS, SEED)
    val_gen = data_flow_from_disk(SOURCE_PATH, validation_dataset, label_indices, False, BS, SEED)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=NUM_EPOCHS
    )

    with open(f"predictions.{iteration}.txt", "w") as file:
        file.write(PREDICTIONS_FILE_HEADER)
        for holdout_item, prediction in zip(holdout_dataset.items(), model.predict(holdout_gen)):
            line = f"{holdout_item[0]},{holdout_item[1]}"
            for probability in prediction:
                line += f",{probability}"
            line += "\n"
            file.write(line)

    iteration += 1
