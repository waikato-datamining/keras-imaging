import os
import sys
from collections import OrderedDict

from tensorflow import keras

from sample import *

INIT_LR = 1e-4
BS = 4
NUM_EPOCHS = 1

SOURCE_PATH, SOURCE_DATASET, SOURCE_EXT = split_arg(sys.argv[1])

source_dataset_without_data = load_dataset(os.path.join(SOURCE_PATH, SOURCE_DATASET + SOURCE_EXT))
source_dataset_with_data = dataset_with_data(SOURCE_PATH, source_dataset_without_data)

label_indices = {}
for label in source_dataset_without_data.values():
    if label not in label_indices:
        label_indices[label] = len(label_indices)

model = ResNet50_for_fine_tuning(num_labels(source_dataset_without_data))
opt = keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

iteration = 0
while True:
    print(f"ITERATION {iteration}")

    if not os.path.exists(f"train.{iteration}.txt") or not os.path.exists(f"validation.{iteration}.txt"):
        break

    with open(f"train.{iteration}.txt", "r") as file:
        train_dataset = OrderedDict((
            (filename.strip(), source_dataset_with_data[filename.strip()])
            for filename in file.readlines()
        ))

    with open(f"validation.{iteration}.txt", "r") as file:
        validation_dataset = OrderedDict((
            (filename.strip(), source_dataset_with_data[filename.strip()])
            for filename in file.readlines()
        ))

    train_gen = data_flow(train_dataset, label_indices, True, BS)
    val_gen = data_flow(validation_dataset, label_indices, False, BS)

    model.fit(
        train_gen,
        steps_per_epoch=len(train_dataset) // BS,
        validation_data=val_gen,
        validation_steps=len(validation_dataset) // BS,
        epochs=NUM_EPOCHS
    )

    iteration += 1
