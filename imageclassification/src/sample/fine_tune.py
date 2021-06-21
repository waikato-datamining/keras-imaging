import os
import sys
from collections import OrderedDict
from random import Random

from tensorflow import keras

from sample import *
from sample.splitters import RandomSplitter

INIT_LR = 1e-4
BS = 5
NUM_EPOCHS = 10
SEED = 42
VALIDATION_PERCENT = 0.15

RANDOM = Random(SEED)

SOURCE_PATH, SOURCE_DATASET, SOURCE_EXT = split_arg(sys.argv[1])

source_dataset = load_dataset("schedule.txt")

label_indices = label_indices(load_dataset(os.path.join(SOURCE_PATH, SOURCE_DATASET + SOURCE_EXT)))

PREDICTIONS_FILE_HEADER = predictions_file_header(label_indices)

holdout_dataset = load_dataset("holdout.txt")

holdout_gen = data_flow_from_disk(SOURCE_PATH, holdout_dataset, label_indices, False, BS, SEED)

iteration = 0
while True:
    subset_size = iteration + 2

    if subset_size > len(source_dataset):
        break

    print(f"ITERATION {iteration}")

    validation_size = max(int(subset_size * VALIDATION_PERCENT), 1)

    train_dataset = top_n(source_dataset, subset_size)
    validation_dataset, train_dataset = RandomSplitter(validation_size, RANDOM)(train_dataset)

    model = ResNet50_for_fine_tuning(len(label_indices))
    opt = keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    train_gen = data_flow_from_disk(SOURCE_PATH, train_dataset, label_indices, True, BS, SEED)
    val_gen = data_flow_from_disk(SOURCE_PATH, validation_dataset, label_indices, False, BS, SEED)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=NUM_EPOCHS
    )

    predictions: Predictions = OrderedDict()
    for holdout_item, prediction in zip(holdout_dataset.keys(), model.predict(holdout_gen)):
        predictions[holdout_item] = prediction

    write_predictions(predictions, PREDICTIONS_FILE_HEADER, f"predictions.{iteration}.txt")

    iteration += 1
