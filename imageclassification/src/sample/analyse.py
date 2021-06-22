import os
import sys

from sklearn.metrics import top_k_accuracy_score, hinge_loss

from sample import split_arg, load_dataset, load_predictions, label_indices

SOURCE_PATH, SOURCE_DATASET, SOURCE_EXT = split_arg(sys.argv[1])

source_dataset_without_data = load_dataset(os.path.join(SOURCE_PATH, SOURCE_DATASET + SOURCE_EXT))

label_indices = label_indices(source_dataset_without_data)

numeric_labels = list(range(len(label_indices)))

holdout_dataset = load_dataset("holdout.txt")

y_true = [label_indices[label] for label in holdout_dataset.values()]

iteration = 0
while True:
    filename = f"predictions.{iteration}.txt"

    if not os.path.exists(filename):
        break

    predictions = load_predictions(filename)

    y_score = list(predictions.values())

    top_1 = top_k_accuracy_score(
        y_true,
        y_score,
        k=1,
        labels=numeric_labels
    )

    hinge = hinge_loss(
        y_true,
        y_score,
        labels=numeric_labels
    )

    print(f"{hinge},{top_1}")

    iteration += 1
