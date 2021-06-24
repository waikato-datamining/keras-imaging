import os
import sys

from sklearn.metrics import top_k_accuracy_score

from sample import split_arg, load_dataset, load_predictions, label_indices
from sample.splitters import TopNSplitter

SOURCE_PATH, SOURCE_DATASET, SOURCE_EXT = split_arg(sys.argv[1])

source_dataset = load_dataset(os.path.join(SOURCE_PATH, SOURCE_DATASET + SOURCE_EXT))

label_indices = label_indices(source_dataset)

numeric_labels = list(range(len(label_indices)))

holdout_dataset = load_dataset("holdout.txt")

schedule_dataset = load_dataset("schedule.txt")

y_true = [label_indices[label] for label in holdout_dataset.values()]

splitter = TopNSplitter(50)

iteration = 0
_, remaining_dataset = splitter(schedule_dataset)
while True:
    holdout_predictions_filename = f"predictions.{iteration}.txt"

    if not os.path.exists(holdout_predictions_filename):
        break

    holdout_predictions = load_predictions(holdout_predictions_filename)

    y_score = list(holdout_predictions.values())

    top_1 = top_k_accuracy_score(
        y_true,
        y_score,
        k=1,
        labels=numeric_labels
    )

    # hinge = hinge_loss(y_true, y_score, labels=numeric_labels)

    update_dataset, remaining_dataset = splitter(remaining_dataset)

    update_predictions_filename = f"update_predictions.{iteration}.txt"

    update_top_1 = None
    if os.path.exists(update_predictions_filename):
        update_y_true = [label_indices[label] for label in update_dataset.values()]
        update_predictions = load_predictions(update_predictions_filename)
        update_y_score = list(update_predictions.values())
        update_top_1 = top_k_accuracy_score(update_y_true, update_y_score, k=1, labels=numeric_labels)

    print(top_1 if update_top_1 is None else f"{top_1},{update_top_1}")

    iteration += 1
