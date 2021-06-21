import os
import sys
from collections import OrderedDict

from sklearn.metrics import top_k_accuracy_score, hinge_loss

from sample import split_arg, load_dataset, load_predictions, per_label

SOURCE_PATH, SOURCE_DATASET, SOURCE_EXT = split_arg(sys.argv[1])

source_dataset_without_data = load_dataset(os.path.join(SOURCE_PATH, SOURCE_DATASET + SOURCE_EXT))

pl = per_label(source_dataset_without_data)

for label, ds in pl.items():
    print(f"{label}\t{len(ds)}")

label_indices = OrderedDict()
for label in source_dataset_without_data.values():
    if label not in label_indices:
        label_indices[label] = len(label_indices)

iteration = 0
while True:
    filename = f"predictions.{iteration}.txt"

    if not os.path.exists(filename):
        break

    predictions = load_predictions(filename)

    y_true = [label_indices[label] for label, _ in predictions.values()]
    y_score = [probs for _, probs in predictions.values()]

    top_1 = top_k_accuracy_score(
        y_true,
        y_score,
        k=1,
        labels=list(range(len(label_indices)))
    )

    hinge = hinge_loss(
        y_true,
        y_score,
        labels=list(range(len(label_indices)))
    )

    print(hinge)

    iteration += 1
