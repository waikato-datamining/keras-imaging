import os

from sklearn.metrics import top_k_accuracy_score

from sample import load_dataset, load_predictions, label_indices as get_label_indices
from sample.splitters import TopNSplitter

models = ["mobilenet", "resnet50", "resnet152"]
datasets = ["asl", "flowers-files", "dog_breeds"]
splits = ["rand", "uni", "kh"]

with open("pivot.txt", "w") as pivot_file:
    pivot_file.write("model,dataset,split,iteration,predicted_on,type,value\n")
    for model in models:
        for dataset in datasets:
            for split in splits:

                source_dataset = load_dataset(f"{dataset}.txt")

                label_indices = get_label_indices(source_dataset)

                numeric_labels = list(range(len(label_indices)))

                split_name = split if split != "kh" else f"kh-{model}"
                split_path = f"{dataset}.strat-0.15.{split_name}.splits"

                holdout_dataset = load_dataset(os.path.join(split_path, "holdout.txt"))

                schedule_dataset = load_dataset(os.path.join(split_path, "schedule.txt"))

                y_true = [label_indices[label] for label in holdout_dataset.values()]

                splitter = TopNSplitter(50)

                iteration = 0
                cumulative_corrections = 0
                _, remaining_dataset = splitter(schedule_dataset)
                while True:
                    holdout_predictions_filename = os.path.join(split_path, f"predictions.{model}.{iteration}.txt")

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

                    pivot_file.write(",".join(map(str, [model, dataset, split, iteration, "holdout", "accuracy", top_1])) + "\n")

                    update_dataset, remaining_dataset = splitter(remaining_dataset)

                    update_predictions_filename = os.path.join(split_path, f"update_predictions.{model}.{iteration}.txt")

                    if os.path.exists(update_predictions_filename):
                        update_y_true = [label_indices[label] for label in update_dataset.values()]
                        update_predictions = load_predictions(update_predictions_filename)
                        update_y_score = list(update_predictions.values())
                        update_top_1 = top_k_accuracy_score(update_y_true, update_y_score, k=1, labels=numeric_labels)
                        pivot_file.write(",".join(map(str, [model, dataset, split, iteration, "update", "accuracy", update_top_1])) + "\n")
                        cumulative_corrections += int((1 - update_top_1) * 50)
                        pivot_file.write(",".join(map(str, [model, dataset, split, iteration, "update", "cumulative_corrections", cumulative_corrections])) + "\n")

                    iteration += 1
