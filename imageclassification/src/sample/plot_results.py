import itertools
import sys
from collections import OrderedDict

from matplotlib import pyplot

SOURCE = sys.argv[1]
TARGET = sys.argv[2]

results = OrderedDict()

with open(SOURCE, "r") as source_file:
    source_file.readline()
    for line in source_file.readlines():
        model, dataset, split, iteration, predicted_on, type, value = line.strip().split(",")
        prediction = f"{predicted_on}_{type}"
        iteration = int(iteration)
        value = float(value)
        if "." in model:
            if split != "uni":
                continue
            model, split = model.split(".", 1)
            split = f"active-{split}"
        elif split == "uni":
            continue

        if dataset not in results:
            dataset_dict = OrderedDict()
            results[dataset] = dataset_dict
        else:
            dataset_dict = results[dataset]

        if model not in dataset_dict:
            model_dict = OrderedDict()
            dataset_dict[model] = model_dict
        else:
            model_dict = dataset_dict[model]

        if split not in model_dict:
            split_dict = OrderedDict()
            model_dict[split] = split_dict
        else:
            split_dict = model_dict[split]

        if prediction not in split_dict:
            prediction_list = []
            split_dict[prediction] = prediction_list
        else:
            prediction_list = split_dict[prediction]

        while len(prediction_list) <= iteration:
            prediction_list.append(0.0)

        prediction_list[iteration] = value

Y_LABEL = TARGET[TARGET.index("_") + 1:]

fig, axes = pyplot.subplots(3, 3, sharey="row", sharex="row", constrained_layout=True, subplot_kw={"xmargin": 0, "ymargin": 0})
fig.suptitle(TARGET)
fig.supylabel("Dataset")
fig.supxlabel("Model")
colour_cycle: itertools.cycle = pyplot.rcParams['axes.prop_cycle']()
split_colours = {}
axes_iter = axes.flat
lines = []
for dataset, dataset_dict in results.items():
    for model, model_dict in dataset_dict.items():
        axis = next(axes_iter)
        axis.set_title(f"{model} - {dataset}")
        axis.set_xlabel('Iteration')
        axis.set_ylabel(Y_LABEL)
        for split, split_dict in model_dict.items():
            if split not in split_colours:
                split_colours[split] = next(colour_cycle)['color']
            prediction_list = split_dict[TARGET]
            lines.append(axis.plot(prediction_list, label=split, color=split_colours[split]))
        axis.set_ybound(lower=0.0)

fig.legend(*axis.get_legend_handles_labels())
pyplot.show()
