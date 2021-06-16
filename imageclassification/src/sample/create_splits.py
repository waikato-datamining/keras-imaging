import os
from collections import OrderedDict
from itertools import count
from random import Random
import sys

from sample import *

RANDOM = Random(42)

SOURCE_PATH, SOURCE_DATASET, SOURCE_EXT = split_arg(sys.argv[1])

print(f"SOURCE PATH = {SOURCE_PATH}")
print(f"SOURCE DATASET = {SOURCE_DATASET}{SOURCE_EXT}")

NUM_HOLDOUTS_PER_LABEL = 5
NUM_ITEMS_PER_LABEL_PER_ITERATION = 5
NUM_VALIDATION_ITEMS_PER_LABEL_PER_ITERATION = 2

print(f"HOLDOUTS PER LABEL = {NUM_HOLDOUTS_PER_LABEL}")
print(f"DATASET ITEMS PER LABEL = {NUM_ITEMS_PER_LABEL_PER_ITERATION}")
print(f"VALIDATION ITEMS PER LABEL = {NUM_VALIDATION_ITEMS_PER_LABEL_PER_ITERATION}")

source_dataset = load_dataset(os.path.join(SOURCE_PATH, SOURCE_DATASET + SOURCE_EXT))

print(f"NUM ITEMS = {len(source_dataset)}")
for label, label_dataset in per_label(source_dataset).items():
    print(f"  {label}: {len(label_dataset)}")

LABELS = set(source_dataset.values())
NUM_LABELS = len(LABELS)

print(f"NUM LABELS = {NUM_LABELS}")

HOLDOUT_SPLITTER: Splitter = StratifiedSplit(NUM_HOLDOUTS_PER_LABEL, LABELS, RANDOM)
TRAIN_SPLITTER: Splitter = StratifiedSplit(NUM_ITEMS_PER_LABEL_PER_ITERATION, LABELS, RANDOM)
VALIDATION_SPLITTER: Splitter = StratifiedSplit(NUM_VALIDATION_ITEMS_PER_LABEL_PER_ITERATION, LABELS, RANDOM)

print(f"HOLDOUT SPLITTER = {HOLDOUT_SPLITTER}")
print(f"TRAIN SPLITTER = {TRAIN_SPLITTER}")
print(f"VALIDATION SPLITTER = {VALIDATION_SPLITTER}")

DEST_PATH = os.path.join(SOURCE_PATH, f"{SOURCE_DATASET}.{HOLDOUT_SPLITTER}.{TRAIN_SPLITTER}.{VALIDATION_SPLITTER}.splits")
os.makedirs(DEST_PATH, exist_ok=True)

print(f"DEST PATH = {DEST_PATH}")

holdout_dataset, left_in_dataset = HOLDOUT_SPLITTER(source_dataset)
holdout_dataset_dest = os.path.join(DEST_PATH, "holdout" + SOURCE_EXT)
write_dataset(holdout_dataset, holdout_dataset_dest)
print(f"WROTE HOLDOUT DATASET TO {holdout_dataset_dest}")

train_proper_dataset, validation_dataset = OrderedDict(), OrderedDict()

for iteration in count():
    print(f"ITERATION {iteration}")

    train_addition_dataset, left_in_dataset = TRAIN_SPLITTER(left_in_dataset)
    print("SELECTED TRAIN DATASET")

    validation_addition_dataset, train_proper_addition_dataset = VALIDATION_SPLITTER(train_addition_dataset)
    print("SELECTED VALIDATION DATASET")

    train_proper_dataset = merge(train_proper_dataset, train_proper_addition_dataset)
    validation_dataset = merge(validation_dataset, validation_addition_dataset)

    train_proper_dataset_dest = os.path.join(DEST_PATH, f"train.{iteration}" + SOURCE_EXT)
    write_dataset(train_proper_dataset, train_proper_dataset_dest)
    print(f"WROTE TRAIN DATASET FOR ITERATION {iteration} TO {train_proper_dataset_dest}")

    validation_dataset_dest = os.path.join(DEST_PATH, f"validation.{iteration}" + SOURCE_EXT)
    write_dataset(validation_dataset, validation_dataset_dest)
    print(f"WROTE VALIDATION DATASET FOR ITERATION {iteration} TO {validation_dataset_dest}")
