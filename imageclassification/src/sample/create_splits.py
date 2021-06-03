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

DEST_PATH = os.path.join(SOURCE_PATH, "splits")
os.makedirs(DEST_PATH, exist_ok=True)

print(f"DEST PATH = {DEST_PATH}")

NUM_HOLDOUTS_PER_LABEL = 5
NUM_ITEMS_PER_LABEL_PER_ITERATION = 5
NUM_VALIDATION_ITEMS_PER_LABEL_PER_ITERATION = 2

print(f"HOLDOUTS PER LABEL = {NUM_HOLDOUTS_PER_LABEL}")
print(f"DATASET ITEMS PER LABEL = {NUM_ITEMS_PER_LABEL_PER_ITERATION}")
print(f"VALIDATION ITEMS PER LABEL = {NUM_VALIDATION_ITEMS_PER_LABEL_PER_ITERATION}")

source_dataset = load_dataset(os.path.join(SOURCE_PATH, SOURCE_DATASET + SOURCE_EXT))

print(f"NUM ITEMS = {len(source_dataset)}")

NUM_LABELS = num_labels(source_dataset)

print(f"NUM LABELS = {NUM_LABELS}")

HOLDOUT_SPLITTER: Splitter = StratifiedSplit(NUM_HOLDOUTS_PER_LABEL, RANDOM)
TRAIN_SPLITTER: Splitter = KernelHerdingSplit(SOURCE_PATH, NUM_ITEMS_PER_LABEL_PER_ITERATION * NUM_LABELS)
VALIDATION_SPLITTER: Splitter = KernelHerdingSplit(SOURCE_PATH, NUM_VALIDATION_ITEMS_PER_LABEL_PER_ITERATION * NUM_LABELS)

print(f"HOLDOUT SPLITTER = {HOLDOUT_SPLITTER}")
print(f"TRAIN SPLITTER = {TRAIN_SPLITTER}")
print(f"VALIDATION SPLITTER = {VALIDATION_SPLITTER}")

holdout_dataset, left_in_dataset = HOLDOUT_SPLITTER(source_dataset)
holdout_dataset_dest = os.path.join(DEST_PATH, SOURCE_DATASET + f".holdout.{HOLDOUT_SPLITTER}" + SOURCE_EXT)
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

    train_proper_dataset_dest = os.path.join(DEST_PATH, SOURCE_DATASET + f".train.{TRAIN_SPLITTER}.{iteration}" + SOURCE_EXT)
    write_dataset(train_proper_dataset, train_proper_dataset_dest)
    print(f"WROTE TRAIN DATASET FOR ITERATION {iteration} TO {train_proper_dataset_dest}")

    validation_dataset_dest = os.path.join(DEST_PATH, SOURCE_DATASET + f".validation.{VALIDATION_SPLITTER}.{iteration}" + SOURCE_EXT)
    write_dataset(validation_dataset, validation_dataset_dest)
    print(f"WROTE VALIDATION DATASET FOR ITERATION {iteration} TO {validation_dataset_dest}")
