import os
from collections import OrderedDict
from itertools import count
from random import Random
import sys

from sample import *

RANDOM = Random(42)

SOURCE_PATH, SOURCE_DATASET, SOURCE_EXT = split_arg(sys.argv[1])

DEST_PATH = os.path.join(SOURCE_PATH, "splits")
os.makedirs(DEST_PATH, exist_ok=True)

NUM_HOLDOUTS_PER_LABEL = 5
NUM_ITEMS_PER_LABEL_PER_ITERATION = 5
NUM_VALIDATION_ITEMS_PER_LABEL_PER_ITERATION = 2

source_dataset = load_dataset(os.path.join(SOURCE_PATH, SOURCE_DATASET + SOURCE_EXT))

print(f"NUM_ITEMS = {len(source_dataset)}")

NUM_LABELS = num_labels(source_dataset)

print(f"NUM_LABELS = {NUM_LABELS}")

HOLDOUT_SPLITTER: Splitter = StratifiedSplit(NUM_HOLDOUTS_PER_LABEL, RANDOM)
TRAIN_SPLITTER: Splitter = StratifiedSplit(NUM_ITEMS_PER_LABEL_PER_ITERATION, RANDOM)
VALIDATION_SPLITTER: Splitter = StratifiedSplit(NUM_VALIDATION_ITEMS_PER_LABEL_PER_ITERATION, RANDOM)

holdout_dataset, left_in_dataset = HOLDOUT_SPLITTER(source_dataset)

write_dataset(holdout_dataset, os.path.join(DEST_PATH, SOURCE_DATASET + f".holdout.{HOLDOUT_SPLITTER}" + SOURCE_EXT))

train_proper_dataset, validation_dataset = OrderedDict(), OrderedDict()

for iteration in count():
    train_addition_dataset, left_in_dataset = TRAIN_SPLITTER(left_in_dataset)
    validation_addition_dataset, train_proper_addition_dataset = VALIDATION_SPLITTER(train_addition_dataset)
    train_proper_dataset = merge(train_proper_dataset, train_proper_addition_dataset)
    validation_dataset = merge(validation_dataset, validation_addition_dataset)
    write_dataset(train_proper_dataset, os.path.join(DEST_PATH, SOURCE_DATASET + f".train.{TRAIN_SPLITTER}.{iteration}" + SOURCE_EXT))
    write_dataset(validation_dataset, os.path.join(DEST_PATH, SOURCE_DATASET + f".validation.{VALIDATION_SPLITTER}.{iteration}" + SOURCE_EXT))