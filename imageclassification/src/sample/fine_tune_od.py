import os
import sys
from random import Random

from wai.annotations.main import main as wai_annotations_main

from sample import *
from sample.splitters import RandomSplitter, TopNSplitter

INIT_LR = 1e-4
BS = 5
NUM_EPOCHS = int(sys.argv[2])
SEED = 42
VALIDATION_PERCENT = 0.15
GPU = sys.argv[5]


RANDOM = Random(SEED)
SHUFFLE_RANDOM = Random(SEED)

SOURCE_PATH, SOURCE_DATASET, SOURCE_EXT = split_arg(sys.argv[1])

RELATIVE_DIR = os.path.join(SOURCE_PATH, sys.argv[4])

MODEL = sys.argv[3]

CWD = os.getcwd()

source_dataset = load_dataset("schedule.txt")

label_indices = label_indices(load_dataset(os.path.join(SOURCE_PATH, SOURCE_DATASET + SOURCE_EXT)))

os.makedirs(MODEL)

MODEL_DIR = os.path.join(CWD, MODEL)

with open(f"{MODEL_DIR}/labels.txt", "w") as file:
    file.write(",".join(label_indices.keys()))

holdout_dataset = load_dataset("holdout.txt")

splitter = TopNSplitter(50)

iteration = 0
iteration_dataset, remaining_dataset = splitter(source_dataset)
while True:
    print(f"ITERATION {iteration}")

    ITERATION_DIR = os.path.join(MODEL_DIR, str(iteration))

    os.makedirs(ITERATION_DIR)

    validation_size = max(int(len(iteration_dataset) * VALIDATION_PERCENT), 1)

    validation_dataset, train_dataset = RandomSplitter(validation_size, RANDOM)(iteration_dataset)

    train_dataset = shuffle_dataset(train_dataset, SHUFFLE_RANDOM)

    write_dataset(change_path(validation_dataset, RELATIVE_DIR), f"{ITERATION_DIR}/validation.txt")
    write_dataset(change_path(train_dataset, RELATIVE_DIR), f"{ITERATION_DIR}/train.txt")

    os.makedirs(f"{ITERATION_DIR}/validation")
    os.makedirs(f"{ITERATION_DIR}/train")
    os.makedirs(f"{ITERATION_DIR}/output")

    wai_annotations_main([
        "convert",
        "from-voc-od",
        "-I",
        f"{ITERATION_DIR}/validation.txt",
        "to-coco-od",
        "-o",
        f"{ITERATION_DIR}/validation/annotations.json",
        "--pretty",
        "--categories",
        *label_indices.keys()
    ])

    wai_annotations_main([
        "convert",
        "from-voc-od",
        "-I",
        f"{ITERATION_DIR}/train.txt",
        "to-coco-od",
        "-o",
        f"{ITERATION_DIR}/train/annotations.json",
        "--pretty",
        "--categories",
        *label_indices.keys()
    ])

    os.system(
        f"docker run "
        f"--gpus device={GPU} "
        f"--shm-size 8G "
        f"-e USER=$USER "
        f"-e MMDET_CLASSES=\"'/labels.txt'\" "
        f"-e MMDET_OUTPUT=/data/output "
        f"-v {MODEL_DIR}/labels.txt:/labels.txt "
        f"-v {os.path.join(CWD, '..', f'setup_{MODEL}.py')}:/setup.py "
        f"-v {os.path.join(CWD, '..', f'base_{MODEL}.pth')}:/model.pth "
        f"-v {ITERATION_DIR}:/data "
        f"public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2020-03-01_cuda10 "
        f"mmdet_train /setup.py"
    )

    write_dataset(change_path(holdout_dataset, RELATIVE_DIR), f"{ITERATION_DIR}/holdout.txt")

    os.makedirs(f"{ITERATION_DIR}/predictions")
    os.makedirs(f"{ITERATION_DIR}/predictions_in")

    wai_annotations_main([
        "convert",
        "from-voc-od",
        "-I",
        f"{ITERATION_DIR}/holdout.txt",
        "to-coco-od",
        "-o",
        f"{ITERATION_DIR}/predictions_in/annotations.json",
        "--pretty",
        "--categories",
        *label_indices.keys()
    ])

    os.system(
        f"docker run "
        f"--gpus device={GPU} "
        f"--shm-size 8G "
        f"-e USER=$USER "
        f"-e MMDET_CLASSES=\"'/labels.txt'\" "
        f"-e MMDET_OUTPUT=/data/output "
        f"-v {MODEL_DIR}/labels.txt:/labels.txt "
        f"-v {os.path.join(CWD, '..', f'setup_{MODEL}.py')}:/setup.py "
        f"-v {os.path.join(CWD, '..', f'base_{MODEL}.pth')}:/model.pth "
        f"-v {ITERATION_DIR}:/data "
        f"public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2020-03-01_cuda10 "
        f"mmdet_predict "
        f"--checkpoint /data/output/latest.pth "
        f"--config /setup.py "
        f"--prediction_in /data/predictions_in/ "
        f"--prediction_out /data/predictions/ "
        f"--labels /labels.txt "
        f"--score 0 "
        f"--delete_input"
    )

    #model = model_for_fine_tuning(MODEL, len(label_indices))
    #opt = keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
    #model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    #train_gen = data_flow_from_disk(SOURCE_PATH, train_dataset, label_indices, True, BS, SEED, MODEL)
    #val_gen = data_flow_from_disk(SOURCE_PATH, validation_dataset, label_indices, False, BS, SEED, MODEL)

    #model.fit(
    #    train_gen,
    #    validation_data=val_gen,
    #    epochs=NUM_EPOCHS
    #)

    #predictions: Predictions = OrderedDict()
    #for holdout_item, prediction in zip(holdout_dataset.keys(), model.predict(holdout_gen)):
    #    predictions[holdout_item] = prediction

    #write_predictions(predictions, PREDICTIONS_FILE_HEADER, f"predictions.{MODEL}.{iteration}.txt")

    #os.makedirs("data")
    #os.makedirs("output")
    #os.makedirs("predictions")

    #os.remove("train.txt")
    #os.remove("validation.txt")
    #rm_dir("data")

    break

    if len(remaining_dataset) == 0:
        break

    update_dataset, remaining_dataset = splitter(remaining_dataset)

    #update_gen = data_flow_from_disk(SOURCE_PATH, update_dataset, label_indices, False, BS, SEED, MODEL)

    #predictions: Predictions = OrderedDict()
    #for update_item, prediction in zip(update_dataset.keys(), model.predict(update_gen)):
    #    predictions[update_item] = prediction

    #write_predictions(predictions, PREDICTIONS_FILE_HEADER, f"update_predictions.{MODEL}.{iteration}.txt")

    iteration_dataset = merge(iteration_dataset, update_dataset)

    iteration += 1
