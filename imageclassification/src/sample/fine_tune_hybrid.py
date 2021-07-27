import os
import shutil
import sys
from collections import OrderedDict
from random import Random

from tensorflow import keras
from tensorflow import distribute
from wai.annotations.main import main as wai_annotations_main

from sample import *
from sample.splitters import RandomSplitter, TopNSplitter

INIT_LR = 1e-4
BS = 5
SEED = 42
VALIDATION_PERCENT = 0.15
LR = 0.02

RANDOM = Random(SEED)
SHUFFLE_RANDOM = Random(SEED)

SOURCE_PATH, SOURCE_DATASET, SOURCE_EXT = split_arg(sys.argv[1])
NUM_EPOCHS = int(sys.argv[2])
MODEL = sys.argv[3]
RELATIVE_DIR = os.path.join(SOURCE_PATH, sys.argv[4])
GPU = sys.argv[5]
IC_MODEL = sys.argv[6]

CWD = os.getcwd()

schedule_dataset = load_dataset("schedule.txt")

source_dataset = load_dataset(os.path.join(SOURCE_PATH, SOURCE_DATASET + SOURCE_EXT))
label_indices = label_indices(source_dataset)

PREDICTIONS_FILE_HEADER = predictions_file_header(label_indices)

MODEL_DIR = os.path.join(CWD, f"{MODEL}-{IC_MODEL}")

os.makedirs(MODEL_DIR)

with open(f"{MODEL_DIR}/setup.py", "w") as file:
    file.write(f"NUM_EPOCHS = {NUM_EPOCHS}\n")
    file.write(f"LR = {LR}\n")
    with open(os.path.join(CWD, '..', f'setup_{MODEL}_{SOURCE_DATASET}.py')) as source_file:
        file.writelines(source_file.readlines())

with open(f"{MODEL_DIR}/labels.txt", "w") as file:
    file.write(",".join(label_indices.keys()))

with open(f"{MODEL_DIR}/object_labels.txt", "w") as file:
    file.write("object")

# Crop all images in advance
write_dataset(change_path(source_dataset, RELATIVE_DIR), f"{MODEL_DIR}/dataset_voc.txt")
os.makedirs(f"{MODEL_DIR}/dataset_rois")
os.makedirs(f"{MODEL_DIR}/dataset_cropped")
wai_annotations_main([
    "convert",
    "from-voc-od",
    "-I",
    f"{MODEL_DIR}/dataset_voc.txt",
    "to-roi-od",
    "--annotations-only",
    "-o",
    f"{MODEL_DIR}/dataset_rois"
])
bboxes = get_highest_score_bbox(
    f"{MODEL_DIR}/dataset_rois",
    schedule_dataset
)
dest_path = f"{MODEL_DIR}/dataset_cropped"
for index, filename in enumerate(source_dataset, 1):
    print(f"Cropping image {filename} ({index} of {len(source_dataset)})")
    if filename in bboxes:
        crop_image(
            os.path.join(SOURCE_PATH, filename),
            dest_path,
            bboxes[filename]
        )
    else:
        shutil.copy(
            os.path.join(SOURCE_PATH, filename),
            dest_path
        )
    
holdout_dataset = load_dataset("holdout.txt")

splitter = TopNSplitter(50)

iteration = 0
iteration_dataset, remaining_dataset = splitter(schedule_dataset)
while True:
    # Debug
    print(f"ITERATION {iteration}")

    # Create a new directory for this iteration
    ITERATION_DIR = os.path.join(MODEL_DIR, str(iteration))
    os.makedirs(ITERATION_DIR)

    # Select a random subset for validation
    validation_size = max(int(len(iteration_dataset) * VALIDATION_PERCENT), 1)
    validation_dataset, train_dataset = RandomSplitter(validation_size, RANDOM)(iteration_dataset)

    # Randomly re-order the training dataset
    train_dataset = shuffle_dataset(train_dataset, SHUFFLE_RANDOM)

    # Train the boxer on the data, with only 'object' labels
    write_dataset(change_path(validation_dataset, RELATIVE_DIR), f"{ITERATION_DIR}/validation.txt")
    write_dataset(change_path(train_dataset, RELATIVE_DIR), f"{ITERATION_DIR}/train.txt")
    os.makedirs(f"{ITERATION_DIR}/val")
    os.makedirs(f"{ITERATION_DIR}/train")
    os.makedirs(f"{ITERATION_DIR}/output")
    wai_annotations_main([
        "convert",
        "from-voc-od",
        "-I",
        f"{ITERATION_DIR}/validation.txt",
        "map-labels",
        *(f"{label}=object" for label in label_indices.keys()),
        "to-coco-od",
        "-o",
        f"{ITERATION_DIR}/val/annotations.json",
        "--pretty",
        "--categories",
        "object"
    ])
    wai_annotations_main([
        "convert",
        "from-voc-od",
        "-I",
        f"{ITERATION_DIR}/train.txt",
        "map-labels",
        *(f"{label}=object" for label in label_indices.keys()),
        "to-coco-od",
        "-o",
        f"{ITERATION_DIR}/train/annotations.json",
        "--pretty",
        "--categories",
        "object"
    ])
    run_command(
        f"docker run "
        f"--gpus device={GPU} "
        f"--shm-size 8G "
        f"-u $(id -u):$(id -g) "
        f"-e USER=$USER "
        f"-e MMDET_CLASSES=\"'/labels.txt'\" "
        f"-e MMDET_OUTPUT=/data/output "
        f"-v {MODEL_DIR}/object_labels.txt:/labels.txt "
        f"-v {MODEL_DIR}/setup.py:/setup.py "
        f"-v {os.path.join(CWD, '..', f'base_{MODEL}.pth')}:/model.pth "
        f"-v {ITERATION_DIR}:/data "
        f"public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2020-03-01_cuda10 "
        f"mmdet_train /setup.py"
    )

    # Train the image classifier on the cropped images
    with distribute.MirroredStrategy([f"GPU:{GPU}"]).scope():
        model = model_for_fine_tuning(MODEL, len(label_indices), "imagenet")
        opt = keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        train_gen = data_flow_from_disk(SOURCE_PATH, change_path(train_dataset, f"{MODEL_DIR}/dataset_cropped"), label_indices, True, BS, SEED, MODEL)
        val_gen = data_flow_from_disk(SOURCE_PATH, change_path(validation_dataset, f"{MODEL_DIR}/dataset_cropped"), label_indices, False, BS, SEED, MODEL)
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=NUM_EPOCHS
        )

    write_dataset(change_path(holdout_dataset, RELATIVE_DIR), f"{ITERATION_DIR}/holdout.txt")

    os.makedirs(f"{ITERATION_DIR}/predictions")
    os.makedirs(f"{ITERATION_DIR}/predictions_in")

    wai_annotations_main([
        "convert",
        "from-voc-od",
        "-I",
        f"{ITERATION_DIR}/holdout.txt",
        "map-labels",
        *(f"{label}=object" for label in label_indices.keys()),
        "to-coco-od",
        "-o",
        f"{ITERATION_DIR}/predictions_in/annotations.json",
        "--pretty",
        "--categories",
        "object"
    ])

    run_command(
        f"docker run "
        f"--gpus device={GPU} "
        f"--shm-size 8G "
        f"-u $(id -u):$(id -g) "
        f"-e USER=$USER "
        f"-e MMDET_CLASSES=\"'/labels.txt'\" "
        f"-e MMDET_OUTPUT=/data/output "
        f"-v {MODEL_DIR}/object_labels.txt:/labels.txt "
        f"-v {MODEL_DIR}/setup.py:/setup.py "
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

    bboxes = get_highest_score_bbox(
        f"{ITERATION_DIR}/predictions",
        holdout_dataset
    )

    dest_path = f"{ITERATION_DIR}/holdout_cropped"
    os.makedirs(dest_path, exist_ok=True)
    for filename in holdout_dataset:
        if filename in bboxes:
            crop_image(
                os.path.join(SOURCE_PATH, filename),
                dest_path,
                bboxes[filename]
            )
        else:
            shutil.copy(
                os.path.join(SOURCE_PATH, filename),
                dest_path
            )

    with distribute.MirroredStrategy([f"GPU:{GPU}"]).scope():
        predict_gen = data_flow_from_disk(SOURCE_PATH, change_path(holdout_dataset, dest_path), label_indices, False, BS, SEED, MODEL)

        predictions: Predictions = OrderedDict()
        for holdout_item, prediction in zip(holdout_dataset.keys(), model.predict(predict_gen)):
            predictions[holdout_item] = prediction

        write_predictions(
            predictions,
            PREDICTIONS_FILE_HEADER,
            f"{ITERATION_DIR}/predictions.txt"
        )

    if len(remaining_dataset) == 0:
        break

    update_dataset, remaining_dataset = splitter(remaining_dataset)

    write_dataset(change_path(update_dataset, RELATIVE_DIR), f"{ITERATION_DIR}/update.txt")

    os.makedirs(f"{ITERATION_DIR}/update_predictions")
    os.makedirs(f"{ITERATION_DIR}/update_predictions_in")

    wai_annotations_main([
        "convert",
        "from-voc-od",
        "-I",
        f"{ITERATION_DIR}/update.txt",
        "map-labels",
        *(f"{label}=object" for label in label_indices.keys()),
        "to-coco-od",
        "-o",
        f"{ITERATION_DIR}/update_predictions_in/annotations.json",
        "--pretty",
        "--categories",
        "object"
    ])

    run_command(
        f"docker run "
        f"--gpus device={GPU} "
        f"--shm-size 8G "
        f"-u $(id -u):$(id -g) "
        f"-e USER=$USER "
        f"-e MMDET_CLASSES=\"'/labels.txt'\" "
        f"-e MMDET_OUTPUT=/data/output "
        f"-v {MODEL_DIR}/object_labels.txt:/labels.txt "
        f"-v {MODEL_DIR}/setup.py:/setup.py "
        f"-v {os.path.join(CWD, '..', f'base_{MODEL}.pth')}:/model.pth "
        f"-v {ITERATION_DIR}:/data "
        f"public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2020-03-01_cuda10 "
        f"mmdet_predict "
        f"--checkpoint /data/output/latest.pth "
        f"--config /setup.py "
        f"--prediction_in /data/update_predictions_in/ "
        f"--prediction_out /data/update_predictions/ "
        f"--labels /labels.txt "
        f"--score 0 "
        f"--delete_input"
    )

    bboxes = get_highest_score_bbox(
        f"{ITERATION_DIR}/update_predictions",
        update_dataset
    )

    dest_path = f"{ITERATION_DIR}/update_cropped"
    os.makedirs(dest_path, exist_ok=True)
    for filename in update_dataset:
        if filename in bboxes:
            crop_image(
                os.path.join(SOURCE_PATH, filename),
                dest_path,
                bboxes[filename]
            )
        else:
            shutil.copy(
                os.path.join(SOURCE_PATH, filename),
                dest_path
            )

    with distribute.MirroredStrategy([f"GPU:{GPU}"]).scope():
        predict_gen = data_flow_from_disk(SOURCE_PATH, change_path(update_dataset, dest_path), label_indices, False, BS, SEED, MODEL)

        predictions: Predictions = OrderedDict()
        for holdout_item, prediction in zip(holdout_dataset.keys(), model.predict(predict_gen)):
            predictions[holdout_item] = prediction

        write_predictions(
            predictions,
            PREDICTIONS_FILE_HEADER,
            f"{ITERATION_DIR}/update_predictions.txt"
        )

    # Clean up
    rm_dir(f"{ITERATION_DIR}/output")
    rm_dir(f"{ITERATION_DIR}/predictions_in")
    rm_dir(f"{ITERATION_DIR}/update_predictions_in")
    rm_dir(f"{ITERATION_DIR}/train")
    rm_dir(f"{ITERATION_DIR}/val")
    os.remove(f"{ITERATION_DIR}/validation.txt")
    os.remove(f"{ITERATION_DIR}/train.txt")
    os.remove(f"{ITERATION_DIR}/update.txt")
    os.remove(f"{ITERATION_DIR}/holdout.txt")

    iteration_dataset = merge(iteration_dataset, update_dataset)

    iteration += 1
