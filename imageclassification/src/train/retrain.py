# Based on code from "Fine-tune InceptionV3 on a new set of classes"
# https://keras.io/applications/
# (C) Copyright Keras
# (C) Copyright 2019 University of Waikato, Hamilton, NZ

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
import argparse
import os
import matplotlib.pyplot as plt


def retrain(img_dir, img_width, img_height, batch_size, num_epochs, validation_split, base_model, output_dir, verbose):
    """
    Retrains a pre-trained model (ie the base model) on new data and saves the new model.

    :param img_dir: the directory with the images (on sub-dir per class)
    :type img_dir: str
    :param img_width: the image width to use
    :type img_width: int
    :param img_height: the image height to use
    :type img_height: int
    :param batch_size: the batch size to use
    :type batch_size: int
    :param num_epochs: the number of epochs to use for training
    :type num_epochs: int
    :param validation_split: the percentage (0-1) to use for validation
    :type validation_split: float
    :param base_model: the model class to use
    :type base_model: str
    :param output_dir: the file to store the final model to
    :type output_dir: str
    :param verbose: whether to be verbose
    :type verbose: bool
    """

    if not os.path.exists(img_dir):
        raise IOError("Image directory does not exist: %s" % img_dir)

    # determine number of classes (ie sub-dirs)
    sub_dirs = [f.path for f in os.scandir(img_dir) if f.is_dir()]
    num_classes = len(sub_dirs)
    if verbose:
        print("%d sub-directories: %s" %(num_classes, str(sub_dirs)))

    # set up generators
    train_datagen = ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split)

    train_generator = train_datagen.flow_from_directory(
        img_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        img_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

    # create the base pre-trained model
    model_module = base_model[:base_model.rindex(".")]
    model_class = base_model[base_model.rindex(".") + 1:]
    _temp = __import__(model_module, globals(), locals(), [model_class], 0)
    base_model_class = getattr(_temp, model_class)
    base_model = base_model_class(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # train the model on the new data for a few epochs
    #model.fit_generator(...)
    history = model.fit_generator(
        train_generator,
        steps_per_epoch = train_generator.samples // batch_size,
        validation_data = validation_generator,
        validation_steps = validation_generator.samples // batch_size,
        epochs = num_epochs)

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    # if verbose:
    #     for i, layer in enumerate(base_model.layers):
    #        print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    # TODO number of layers??
    # for layer in model.layers[:249]:
    #    layer.trainable = False
    # for layer in model.layers[249:]:
    #    layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

    # we train our model again
    # alongside the top Dense layers
    #model.fit_generator(...)
    # history = model.fit_generator(
    #     train_generator,
    #     steps_per_epoch = train_generator.samples // batch_size,
    #     validation_data = validation_generator,
    #     validation_steps = validation_generator.samples // batch_size,
    #     epochs = num_epochs)

    # model
    model.save(output_dir + "/model.hdf5", overwrite=True)

    # loss visualization
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']
    epoch_count = range(1, len(training_loss) + 1)
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(output_dir + "/" + "loss.png")
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='', help='Path to folders of images.')
    parser.add_argument('--num_epochs', type=int, default=50, help='The number of epochs to perform.')
    parser.add_argument('--validation_split', type=float, default=0.2, help='What percentage of images to use as a validation set.')
    parser.add_argument('--batch_size', type=int, default=16, help='How many images to train on at a time.')
    parser.add_argument('--image_width', type=int, default=200, help='The image width to use.')
    parser.add_argument('--image_height', type=int, default=200, help='The image height to use.')
    parser.add_argument('--base_model', type=str, default="keras.applications.inception_v3.InceptionV3", help='The model class to use.')
    parser.add_argument('--output_dir', type=str, default='/tmp', help='Where to save the trained model and statistics.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Whether to print out some debugging information.')
    params = parser.parse_args()

    retrain(img_dir=params.image_dir,
            img_width=params.image_width, img_height=params.image_height,
            batch_size=params.batch_size, num_epochs=params.num_epochs,
            validation_split=params.validation_split,
            base_model=params.base_model, output_dir=params.output_dir,
            verbose=params.verbose)
