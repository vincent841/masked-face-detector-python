
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from config import Config
from model import MaskedFaceDetector

# to figure out gpu memory issue
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

class TrainConfig(Config):
    # initialize the initial learning rate, number of epochs to train for,
    # and batch size
    # learning rate
    INIT_LR = 1e-4

    # max # of epochs
    EPOCHS = 20

    # batch size
    BS = 32

class MaskedFaceTrain(object):

    def __init__(self, dataset_dir):
        # grab the list of images in our dataset directory, then initialize
        # the list of data (i.e., images) and class images
        print("[INFO] loading images...")
        self.imagePaths = list(paths.list_images(args["dataset"]))
        data = []
        labels = []   

    def prepare(self):
        # loop over the image paths
        for imagePath in imagePaths:
            # extract the class label from the filename
            label = imagePath.split(os.path.sep)[-2]

            # load the input image (224x224) and preprocess it
            image = load_img(imagePath, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)

            # update the data and labels lists, respectively
            data.append(image)
            labels.append(label)

            # convert the data and labels to NumPy arrays
            data = np.array(data, dtype="float32")
            labels = np.array(labels)

            # perform one-hot encoding on the labels
            lb = LabelBinarizer()
            labels = lb.fit_transform(labels)
            labels = to_categorical(labels)

            # partition the data into training and testing splits using 75% of
            # the data for training and the remaining 25% for testing
            (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                            test_size=0.20, stratify=labels, random_state=42)

            # construct the training image generator for data augmentation
            aug = ImageDataGenerator(
                rotation_range=20,
                zoom_range=0.15,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                horizontal_flip=True,
                fill_mode="nearest")

    def compile(self):
        # compile our model
        print("[INFO] compiling model...")
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        model.compile(loss="binary_crossentropy", optimizer=opt,
                    metrics=["accuracy"])

    def train(self):
        self.compile()

        # train the head of the network
        print("[INFO] training head...")
        H = model.fit(
            aug.flow(trainX, trainY, batch_size=BS),
            steps_per_epoch=len(trainX) // BS,
            validation_data=(testX, testY),
            validation_steps=len(testX) // BS,
            epochs=EPOCHS)

    def evaluate(self):
        # make predictions on the testing set
        print("[INFO] evaluating network...")
        predIdxs = model.predict(testX, batch_size=BS)

        # for each image in the testing set we need to find the index of the
        # label with corresponding largest predicted probability
        predIdxs = np.argmax(predIdxs, axis=1)

        # show a nicely formatted classification report
        print(classification_report(testY.argmax(axis=1), predIdxs,
                                    target_names=lb.classes_))

        # serialize the model to disk
        print("[INFO] saving mask detector model...")
        model.save(args["model"], save_format="h5")

        # plot the training loss and accuracy
        N = EPOCHS
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(args["plot"])

    def save_model(self):
        # serialize the model to disk
        print("[INFO] saving mask detector model...")
        model.save(args["model"], save_format="h5")




if __name__ == '__main__':
    model = MaskedFaceTrain('./dataset')
    model.evaluate()









