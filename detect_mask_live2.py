# USAGE
# python detect_mask_image.py --image examples/example_01.png

# import the necessary packages
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import os
import time

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

# create opencv
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
cap = cv2.VideoCapture(0)

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image")
# ap.add_argument("-f", "--face", type=str,
#                 default="resources",
#                 help="path to face detector model directory")
# ap.add_argument("-m", "--model", type=str,
#                 default="mask_detector.model",
#                 help="path to trained face mask detector model")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
#                 help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())

# load our serialized face detector model from disk

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model("mask_detector.model")

while (True):
    ret, frame = cap.read()

    try:

        start = time.time()

        orig = frame.copy()
        (h, w) = frame.shape[: 2]

        # extract the face ROI, convert it from BGR to RGB channel
        # ordering, resize it to 224x224, and preprocess it
        face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # pass the face through the model to determine if the face
        # has a mask or not
        stime = time.time()
        (mask, withoutMask) = model.predict(face)[0]
        print('elapsed_time: ', time.time() - stime)

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(
            label, max(mask, withoutMask) * 100)

        print(label)
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # print('mask: ', mask)
        # print('withoutMask: ', withoutMask)

    except Exception as ex:
        print("Error :", ex)

    # display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# # load the input image from disk, clone it, and grab the image spatial
# # dimensions
# image = cv2.imread(args["image"])
# orig = image.copy()
# (h, w) = image.shape[:2]

# # construct a blob from the image
# blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
#                              (104.0, 177.0, 123.0))

# # pass the blob through the network and obtain the face detections
# print("[INFO] computing face detections...")
# net.setInput(blob)
# detections = net.forward()

# # loop over the detections
# for i in range(0, detections.shape[2]):
#     # extract the confidence (i.e., probability) associated with
#     # the detection
#     confidence = detections[0, 0, i, 2]

#     # filter out weak detections by ensuring the confidence is
#     # greater than the minimum confidence
#     if confidence > args["confidence"]:
#         # compute the (x, y)-coordinates of the bounding box for
#         # the object
#         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#         (startX, startY, endX, endY) = box.astype("int")

#         # ensure the bounding boxes fall within the dimensions of
#         # the frame
#         (startX, startY) = (max(0, startX), max(0, startY))
#         (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

#         # extract the face ROI, convert it from BGR to RGB channel
#         # ordering, resize it to 224x224, and preprocess it
#         face = image[startY:endY, startX:endX]
#         face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#         face = cv2.resize(face, (224, 224))
#         face = img_to_array(face)
#         face = preprocess_input(face)
#         face = np.expand_dims(face, axis=0)

#         # pass the face through the model to determine if the face
#         # has a mask or not
#         (mask, withoutMask) = model.predict(face)[0]

#         # determine the class label and color we'll use to draw
#         # the bounding box and text
#         label = "Mask" if mask > withoutMask else "No Mask"
#         color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

#         # include the probability in the label
#         label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

#         # display the label and bounding box rectangle on the output
#         # frame
#         cv2.putText(image, label, (startX, startY - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
#         cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# # show the output image
# cv2.imshow("Output", image)
# cv2.waitKey(0)
