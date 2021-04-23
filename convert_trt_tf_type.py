# USAGE
# python detect_mask_image.py --image examples/example_01.png

# import the necessary packages
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np
import argparse
import cv2
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
        from tensorflow.python.compiler.tensorrt import trt_convert as trt


# prepare conveter parameters
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(
    max_workspace_size_bytes=(1 << 32))
conversion_params = conversion_params._replace(precision_mode="FP16")
conversion_params = conversion_params._replace(
    maximum_cached_engines=100)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='./saved_model/1',
    conversion_params=conversion_params)

converter.convert()

converter.save('./converted_model/1')
