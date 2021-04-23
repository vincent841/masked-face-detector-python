
# import tensorflow as tf
# from tensorflow.python.framework import graph_io
# from tensorflow.keras.models import load_model

# import tensorflow.python.keras.backend as kb

# # Clear any previous session.
# tf.keras.backend.clear_session()

# save_pb_dir = './'
# model_fname = './mask_detector.model'


# def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
#     with graph.as_default():
#         graphdef_inf = tf.graph_util.remove_training_nodes(
#             graph.as_graph_def())
#         graphdef_frozen = tf.graph_util.convert_variables_to_constants(
#             session, graphdef_inf, output)
#         graph_io.write_graph(graphdef_frozen, save_pb_dir,
#                              save_pb_name, as_text=save_pb_as_text)
#         return graphdef_frozen


# # This line must be executed before loading Keras model.
# tf.keras.backend.set_learning_phase(0)

# model = load_model(model_fname)

# session = kb.get_session()

# input_names = [t.op.name for t in model.inputs]
# output_names = [t.op.name for t in model.outputs]

# # Prints input and output nodes names, take notes of them.
# print(input_names, output_names)

# frozen_graph = freeze_graph(session.graph, session, [
#                             out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)


# import logging
# import tensorflow as tf
# from tensorflow.compat.v1 import graph_util
# from tensorflow.python.keras import backend as K
# from tensorflow import keras

# # necessary !!!
# tf.compat.v1.disable_eager_execution()

# h5_path = './mask_detector.model'
# model = keras.models.load_model(h5_path)
# model.summary()
# # save pb
# with K.get_session() as sess:
#     output_names = [out.op.name for out in model.outputs]
#     input_graph_def = sess.graph.as_graph_def()
#     for node in input_graph_def.node:
#         node.device = ""
#     graph = graph_util.remove_training_nodes(input_graph_def)
#     graph_frozen = graph_util.convert_variables_to_constants(
#         sess, graph, output_names)
#     tf.io.write_graph(graph_frozen, './model.pb', as_text=False)
# logging.info("save pb successfully！")


import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np

h5_path = './mask_detector.model'
model = keras.models.load_model(h5_path)
model.summary()

full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="mask_detector.pb",
                  as_text=False)
