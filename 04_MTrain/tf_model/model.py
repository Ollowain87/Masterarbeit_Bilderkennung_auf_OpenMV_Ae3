#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    02-Oct-2026 15:54:58

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    input = keras.Input(shape=(48,48,3))
    conv_1 = layers.Conv2D(16, (3,3), padding="same", name="conv_1_")(input)
    relu_1 = layers.ReLU()(conv_1)
    maxpool_1 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(relu_1)
    conv_2 = layers.Conv2D(32, (3,3), padding="same", name="conv_2_")(maxpool_1)
    relu_2 = layers.ReLU()(conv_2)
    maxpool_2 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(relu_2)
    conv_3 = layers.Conv2D(64, (3,3), padding="same", name="conv_3_")(maxpool_2)
    relu_3 = layers.ReLU()(conv_3)
    conv_4 = layers.Conv2D(64, (3,3), padding="same", name="conv_4_")(relu_3)
    relu_4 = layers.ReLU()(conv_4)
    conv_out = layers.Conv2D(13, (1,1), name="conv_out_")(relu_4)
    gap = layers.GlobalAveragePooling2D(keepdims=True)(conv_out)
    softmax = layers.Softmax()(gap)

    model = keras.Model(inputs=[input], outputs=[softmax])
    return model
