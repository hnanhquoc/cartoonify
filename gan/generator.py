# The generator network  𝐺  begins with a flat convolution stage followed by two down-convolution blocks
# to spatially compress and encode the images.
#
# Afterwards, eight residual blocks with identical layout are used to construct the content and manifold feature.
#
# Finally, the output cartoon style images are reconstructed by two up-convolution blocks.
import tensorflow as tf
from keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Conv2D

from gan.layers import base_block, residual_block
from util.contants import IMG_SIZE


def generator(base_filters=64):
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])
    end_kernel_size = 7
    end_padding = (end_kernel_size - 1) // 2
    end_padding = (end_padding, end_padding)

    architect = [
        # Flat Convolution
        base_block(filters=base_filters, kernel_size=end_kernel_size),
        # Down Convolution
        base_block(filters=base_filters*2, kernel_size=3, stride_1=2, stride_2=1),
        base_block(filters=base_filters*4, kernel_size=3, stride_1=2, stride_2=1),
        # Residual Blocks
        residual_block(filters=base_filters*4, kernel_size=3),
        residual_block(filters=base_filters*4, kernel_size=3),
        residual_block(filters=base_filters*4, kernel_size=3),
        residual_block(filters=base_filters*4, kernel_size=3),
        residual_block(filters=base_filters*4, kernel_size=3),
        residual_block(filters=base_filters*4, kernel_size=3),
        residual_block(filters=base_filters*4, kernel_size=3),
        residual_block(filters=base_filters*4, kernel_size=3),
        # Up Convolution
        base_block(filters=base_filters * 2, kernel_size=3, stride_1=0.5, stride_2=1),
        base_block(filters=base_filters, kernel_size=3, stride_1=0.5, stride_2=1),
        # Final Convolution
        ZeroPadding2D(end_padding),
        Conv2D(filters=3, kernel_size=end_kernel_size, activation="tanh")
    ]

    x = inputs
    for layer in architect:
        layer.summary()
        x = layer(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
