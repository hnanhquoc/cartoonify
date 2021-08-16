# The generator network  𝐺  begins with a flat convolution stage followed by two down-convolution blocks
# to spatially compress and encode the images.
#
# Afterwards, eight residual blocks with identical layout are used to construct the content and manifold feature.
#
# Finally, the output cartoon style images are reconstructed by two up-convolution blocks.
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.python.keras.losses import MeanAbsoluteError, BinaryCrossentropy

from gan.layers import base_block, residual_block
from util.contants import IMG_SIZE, BATCH_SIZE
from util.utils import gram


def generator(base_filters=64, input_size=IMG_SIZE, batch_size=BATCH_SIZE):
    inputs = tf.keras.layers.Input(shape=[input_size, input_size, 3], batch_size=batch_size)
    end_kernel_size = 7
    end_padding = (end_kernel_size - 1) // 2
    end_padding = (end_padding, end_padding)

    down_stack = [
        # Flat Convolution
        base_block(filters=base_filters, kernel_size=end_kernel_size),
        # Down Convolution
        base_block(filters=base_filters * 2, kernel_size=3, stride_1=2, stride_2=1),
        base_block(filters=base_filters * 4, kernel_size=3, stride_1=2, stride_2=1),
        # Residual Blocks
        residual_block(filters=base_filters * 4, kernel_size=3),
        residual_block(filters=base_filters * 4, kernel_size=3),
        residual_block(filters=base_filters * 4, kernel_size=3),
        residual_block(filters=base_filters * 4, kernel_size=3),
        residual_block(filters=base_filters * 4, kernel_size=3),
        residual_block(filters=base_filters * 4, kernel_size=3),
        residual_block(filters=base_filters * 4, kernel_size=3),
        residual_block(filters=base_filters * 4, kernel_size=3),
    ]

    x = inputs
    for down in down_stack:
        x = down(x)

    up_stack = [
        # Up Convolution
        base_block(filters=base_filters * 2, kernel_size=3, stride_2=1),
        base_block(filters=base_filters, kernel_size=3, stride_2=1),
    ]
    for up in up_stack:
        x = tf.keras.backend.resize_images(x, 2, 2, "channels_last", 'bilinear')
        x = up(x)

    last = Conv2D(filters=3, kernel_size=end_kernel_size, padding="same", activation="tanh")

    return tf.keras.Model(inputs=inputs, outputs=last(x))

