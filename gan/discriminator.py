import tensorflow as tf
from tensorflow.keras.layers import Conv2D

from gan.layers import base_block, stride_block
from util.contants import IMG_SIZE, BATCH_SIZE


def discriminator(base_filters=32, lrelu_alpha=0.2, img_size=IMG_SIZE):
    inp = tf.keras.layers.Input(shape=[img_size, img_size, 3], name='input_image', batch_size=BATCH_SIZE)

    architect = [
        base_block(filters=base_filters, kernel_size=3, norm=None, leaky_relu_alpha=lrelu_alpha),
        stride_block(filters=base_filters * 2, kernel_size=3, leaky_relu_alpha=lrelu_alpha),
        stride_block(filters=base_filters * 4, kernel_size=3, leaky_relu_alpha=lrelu_alpha),
        base_block(filters=base_filters * 4, kernel_size=3, leaky_relu_alpha=lrelu_alpha),
        Conv2D(1, 3)
    ]

    x = inp
    for layer in architect:
        x = layer(x)

    return tf.keras.Model(inputs=inp, outputs=x)
