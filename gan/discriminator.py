import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.python.keras.losses import MeanSquaredError

from gan.layers import base_block, stride_block
from util.contants import IMG_SIZE


def discriminator(base_filters=32, lrelu_alpha=0.2, img_size=IMG_SIZE):
    inp = tf.keras.layers.Input(shape=[img_size, img_size, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[img_size, img_size, 3], name='target_image')

    architect = [
        base_block(filters=base_filters, kernel_size=3, apply_norm=False, leaky_relu_alpha=lrelu_alpha),
        stride_block(filters=base_filters * 2, kernel_size=3, leaky_relu_alpha=lrelu_alpha),
        stride_block(filters=base_filters * 4, kernel_size=3, leaky_relu_alpha=lrelu_alpha),
        base_block(filters=base_filters * 4, kernel_size=3, leaky_relu_alpha=lrelu_alpha),
        Conv2D(1, 3)
    ]

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)
    for layer in architect:
        x = layer(x)

    return tf.keras.Model(inputs=[inp, tar], outputs=x)
