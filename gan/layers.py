import tensorflow as tf
from tensorflow import pad
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import ReLU, Dropout, Add, ZeroPadding2D
from tensorflow.python.keras.layers import Conv2D, LeakyReLU, BatchNormalization
from tensorflow_addons.layers import InstanceNormalization


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return pad(input_tensor, [[0, 0], [padding_height, padding_height], [padding_width, padding_width], [0, 0]],
                   'REFLECT')


def get_norm(norm_type):
    if norm_type == "instance":
        return InstanceNormalization()
    elif norm_type == 'batch':
        return BatchNormalization()
    else:
        raise ValueError(f"Unrecognized norm_type {norm_type}")


def base_block(filters, kernel_size,
               stride_1=1,
               stride_2=0,
               norm="instance",
               dropout=0,
               apply_relu=True,
               leaky_relu_alpha=0):
    padding = (kernel_size - 1) // 2
    padding = (padding, padding)

    result = tf.keras.Sequential([
        ZeroPadding2D(padding),
        Conv2D(filters, kernel_size, strides=stride_1),
    ])

    if stride_2 > 0:
        result.add(ZeroPadding2D(padding))
        result.add(Conv2D(filters, kernel_size, strides=stride_2))

    if norm is not None:
        result.add(get_norm(norm))

    if dropout > 0:
        result.add(Dropout(dropout))

    if leaky_relu_alpha > 0:
        result.add(LeakyReLU(leaky_relu_alpha))
    elif apply_relu:
        result.add(ReLU())

    return result


def residual_block(filters, kernel_size):
    padding = (kernel_size - 1) // 2
    padding = (padding, padding)
    result = tf.keras.Sequential([
        ZeroPadding2D(padding),
        Conv2D(filters, kernel_size, strides=1),
        InstanceNormalization(),
        ReLU(),
        ZeroPadding2D(padding),
        Conv2D(filters, kernel_size, strides=1),
        InstanceNormalization(),
    ])

    result.add = Add()

    return result


def stride_block(filters, kernel_size,
                 stride_1=2,
                 stride_2=1,
                 leaky_relu_alpha=0):
    return tf.keras.Sequential([
        ReflectionPadding2D(),
        Conv2D(filters, kernel_size, strides=stride_1),
        LeakyReLU(leaky_relu_alpha),
        ReflectionPadding2D(),
        Conv2D(filters * 2, kernel_size, strides=stride_2),
        BatchNormalization(),
        LeakyReLU(leaky_relu_alpha),
    ])
