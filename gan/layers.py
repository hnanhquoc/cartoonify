import tensorflow as tf
# def channel_shuffle_2(x):
#     dyn_shape = tf.shape(x)
#     h, w = dyn_shape[1], dyn_shape[2]
#     c = x.shape[3]
#     x = K.reshape(x, [-1, h, w, 2, c // 2])
#     x = K.permute_dimensions(x, [0, 1, 2, 4, 3])
#     x = K.reshape(x, [-1, h, w, c])
#     return x
from tensorflow import pad
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import ReLU, Dropout, Add, BatchNormalization, ZeroPadding2D


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


# def get_padding(pad_type, padding):
#     if pad_type == "reflect":
#         return ReflectionPadding2D(padding)
#     elif pad_type == "constant":
#         return ZeroPadding2D(padding)
#     else:
#         raise ValueError(f"Unrecognized pad_type {pad_type}")


# def get_norm(norm_type):
#     if norm_type == 'batch':
#         return BatchNormalization()
#     else:
#         raise ValueError(f"Unrecognized norm_type {norm_type}")


# class UpSampleConv(Model):
#     def __init__(self,
#                  filters,
#                  kernel_size,
#                  norm_type="batch",
#                  pad_type="constant"):
#         super(UpSampleConv, self).__init__(name="UpSampleConv")
#         self.model = ConvBlock(filters, kernel_size, 1, norm_type, pad_type)
#
#     def build(self, input_shape):
#         super(UpSampleConv, self).build(input_shape)
#
#     def call(self, x, training=False):
#         x = tf.keras.backend.resize_images(x, 2, 2, "channels_last", 'bilinear')
#         return self.model(x, training=training)


# class ConvBlock(Model):
#     def __init__(self,
#                  filters,
#                  kernel_size,
#                  stride=1,
#                  norm_type="batch",
#                  pad_type="constant"):
#         super(ConvBlock, self).__init__(name="ConvBlock")
#         padding = (kernel_size - 1) // 2
#         padding = (padding, padding)
#
#         self.model = tf.keras.models.Sequential()
#         self.model.add(get_padding(pad_type, padding))
#         self.model.add(Conv2D(filters, kernel_size, stride))
#         self.model.add(get_padding(pad_type, padding))
#         self.model.add(Conv2D(filters, kernel_size))
#         self.model.add(get_norm(norm_type))
#         self.model.add(ReLU())
#
#     def build(self, input_shape):
#         super(ConvBlock, self).build(input_shape)
#
#     def call(self, x, training=False):
#         return self.model(x, training=training)
#
#
# class FlatConv(Model):
#     def __init__(self,
#                  filters,
#                  kernel_size,
#                  norm_type="batch",
#                  pad_type="constant"):
#         super(FlatConv, self).__init__(name="FlatConv")
#         padding = (kernel_size - 1) // 2
#         padding = (padding, padding)
#         self.model = tf.keras.models.Sequential()
#         self.model.add(get_padding(pad_type, padding))
#         self.model.add(Conv2D(filters, kernel_size))
#         self.model.add(get_norm(norm_type))
#         self.model.add(ReLU())
#
#     def build(self, input_shape):
#         super(FlatConv, self).build(input_shape)
#
#     def call(self, x, training=False):
#         return self.model(x, training=training)
#
#
# class ResBlock(Model):
#     def __init__(self,
#                  filters,
#                  kernel_size,
#                  norm_type="batch",
#                  pad_type="constant"):
#         super(ResBlock, self).__init__(name="ResBlock")
#         padding = (kernel_size - 1) // 2
#         padding = (padding, padding)
#         self.model = tf.keras.models.Sequential()
#         self.model.add(get_padding(pad_type, padding))
#         self.model.add(Conv2D(filters, kernel_size))
#         self.model.add(get_norm(norm_type))
#         self.model.add(ReLU())
#         self.model.add(get_padding(pad_type, padding))
#         self.model.add(Conv2D(filters, kernel_size))
#         self.model.add(get_norm(norm_type))
#         self.add = Add()
#
#     def build(self, input_shape):
#         super(ResBlock, self).build(input_shape)
#
#     def call(self, x, training=False):
#         return self.add([self.model(x, training=training), x])
#
#
# class StridedConv(Model):
#     def __init__(self,
#                  filters=64,
#                  lrelu_alpha=0.2,
#                  pad_type="constant",
#                  norm_type="batch"):
#         super(StridedConv, self).__init__(name="StridedConv")
#
#         self.model = tf.keras.models.Sequential()
#         self.model.add(get_padding(pad_type, (1, 1)))
#         self.model.add(Conv2D(filters, 3, strides=(2, 2)))
#         self.model.add(LeakyReLU(lrelu_alpha))
#         self.model.add(get_padding(pad_type, (1, 1)))
#         self.model.add(Conv2D(filters * 2, 3))
#         self.model.add(get_norm(norm_type))
#         self.model.add(LeakyReLU(lrelu_alpha))
#
#     def build(self, input_shape):
#         super(StridedConv, self).build(input_shape)
#
#     def call(self, x, training=False):
#         return self.model(x, training=training)
from tensorflow.python.keras.layers import Conv2D, LeakyReLU


def base_block(filters, kernel_size,
               stride_1=1,
               stride_2=0,
               apply_norm=True,
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

    if apply_norm:
        result.add(BatchNormalization())

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
        BatchNormalization(),
        ReLU(),
        ZeroPadding2D(padding),
        Conv2D(filters, kernel_size, strides=1),
        BatchNormalization(),
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
