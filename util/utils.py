# Processing functions using Tensorflow
import gc
import os
from glob import glob
from itertools import product
from random import choice

import cv2
import numpy
import tensorflow as tf


def load(path):
    """Load, decode paths into images and cast them into float32
    """
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, 'float32')
    return image


def normalize(image):
    """ Cast to float32 and
        Normalize the images to [-1, 1]
    """
    image = (image / 127.5) - 1
    return image


def resize(image, height, width):
    """ Resize the two image to heigh, width
    """
    image = tf.image.resize(image,
                            [height, width],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image


def random_crop(image, height, width):
    """ Crop image
    """
    cropped_image = tf.image.random_crop(image, size=[height, width, 3])

    return cropped_image[0], cropped_image[1]


@tf.function()
def random_jitter(draw, height, width, crop_height, crop_width):
    draw = resize(draw, height, width)
    draw, original = random_crop(draw, crop_height, crop_width)

    # Augmentation to random flip
    if tf.random.uniform(()) > 0.5:
        draw = tf.image.flip_left_right(draw)
        original = tf.image.flip_left_right(original)

    return draw


def _save_generated_images(result_dir, batch_x, image_name, nrow=2, ncol=4):
    # NOTE: 0 <= batch_x <= 1, float32, numpy.ndarray
    if not isinstance(batch_x, np.ndarray):
        batch_x = batch_x.numpy()
    n, h, w, c = batch_x.shape
    out_arr = np.zeros([h * nrow, w * ncol, 3], dtype=np.uint8)
    for (i, j), k in zip(product(range(nrow), range(ncol)), range(n)):
        out_arr[(h * i):(h * (i + 1)), (w * j):(w * (j + 1))] = batch_x[k]
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    cv2.imwrite(os.path.join(result_dir, image_name), out_arr)
    gc.collect()
    return out_arr


@tf.function
def gram(x):
    shape_x = tf.shape(x)
    b = shape_x[0]
    c = shape_x[3]
    x = tf.reshape(x, [b, -1, c])
    return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast((tf.size(x) // b), tf.float32)


@tf.function
def random_resize(image):
    size = choice([IMG_SIZE - 32, IMG_SIZE, IMG_SIZE + 32])
    return resize(image, (size, size))


@tf.function
def image_processing(self, filename, is_train=True):
    crop_size = self.input_size
    if self.multi_scale and is_train:
        crop_size += 32
    x = tf.io.read_file(filename)
    x = tf.image.decode_jpeg(x, channels=3)
    if is_train:
        sizes = tf.cast(
            crop_size * tf.random.uniform([2], 0.9, 1.1), tf.int32)
        shape = tf.shape(x)[:2]
        sizes = tf.minimum(sizes, shape)
        x = tf.image.random_crop(x, (sizes[0], sizes[1], 3))
        x = tf.image.random_flip_left_right(x)
    x = tf.image.resize(x, (crop_size, crop_size))
    img = tf.cast(x, tf.float32) / 127.5 - 1
    return img

def get_dataset(self, dataset_name, domain, _type, batch_size):
    files = glob(os.path.join(self.data_dir, dataset_name, f"{_type}{domain}", "*"))
    num_images = len(files)
    self.logger.info(
        f"Found {num_images} domain{domain} images in {_type}{domain} folder."
    )
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.shuffle(buffer_size=400).repeat(num_images)

    def fn(fname):
        if self.multi_scale:
            return self.random_resize(self.image_processing(fname, True))
        else:
            return self.image_processing(fname, True)

    ds = ds.map(fn).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    steps = int(np.ceil(num_images / batch_size))
    # user iter(ds) to avoid generating iterator every epoch
    return iter(ds), steps
