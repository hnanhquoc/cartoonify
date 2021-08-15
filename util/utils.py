# Processing functions using Tensorflow
import os
from glob import glob
from itertools import product

import cv2
import numpy as np
import tensorflow as tf

from util.contants import IMG_SIZE, DATA_DIR


def load(path):
    """
    Load, decode paths into images and cast them into float32
    :param path: path to the image
    :return: an image vector.
    """
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, 'float32')
    return image


def normalize(image):
    """
    Cast to float32 and Normalize the images to [-1, 1]
    :param image:
    :return:
    """
    image = (image / 127.5) - 1
    return image


def resize(image, height, width):
    """
    Resize the two image to height, width
    :param image:
    :param height:
    :param width:
    :return:
    """
    image = tf.image.resize(image,
                            [height, width],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image


def random_crop(image, height, width):
    """
    Crop image
    :param image:
    :param height:
    :param width:
    :return:
    """
    return tf.image.random_crop(image, size=[height, width, 3])


@tf.function
def image_processing(x, is_train=True):
    """
    Preprocess the image.
    :param x: target image.
    :param is_train: is training?
    :return: a processed image.
    """
    crop_size = IMG_SIZE
    if is_train:
        sizes = tf.cast(crop_size * tf.random.uniform([2], 0.9, 1.1), tf.int32)
        shape = tf.shape(x)[:2]
        sizes = tf.minimum(sizes, shape)
        # Adding random noise to the image.
        x = random_crop(x, sizes[0], sizes[1])
        x = tf.image.random_flip_left_right(x)
    x = resize(x, crop_size, crop_size)

    return normalize(x)


def get_dataset(dataset_name, domain, _type, batch_size):
    files = glob(os.path.join(DATA_DIR, dataset_name, f"{_type}{domain}", "*"))
    num_images = len(files)
    print(f"Found {num_images} domain{domain} images in {_type}{domain} folder.")

    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.shuffle(buffer_size=400).repeat(num_images)

    ds = ds.map(image_processing, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)
    steps = int(np.ceil(num_images / batch_size))
    # user iter(ds) to avoid generating iterator every epoch
    return iter(ds), steps


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
    # gc.collect()
    return out_arr


@tf.function
def gram(x):
    """
    Gram matrix.
    If the vectors v_1, v_2, ..., v_n are real and the columns of matrix X,
    then the Gram matrix is X^TX.
    :param x: an image.
    :return: a matrix.
    """
    shape_x = tf.shape(x)
    b = shape_x[0]
    c = shape_x[3]
    x = tf.reshape(x, [b, -1, c])
    return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast((tf.size(x) // b), tf.float32)

