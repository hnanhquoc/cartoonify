from gan.discriminator import discriminator
from gan.generator import generator

import tensorflow as tf
import matplotlib.pyplot as plt

from util.contants import IMG_SIZE
from util.utils import load, image_processing, resize

if __name__ == '__main__':
    g = generator()
    g.summary()

    d = discriminator()
    d.summary()

    # Load one sample
    draw_sample = load("dataset/animeGAN/trainA/6.jpg")
    # processed_sample = resize(draw_sample, IMG_SIZE, IMG_SIZE)
    processed_sample = image_processing("dataset/animeGAN/trainA/6.jpg")

    gen_output = g(processed_sample[tf.newaxis, ...])
    disc_out = d([processed_sample[tf.newaxis, ...], gen_output])

    plt.figure(figsize=(10, 8))
    plt.subplot(231)
    plt.imshow(draw_sample / 255.0)
    plt.subplot(232)
    plt.imshow(gen_output[0, ...])
    plt.subplot(233)
    plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
    plt.colorbar()
    # plt.imshow(original_sample / 255.0)
    plt.show()
    # tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
