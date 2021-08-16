import argparse
import gc
import os
from glob import glob

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from gan.discriminator import discriminator
from gan.generator import generator
from util.contants import EPOCHS, CONTENT_LAMBDA, STYLE_LAMBDA, CHECKPOINT_DIR, PRETRAIN_PREFIX, GENERATOR_NAME, \
    DISCRIMINATOR_NAME, LOG_DIR, PRETRAIN_EPOCHS, SOURCE_DOMAIN, BATCH_SIZE, IMG_SIZE, PRETRAIN_LEARNING_RATE, \
    RESULT_DIR, DATA_DIR, SAMPLE_SIZE, PRETRAIN_REPORTING_STEPS, PRETRAIN_SAVING_EPOCHS, G_ADV_LAMBDA, GENERATOR_LR, \
    DISCRIMINATOR_LR, TARGET_DOMAIN, REPORTING_STEPS, MODEL_DIR
from util.logger import get_logger
from util.utils import load_vgg19, get_dataset, save_image, image_processing, gram


class Trainer:
    def __init__(
            self,
            debug,
            dataset_name,
            epochs=EPOCHS,
    ):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ["SM_FRAMEWORK"] = "tf.keras"
        self.ascii = os.name == "nt"
        self.debug = debug
        self.dataset_name = dataset_name
        self.epochs = epochs

        self.logger = get_logger("Trainer", debug=debug)
        self.vgg = load_vgg19()

        self.logger.info(f"Setting up objective functions and metrics using lsgan...")
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.generator_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.discriminator_loss_object = tf.keras.losses.MeanSquaredError()

        self.g_total_loss_metric = tf.keras.metrics.Mean("g_total_loss", dtype=tf.float32)
        self.g_adv_loss_metric = tf.keras.metrics.Mean("g_adversarial_loss", dtype=tf.float32)
        self.d_total_loss_metric = tf.keras.metrics.Mean("d_total_loss", dtype=tf.float32)
        self.d_real_loss_metric = tf.keras.metrics.Mean("d_real_loss", dtype=tf.float32)
        self.d_fake_loss_metric = tf.keras.metrics.Mean("d_fake_loss", dtype=tf.float32)
        self.d_smooth_loss_metric = tf.keras.metrics.Mean("d_smooth_loss", dtype=tf.float32)

        self.metric_and_names = [
            (self.g_total_loss_metric, "g_total_loss"),
            (self.g_adv_loss_metric, "g_adversarial_loss"),
            (self.d_total_loss_metric, "d_total_loss"),
            (self.d_real_loss_metric, "d_real_loss"),
            (self.d_fake_loss_metric, "d_fake_loss"),
            (self.d_smooth_loss_metric, "d_smooth_loss"),
        ]
        if CONTENT_LAMBDA != 0.:
            self.content_loss_metric = tf.keras.metrics.Mean("content_loss", dtype=tf.float32)
            self.metric_and_names.append((self.content_loss_metric, "content_loss"))
        if STYLE_LAMBDA != 0.:
            self.style_loss_metric = tf.keras.metrics.Mean("style_loss", dtype=tf.float32)
            self.metric_and_names.append((self.style_loss_metric, "style_loss"))

        self.logger.info("Setting up generator, discriminator, and optimizers...")
        self.g = generator(base_filters=2 if self.debug else 64)
        self.d = discriminator(base_filters=2 if self.debug else 32)
        self.pre_optimizer = tf.keras.optimizers.Adam(learning_rate=PRETRAIN_LEARNING_RATE, beta_1=0.5)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=GENERATOR_LR, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=DISCRIMINATOR_LR, beta_1=0.5)

        self.logger.info("Setting up checkpoint paths...")
        self.pretrain_checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "pretrain", PRETRAIN_PREFIX)
        self.generator_checkpoint_dir = os.path.join(CHECKPOINT_DIR, GENERATOR_NAME)
        self.generator_checkpoint_prefix = os.path.join(self.generator_checkpoint_dir, GENERATOR_NAME)
        self.discriminator_checkpoint_dir = os.path.join(CHECKPOINT_DIR, DISCRIMINATOR_NAME)
        self.discriminator_checkpoint_prefix = os.path.join(self.discriminator_checkpoint_dir, DISCRIMINATOR_NAME)

    @tf.function
    def pass_to_vgg(self, tensor):
        if self.vgg is not None:
            tensor = self.vgg(tensor)
        return tensor

    @tf.function
    def content_loss(self, input_images, generated_images):
        return self.mae(input_images, generated_images)

    @tf.function
    def style_loss(self, input_images, generated_images):
        input_images = gram(input_images)
        generated_images = gram(generated_images)
        return self.mae(input_images, generated_images)

    @tf.function
    def generator_adversarial_loss(self, fake_output):
        return self.generator_loss_object(tf.ones_like(fake_output), fake_output)

    @tf.function
    def discriminator_loss(self, real_output, fake_output, smooth_output):
        real_loss = self.discriminator_loss_object(tf.ones_like(real_output), real_output)
        fake_loss = self.discriminator_loss_object(tf.zeros_like(fake_output), fake_output)
        smooth_loss = self.discriminator_loss_object(tf.zeros_like(smooth_output), smooth_output)
        total_loss = real_loss + fake_loss + smooth_loss
        return real_loss, fake_loss, smooth_loss, total_loss

    @tf.function
    def pretrain_step(self, input_images, g, optimizer):

        with tf.GradientTape() as tape:
            generated_images = g(input_images, training=True)
            c_loss = CONTENT_LAMBDA * self.content_loss(self.pass_to_vgg(input_images),
                                                        self.pass_to_vgg(generated_images))

        gradients = tape.gradient(c_loss, g.trainable_variables)
        optimizer.apply_gradients(zip(gradients, g.trainable_variables))

        self.content_loss_metric(c_loss)

    @tf.function
    def train_step(self, source_images, target_images, smooth_images,
                   g, d, g_optimizer, d_optimizer):
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            generated_images = g(source_images)

            real_output = d(target_images)
            fake_output = d(generated_images)
            smooth_out = d(smooth_images)
            d_real_loss, d_fake_loss, d_smooth_loss, d_total_loss = \
                self.discriminator_loss(real_output, fake_output, smooth_out)

            g_adv_loss = G_ADV_LAMBDA * self.generator_adversarial_loss(fake_output)

            g_total_loss = g_adv_loss
            if CONTENT_LAMBDA != 0. or STYLE_LAMBDA != 0.:
                vgg_generated_images = self.pass_to_vgg(generated_images)
                if CONTENT_LAMBDA != 0.:
                    c_loss = CONTENT_LAMBDA * self.content_loss(
                        self.pass_to_vgg(source_images), vgg_generated_images)
                    g_total_loss = g_total_loss + c_loss
                if STYLE_LAMBDA != 0.:
                    s_loss = STYLE_LAMBDA * self.style_loss(
                        self.pass_to_vgg(target_images[:vgg_generated_images.shape[0]]),
                        vgg_generated_images)
                    g_total_loss = g_total_loss + s_loss

        d_grads = d_tape.gradient(d_total_loss, d.trainable_variables)
        g_grads = g_tape.gradient(g_total_loss, g.trainable_variables)

        d_optimizer.apply_gradients(zip(d_grads, d.trainable_variables))
        g_optimizer.apply_gradients(zip(g_grads, g.trainable_variables))

        self.g_total_loss_metric(g_total_loss)
        self.g_adv_loss_metric(g_adv_loss)
        if CONTENT_LAMBDA != 0.:
            self.content_loss_metric(c_loss)
        if STYLE_LAMBDA != 0.:
            self.style_loss_metric(s_loss)
        self.d_total_loss_metric(d_total_loss)
        self.d_real_loss_metric(d_real_loss)
        self.d_fake_loss_metric(d_fake_loss)
        self.d_smooth_loss_metric(d_smooth_loss)

    def pretrain_generator(self):
        summary_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, "pretrain"))
        self.logger.info(f"Starting to pretrain generator with {PRETRAIN_EPOCHS} epochs...")
        self.logger.info(
            f"Building `{self.dataset_name}` dataset with domain `{SOURCE_DOMAIN}`..."
        )
        dataset, steps_per_epoch = get_dataset(dataset_name=self.dataset_name,
                                               domain=SOURCE_DOMAIN,
                                               _type="train",
                                               batch_size=BATCH_SIZE)
        self.logger.info(f"Initializing generator with batch_size: {BATCH_SIZE}, input_size: {IMG_SIZE}...")
        self.g.summary()

        self.logger.info(f"Try restoring checkpoint: `{self.pretrain_checkpoint_prefix}`...")
        try:
            checkpoint = tf.train.Checkpoint(generator=self.g)
            status = checkpoint.restore(tf.train.latest_checkpoint(os.path.join(CHECKPOINT_DIR, "pretrain")))
            status.assert_consumed()

            self.logger.info(f"Previous checkpoints has been restored.")
            trained_epochs = checkpoint.save_counter.numpy()
            epochs = PRETRAIN_EPOCHS - trained_epochs
            if epochs <= 0:
                self.logger.info(f"Already trained {trained_epochs} epochs. "
                                 "Set a larger `pretrain_epochs`...")
                return
            else:
                self.logger.info(f"Already trained {trained_epochs} epochs, "
                                 f"{epochs} epochs left to be trained...")
        except AssertionError:
            self.logger.info(f"Checkpoint is not found, "
                             f"training from scratch with {PRETRAIN_EPOCHS} epochs...")
            trained_epochs = 0
            epochs = PRETRAIN_EPOCHS

        val_files = glob(os.path.join(DATA_DIR, self.dataset_name, f"test{SOURCE_DOMAIN}", "*"))
        val_real_batch = tf.map_fn(
            lambda fname: image_processing(fname, False),
            tf.constant(val_files), tf.float32, back_prop=False)
        real_batch = next(dataset)
        while real_batch.shape[0] < SAMPLE_SIZE:
            real_batch = tf.concat((real_batch, next(dataset)), 0)
        real_batch = real_batch[:SAMPLE_SIZE]
        with summary_writer.as_default():
            img = np.expand_dims(save_image(RESULT_DIR,
                                            tf.cast((real_batch + 1) * 127.5, tf.uint8),
                                            image_name="pretrain_sample_images.png"), 0, )
            tf.summary.image("pretrain_sample_images", img, step=0)
            img = np.expand_dims(save_image(RESULT_DIR,
                                            tf.cast((val_real_batch + 1) * 127.5, tf.uint8),
                                            image_name="pretrain_val_sample_images.png"), 0, )
            tf.summary.image("pretrain_val_sample_images", img, step=0)
        gc.collect()

        self.logger.info("Starting pre-training loop, "
                         "setting up summary writer to record progress on TensorBoard...")

        for epoch in range(epochs):
            epoch_idx = trained_epochs + epoch + 1

            for step in tqdm(
                    range(1, steps_per_epoch + 1),
                    desc=f"Pretrain Epoch {epoch + 1}/{epochs}"):
                # NOTE: not following official "for img in dataset" example
                #       since it generates new iterator every epoch and can
                #       hardly be garbage-collected by python
                image_batch = dataset.next()
                self.pretrain_step(image_batch, self.g, self.pre_optimizer)

                if step % PRETRAIN_REPORTING_STEPS == 0:
                    global_step = (epoch_idx - 1) * steps_per_epoch + step
                    with summary_writer.as_default():
                        tf.summary.scalar('content_loss',
                                          self.content_loss_metric.result(),
                                          step=global_step)
                        fake_batch = tf.cast((self.g(real_batch) + 1) * 127.5, tf.uint8)
                        img = np.expand_dims(
                            save_image(
                                RESULT_DIR,
                                fake_batch,
                                image_name=(f"pretrain_generated_images_at_epoch_{epoch_idx}"
                                            f"_step_{step}.png")), 0, )
                        tf.summary.image('pretrain_generated_images', img, step=global_step)
                    self.content_loss_metric.reset_states()
            with summary_writer.as_default():
                val_fake_batch = tf.cast((self.g(val_real_batch) + 1) * 127.5, tf.uint8)
                img = np.expand_dims(
                    save_image(
                        RESULT_DIR,
                        val_fake_batch,
                        image_name=("pretrain_val_generated_images_at_epoch_"
                                    f"{epoch_idx}_step_{step}.png")), 0, )
                tf.summary.image('pretrain_val_generated_images', img, step=epoch)

            if epoch % PRETRAIN_SAVING_EPOCHS == 0:
                self.logger.info(f"Saving checkpoints after epoch {epoch_idx} ended...")
                checkpoint.save(file_prefix=self.pretrain_checkpoint_prefix)
            gc.collect()
        del dataset
        gc.collect()

    def train_gan(self):
        self.logger.info("Setting up summary writer to record progress on TensorBoard...")
        summary_writer = tf.summary.create_file_writer(LOG_DIR)
        self.logger.info(
            f"Starting adversarial training with {self.epochs} epochs, "
            f"batch size: {BATCH_SIZE}..."
        )
        self.logger.info(f"Building `{self.dataset_name}` "
                         "datasets for source/target/smooth domains...")
        ds_source, steps_per_epoch = get_dataset(dataset_name=self.dataset_name,
                                                 domain=SOURCE_DOMAIN,
                                                 _type="train",
                                                 batch_size=BATCH_SIZE)
        ds_target, _ = get_dataset(dataset_name=self.dataset_name,
                                   domain=TARGET_DOMAIN,
                                   _type="train",
                                   batch_size=BATCH_SIZE)
        ds_smooth, _ = get_dataset(dataset_name=self.dataset_name,
                                   domain=f"{TARGET_DOMAIN}_smooth",
                                   _type="train",
                                   batch_size=BATCH_SIZE)

        self.logger.info(f"Searching existing checkpoints: `{self.generator_checkpoint_prefix}`...")
        try:
            g_checkpoint = tf.train.Checkpoint(generator=self.g)
            g_checkpoint.restore(
                tf.train.latest_checkpoint(
                    self.generator_checkpoint_dir)).assert_existing_objects_matched()
            self.logger.info(f"Previous checkpoints has been restored.")
            trained_epochs = g_checkpoint.save_counter.numpy()
            epochs = self.epochs - trained_epochs
            if epochs <= 0:
                self.logger.info(f"Already trained {trained_epochs} epochs. "
                                 "Set a larger `epochs`...")
                return
            else:
                self.logger.info(f"Already trained {trained_epochs} epochs, "
                                 f"{epochs} epochs left to be trained...")
        except AssertionError as e:
            self.logger.warning(e)
            self.logger.warning("Previous checkpoints are not found, trying to load checkpoints from pretraining...")

            try:
                g_checkpoint = tf.train.Checkpoint(generator=self.g)
                g_checkpoint.restore(tf.train.latest_checkpoint(
                    os.path.join(CHECKPOINT_DIR, "pretrain"))).assert_existing_objects_matched()
                self.logger.info("Successfully loaded `{self.pretrain_checkpoint_prefix}`...")
            except AssertionError:
                self.logger.warning("specified pretrained checkpoint is not found, training from scratch...")

            trained_epochs = 0
            epochs = self.epochs

        self.logger.info("Searching existing checkpoints: "
                         f"`{self.discriminator_checkpoint_prefix}`...")
        try:
            d_checkpoint = tf.train.Checkpoint(d=self.d)
            d_checkpoint.restore(
                tf.train.latest_checkpoint(self.discriminator_checkpoint_dir)).assert_existing_objects_matched()
            self.logger.info(f"Previous checkpoints has been restored.")
        except AssertionError:
            self.logger.info("specified checkpoint is not found, training from scratch...")

        val_files = glob(os.path.join(DATA_DIR, self.dataset_name, f"test{SOURCE_DOMAIN}", "*"))
        val_real_batch = tf.map_fn(
            lambda fname: image_processing(fname, False),
            tf.constant(val_files), tf.float32, back_prop=False)
        real_batch = next(ds_source)
        while real_batch.shape[0] < SAMPLE_SIZE:
            real_batch = tf.concat((real_batch, next(ds_source)), 0)
        real_batch = real_batch[:SAMPLE_SIZE]
        with summary_writer.as_default():
            img = np.expand_dims(save_image(RESULT_DIR,
                                            tf.cast((real_batch + 1) * 127.5, tf.uint8),
                                            image_name="gan_sample_images.png"), 0, )
            tf.summary.image("gan_sample_images", img, step=0)
            img = np.expand_dims(save_image(RESULT_DIR,
                                            tf.cast((val_real_batch + 1) * 127.5, tf.uint8),
                                            image_name="gan_val_sample_images.png"), 0, )
            tf.summary.image("gan_val_sample_images", img, step=0)
        gc.collect()

        self.logger.info("Starting training loop...")

        self.logger.info(f"Number of trained epochs: {trained_epochs}, "
                         f"epochs to be trained: {epochs}, "
                         f"batch size: {BATCH_SIZE}")
        for epoch in range(epochs):
            epoch_idx = trained_epochs + epoch + 1

            for step in tqdm(
                    range(1, steps_per_epoch + 1),
                    desc=f'Train {epoch + 1}/{epochs}',
                    total=steps_per_epoch):
                source_images, target_images, smooth_images = (
                    ds_source.next(), ds_target.next(), ds_smooth.next())
                self.train_step(source_images, target_images, smooth_images,
                                self.g, self.d, self.g_optimizer, self.d_optimizer)

                if step % REPORTING_STEPS == 0:

                    global_step = (epoch_idx - 1) * steps_per_epoch + step
                    with summary_writer.as_default():
                        for metric, name in self.metric_and_names:
                            tf.summary.scalar(name, metric.result(), step=global_step)
                            metric.reset_states()
                        fake_batch = tf.cast((self.g(real_batch) + 1) * 127.5, tf.uint8)
                        img = np.expand_dims(save_image(RESULT_DIR,
                                                        fake_batch,
                                                        image_name=("gan_generated_images_at_epoch_"
                                                                    f"{epoch_idx}_step_{step}.png")), 0, )
                        tf.summary.image('gan_generated_images', img, step=global_step)

                    self.logger.debug(f"Epoch {epoch_idx}, Step {step} finished, "
                                      f"{global_step * BATCH_SIZE} images processed.")

            with summary_writer.as_default():
                val_fake_batch = tf.cast(
                    (self.g(val_real_batch) + 1) * 127.5, tf.uint8)
                img = np.expand_dims(
                    save_image(RESULT_DIR,
                               val_fake_batch,
                               image_name=("gan_val_generated_images_at_epoch_"
                                           f"{epoch_idx}_step_{step}.png")), 0, )
                tf.summary.image('gan_val_generated_images', img, step=epoch)
            self.logger.info(f"Saving checkpoints after epoch {epoch_idx} ended...")
            g_checkpoint.save(file_prefix=self.generator_checkpoint_prefix)
            d_checkpoint.save(file_prefix=self.discriminator_checkpoint_prefix)

            self.g.save_weights(os.path.join(MODEL_DIR, "generator"))
            gc.collect()
        del ds_source, ds_target, ds_smooth
        gc.collect()


def main(**kwargs):
    mode = kwargs["mode"]
    debug = kwargs["debug"]
    dataset_name = kwargs["dataset_name"]
    t = Trainer(debug, dataset_name)
    if mode == "full":
        t.pretrain_generator()
        gc.collect()
        t.train_gan()
    elif mode == "pretrain":
        t.pretrain_generator()
    elif mode == "gan":
        t.train_gan()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "pretrain", "gan"])
    parser.add_argument("--dataset_name", type=str, default="animeGAN")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--not_show_progress_bar", action="store_true")

    args = parser.parse_args()

    args.show_progress = not args.not_show_progress_bar

    kwargs = vars(args)
    main(**kwargs)
