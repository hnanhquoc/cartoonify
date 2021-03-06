"""
Minimum inference code
"""
import cv2
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from imageio import imwrite

from util.logger import get_logger

# NOTE: TF warnings are too noisy without this
from util.utils import load, resize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(40)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def main(m_path, img_path, out_dir):
    logger = get_logger("inference")
    logger.info(f"generating image from {img_path}")
    imported = tf.saved_model.load(m_path)
    f = imported.signatures["serving_default"]
    img = np.array(Image.open(img_path).convert("RGB"))
    img = np.expand_dims(img, 0).astype(np.float32) / 127.5 - 1
    out = f(tf.constant(img))['conv2d_25']
    out = ((out.numpy().squeeze() + 1) * 127.5).astype(np.uint8)
    if out_dir != "" and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if out_dir == "":
        out_dir = "util"
    out_path = os.path.join(out_dir, os.path.split(img_path)[1])
    imwrite(out_path, out)
    logger.info(f"generated image saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--m_path", type=str,
                        default=os.path.join("exported_models", "SavedModel_0000"))
    parser.add_argument("--img_path", type=str,
                        default=os.path.join("input_images", "2808.jpg"))
    parser.add_argument("--out_dir", type=str, default='out')
    args = parser.parse_args()
    main(args.m_path, args.img_path, args.out_dir)
