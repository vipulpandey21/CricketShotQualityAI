"""
video_utils.py
Shared video processing utilities.
"""

import cv2
import numpy as np
import tensorflow as tf


def extract_frames(video_path: str, n_frames: int = 30, output_size: tuple = (224, 224)) -> np.ndarray:
    """
    Extract n_frames from a video, resize to output_size, return RGB numpy array.
    Shape: (n_frames, H, W, 3)
    """
    result = []
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = cap.read()
    if ret:
        frame = tf.image.resize_with_pad(
            tf.image.convert_image_dtype(frame, tf.uint8), *output_size
        ).numpy()
        result.append(frame)
    else:
        result.append(np.zeros((*output_size, 3), dtype=np.uint8))

    for _ in range(n_frames - 1):
        ret, frame = cap.read()
        if ret:
            frame = tf.image.resize_with_pad(
                tf.image.convert_image_dtype(frame, tf.uint8), *output_size
            ).numpy()
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))

    cap.release()
    return np.array(result)[..., [2, 1, 0]]  # BGR → RGB
