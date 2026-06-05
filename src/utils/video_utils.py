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


def extract_raw_frames(video_path: str, max_frames: int = 60) -> list:
    """
    Extract up to max_frames raw BGR frames at full resolution.
    Used by MediaPipe — it needs the original image size to detect joints.

    Returns:
        List of BGR numpy arrays (variable H×W×3).
    """
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Sample evenly across the whole video so we don't miss the actual shot
    step = max(1, total // max_frames)
    idx = 0
    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        idx += step
    cap.release()
    return frames
