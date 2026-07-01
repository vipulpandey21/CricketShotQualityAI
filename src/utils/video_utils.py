"""
video_utils.py
Shared video processing utilities.
"""

import cv2
import numpy as np
import tensorflow as tf


def extract_frames(video_path: str, n_frames: int = 30, output_size: tuple = (224, 224)) -> np.ndarray:
    """
    Extract n_frames from the middle 60% of a video (where the actual shot happens),
    resize to output_size, return RGB numpy array. Shape: (n_frames, H, W, 3)
    """
    result = []
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample from middle 60% — skip intro and outro
    start = int(total * 0.20)
    end   = int(total * 0.80)
    usable = max(end - start, 1)
    step  = max(1, usable // n_frames)

    for i in range(n_frames):
        idx = start + i * step
        if idx >= end:
            idx = end - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = tf.image.resize_with_pad(
                tf.image.convert_image_dtype(frame, tf.uint8), *output_size
            ).numpy()
            result.append(frame)
        else:
            if result:
                result.append(np.zeros_like(result[0]))
            else:
                result.append(np.zeros((*output_size, 3), dtype=np.uint8))

    cap.release()
    return np.array(result)[..., [2, 1, 0]]  # BGR → RGB


def extract_raw_frames(video_path: str, max_frames: int = 60) -> list:
    """
    Extract up to max_frames raw BGR frames from middle 60% of video at full resolution.
    Used by MediaPipe — needs original image size to detect joints accurately.
    """
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start = int(total * 0.20)
    end   = int(total * 0.80)
    usable = max(end - start, 1)
    step  = max(1, usable // max_frames)

    for i in range(max_frames):
        idx = start + i * step
        if idx >= end:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames
