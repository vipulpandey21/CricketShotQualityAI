"""
video_utils.py
Shared video processing utilities used across classifier, pose, and quality modules.
"""

import cv2
import numpy as np
import tensorflow as tf


def format_frame(frame: np.ndarray, output_size: tuple = (224, 224)) -> np.ndarray:
    """Resize and pad a single BGR frame to output_size, returns uint8 RGB numpy array."""
    frame_tensor = tf.image.convert_image_dtype(frame, tf.uint8)
    frame_tensor = tf.image.resize_with_pad(frame_tensor, *output_size)
    return frame_tensor.numpy()


def extract_frames(
    video_path: str,
    n_frames: int = 30,
    output_size: tuple = (224, 224),
    frame_step: int = 1,
) -> np.ndarray:
    """
    Extract n_frames from a video file with a given step between frames.

    Args:
        video_path:  Path to the video file.
        n_frames:    Number of frames to extract.
        output_size: (height, width) to resize each frame to.
        frame_step:  How many raw frames to skip between each extracted frame.

    Returns:
        np.ndarray of shape (n_frames, H, W, 3) in RGB order.
    """
    result = []
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = cap.read()
    if ret:
        result.append(format_frame(frame, output_size))
    else:
        result.append(np.zeros((*output_size, 3), dtype=np.uint8))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = cap.read()
        if ret:
            result.append(format_frame(frame, output_size))
        else:
            result.append(np.zeros_like(result[0]))

    cap.release()

    # BGR → RGB
    frames = np.array(result)[..., [2, 1, 0]]
    return frames


def extract_raw_frames(video_path: str, max_frames: int = 300) -> list:
    """
    Extract raw BGR frames (no resize) for pose estimation.

    Args:
        video_path: Path to the video file.
        max_frames: Maximum number of frames to extract.

    Returns:
        List of BGR numpy arrays.
    """
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames
