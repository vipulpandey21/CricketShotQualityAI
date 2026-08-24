"""
video_utils.py
Shared video processing utilities.
"""

import cv2
import numpy as np

# Tensorflow import made optional - only needed for extract_frames()
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def extract_frames(video_path: str, n_frames: int = 30, output_size: tuple = (224, 224)) -> np.ndarray:
    """
    Extract the classifier's input: n_frames CONSECUTIVE frames from the start
    of the video, resized with padding, as uint8 RGB. Shape (n_frames, H, W, 3).

    This deliberately mirrors `frames_from_video_file` in
    Notebooks/Cricket_Shot_Classification_EfNetB0.ipynb, which is how
    model_weights.h5 was trained: 30 consecutive frames from frame 0
    (frame_step=1), cast to uint8.

    It previously sampled 30 STRIDED frames from the middle 60% of the clip
    instead. That fed the model a different kind of input than it was trained
    on, and measurably wrecked its predictions on the demo clips:

        strided, middle 60%   top-1 32%   top-3 66%
        consecutive from 0    top-1 46%   top-3 88%

    Other placements were tried and are worse — starting 20% in gave 38/70,
    centring the window gave 34/60, and averaging predictions over several
    start offsets gave 36/72. Matching the training convention wins.

    Requires tensorflow for resizing.
    """
    if not TF_AVAILABLE:
        raise ImportError("tensorflow is required for extract_frames(). Use extract_raw_frames() instead.")

    def _format(frame):
        frame = tf.image.convert_image_dtype(frame, tf.uint8)
        return tf.image.resize_with_pad(frame, *output_size).numpy()

    result = []
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = cap.read()
    result.append(_format(frame) if ret
                  else np.zeros((*output_size, 3), dtype=np.uint8))

    for _ in range(n_frames - 1):
        ret, frame = cap.read()
        result.append(_format(frame) if ret else np.zeros_like(result[0]))

    cap.release()
    # BGR → RGB, and uint8 to match the dtype the model was trained against
    return np.array(result)[..., [2, 1, 0]].astype(np.uint8)


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
