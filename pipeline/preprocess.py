from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np


# output BGR channel order, shape (H, W, 3), dtype uint8
BGRImage = np.ndarray


def preprocess_image(
    source: Union[str, np.ndarray],
    target_width: int = 1200
) -> BGRImage:
    """
    Full pre-processing pipeline for a single figure.

    Parameters
    ----------
    source : str | np.ndarray
        File path to an image or a raw BGR numpy array.
    target_width : int
        Minimum width; images narrower than this are upscaled.

    Returns
    -------
    BGRImage
        Pre-processed image as (H, W, 3) uint8 BGR array.
    """
    img = _load(source)
    img = _upscale_if_needed(img, target_width=target_width)
    img = _apply_clahe(img)
    img = _deskew(img)
    return img


# Load helpers
def _load(source: Union[str, np.ndarray]) -> BGRImage:
    """Load the input into a BGR NumPy array."""
    if isinstance(source, np.ndarray):
        if source.ndim == 3 and source.shape[2] == 3:  # fixed: was source.shape()
            return source.copy()
        raise ValueError("Input array is not a valid BGR image.")
    elif isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"Could not read image from: {source}")
        return img
    else:
        raise TypeError("Source must be either numpy.ndarray or a file path string.")


def _upscale_if_needed(img: BGRImage, target_width: int = 1200) -> BGRImage:
    """Upscale low-resolution images using INTER_CUBIC."""
    h, w = img.shape[:2]  # fixed: was img[:2]
    if w >= target_width:
        return img
    scale = target_width / w
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _apply_clahe(img: BGRImage) -> BGRImage:
    """Apply CLAHE on the L-channel in LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def _deskew(img: BGRImage) -> BGRImage:
    """Correct small rotational misalignment using Probabilistic Hough."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 100)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=100, minLineLength=100, maxLineGap=10
    )
    if lines is None:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]           # fixed: was missing indentation
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))  # fixed: was outside loop
        if abs(angle) < 10:  # only near-horizontal lines for deskew
            angles.append(angle)

    if not angles:
        return img

    median_angle = np.median(angles)
    if abs(median_angle) < 0.3:  # skip trivial rotations
        return img

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -median_angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def load_image_from_bytes(data: bytes) -> BGRImage:
    """Decode an in-memory image byte buffer into a BGR NumPy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes.")
    return img
